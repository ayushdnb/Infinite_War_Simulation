from __future__ import annotations
from dataclasses import dataclass
import collections
from typing import Dict, Optional, List, Tuple, TYPE_CHECKING

import torch

import config
from simulation.stats import SimulationStats
from engine.agent_registry import (
    AgentsRegistry,
    COL_ALIVE, COL_TEAM, COL_X, COL_Y, COL_HP, COL_ATK, COL_UNIT, COL_VISION, COL_HP_MAX, COL_AGENT_ID
)
from engine.ray_engine.raycast_32 import raycast32_firsthit
from engine.game.move_mask import build_mask, DIRS8
from engine.respawn import RespawnController, RespawnCfg
from engine.mapgen import Zones

from agent.ensemble import ensemble_forward
from agent.transformer_brain import TransformerBrain

if TYPE_CHECKING:
    from rl.ppo_runtime import PerAgentPPORuntime

try:
    from rl.ppo_runtime import PerAgentPPORuntime as _PerAgentPPORuntimeRT
except Exception:
    _PerAgentPPORuntimeRT = None


@dataclass
class TickMetrics:
    alive: int = 0
    moved: int = 0
    attacks: int = 0
    deaths: int = 0
    tick: int = 0
    cp_red_tick: float = 0.0
    cp_blue_tick: float = 0.0


class TickEngine:
    def __init__(self, registry: AgentsRegistry, grid: torch.Tensor,
                 stats: SimulationStats, zones: Optional[Zones] = None) -> None:
        self.registry = registry
        self.grid = grid
        self.stats = stats
        self.device = grid.device
        self.H, self.W = int(grid.size(1)), int(grid.size(2))
        self.respawner = RespawnController(RespawnCfg())
        self.agent_scores: Dict[int, float] = collections.defaultdict(float)
        self.zones: Optional[Zones] = zones
        self._z_heal: Optional[torch.Tensor] = None
        self._z_cp_masks: List[torch.Tensor] = []
        self._ensure_zone_tensors()
        self.DIRS8_dev = DIRS8.to(self.device)
        self._ACTIONS = int(getattr(config, "NUM_ACTIONS", 41))
        self._OBS_DIM = config.OBS_DIM
        self._grid_dt = self.grid.dtype
        self._data_dt = self.registry.agent_data.dtype

        # ================================================================
        # Instinct cache (computed under no_grad) — NEW
        # ================================================================
        self._instinct_cached_r: int = -999999
        self._instinct_offsets: Optional[torch.Tensor] = None  # (M,2) long dx,dy within radius
        self._instinct_area: float = 1.0

        self._g0 = torch.tensor(0.0, device=self.device, dtype=self._grid_dt)
        self._gneg = torch.tensor(-1.0, device=self.device, dtype=self._grid_dt)
        self._d0 = torch.tensor(0.0, device=self.device, dtype=self._data_dt)
        self._ppo_enabled = bool(getattr(config, "PPO_ENABLED", False))
        self._ppo: Optional["PerAgentPPORuntime"] = None
        if self._ppo_enabled and _PerAgentPPORuntimeRT is not None:
            self._ppo = _PerAgentPPORuntimeRT(
                registry=self.registry, device=self.device,
                obs_dim=self._OBS_DIM, act_dim=self._ACTIONS,
            )

    def _ppo_reset_on_respawn(self, was_dead: torch.Tensor) -> None:
        """Reset per-slot PPO state for any slot that was dead before respawn and is alive after."""
        if self._ppo is None:
            return
        data = self.registry.agent_data
        now_alive = (data[:, COL_ALIVE] > 0.5)
        spawned_slots = (was_dead & now_alive).nonzero(as_tuple=False).squeeze(1)
        if spawned_slots.numel() == 0:
            return
        self._ppo.reset_agents(spawned_slots)
        if bool(getattr(config, "PPO_RESET_LOG", False)):
            # Keep logs short to avoid spam; show up to 16 slots.
            sl = spawned_slots[:16].tolist()
            suffix = "" if spawned_slots.numel() <= 16 else "..."
            print(f"[ppo] reset state for {int(spawned_slots.numel())} respawned slots: {sl}{suffix}")

    def _ensure_zone_tensors(self) -> None:
        self._z_heal, self._z_cp_masks = None, []
        if self.zones is None: return
        try:
            if getattr(self.zones, "heal_mask", None) is not None:
                self._z_heal = self.zones.heal_mask.to(self.device, non_blocking=True).bool()
            self._z_cp_masks = [m.to(self.device, non_blocking=True).bool() for m in getattr(self.zones, "cp_masks", [])]
        except Exception as e:
            print(f"[tick] WARN: zone tensor setup failed ({e}); zones disabled.")

    @staticmethod
    def _as_long(x: torch.Tensor) -> torch.Tensor: return x.to(torch.long)

    def _recompute_alive_idx(self) -> torch.Tensor:
        return (self.registry.agent_data[:, COL_ALIVE] > 0.5).nonzero(as_tuple=False).squeeze(1)

    @torch.no_grad()
    def _get_instinct_offsets(self) -> Tuple[torch.Tensor, float]:
        """
        Returns cached integer (dx,dy) offsets inside a discrete circle of radius R (cells),
        plus the offset-count area used for density normalization.
        """
        R = int(getattr(config, "INSTINCT_RADIUS", 6))
        if R < 0:
            R = 0

        if self._instinct_offsets is None or self._instinct_cached_r != R:
            # Build once per radius change. Keep on engine device.
            if R == 0:
                offsets = torch.zeros((1, 2), device=self.device, dtype=torch.long)
            else:
                r = torch.arange(-R, R + 1, device=self.device, dtype=torch.long)
                dx, dy = torch.meshgrid(r, r, indexing="xy")  # dx: (S,S), dy: (S,S)
                mask = (dx * dx + dy * dy) <= (R * R)
                offsets = torch.stack([dx[mask], dy[mask]], dim=1).contiguous()  # (M,2)
                if offsets.numel() == 0:
                    offsets = torch.zeros((1, 2), device=self.device, dtype=torch.long)

            self._instinct_offsets = offsets
            self._instinct_area = float(int(offsets.size(0)))
            self._instinct_cached_r = R

        return self._instinct_offsets, self._instinct_area

    @torch.no_grad()
    def _compute_instinct_context(
        self,
        alive_idx: torch.Tensor,
        pos_xy: torch.Tensor,
        unit_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Instinct token (4 floats) per alive agent:
          1) ally_archer_density
          2) ally_soldier_density
          3) noisy_enemy_density
          4) threat_ratio = enemy_density / (ally_total_density + eps)
        Densities are counts / area, where area = number of discrete cells in the radius mask.
        """
        N = int(alive_idx.numel())
        if N == 0:
            return torch.empty((0, 4), device=self.device, dtype=self._data_dt)

        data = self.registry.agent_data
        offsets, area = self._get_instinct_offsets()
        M = int(offsets.size(0))
        if M <= 0 or area <= 0.0:
            return torch.zeros((N, 4), device=self.device, dtype=self._data_dt)

        # Broadcasted neighborhood coords (N,M)
        x0 = pos_xy[:, 0].to(torch.long).view(N, 1)
        y0 = pos_xy[:, 1].to(torch.long).view(N, 1)
        ox = offsets[:, 0].view(1, M)
        oy = offsets[:, 1].view(1, M)
        xx = (x0 + ox).clamp(0, self.W - 1)
        yy = (y0 + oy).clamp(0, self.H - 1)

        occ = self.grid[0][yy, xx]      # (N,M) float, 0 empty, 1 wall, 2 red, 3 blue
        uid = unit_map[yy, xx]          # (N,M) long/int, -1 none, 1 soldier, 2 archer

        teams = data[alive_idx, COL_TEAM]  # (N,) float: 2.0 red, 3.0 blue
        team_is_red = (teams == 2.0)
        ally_occ = torch.where(team_is_red, occ.new_full((N,), 2.0), occ.new_full((N,), 3.0)).view(N, 1)
        enemy_occ = torch.where(team_is_red, occ.new_full((N,), 3.0), occ.new_full((N,), 2.0)).view(N, 1)

        ally_mask = (occ == ally_occ)
        enemy_mask = (occ == enemy_occ)

        ally_arch = ally_mask & (uid == 2)
        ally_sold = ally_mask & (uid == 1)

        ally_arch_c = ally_arch.sum(dim=1).to(torch.float32)
        ally_sold_c = ally_sold.sum(dim=1).to(torch.float32)
        enemy_c = enemy_mask.sum(dim=1).to(torch.float32)

        # Exclude self cell (offset (0,0) is included by construction).
        self_unit = data[alive_idx, COL_UNIT]  # (N,) float: 1 soldier, 2 archer
        ally_arch_c = (ally_arch_c - (self_unit == 2.0).to(torch.float32)).clamp_min(0.0)
        ally_sold_c = (ally_sold_c - (self_unit == 1.0).to(torch.float32)).clamp_min(0.0)

        # Add small noise to enemy count (requirement).
        noise = torch.randn((N,), device=self.device, dtype=torch.float32) * 0.25
        enemy_c_noisy = (enemy_c + noise).clamp_min(0.0)

        inv_area = 1.0 / float(area)
        ally_arch_d = ally_arch_c * inv_area
        ally_sold_d = ally_sold_c * inv_area
        enemy_d = enemy_c_noisy * inv_area

        eps = 1e-4 if self._data_dt == torch.float16 else 1e-6
        ally_total_d = ally_arch_d + ally_sold_d
        threat = enemy_d / (ally_total_d + eps)

        out = torch.stack([ally_arch_d, ally_sold_d, enemy_d, threat], dim=1)
        return out.to(dtype=self._data_dt)

    def _apply_deaths(self, sel: torch.Tensor, metrics: TickMetrics, credit_kills: bool = True) -> Tuple[int, int]:
        data = self.registry.agent_data
        dead_idx = sel.nonzero(as_tuple=False).squeeze(1) if sel.dtype == torch.bool else sel.view(-1)
        if dead_idx.numel() == 0:
            return 0, 0

        dead_team = data[dead_idx, COL_TEAM]
        red_deaths = int((dead_team == 2.0).sum().item())
        blue_deaths = int((dead_team == 3.0).sum().item())

        if red_deaths:
            self.stats.add_death("red", red_deaths)
            if credit_kills:
                self.stats.add_kill("blue", red_deaths)

        if blue_deaths:
            self.stats.add_death("blue", blue_deaths)
            if credit_kills:
                self.stats.add_kill("red", blue_deaths)

        gx, gy = self._as_long(data[dead_idx, COL_X]), self._as_long(data[dead_idx, COL_Y])
        self.grid[0][gy, gx], self.grid[1][gy, gx], self.grid[2][gy, gx] = self._g0, self._g0, self._gneg
        data[dead_idx, COL_ALIVE] = self._d0
        metrics.deaths += int(dead_idx.numel())
        return red_deaths, blue_deaths

    @torch.no_grad()
    def _build_transformer_obs(self, alive_idx: torch.Tensor, pos_xy: torch.Tensor) -> torch.Tensor:
        from engine.ray_engine.raycast_firsthit import build_unit_map
        data = self.registry.agent_data
        N = alive_idx.numel()

        # --- Zone flags for rich features ---
        if self._z_heal is not None:
            on_heal = self._z_heal[pos_xy[:, 1], pos_xy[:, 0]]
        else:
            on_heal = torch.zeros(N, device=self.device, dtype=torch.bool)

        on_cp = torch.zeros(N, device=self.device, dtype=torch.bool)
        if self._z_cp_masks:
            for cp_mask in self._z_cp_masks:
                on_cp |= cp_mask[pos_xy[:, 1], pos_xy[:, 0]]

        def _norm_const(v: float, scale: float) -> torch.Tensor:
            s = scale if scale > 0 else 1.0
            return torch.full((N,), v / s, dtype=self._data_dt, device=self.device)

        expected_ray_dim = 32 * 8
        unit_map = build_unit_map(data, self.grid)
        rays = raycast32_firsthit(
            pos_xy, self.grid, unit_map,
            max_steps_each=data[alive_idx, COL_VISION].long()
        )
        if rays.shape != (N, expected_ray_dim):
            raise RuntimeError(
                f"[obs] ray tensor shape mismatch: got {tuple(rays.shape)}, "
                f"expected ({N}, {expected_ray_dim})."
            )

        hp_max = data[alive_idx, COL_HP_MAX].clamp_min(1.0)

        rich_base = torch.stack([
            data[alive_idx, COL_HP] / hp_max,
            data[alive_idx, COL_X] / (self.W - 1),
            data[alive_idx, COL_Y] / (self.H - 1),
            (data[alive_idx, COL_TEAM] == 2.0),
            (data[alive_idx, COL_TEAM] == 3.0),
            (data[alive_idx, COL_UNIT] == 1.0),
            (data[alive_idx, COL_UNIT] == 2.0),
            data[alive_idx, COL_ATK] / (config.MAX_ATK or 1.0),
            data[alive_idx, COL_VISION] / (config.RAYCAST_MAX_STEPS or 15.0),
            on_heal.to(self._data_dt),
            on_cp.to(self._data_dt),
            _norm_const(float(self.stats.tick), 50000.0),
            _norm_const(self.stats.red.score, 1000.0), _norm_const(self.stats.blue.score, 1000.0),
            _norm_const(self.stats.red.cp_points, 500.0), _norm_const(self.stats.blue.cp_points, 500.0),
            _norm_const(self.stats.red.kills, 500.0), _norm_const(self.stats.blue.kills, 500.0),
            _norm_const(self.stats.red.deaths, 500.0), _norm_const(self.stats.blue.deaths, 500.0),
            # Padding to preserve RICH_BASE_DIM=23 layout (reserved slots).
            torch.zeros(N, device=self.device, dtype=self._data_dt),
            torch.zeros(N, device=self.device, dtype=self._data_dt),
            torch.zeros(N, device=self.device, dtype=self._data_dt),
        ], dim=1).to(dtype=self._data_dt)

        instinct = self._compute_instinct_context(alive_idx=alive_idx, pos_xy=pos_xy, unit_map=unit_map)
        # Hard invariant
        if instinct.shape != (N, 4):
            raise RuntimeError(f"instinct shape {tuple(instinct.shape)} != (N,4)")

        rich = torch.cat([rich_base, instinct], dim=1)

        expected_rich_dim = int(self._OBS_DIM) - expected_ray_dim
        if rich.shape != (N, expected_rich_dim):
            raise RuntimeError(
                f"[obs] rich tensor shape mismatch: got {tuple(rich.shape)}, "
                f"expected ({N}, {expected_rich_dim})."
            )

        obs = torch.cat([rays, rich.to(rays.dtype)], dim=1)
        if obs.shape != (N, int(self._OBS_DIM)):
            raise RuntimeError(
                f"[obs] final obs shape mismatch: got {tuple(obs.shape)}, "
                f"expected ({N}, {int(self._OBS_DIM)})."
            )
        return obs

    @torch.no_grad()
    def run_tick(self) -> Dict[str, float]:
        data = self.registry.agent_data
        metrics = TickMetrics()
        alive_idx = self._recompute_alive_idx()
        if alive_idx.numel() == 0:
            self.stats.on_tick_advanced(1)
            metrics.tick = int(self.stats.tick)
            was_dead = (data[:, COL_ALIVE] <= 0.5) if self._ppo is not None else None
            self.respawner.step(self.stats.tick, self.registry, self.grid)
            if was_dead is not None:
                self._ppo_reset_on_respawn(was_dead)
            return vars(metrics)

        pos_xy = self.registry.positions_xy(alive_idx)
        obs = self._build_transformer_obs(alive_idx, pos_xy)
        # ABSOLUTE invariant: obs width must match config.OBS_DIM
        if obs.dim() != 2 or int(obs.shape[1]) != int(config.OBS_DIM):
            raise RuntimeError(
                f"[obs] shape mismatch: got {tuple(obs.shape)}, expected (N,{int(config.OBS_DIM)})"
            )
        mask = build_mask(pos_xy, data[alive_idx, COL_TEAM], self.grid, unit=self._as_long(data[alive_idx, COL_UNIT]))
        actions = torch.zeros_like(alive_idx, dtype=torch.long)
        rec_agent_ids, rec_obs, rec_logits, rec_values, rec_actions, rec_teams = [], [], [], [], [], []

        for bucket in self.registry.build_buckets(alive_idx):
            loc = torch.searchsorted(alive_idx, bucket.indices)
            dist, vals = ensemble_forward(bucket.models, obs[loc])
            logits32 = torch.where(mask[loc], dist.logits, torch.finfo(torch.float32).min).to(torch.float32)
            a = torch.distributions.Categorical(logits=logits32).sample()
            if self._ppo:
                rec_agent_ids.append(bucket.indices)
                rec_obs.append(obs[loc])
                rec_logits.append(logits32)
                rec_values.append(vals)
                rec_actions.append(a)
                rec_teams.append(data[bucket.indices, COL_TEAM])
            actions[loc] = a

        metrics.alive = int(alive_idx.numel())

        # ----- MOVE HANDLING WITH CONFLICT RESOLUTION (Law 1) -----
        is_move = (actions >= 1) & (actions <= 8)
        if is_move.any():
            move_idx, dir_idx = alive_idx[is_move], actions[is_move] - 1
            x0, y0 = pos_xy[is_move].T
            nx, ny = (x0 + self.DIRS8_dev[dir_idx, 0]).clamp(0, self.W - 1), (y0 + self.DIRS8_dev[dir_idx, 1]).clamp(0, self.H - 1)
            can_move = (self.grid[0][ny, nx] == self._g0)
            if can_move.any():
                move_idx, x0, y0, nx, ny = move_idx[can_move], x0[can_move], y0[can_move], nx[can_move], ny[can_move]

                # -------------------- MOVE CONFLICT RESOLUTION (Law 1) --------------------
                # Candidates are already filtered by can_move (destination cell empty).
                # If multiple candidates target the same destination in the same tick:
                #   winner = highest HP; tie for highest HP -> nobody moves to that cell.
                dest_key = (ny * self.W + nx).to(torch.long)   # (M,) unique cell id
                hp = data[move_idx, COL_HP]                   # (M,) HP used for winner rule

                # Fast path (vectorized): per-destination max HP + count of max-HP claimants.
                # Deterministic: ties never pick a random winner; they block the move.
                try:
                    num_cells = self.H * self.W
                    max_hp = torch.full((num_cells,), torch.finfo(hp.dtype).min, device=self.device, dtype=hp.dtype)
                    max_hp.scatter_reduce_(0, dest_key, hp, reduce="amax", include_self=True)
                    is_max = (hp == max_hp[dest_key])
                    max_cnt = torch.zeros((num_cells,), device=self.device, dtype=torch.int32)
                    max_cnt.scatter_add_(0, dest_key, is_max.to(torch.int32))
                    winner_mask = is_max & (max_cnt[dest_key] == 1)
                except Exception:
                    # Fallback: deterministic group scan after sorting by destination.
                    # Only used if scatter_reduce_ is unavailable in the runtime Torch build.
                    winner_mask = torch.zeros_like(dest_key, dtype=torch.bool)
                    order = torch.argsort(dest_key)
                    dest_s = dest_key[order]
                    hp_s = hp[order]
                    if dest_s.numel() > 0:
                        starts = torch.cat([
                            torch.zeros(1, device=self.device, dtype=torch.long),
                            (dest_s[1:] != dest_s[:-1]).nonzero(as_tuple=False).squeeze(1) + 1
                        ])
                        ends = torch.cat([starts[1:], torch.tensor([dest_s.numel()], device=self.device, dtype=torch.long)])
                        for s, e in zip(starts.tolist(), ends.tolist()):
                            group_hp = hp_s[s:e]
                            m = group_hp.max()
                            is_m = (group_hp == m)
                            if int(is_m.sum().item()) == 1:
                                win_off = int(is_m.nonzero(as_tuple=False)[0].item()) + s
                                winner_mask[order[win_off]] = True

                if winner_mask.any():
                    w_move_idx = move_idx[winner_mask]
                    w_x0, w_y0, w_nx, w_ny = x0[winner_mask], y0[winner_mask], nx[winner_mask], ny[winner_mask]

                    # Commit movement ONLY for winners; losers keep their original cells.
                    self.grid[0, w_y0, w_x0], self.grid[1, w_y0, w_x0], self.grid[2, w_y0, w_x0] = self._g0, self._g0, self._gneg
                    data[w_move_idx, COL_X], data[w_move_idx, COL_Y] = w_nx.to(self._data_dt), w_ny.to(self._data_dt)
                    self.grid[0, w_ny, w_nx] = data[w_move_idx, COL_TEAM].to(self._grid_dt)
                    self.grid[1, w_ny, w_nx] = data[w_move_idx, COL_HP].to(self._grid_dt)
                    self.grid[2, w_ny, w_nx] = w_move_idx.to(self._grid_dt)

                    # Count *actual* movement winners (not just candidates).
                    metrics.moved = int(w_move_idx.numel())

                    # Optional debug-only invariant checks (default off; no cost unless enabled).
                    # Enable with: FWS_DEBUG_MOVE=1
                    import os
                    if os.getenv("FWS_DEBUG_MOVE", "0") in {"1", "true", "True"}:
                        # Each alive slot should appear exactly once in grid[2].
                        ids = self._as_long(self.grid[2]).view(-1)
                        present = ids[ids >= 0]
                        counts = torch.bincount(present, minlength=self.registry.capacity)
                        alive_slots = (data[:, COL_ALIVE] > 0.5).nonzero(as_tuple=False).squeeze(1)
                        bad = alive_slots[counts[alive_slots] != 1]
                        if bad.numel() > 0:
                            sl = bad[:16].tolist()
                            suffix = "" if bad.numel() <= 16 else "..."
                            raise RuntimeError(f"[move invariant] alive slots not exactly once in grid: {sl}{suffix}")
        # ----- END MOVE HANDLING -----

        combat_rd, combat_bd = 0, 0
        meta_rd, meta_bd = 0, 0

        individual_rewards = torch.zeros(self.registry.capacity, device=self.device, dtype=self._data_dt)

        if (is_attack := actions >= 9).any():
            atk_idx, atk_act = alive_idx[is_attack], actions[is_attack]
            r, dir_idx = ((atk_act - 9) % 4) + 1, (atk_act - 9) // 4
            dxy = self.DIRS8_dev[dir_idx] * r.unsqueeze(1)
            ax, ay = pos_xy[is_attack].T
            tx, ty = (ax + dxy[:, 0]).clamp(0, self.W - 1), (ay + dxy[:, 1]).clamp(0, self.H - 1)
            victims = self._as_long(self.grid[2][ty, tx])
            if (valid_hit := victims >= 0).any():
                atk_idx, victims = atk_idx[valid_hit], victims[valid_hit]
                if (is_enemy := data[atk_idx, COL_TEAM] != data[victims, COL_TEAM]).any():
                    atk_idx, victims = atk_idx[is_enemy], victims[is_enemy]
                    was_alive = data[victims, COL_HP] > 0
                    data[victims, COL_HP] -= data[atk_idx, COL_ATK]
                    now_dead = data[victims, COL_HP] <= 0
                    if (killers := atk_idx[was_alive & now_dead]).numel() > 0:
                        reward_val = float(config.PPO_REWARD_KILL_INDIVIDUAL)
                        individual_rewards.index_add_(0, killers, torch.full_like(killers, reward_val, dtype=self._data_dt))
                        for killer_slot in killers:
                            uid = int(data[killer_slot.item(), COL_AGENT_ID].item())
                            self.agent_scores[uid] += reward_val
                    vy, vx = self._as_long(data[victims, COL_Y]), self._as_long(data[victims, COL_X])
                    self.grid[1, vy, vx] = data[victims, COL_HP].to(self._grid_dt)
                    metrics.attacks += int(is_enemy.sum().item())

        rD, bD = self._apply_deaths((data[:, COL_ALIVE] > 0.5) & (data[:, COL_HP] <= 0.0), metrics)
        combat_rd += rD
        combat_bd += bD

        if (alive_idx := self._recompute_alive_idx()).numel() > 0:
            pos_xy = self.registry.positions_xy(alive_idx)
            if self._z_heal is not None and (on_heal := self._z_heal[pos_xy[:, 1], pos_xy[:, 0]]).any():
                heal_idx = alive_idx[on_heal]
                data[heal_idx, COL_HP] = (data[heal_idx, COL_HP] + config.HEAL_RATE).clamp_max(data[heal_idx, COL_HP_MAX])
                self.grid[1, pos_xy[on_heal, 1], pos_xy[on_heal, 0]] = data[heal_idx, COL_HP].to(self._grid_dt)

            if meta_drain := getattr(config, "METABOLISM_ENABLED", True):
                drain = torch.where(data[alive_idx, COL_UNIT] == 1.0, config.META_SOLDIER_HP_PER_TICK, config.META_ARCHER_HP_PER_TICK)
                data[alive_idx, COL_HP] -= drain.to(self._data_dt)
                self.grid[1, pos_xy[:, 1], pos_xy[:, 0]] = data[alive_idx, COL_HP].to(self._grid_dt)
                if (data[alive_idx, COL_HP] <= 0.0).any():
                    rD, bD = self._apply_deaths(
                        alive_idx[data[alive_idx, COL_HP] <= 0.0],
                        metrics,
                        credit_kills=False,
                    )
                    meta_rd += rD
                    meta_bd += bD

            if self._z_cp_masks and (alive_idx := self._recompute_alive_idx()).numel() > 0:
                pos_xy, teams_alive = self.registry.positions_xy(alive_idx), data[alive_idx, COL_TEAM]
                for cp_mask in self._z_cp_masks:
                    if (on_cp := cp_mask[pos_xy[:, 1], pos_xy[:, 0]]).any():
                        red_on = (on_cp & (teams_alive == 2.0)).sum().item()
                        blue_on = (on_cp & (teams_alive == 3.0)).sum().item()
                        if red_on > blue_on:
                            self.stats.add_capture_points("red", config.CP_REWARD_PER_TICK)
                            metrics.cp_red_tick += config.CP_REWARD_PER_TICK
                        elif blue_on > red_on:
                            self.stats.add_capture_points("blue", config.CP_REWARD_PER_TICK)
                            metrics.cp_blue_tick += config.CP_REWARD_PER_TICK

                        # Individual reward for agents on a contested CP
                        if red_on > 0 and blue_on > 0:
                            winners_on_cp = None
                            if red_on > blue_on:
                                winners_on_cp = on_cp & (teams_alive == 2.0)
                            elif blue_on > red_on:
                                winners_on_cp = on_cp & (teams_alive == 3.0)
                            if winners_on_cp is not None and winners_on_cp.any():
                                winners_idx = alive_idx[winners_on_cp]
                                reward_val = config.PPO_REWARD_CONTESTED_CP
                                individual_rewards.index_add_(
                                    0,
                                    winners_idx,
                                    torch.full_like(winners_idx, reward_val, dtype=self._data_dt),
                                )

        if self._ppo and rec_agent_ids:
            agent_ids = torch.cat(rec_agent_ids)
            team_r_rew = (combat_bd * config.TEAM_KILL_REWARD) + ((combat_rd + meta_rd) * config.PPO_REWARD_DEATH) + metrics.cp_red_tick
            team_b_rew = (combat_rd * config.TEAM_KILL_REWARD) + ((combat_bd + meta_bd) * config.PPO_REWARD_DEATH) + metrics.cp_blue_tick

            current_hp = data[agent_ids, COL_HP]
            hp_reward = (current_hp * config.PPO_REWARD_HP_TICK).to(self._data_dt)
            final_rewards = individual_rewards[agent_ids] + torch.where(torch.cat(rec_teams) == 2.0, team_r_rew, team_b_rew) + hp_reward

            with torch.enable_grad():
                self._ppo.record_step(
                    agent_ids=agent_ids,
                    obs=torch.cat(rec_obs),
                    logits=torch.cat(rec_logits),
                    values=torch.cat(rec_values),
                    actions=torch.cat(rec_actions),
                    rewards=final_rewards,
                    done=(data[agent_ids, COL_ALIVE] <= 0.5)
                )

        self.stats.on_tick_advanced(1)
        metrics.tick = int(self.stats.tick)
        was_dead = (data[:, COL_ALIVE] <= 0.5) if self._ppo is not None else None
        self.respawner.step(self.stats.tick, self.registry, self.grid)
        if was_dead is not None:
            self._ppo_reset_on_respawn(was_dead)
        return vars(metrics)