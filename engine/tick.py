from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, TYPE_CHECKING

import torch

import config
from simulation.stats import SimulationStats
from engine.agent_registry import (
    AgentsRegistry,
    COL_ALIVE, COL_TEAM, COL_X, COL_Y, COL_HP, COL_ATK, COL_UNIT, COL_VISION, COL_HP_MAX
)
from engine.ray_engine.raycast_64 import raycast64_firsthit
from engine.game.move_mask import build_mask, DIRS8
from engine.respawn import RespawnController
from engine.mapgen import Zones

from agent.ensemble import ensemble_forward
from agent.transformer_brain import TransformerBrain  # kept for compatibility

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
    """
    Per-tick simulation core using the TransformerBrain and ensemble forward.

    Handles all primary game logic including:
    - Agent observation building (V2: 64 rays + 21 rich features)
    - Action sampling via batched model inference
    - Movement and Combat resolution
    - Post-action effects: Healing, Metabolism, and Capture Point scoring
    - PPO data recording with team-based reward shaping
    """
    def __init__(self, registry: AgentsRegistry, grid: torch.Tensor,
                 stats: SimulationStats, zones: Optional[Zones] = None) -> None:
        self.registry = registry
        self.grid = grid
        self.stats = stats
        self.device = grid.device
        self.H, self.W = int(grid.size(1)), int(grid.size(2))
        self.respawner = RespawnController()

        # Zones
        self.zones: Optional[Zones] = zones
        self._z_heal: Optional[torch.Tensor] = None
        self._z_cp_masks: List[torch.Tensor] = []
        self._ensure_zone_tensors()

        # Directions on device
        self.DIRS8_dev = DIRS8.to(self.device)

        # Config cache
        self._ACTIONS = int(getattr(config, "NUM_ACTIONS", 41))
        self._OBS_DIM = (64 * 8) + 21  # rays (64*8) + rich features (21)
        self._MAX_HP = float(getattr(config, "MAX_HP", 1.0))
        self._HEAL_RATE = float(getattr(config, "HEAL_RATE", 0.02))
        self._CP_REWARD = float(getattr(config, "CP_REWARD_PER_TICK", 0.05)) # Tuned from 5.0
        self._LOS_BLOCKS = bool(getattr(config, "ARCHER_LOS_BLOCKS_WALLS", False))

        self._META_ON = bool(getattr(config, "METABOLISM_ENABLED", True))
        self._META_SOLDIER = float(getattr(config, "META_SOLDIER_HP_PER_TICK", 0.0001))
        self._META_ARCHER = float(getattr(config, "META_ARCHER_HP_PER_TICK", 0.00001))

        # Dtype helpers
        self._grid_dt = self.grid.dtype
        self._data_dt = self.registry.agent_data.dtype
        # Typed scalars for grid & data assignments
        self._g0 = torch.tensor(0.0, device=self.device, dtype=self._grid_dt)
        self._gneg = torch.tensor(-1.0, device=self.device, dtype=self._grid_dt)
        self._d0 = torch.tensor(0.0, device=self.device, dtype=self._data_dt)

        # PPO runtime (optional)
        self._ppo_enabled = bool(getattr(config, "PPO_ENABLED", False))
        self._ppo: Optional["PerAgentPPORuntime"] = None
        if self._ppo_enabled and _PerAgentPPORuntimeRT is not None:
            self._ppo = _PerAgentPPORuntimeRT(
                registry=self.registry,
                device=self.device,
                obs_dim=self._OBS_DIM,
                act_dim=self._ACTIONS,
            )

    # -------------------------- internals & utils --------------------------

    def _ensure_zone_tensors(self) -> None:
        self._z_heal, self._z_cp_masks = None, []
        if self.zones is None:
            return
        try:
            if getattr(self.zones, "heal_mask", None) is not None:
                self._z_heal = self.zones.heal_mask.to(self.device, non_blocking=True).bool()
            self._z_cp_masks = [m.to(self.device, non_blocking=True).bool()
                                for m in getattr(self.zones, "cp_masks", [])]
        except Exception as e:
            print(f"[tick] WARN: zone tensor setup failed ({e}); zones disabled.")
            self._z_heal, self._z_cp_masks = None, []

    @staticmethod
    def _as_long(x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.long)

    def _recompute_alive_idx(self) -> torch.Tensor:
        return (self.registry.agent_data[:, COL_ALIVE] > 0.5).nonzero(as_tuple=False).squeeze(1)

    def _apply_deaths(self, sel: torch.Tensor, metrics: TickMetrics) -> Tuple[int, int]:
        """
        Accepts either a boolean mask over all agents OR a 1-D tensor of indices to kill.
        Clears grid cells, marks agents dead, updates stats.
        """
        data = self.registry.agent_data

        # Normalize selector
        if sel.dtype == torch.bool:
            dead_idx = sel.nonzero(as_tuple=False).squeeze(1)
        else:
            # assume indices tensor
            dead_idx = sel.view(-1)

        if dead_idx.numel() == 0:
            return 0, 0

        # Count team deaths
        dead_team = data[dead_idx, COL_TEAM]
        red_deaths_tick = int((dead_team == 2.0).sum().item())
        blue_deaths_tick = int((dead_team == 3.0).sum().item())

        if red_deaths_tick:
            self.stats.add_death("red", red_deaths_tick)
            self.stats.add_kill("blue", red_deaths_tick)
        if blue_deaths_tick:
            self.stats.add_death("blue", blue_deaths_tick)
            self.stats.add_kill("red", blue_deaths_tick)

        # Clear grid cells with typed scalars
        gx = self._as_long(data[dead_idx, COL_X])
        gy = self._as_long(data[dead_idx, COL_Y])
        self.grid[0][gy, gx] = self._g0
        self.grid[1][gy, gx] = self._g0
        self.grid[2][gy, gx] = self._gneg

        # Mark agents dead
        data[dead_idx, COL_ALIVE] = self._d0

        metrics.deaths += int(dead_idx.numel())
        return red_deaths_tick, blue_deaths_tick

    # ------------------------- observations for brain -------------------------

    @torch.no_grad()
    def _build_transformer_obs(self, alive_idx: torch.Tensor) -> torch.Tensor:
        """
        Builds observations for the TransformerBrain.
        Ensures a consistent dtype for concatenation regardless of AMP.
        """
        from engine.ray_engine.raycast_firsthit import build_unit_map
        data = self.registry.agent_data
        pos_xy = self.registry.positions_xy(alive_idx)

        # vision per agent from data (int steps)
        max_steps_each = data[alive_idx, COL_VISION].to(self.device, dtype=torch.long)

        unit_map = build_unit_map(data, self.grid)
        rays = raycast64_firsthit(pos_xy, self.grid, unit_map, max_steps_each=max_steps_each)
        obs_dt = rays.dtype  # keep obs dtype consistent

        N = alive_idx.numel()
        x = (data[alive_idx, COL_X] / float(self.W - 1)).to(dtype=obs_dt)
        y = (data[alive_idx, COL_Y] / float(self.H - 1)).to(dtype=obs_dt)

        hp = data[alive_idx, COL_HP]
        hp_max = data[alive_idx, COL_HP_MAX].clamp_min(1.0)
        hp_norm = (hp / hp_max).to(dtype=obs_dt)

        team = data[alive_idx, COL_TEAM]
        unit = data[alive_idx, COL_UNIT]

        team_red = (team == 2.0).to(obs_dt)
        team_blue = (team == 3.0).to(obs_dt)
        unit_sold = (unit == 1.0).to(obs_dt)
        unit_arch = (unit == 2.0).to(obs_dt)

        # Agent's own stats
        own_atk = data[alive_idx, COL_ATK]
        own_vision = data[alive_idx, COL_VISION]
        max_atk = float(getattr(config, "MAX_ATK", 1.0)) or 1.0
        max_vision = float(getattr(config, "RAYCAST_MAX_STEPS", 15)) or 15.0

        own_atk_norm = (own_atk / max_atk).to(obs_dt)
        own_vision_norm = (own_vision / max_vision).to(obs_dt)

        # Game state stats (broadcast as constants)
        red_score = float(self.stats.red.score)
        blue_score = float(self.stats.blue.score)

        def _norm_const(v: float, scale: float) -> torch.Tensor:
            s = scale if scale > 0 else 1.0
            return torch.full((N,), v / s, dtype=obs_dt, device=self.device)

        # Rich features vector (21 elements)
        rich_features = torch.stack([
            hp_norm, x, y,
            team_red, team_blue,
            unit_sold, unit_arch,
            own_atk_norm, own_vision_norm,
            _norm_const(float(self.stats.tick), 50000.0),  # time awareness
            _norm_const(red_score, 1000.0),
            _norm_const(blue_score, 1000.0),
            _norm_const(float(self.stats.red.cp_points), 500.0),
            _norm_const(float(self.stats.blue.cp_points), 500.0),
            _norm_const(float(self.stats.red.kills), 200.0),
            _norm_const(float(self.stats.blue.kills), 200.0),
            _norm_const(float(self.stats.red.deaths), 200.0),
            _norm_const(float(self.stats.blue.deaths), 200.0),
            torch.zeros(N, device=self.device, dtype=obs_dt),  # placeholder
            torch.zeros(N, device=self.device, dtype=obs_dt),  # placeholder
            torch.zeros(N, device=self.device, dtype=obs_dt),  # placeholder
        ], dim=1)

        return torch.cat([rays, rich_features], dim=1)

    # -------------------------------- run tick --------------------------------

    @torch.no_grad()
    def run_tick(self) -> Dict[str, float]:
        data = self.registry.agent_data
        metrics = TickMetrics()
        alive_idx = self._recompute_alive_idx()

        if alive_idx.numel() == 0:
            self.stats.on_tick_advanced(1)
            metrics.tick = int(self.stats.tick)
            self.respawner.step(self.stats.tick, self.registry, self.grid)
            return vars(metrics)

        teams = data[alive_idx, COL_TEAM]
        units = self._as_long(data[alive_idx, COL_UNIT])
        pos_xy = self.registry.positions_xy(alive_idx)

        # Observations + mask
        obs = self._build_transformer_obs(alive_idx)
        mask = build_mask(pos_xy, teams, self.grid, unit=units)

        # Action buffer
        actions = torch.zeros(alive_idx.numel(), device=self.device, dtype=torch.long)

        # PPO record buffers
        rec_agent_ids: List[torch.Tensor] = []
        rec_obs: List[torch.Tensor] = []
        rec_logits: List[torch.Tensor] = []
        rec_values: List[torch.Tensor] = []
        rec_actions: List[torch.Tensor] = []
        rec_teams: List[torch.Tensor] = []

        # Run per bucket
        buckets = self.registry.build_buckets(alive_idx)
        for bucket in buckets:
            loc = torch.searchsorted(alive_idx, bucket.indices)
            o, m = obs[loc], mask[loc]

            dist, vals = ensemble_forward(bucket.models, o)
            logits = dist.logits

            # Mask logits, then cast to float32 for Categorical stability
            neg_inf = torch.finfo(logits.dtype).min
            masked_logits = torch.where(m, logits, neg_inf)
            logits32 = masked_logits.to(torch.float32)

            cat = torch.distributions.Categorical(logits=logits32)
            a = cat.sample()

            if self._ppo:
                rec_agent_ids.append(bucket.indices.clone())
                rec_obs.append(o.detach().clone().to(torch.float32))
                rec_logits.append(logits32.detach().clone())
                rec_values.append(vals.detach().clone().to(torch.float32))
                rec_actions.append(a.detach().clone())
                rec_teams.append(teams[loc].detach().clone())

            actions[loc] = a

        metrics.alive = int(alive_idx.numel())

        # ------------------------------- movement -------------------------------
        is_move = (actions >= 1) & (actions <= 8)
        if torch.any(is_move):
            move_idx = alive_idx[is_move]
            dir_idx = (actions[is_move] - 1)
            dxy = self.DIRS8_dev[dir_idx]
            x0, y0 = pos_xy[is_move].T
            nx = (x0 + dxy[:, 0]).clamp(0, self.W - 1)
            ny = (y0 + dxy[:, 1]).clamp(0, self.H - 1)

            # Free cell?
            target_occ = self.grid[0][ny, nx]
            can_move = (target_occ == self._g0)

            if can_move.any():
                move_idx_final = move_idx[can_move]
                x0_final, y0_final = x0[can_move], y0[can_move]
                nx_final, ny_final = nx[can_move], ny[can_move]

                # Clear old pos (typed)
                self.grid[0][y0_final, x0_final] = self._g0
                self.grid[1][y0_final, x0_final] = self._g0
                self.grid[2][y0_final, x0_final] = self._gneg

                # Update registry (cast RHS to data dtype)
                data[move_idx_final, COL_X] = nx_final.to(self._data_dt)
                data[move_idx_final, COL_Y] = ny_final.to(self._data_dt)

                # Set new pos on grid (cast RHS to grid dtype)
                self.grid[0][ny_final, nx_final] = data[move_idx_final, COL_TEAM].to(self._grid_dt)
                self.grid[1][ny_final, nx_final] = data[move_idx_final, COL_HP].to(self._grid_dt)
                self.grid[2][ny_final, nx_final] = move_idx_final.to(self._grid_dt)

                metrics.moved = int(can_move.sum().item())

        # -------------------------------- combat --------------------------------
        combat_red_deaths, combat_blue_deaths = 0, 0
        is_attack = (actions >= 9) & (actions < self._ACTIONS)
        if torch.any(is_attack):
            atk_idx = alive_idx[is_attack]
            atk_act = actions[is_attack]

            rel = atk_act - 9
            dir_idx = rel // 4
            r = (rel % 4) + 1
            dxy = self.DIRS8_dev[dir_idx] * r.unsqueeze(1)

            ax, ay = pos_xy[is_attack].T
            tx = (ax + dxy[:, 0]).clamp(0, self.W - 1)
            ty = (ay + dxy[:, 1]).clamp(0, self.H - 1)

            victims = self._as_long(self.grid[2][ty, tx])
            valid_hit = (victims >= 0)
            if valid_hit.any():
                atk_idx_final = atk_idx[valid_hit]
                victims_final = victims[valid_hit]

                atk_team = data[atk_idx_final, COL_TEAM]
                vic_team = data[victims_final, COL_TEAM]
                is_enemy = (atk_team != vic_team)

                if is_enemy.any():
                    atk_idx_enemy = atk_idx_final[is_enemy]
                    victims_enemy = victims_final[is_enemy]

                    dmg = data[atk_idx_enemy, COL_ATK]
                    data[victims_enemy, COL_HP] -= dmg

                    # write back HP to grid with proper dtype
                    vy = data[victims_enemy, COL_Y].long()
                    vx = data[victims_enemy, COL_X].long()
                    self.grid[1][vy, vx] = data[victims_enemy, COL_HP].to(self._grid_dt)

                    metrics.attacks += int(is_enemy.sum().item())

                    # deaths from combat: compute mask over all agents, then apply
                    was_alive = (data[:, COL_ALIVE] > 0.5)
                    now_dead_mask = (data[:, COL_HP] <= 0.0) & was_alive
                    rD, bD = self._apply_deaths(now_dead_mask, metrics)
                    combat_red_deaths += rD
                    combat_blue_deaths += bD

        # ---------------------------- post-action phase --------------------------
        # Re-evaluate who is still alive after combat
        alive_idx = self._recompute_alive_idx()
        if alive_idx.numel() == 0:
            self.stats.on_tick_advanced(1)
            metrics.tick = int(self.stats.tick)
            self.respawner.step(self.stats.tick, self.registry, self.grid)
            return vars(metrics)

        # Current positions of survivors
        pos_xy = self.registry.positions_xy(alive_idx)

        # Healing
        if self._z_heal is not None:
            on_heal_mask = self._z_heal[pos_xy[:, 1].long(), pos_xy[:, 0].long()]
            if on_heal_mask.any():
                heal_idx = alive_idx[on_heal_mask]
                hp_max = data[heal_idx, COL_HP_MAX]
                data[heal_idx, COL_HP] = (data[heal_idx, COL_HP] + self._HEAL_RATE).clamp_max(hp_max)

                heal_pos = pos_xy[on_heal_mask]
                self.grid[1][heal_pos[:, 1].long(), heal_pos[:, 0].long()] = \
                    data[heal_idx, COL_HP].to(self._grid_dt)

        # Metabolism
        if self._META_ON:
            units_alive = data[alive_idx, COL_UNIT]
            drain = torch.where(units_alive == 1.0,
                                 torch.tensor(self._META_SOLDIER, device=self.device, dtype=self._data_dt),
                                 torch.tensor(self._META_ARCHER, device=self.device, dtype=self._data_dt))
            data[alive_idx, COL_HP] -= drain

            # Update grid HP
            self.grid[1][pos_xy[:, 1].long(), pos_xy[:, 0].long()] = \
                data[alive_idx, COL_HP].to(self._grid_dt)

            # Deaths from metabolism
            now_dead_from_meta = (data[alive_idx, COL_HP] <= 0.0)
            if now_dead_from_meta.any():
                dead_from_meta_idx = alive_idx[now_dead_from_meta]
                rD, bD = self._apply_deaths(dead_from_meta_idx, metrics)
                combat_red_deaths += rD
                combat_blue_deaths += bD

        # --- NEW SECTION: CAPTURE POINT SCORING ---
        if self._z_cp_masks:
            # Re-fetch alive agents in case of metabolism deaths
            alive_idx = self._recompute_alive_idx()
            if alive_idx.numel() > 0:
                pos_xy_after_meta = self.registry.positions_xy(alive_idx)
                teams_alive = data[alive_idx, COL_TEAM]
                is_red_alive = (teams_alive == 2.0)
                is_blue_alive = (teams_alive == 3.0)

                for cp_mask in self._z_cp_masks:
                    # Find which agents are on the current capture point
                    on_cp_mask = cp_mask[pos_xy_after_meta[:, 1].long(), pos_xy_after_meta[:, 0].long()]
                    
                    if on_cp_mask.any():
                        # Count red and blue agents on this point
                        red_on_cp = (on_cp_mask & is_red_alive).sum()
                        blue_on_cp = (on_cp_mask & is_blue_alive).sum()

                        if red_on_cp > blue_on_cp:
                            self.stats.add_capture_points("red", self._CP_REWARD)
                            metrics.cp_red_tick += self._CP_REWARD
                        elif blue_on_cp > red_on_cp:
                            self.stats.add_capture_points("blue", self._CP_REWARD)
                            metrics.cp_blue_tick += self._CP_REWARD
        # --- END OF NEW SECTION ---

        # PPO record and final updates
        if self._ppo and rec_agent_ids:
            R_kill = float(getattr(config, "PPO_REWARD_KILL", 1.0))
            R_death = float(getattr(config, "PPO_REWARD_DEATH", -0.3))
            R_cp = float(getattr(config, "CP_REWARD_PER_TICK", 0.05)) # Match CP_REWARD

            # Integrate CP points into the team reward
            team_reward_red = (combat_blue_deaths * R_kill) + (combat_red_deaths * R_death) + metrics.cp_red_tick
            team_reward_blue = (combat_red_deaths * R_kill) + (combat_blue_deaths * R_death) + metrics.cp_blue_tick

            agent_ids = torch.cat(rec_agent_ids)
            done_step = (self.registry.agent_data[agent_ids, COL_ALIVE] <= 0.5)

            # enable grads only for PPO bookkeeping, but inputs are detached already
            with torch.enable_grad():
                self._ppo.record_step(
                    agent_ids=agent_ids,
                    team_ids=torch.cat(rec_teams),
                    obs=torch.cat(rec_obs),
                    logits=torch.cat(rec_logits),
                    values=torch.cat(rec_values),
                    actions=torch.cat(rec_actions),
                    team_reward_red=team_reward_red,
                    team_reward_blue=team_reward_blue,
                    done=done_step,
                )

        # Advance time, respawn
        self.stats.on_tick_advanced(1)
        metrics.tick = int(self.stats.tick)
        self.respawner.step(self.stats.tick, self.registry, self.grid)

        return vars(metrics)