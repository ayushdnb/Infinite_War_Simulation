from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, TYPE_CHECKING

import torch

import config
from simulation.stats import SimulationStats, TEAM_RED, TEAM_BLUE
from .agent_registry import (
    AgentsRegistry, COL_ALIVE, COL_TEAM, COL_X, COL_Y, COL_HP, COL_ATK, COL_UNIT,
)
from .ray_engine.raycaster2d import raycast8 as raycast8_legacy
from .ray_engine.raycast_firsthit import raycast8_firsthit, build_unit_map
from .game.move_mask import build_mask, DIRS8
from .respawn import RespawnController
from .mapgen import Zones  # optional; all uses guarded

from agent.ensemble import ensemble_forward
from agent.mutation import pick_mutants, mutate_model_inplace

from .ego_frame import (
    compute_heading8,
    rotate_rays_to_heading,
    unrotate_logits_by_heading,
)

# --- PPO runtime: import for runtime, but type-check safely -------------------
if TYPE_CHECKING:
    from ..rl.ppo_runtime import PerAgentPPORuntime  # only for type checkers

try:
    from ..rl.ppo_runtime import PerAgentPPORuntime as _PerAgentPPORuntimeRT
except Exception:
    _PerAgentPPORuntimeRT = None  # runtime fallback


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
    Per-tick simulation core.

    - 17/41 actions, LoS, zones, metabolism, ensembles
    - V2 observations: 64 first-hit rays + 21 rich = 85
    - Ego rotation for rays; inverse rotation for directional logits
    - PPO per-agent with team-shaped rewards (kills, deaths, CP)
    """
    def __init__(self, registry: AgentsRegistry, grid: torch.Tensor, stats: SimulationStats, zones: Optional[Zones] = None) -> None:
        self.registry = registry
        self.grid = grid
        self.stats = stats
        self.device = grid.device
        self.H, self.W = grid.size(1), grid.size(2)
        self.respawner = RespawnController()

        self.zones: Optional[Zones] = zones
        self._z_heal: Optional[torch.Tensor] = None
        self._z_cp_masks: List[torch.Tensor] = []
        self._zones_valid = False
        self._ensure_zone_tensors()

        self.DIRS8_dev = DIRS8.to(self.device)

        self._ACTIONS = int(getattr(config, "NUM_ACTIONS", 17))
        self._OBS_DIM = int(getattr(config, "OBS_DIM", 64))
        self._MAX_HP = float(getattr(config, "MAX_HP", 1.0))
        self._HEAL_RATE = float(getattr(config, "HEAL_RATE", 0.02))
        self._CP_REWARD = float(getattr(config, "CP_REWARD_PER_TICK", 5.0))
        self._LOS_BLOCKS = bool(getattr(config, "ARCHER_LOS_BLOCKS_WALLS", False))
        self._PER_AGENT_BRAINS = bool(getattr(config, "PER_AGENT_BRAINS", False))
        self._MUT_PERIOD = int(getattr(config, "MUTATION_PERIOD_TICKS", 1000))
        self._MUT_FRAC = float(getattr(config, "MUTATION_FRACTION_ALIVE", 0.05))

        self._META_ON = bool(getattr(config, "METABOLISM_ENABLED", True))
        self._META_SOLDIER = float(getattr(config, "META_SOLDIER_HP_PER_TICK", 0.0020))
        self._META_ARCHER = float(getattr(config, "META_ARCHER_HP_PER_TICK", 0.0015))

        self._RELATIVE_DIRS = bool(getattr(config, "RELATIVE_DIRS", True))
        cap = self.registry.capacity
        self._last_pos_xy = torch.zeros((cap, 2), dtype=torch.long, device=self.device)
        self._heading8 = torch.zeros((cap,), dtype=torch.long, device=self.device)
        self._has_heading = torch.zeros((cap,), dtype=torch.bool, device=self.device)

        # PPO runtime (created only if enabled)
        self._ppo_enabled = bool(getattr(config, "PPO_ENABLED", False))
        self._ppo: Optional["PerAgentPPORuntime"] = None
        if self._ppo_enabled and _PerAgentPPORuntimeRT is not None:
            self._ppo = _PerAgentPPORuntimeRT(
                registry=self.registry,
                device=self.device,
                obs_dim=self._OBS_DIM,
                act_dim=self._ACTIONS,
            )

    # -------------------- Zones: device-safe cache --------------------
    def _ensure_zone_tensors(self) -> None:
        self._z_heal = None
        self._z_cp_masks = []
        self._zones_valid = False
        if self.zones is None:
            return
        try:
            H, W = self.H, self.W
            if getattr(self.zones, "heal_mask", None) is not None:
                z = self.zones.heal_mask.to(self.device, non_blocking=True).bool()
                if z.shape[-2:] == (H, W):
                    self._z_heal = z
                else:
                    print("[tick] WARN: heal_mask shape mismatch; disabling heal zone.")
            cps = []
            for m in getattr(self.zones, "cp_masks", []) or []:
                mm = m.to(self.device, non_blocking=True).bool()
                if mm.shape[-2:] == (H, W):
                    cps.append(mm)
                else:
                    print("[tick] WARN: cp mask shape mismatch; skipping one cp mask.")
            self._z_cp_masks = cps
            self._zones_valid = True
        except Exception as e:
            print(f"[tick] WARN: zone tensor setup failed ({type(e).__name__}: {e}); zones disabled.")
            self._z_heal = None
            self._z_cp_masks = []
            self._zones_valid = False

    @staticmethod
    def _as_long(x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.long)

    def _recompute_alive_idx(self) -> torch.Tensor:
        data = self.registry.agent_data
        return (data[:, COL_ALIVE] > 0.5).nonzero(as_tuple=False).squeeze(1)

    def _apply_deaths(self, now_dead_mask: torch.Tensor, metrics: TickMetrics,
                      killer_team_id_val: float | int | None = None) -> Tuple[int, int]:
        red_deaths_tick = 0
        blue_deaths_tick = 0
        if not torch.any(now_dead_mask):
            return red_deaths_tick, blue_deaths_tick

        data = self.registry.agent_data
        dead_idx = now_dead_mask.nonzero(as_tuple=False).squeeze(1)
        if dead_idx.numel() == 0:
            return red_deaths_tick, blue_deaths_tick

        dead_team = data[dead_idx, COL_TEAM]
        red_deaths_tick = int((dead_team == 2.0).sum().item())
        blue_deaths_tick = int((dead_team == 3.0).sum().item())

        if red_deaths_tick:
            self.stats.add_death(TEAM_RED, red_deaths_tick)
            self.stats.add_kill(TEAM_BLUE, red_deaths_tick)
        if blue_deaths_tick:
            self.stats.add_death(TEAM_BLUE, blue_deaths_tick)
            self.stats.add_kill(TEAM_RED, blue_deaths_tick)

        gx = self._as_long(data[dead_idx, COL_X])
        gy = self._as_long(data[dead_idx, COL_Y])
        self.grid[0][gy, gx] = 0.0
        self.grid[1][gy, gx] = 0.0
        self.grid[2][gy, gx] = -1.0
        data[dead_idx, COL_ALIVE] = 0.0

        for i, tid, x_, y_ in zip(dead_idx.tolist(), dead_team.tolist(), gx.tolist(), gy.tolist()):
            if killer_team_id_val is None:
                k = 3.0 if float(tid) == 2.0 else 2.0
            else:
                k = float(killer_team_id_val)
            self.stats.record_death_entry(
                agent_id=int(i), team_id_val=float(tid), x=int(x_), y=int(y_), killer_team_id_val=k,
            )

        metrics.deaths += int(dead_idx.numel())
        return red_deaths_tick, blue_deaths_tick

    # -------------------- Observations (V2) --------------------
    @torch.no_grad()
    def _build_obs_v2(self, alive_idx: torch.Tensor) -> torch.Tensor:
        data = self.registry.agent_data
        pos_xy = self.registry.positions_xy(alive_idx)
        unit_map = build_unit_map(data, self.grid)

        vision_map = getattr(config, "VISION_RANGE_BY_UNIT", {1: 6, 2: 10})
        default_vis = int(getattr(config, "RAYCAST_MAX_STEPS", max(vision_map.values()) if len(vision_map) else 10))
        units_here = data[alive_idx, COL_UNIT].to(torch.long)
        max_steps_each = torch.full((units_here.numel(),), int(default_vis), device=self.device, dtype=torch.long)
        for uid, vis in vision_map.items():
            uid_i = int(uid); vis_i = int(vis)
            if vis_i < 0:
                continue
            mask = (units_here == uid_i)
            if mask.any():
                max_steps_each[mask] = vis_i

        rays = raycast8_firsthit(pos_xy, self.grid, unit_map, max_steps_each=max_steps_each)  # (N,64)

        N = alive_idx.numel()
        x = data[alive_idx, COL_X] / float(self.W - 1)
        y = data[alive_idx, COL_Y] / float(self.H - 1)
        hp = data[alive_idx, COL_HP] / float(self._MAX_HP)
        team = data[alive_idx, COL_TEAM]
        unit = data[alive_idx, COL_UNIT]

        team_red = (team == 2.0).to(rays.dtype)
        team_blue = (team == 3.0).to(rays.dtype)
        unit_sold = (unit == 1.0).to(rays.dtype)
        unit_arch = (unit == 2.0).to(rays.dtype)

        red_score = float(getattr(getattr(self.stats, "red", None), "score", 0.0))
        blue_score = float(getattr(getattr(self.stats, "blue", None), "score", 0.0))
        red_cp = float(getattr(getattr(self.stats, "red", None), "cp", 0.0))
        blue_cp = float(getattr(getattr(self.stats, "blue", None), "cp", 0.0))

        def _norm(v: float, scale: float) -> torch.Tensor:
            s = scale if scale > 0 else 1.0
            return torch.tensor(v / s, dtype=rays.dtype, device=self.device).expand(N)

        red_score_norm  = _norm(red_score, 1000.0)
        blue_score_norm = _norm(blue_score, 1000.0)
        cp_red_norm     = _norm(red_cp,   1000.0)
        cp_blue_norm    = _norm(blue_cp,  1000.0)

        all_data = self.registry.agent_data
        alive_mask_all = (all_data[:, COL_ALIVE] > 0.5)
        t_all = all_data[:, COL_TEAM]
        u_all = all_data[:, COL_UNIT]
        total_red  = max(int((t_all == 2.0).sum().item()), 1)
        total_blue = max(int((t_all == 3.0).sum().item()), 1)
        red_sold_alive   = int(((alive_mask_all) & (t_all == 2.0) & (u_all == 1.0)).sum().item())
        red_arch_alive   = int(((alive_mask_all) & (t_all == 2.0) & (u_all == 2.0)).sum().item())
        blue_sold_alive  = int(((alive_mask_all) & (t_all == 3.0) & (u_all == 1.0)).sum().item())
        blue_arch_alive  = int(((alive_mask_all) & (t_all == 3.0) & (u_all == 2.0)).sum().item())
        red_sold_alive_norm  = torch.tensor(red_sold_alive  / total_red,  dtype=rays.dtype, device=self.device).expand(N)
        red_arch_alive_norm  = torch.tensor(red_arch_alive  / total_red,  dtype=rays.dtype, device=self.device).expand(N)
        blue_sold_alive_norm = torch.tensor(blue_sold_alive / total_blue, dtype=rays.dtype, device=self.device).expand(N)
        blue_arch_alive_norm = torch.tensor(blue_arch_alive / total_blue, dtype=rays.dtype).to(self.device).expand(N)

        meta_val = torch.where(unit == 1.0,
                               torch.tensor(self._META_SOLDIER, dtype=rays.dtype, device=self.device),
                               torch.tensor(self._META_ARCHER,  dtype=rays.dtype, device=self.device))
        max_atk = float(getattr(config, "MAX_ATK", 1.0))
        if max_atk <= 0: max_atk = 1.0
        own_atk_norm = (data[alive_idx, COL_ATK] / max_atk).to(rays.dtype)

        atk_cd   = torch.zeros(N, dtype=rays.dtype, device=self.device)
        took_dmg = torch.zeros(N, dtype=rays.dtype, device=self.device)

        self_feats = torch.stack([
            hp.to(rays.dtype), x.to(rays.dtype), y.to(rays.dtype),
            team_red, team_blue, unit_sold, unit_arch,
            red_score_norm, blue_score_norm, atk_cd, took_dmg,
            meta_val, own_atk_norm,
            blue_sold_alive_norm, red_sold_alive_norm, blue_arch_alive_norm, red_arch_alive_norm,
            cp_blue_norm, cp_red_norm,
            red_score_norm, blue_score_norm
        ], dim=1)

        return torch.cat([rays, self_feats], dim=1)  # (N,85)

    # -------------------- Main tick --------------------
    @torch.no_grad()
    def run_tick(self) -> Dict[str, float]:
        data = self.registry.agent_data
        assert data.device == self.device
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

        required_dim = 64 + 21
        use_v2 = (self._ACTIONS >= 41) and (self._OBS_DIM >= required_dim)
        if use_v2:
            obs = self._build_obs_v2(alive_idx)
        else:
            obs = raycast8_legacy(pos_xy, self.grid)

        try:
            mask = build_mask(pos_xy, teams, self.grid, unit=units)
        except TypeError:
            mask = build_mask(pos_xy, teams, self.grid)

        # === Ego-centric rotation of the first 64 dims =========================
        if use_v2 and self._RELATIVE_DIRS:
            prev_h = self._heading8[alive_idx]
            last_p = self._last_pos_xy[alive_idx]
            # FIX: compute_heading8 has a 3-arg signature in your repo
            h = compute_heading8(pos_xy, last_p, prev_h)
            self._heading8[alive_idx] = h
            self._has_heading[alive_idx] = True
            self._last_pos_xy[alive_idx] = pos_xy
            obs[:, :64] = rotate_rays_to_heading(obs[:, :64], h)

        # ---- per-architecture ensemble → sample actions; capture PPO data ----
        actions = torch.zeros((alive_idx.size(0),), dtype=torch.long, device=self.device)

        rec_agent_ids: List[torch.Tensor] = []
        rec_obs: List[torch.Tensor] = []
        rec_logits: List[torch.Tensor] = []
        rec_values: List[torch.Tensor] = []
        rec_actions: List[torch.Tensor] = []
        rec_teams: List[torch.Tensor] = []

        buckets = self.registry.build_buckets(alive_idx)
        for bucket in buckets:
            loc = torch.searchsorted(alive_idx, bucket.indices)
            o = obs[loc]
            m = mask[loc]
            amp_ok = False
            try:
                amp_ok = bool(getattr(config, "amp_enabled", lambda: False)())
            except Exception:
                amp_ok = False
            with torch.amp.autocast("cuda", enabled=amp_ok):
                dist, vals = ensemble_forward(bucket.models, o)
                logits = dist.logits

                # inverse-rotate directional groups back to GLOBAL before masking
                if use_v2 and self._RELATIVE_DIRS:
                    h_loc = self._heading8[bucket.indices]
                    logits = unrotate_logits_by_heading(logits, h_loc)

                neg_inf = torch.finfo(logits.dtype).min
                masked_logits = torch.where(m, logits, torch.full_like(logits, neg_inf))
                dist = torch.distributions.Categorical(logits=masked_logits)
                a = dist.sample()

            if self._ppo is not None:
                rec_agent_ids.append(bucket.indices.clone())
                rec_obs.append(o.clone())
                rec_logits.append(masked_logits.detach().clone())
                rec_values.append(vals.detach().clone())
                rec_actions.append(a.detach().clone())
                rec_teams.append(teams[loc].detach().clone())

            actions[loc] = a

        metrics.alive = int(alive_idx.numel())

        # ---------------- movement ----------------
        is_move = (actions >= 1) & (actions <= 8)
        if torch.any(is_move):
            move_idx = alive_idx[is_move]
            dir_idx = (actions[is_move] - 1).to(torch.long)
            d8 = self.DIRS8_dev
            dx = d8[dir_idx, 0]; dy = d8[dir_idx, 1]
            x0 = self._as_long(data[move_idx, COL_X]); y0 = self._as_long(data[move_idx, COL_Y])
            nx = (x0 + dx).clamp(0, self.W - 1); ny = (y0 + dy).clamp(0, self.H - 1)
            occ = self.grid[0]; target = occ[ny, nx]; free = (target == 0.0)
            flat = ny * self.W + nx
            counts = torch.zeros((self.H * self.W,), device=self.device, dtype=torch.int32)
            counts.scatter_add_(0, flat, free.to(torch.int32))
            ok = free & (counts[flat] == 1)
            if torch.any(ok):
                ok_idx = move_idx[ok]
                x_from, y_from = x0[ok], y0[ok]
                x_to, y_to = nx[ok], ny[ok]
                self.grid[0][y_from, x_from] = 0.0
                self.grid[1][y_from, x_from] = 0.0
                self.grid[2][y_from, x_from] = -1.0
                data[ok_idx, COL_X] = x_to.to(data.dtype)
                data[ok_idx, COL_Y] = y_to.to(data.dtype)
                team_at = data[ok_idx, COL_TEAM]; hp_at = data[ok_idx, COL_HP]
                self.grid[0][y_to, x_to] = team_at
                self.grid[1][y_to, x_to] = hp_at
                self.grid[2][y_to, x_to] = ok_idx.to(self.grid[2].dtype)
                metrics.moved = int(ok.sum().item())

        # ---------------- attacks ----------------
        A = self._ACTIONS
        is_attack = (actions >= 9) & (actions < (17 if A <= 17 else 41))
        combat_red_deaths = 0
        combat_blue_deaths = 0

        if torch.any(is_attack):
            atk_idx_all = alive_idx[is_attack]
            atk_act_all = actions[is_attack]
            if A <= 17:
                dir_idx = atk_act_all - 9; r = torch.ones_like(dir_idx)
            else:
                rel = atk_act_all - 9; dir_idx = rel // 4; r = (rel % 4) + 1

            d8 = self.DIRS8_dev
            dx_unit = d8[dir_idx, 0].to(torch.long)
            dy_unit = d8[dir_idx, 1].to(torch.long)
            dx = dx_unit * r; dy = dy_unit * r

            ax = self._as_long(data[atk_idx_all, COL_X])
            ay = self._as_long(data[atk_idx_all, COL_Y])
            tx = (ax + dx).clamp(0, self.W - 1); ty = (ay + dy).clamp(0, self.H - 1)

            if self._LOS_BLOCKS and (A >= 41):
                occ = self.grid[0]
                los_blocked = torch.zeros(atk_idx_all.shape[0], dtype=torch.bool, device=self.device)
                for step in (1, 2, 3):
                    step_mask = (r > step)
                    if not torch.any(step_mask): continue
                    ix = (ax + dx_unit * step).clamp(0, self.W - 1)
                    iy = (ay + dy_unit * step).clamp(0, self.H - 1)
                    los_blocked |= (step_mask & (occ[iy, ix] == 1.0))
                keep = ~los_blocked
                if torch.any(keep):
                    atk_idx_all = atk_idx_all[keep]; tx = tx[keep]; ty = ty[keep]
                else:
                    atk_idx_all = atk_idx_all[:0]; tx = tx[:0]; ty = ty[:0]

            if atk_idx_all.numel() > 0:
                victims = self._as_long(self.grid[2][ty, tx])
                valid = victims >= 0
                if torch.any(valid):
                    atk_idx = atk_idx_all[valid]; victims = victims[valid]
                    atk_team = data[atk_idx, COL_TEAM]
                    vic_team = data[victims, COL_TEAM]
                    enemy = (atk_team != vic_team)
                    if torch.any(enemy):
                        atk_idx = atk_idx[enemy]; victims = victims[enemy]
                        atk_team = atk_team[enemy]; vic_team = vic_team[enemy]
                        dmg = data[atk_idx, COL_ATK]
                        hp_view = data[:, COL_HP]; delta = torch.zeros_like(hp_view)
                        delta.scatter_add_(0, victims, -dmg)
                        data[:, COL_HP] = hp_view + delta

                        red_dmg    = float(dmg[atk_team == 2.0].sum().item())
                        blue_dmg   = float(dmg[atk_team == 3.0].sum().item())
                        red_taken  = float(dmg[vic_team == 2.0].sum().item())
                        blue_taken = float(dmg[vic_team == 3.0].sum().item())
                        if red_dmg:   self.stats.add_damage_dealt(TEAM_RED,  red_dmg)
                        if blue_dmg:  self.stats.add_damage_dealt(TEAM_BLUE, blue_dmg)
                        if red_taken: self.stats.add_damage_taken(TEAM_RED,  red_taken)
                        if blue_taken:self.stats.add_damage_taken(TEAM_BLUE, blue_taken)

                        was_alive = data[:, COL_ALIVE] > 0.5
                        now_dead = (data[:, COL_HP] <= 0.0) & was_alive
                        rD, bD = self._apply_deaths(now_dead, metrics, killer_team_id_val=None)
                        combat_red_deaths  += rD
                        combat_blue_deaths += bD
                        metrics.attacks = int(dmg.numel())

        alive_idx = self._recompute_alive_idx()
        if alive_idx.numel() == 0:
            self.stats.on_tick_advanced(1)
            metrics.tick = int(self.stats.tick)
            self.respawner.step(self.stats.tick, self.registry, self.grid)
            return {
                "alive": 0, "moved": metrics.moved, "attacks": metrics.attacks,
                "deaths": metrics.deaths, "tick": metrics.tick,
                "cp_red_tick": metrics.cp_red_tick, "cp_blue_tick": metrics.cp_blue_tick,
            }

        # -------------------- healing --------------------
        if self._zones_valid and (self._z_heal is not None):
            heal = self._z_heal
            ax = self._as_long(data[alive_idx, COL_X]); ay = self._as_long(data[alive_idx, COL_Y])
            on_heal = heal[ay, ax]
            if torch.any(on_heal):
                healers = alive_idx[on_heal]
                data[healers, COL_HP] = torch.clamp(data[healers, COL_HP] + self._HEAL_RATE, max=self._MAX_HP)
                hx = self._as_long(data[healers, COL_X]); hy = self._as_long(data[healers, COL_Y])
                self.grid[1][hy, hx] = data[healers, COL_HP]

        # -------------------- metabolism --------------------
        if self._META_ON:
            ax = self._as_long(data[alive_idx, COL_X]); ay = self._as_long(data[alive_idx, COL_Y])
            units_here = self._as_long(data[alive_idx, COL_UNIT])
            drain_s = torch.tensor(self._META_SOLDIER, dtype=data.dtype, device=self.device)
            drain_a = torch.tensor(self._META_ARCHER,  dtype=data.dtype, device=self.device)
            drain = torch.where(units_here == 1, drain_s, drain_a)
            data[alive_idx, COL_HP] = torch.clamp(data[alive_idx, COL_HP] - drain, min=0.0, max=self._MAX_HP)
            self.grid[1][ay, ax] = data[alive_idx, COL_HP]
            was_alive = data[:, COL_ALIVE] > 0.5
            now_dead = (data[:, COL_HP] <= 0.0) & was_alive
            _ = self._apply_deaths(now_dead, metrics, killer_team_id_val=0.0)
            if torch.any(now_dead):
                alive_idx = self._recompute_alive_idx()

        # -------------------- capture scoring (also used for PPO reward) ---------------
        cp_gain_red = 0.0
        cp_gain_blue = 0.0
        if self._zones_valid and len(self._z_cp_masks) > 0:
            occ = self.grid[0]
            red_mask = (occ == 2.0); blue_mask = (occ == 3.0)
            for m in self._z_cp_masks:
                red_cnt = int((red_mask & m).sum().item())
                blue_cnt = int((blue_mask & m).sum().item())
                if red_cnt > blue_cnt:   cp_gain_red  += self._CP_REWARD
                elif blue_cnt > red_cnt: cp_gain_blue += self._CP_REWARD
            metrics.cp_red_tick = cp_gain_red
            metrics.cp_blue_tick = cp_gain_blue
            if hasattr(self.stats, "add_capture_points"):
                if cp_gain_red:  self.stats.add_capture_points(TEAM_RED,  cp_gain_red)
                if cp_gain_blue: self.stats.add_capture_points(TEAM_BLUE, cp_gain_blue)

        # -------------------- PPO record (team rewards per tick) -----------------------
        if self._ppo is not None and len(rec_agent_ids) > 0:
            R_kill  = float(getattr(config, "PPO_REWARD_KILL", 1.0))
            R_death = float(getattr(config, "PPO_REWARD_DEATH", -0.3))
            R_cp    = float(getattr(config, "PPO_REWARD_CP_MULT", 0.05))  # CP weight

            team_reward_red  = (combat_blue_deaths * R_kill) + (combat_red_deaths * R_death) + (R_cp * metrics.cp_red_tick)
            team_reward_blue = (combat_red_deaths * R_kill) + (combat_blue_deaths * R_death) + (R_cp * metrics.cp_blue_tick)

            agent_ids   = torch.cat(rec_agent_ids, dim=0)
            teams_step  = torch.cat(rec_teams, dim=0)
            obs_step    = torch.cat(rec_obs, dim=0)
            logits_step = torch.cat(rec_logits, dim=0)
            values_step = torch.cat(rec_values, dim=0)
            actions_step= torch.cat(rec_actions, dim=0)

            done_step = (self.registry.agent_data[agent_ids, COL_ALIVE] <= 0.5)
            with torch.enable_grad():
                self._ppo.record_step(
                    agent_ids=agent_ids,
                    team_ids=teams_step,
                    obs=obs_step,
                    logits=logits_step,
                    values=values_step,
                    actions=actions_step,
                    team_reward_red=team_reward_red,
                    team_reward_blue=team_reward_blue,
                    done=done_step,
                )

        # -------------------- tick++ / respawn / mutation ------------------------
        self.stats.on_tick_advanced(1)
        metrics.tick = int(self.stats.tick)

        if self._PER_AGENT_BRAINS and (self.stats.tick % self._MUT_PERIOD == 0):
            alive_now = self._recompute_alive_idx()
            if alive_now.numel() > 0:
                chosen = pick_mutants(alive_now, fraction=self._MUT_FRAC)
                if chosen.numel() > 0:
                    self.registry.apply_mutations(chosen, mutate_model_inplace)

        self.respawner.step(self.stats.tick, self.registry, self.grid)

        return {
            "alive": metrics.alive,
            "moved": metrics.moved,
            "attacks": metrics.attacks,
            "deaths": metrics.deaths,
            "tick": metrics.tick,
            "cp_red_tick": metrics.cp_red_tick,
            "cp_blue_tick": metrics.cp_blue_tick,
        }
