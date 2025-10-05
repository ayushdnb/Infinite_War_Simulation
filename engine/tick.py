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
from engine.ray_engine.raycast_64 import raycast64_firsthit
from engine.game.move_mask import build_mask, DIRS8
from engine.respawn import RespawnController
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
        self.respawner = RespawnController()
        self.agent_scores: Dict[int, float] = collections.defaultdict(float)
        self.zones: Optional[Zones] = zones
        self._z_heal: Optional[torch.Tensor] = None
        self._z_cp_masks: List[torch.Tensor] = []
        self._ensure_zone_tensors()
        self.DIRS8_dev = DIRS8.to(self.device)
        self._ACTIONS = int(getattr(config, "NUM_ACTIONS", 41))
        # --- MODIFIED: Use obs_dim from config ---
        self._OBS_DIM = config.OBS_DIM
        self._grid_dt = self.grid.dtype
        self._data_dt = self.registry.agent_data.dtype
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
    def _recompute_alive_idx(self) -> torch.Tensor: return (self.registry.agent_data[:, COL_ALIVE] > 0.5).nonzero(as_tuple=False).squeeze(1)

    def _apply_deaths(self, sel: torch.Tensor, metrics: TickMetrics) -> Tuple[int, int]:
        data = self.registry.agent_data
        dead_idx = sel.nonzero(as_tuple=False).squeeze(1) if sel.dtype == torch.bool else sel.view(-1)
        if dead_idx.numel() == 0: return 0, 0
        dead_team = data[dead_idx, COL_TEAM]
        red_deaths, blue_deaths = int((dead_team == 2.0).sum().item()), int((dead_team == 3.0).sum().item())
        if red_deaths: self.stats.add_death("red", red_deaths); self.stats.add_kill("blue", red_deaths)
        if blue_deaths: self.stats.add_death("blue", blue_deaths); self.stats.add_kill("red", blue_deaths)
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

        # --- NEW: Calculate zone flags for rich features ---
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
        
        rays = raycast64_firsthit(pos_xy, self.grid, build_unit_map(data, self.grid), max_steps_each=data[alive_idx, COL_VISION].long())
        hp_max = data[alive_idx, COL_HP_MAX].clamp_min(1.0)
        
        # --- MODIFIED: Added on_heal and on_cp to the rich feature stack ---
        rich = torch.stack([
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
            _norm_const(float(self.stats.red.kills), 200.0), _norm_const(float(self.stats.blue.kills), 200.0),
            _norm_const(float(self.stats.red.deaths), 200.0), _norm_const(float(self.stats.blue.deaths), 200.0),
            torch.zeros(N, device=self.device, dtype=self._data_dt),
            torch.zeros(N, device=self.device, dtype=self._data_dt),
            torch.zeros(N, device=self.device, dtype=self._data_dt),
        ], dim=1)
        return torch.cat([rays, rich.to(rays.dtype)], dim=1)

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

        pos_xy = self.registry.positions_xy(alive_idx)
        # --- MODIFIED: Pass pos_xy to the observation builder ---
        obs = self._build_transformer_obs(alive_idx, pos_xy)
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
                rec_obs.append(obs[loc]); rec_logits.append(logits32);
                rec_values.append(vals); rec_actions.append(a); rec_teams.append(data[bucket.indices, COL_TEAM])
            actions[loc] = a
        metrics.alive = int(alive_idx.numel())
        
        is_move = (actions >= 1) & (actions <= 8)
        if is_move.any():
            move_idx, dir_idx = alive_idx[is_move], actions[is_move] - 1
            x0, y0 = pos_xy[is_move].T
            nx, ny = (x0 + self.DIRS8_dev[dir_idx, 0]).clamp(0, self.W - 1), (y0 + self.DIRS8_dev[dir_idx, 1]).clamp(0, self.H - 1)
            can_move = (self.grid[0][ny, nx] == self._g0)
            if can_move.any():
                move_idx, x0, y0, nx, ny = move_idx[can_move], x0[can_move], y0[can_move], nx[can_move], ny[can_move]
                self.grid[0, y0, x0], self.grid[1, y0, x0], self.grid[2, y0, x0] = self._g0, self._g0, self._gneg
                data[move_idx, COL_X], data[move_idx, COL_Y] = nx.to(self._data_dt), ny.to(self._data_dt)
                self.grid[0, ny, nx] = data[move_idx, COL_TEAM].to(self._grid_dt)
                self.grid[1, ny, nx] = data[move_idx, COL_HP].to(self._grid_dt)
                self.grid[2, ny, nx] = move_idx.to(self._grid_dt)
                metrics.moved = int(can_move.sum().item())

        combat_rd, combat_bd = 0, 0
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
        combat_rd += rD; combat_bd += bD

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
                    rD, bD = self._apply_deaths(alive_idx[data[alive_idx, COL_HP] <= 0.0], metrics)
                    combat_rd += rD; combat_bd += bD
            if self._z_cp_masks and (alive_idx := self._recompute_alive_idx()).numel() > 0:
                pos_xy, teams_alive = self.registry.positions_xy(alive_idx), data[alive_idx, COL_TEAM]
                for cp_mask in self._z_cp_masks:
                    if (on_cp := cp_mask[pos_xy[:, 1], pos_xy[:, 0]]).any():
                        red_on, blue_on = (on_cp & (teams_alive == 2.0)).sum(), (on_cp & (teams_alive == 3.0)).sum()
                        if red_on > blue_on: self.stats.add_capture_points("red", config.CP_REWARD_PER_TICK); metrics.cp_red_tick += config.CP_REWARD_PER_TICK
                        elif blue_on > red_on: self.stats.add_capture_points("blue", config.CP_REWARD_PER_TICK); metrics.cp_blue_tick += config.CP_REWARD_PER_TICK
                        
                        # --- NEW: Individual reward for all agents on a contested CP ---
                        # This encourages brawling and holding strategic ground.
                        if red_on > 0 and blue_on > 0:
                            agents_on_this_cp_idx = alive_idx[on_cp]
                            reward_val = config.PPO_REWARD_CONTESTED_CP
                            individual_rewards.index_add_(0, agents_on_this_cp_idx, torch.full_like(agents_on_this_cp_idx, reward_val, dtype=self._data_dt))

        if self._ppo and rec_agent_ids:
            agent_ids = torch.cat(rec_agent_ids)
            team_r_rew = (combat_bd * config.TEAM_KILL_REWARD) + (combat_rd * config.PPO_REWARD_DEATH) + metrics.cp_red_tick
            team_b_rew = (combat_rd * config.TEAM_KILL_REWARD) + (combat_bd * config.PPO_REWARD_DEATH) + metrics.cp_blue_tick
            current_hp = data[agent_ids, COL_HP]
            hp_reward = (current_hp * config.PPO_REWARD_HP_TICK).to(self._data_dt)
            final_rewards = individual_rewards[agent_ids] + torch.where(torch.cat(rec_teams) == 2.0, team_r_rew, team_b_rew) + hp_reward
 
            with torch.enable_grad():
                self._ppo.record_step(agent_ids=agent_ids, obs=torch.cat(rec_obs), logits=torch.cat(rec_logits), values=torch.cat(rec_values), actions=torch.cat(rec_actions), rewards=final_rewards, done=(data[agent_ids, COL_ALIVE] <= 0.5))

        self.stats.on_tick_advanced(1)
        metrics.tick = int(self.stats.tick)
        self.respawner.step(self.stats.tick, self.registry, self.grid)
        return vars(metrics)