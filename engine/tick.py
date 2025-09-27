from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, TYPE_CHECKING

import torch

import config
from simulation.stats import SimulationStats
from engine.agent_registry import AgentsRegistry, COL_ALIVE, COL_TEAM, COL_X, COL_Y, COL_HP, COL_ATK, COL_UNIT, COL_VISION, COL_HP_MAX
from engine.ray_engine.raycast_64 import raycast64_firsthit
from engine.game.move_mask import build_mask, DIRS8
from engine.respawn import RespawnController
from engine.mapgen import Zones

from agent.ensemble import ensemble_forward
from agent.transformer_brain import TransformerBrain # New Brain

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
    Per-tick simulation core using the TransformerBrain.
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
        self._ensure_zone_tensors()

        self.DIRS8_dev = DIRS8.to(self.device)

        self._ACTIONS = int(getattr(config, "NUM_ACTIONS", 41))
        self._OBS_DIM = (64 * 8) + 21 # Ray features + rich features
        self._MAX_HP = float(getattr(config, "MAX_HP", 1.0))
        self._HEAL_RATE = float(getattr(config, "HEAL_RATE", 0.02))
        self._CP_REWARD = float(getattr(config, "CP_REWARD_PER_TICK", 5.0))
        self._LOS_BLOCKS = bool(getattr(config, "ARCHER_LOS_BLOCKS_WALLS", False))
        
        self._META_ON = bool(getattr(config, "METABOLISM_ENABLED", True))
        self._META_SOLDIER = float(getattr(config, "META_SOLDIER_HP_PER_TICK", 0.0001))
        self._META_ARCHER = float(getattr(config, "META_ARCHER_HP_PER_TICK", 0.00001))

        self._ppo_enabled = bool(getattr(config, "PPO_ENABLED", False))
        self._ppo: Optional["PerAgentPPORuntime"] = None
        if self._ppo_enabled and _PerAgentPPORuntimeRT is not None:
            self._ppo = _PerAgentPPORuntimeRT(
                registry=self.registry,
                device=self.device,
                obs_dim=self._OBS_DIM,
                act_dim=self._ACTIONS,
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
            self._z_heal, self._z_cp_masks = None, []

    @staticmethod
    def _as_long(x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.long)

    def _recompute_alive_idx(self) -> torch.Tensor:
        return (self.registry.agent_data[:, COL_ALIVE] > 0.5).nonzero(as_tuple=False).squeeze(1)

    def _apply_deaths(self, now_dead_mask: torch.Tensor, metrics: TickMetrics) -> Tuple[int, int]:
        red_deaths_tick, blue_deaths_tick = 0, 0
        if not torch.any(now_dead_mask): return red_deaths_tick, blue_deaths_tick

        dead_idx = now_dead_mask.nonzero(as_tuple=False).squeeze(1)
        if dead_idx.numel() == 0: return red_deaths_tick, blue_deaths_tick

        data = self.registry.agent_data
        dead_team = data[dead_idx, COL_TEAM]
        red_deaths_tick = int((dead_team == 2.0).sum().item())
        blue_deaths_tick = int((dead_team == 3.0).sum().item())

        if red_deaths_tick: self.stats.add_death("red", red_deaths_tick); self.stats.add_kill("blue", red_deaths_tick)
        if blue_deaths_tick: self.stats.add_death("blue", blue_deaths_tick); self.stats.add_kill("red", blue_deaths_tick)

        gx = self._as_long(data[dead_idx, COL_X]); gy = self._as_long(data[dead_idx, COL_Y])
        self.grid[0][gy, gx] = 0.0; self.grid[1][gy, gx] = 0.0; self.grid[2][gy, gx] = -1.0
        data[dead_idx, COL_ALIVE] = 0.0

        metrics.deaths += int(dead_idx.numel())
        return red_deaths_tick, blue_deaths_tick

    @torch.no_grad()
    def _build_transformer_obs(self, alive_idx: torch.Tensor) -> torch.Tensor:
        """Builds observations for the TransformerBrain."""
        from engine.ray_engine.raycast_firsthit import build_unit_map
        data = self.registry.agent_data
        pos_xy = self.registry.positions_xy(alive_idx)
        
        # Correctly access vision range from agent_data tensor
        max_steps_each = data[alive_idx, COL_VISION].to(self.device, dtype=torch.long)
        
        unit_map = build_unit_map(data, self.grid)
        rays = raycast64_firsthit(pos_xy, self.grid, unit_map, max_steps_each=max_steps_each)

        N = alive_idx.numel()
        x = data[alive_idx, COL_X] / float(self.W - 1)
        y = data[alive_idx, COL_Y] / float(self.H - 1)
        
        hp = data[alive_idx, COL_HP]
        hp_max = data[alive_idx, COL_HP_MAX].clamp_min(1.0)
        hp_norm = hp / hp_max

        team = data[alive_idx, COL_TEAM]
        unit = data[alive_idx, COL_UNIT]

        team_red = (team == 2.0).to(rays.dtype)
        team_blue = (team == 3.0).to(rays.dtype)
        unit_sold = (unit == 1.0).to(rays.dtype)
        unit_arch = (unit == 2.0).to(rays.dtype)

        # Agent's own stats
        own_atk = data[alive_idx, COL_ATK]
        own_vision = data[alive_idx, COL_VISION]
        max_atk = float(getattr(config, "MAX_ATK", 1.0)) or 1.0
        max_vision = float(getattr(config, "RAYCAST_MAX_STEPS", 15)) or 15.0
        
        own_atk_norm = (own_atk / max_atk).to(rays.dtype)
        own_vision_norm = (own_vision / max_vision).to(rays.dtype)

        # Game state stats
        red_score = self.stats.red.score
        blue_score = self.stats.blue.score
        
        def _norm(v, scale):
            s = scale if scale > 0 else 1.0
            return torch.full((N,), v/s, dtype=rays.dtype, device=self.device)

        # Rich features vector (21 elements)
        rich_features = torch.stack([
            hp_norm, x, y,
            team_red, team_blue,
            unit_sold, unit_arch,
            own_atk_norm, own_vision_norm,
            _norm(self.stats.tick, 50000.0), # Time awareness
            _norm(red_score, 1000.0),
            _norm(blue_score, 1000.0),
            _norm(self.stats.red.cp_points, 500.0),
            _norm(self.stats.blue.cp_points, 500.0),
            _norm(self.stats.red.kills, 200.0),
            _norm(self.stats.blue.kills, 200.0),
            _norm(self.stats.red.deaths, 200.0),
            _norm(self.stats.blue.deaths, 200.0),
            torch.zeros(N, device=self.device), # Placeholder
            torch.zeros(N, device=self.device), # Placeholder
            torch.zeros(N, device=self.device)  # Placeholder
        ], dim=1)
        
        return torch.cat([rays, rich_features], dim=1)

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

        obs = self._build_transformer_obs(alive_idx)
        mask = build_mask(pos_xy, teams, self.grid, unit=units)
        
        actions = torch.zeros_like(alive_idx, dtype=torch.long)
        
        # PPO data recording setup
        rec_agent_ids, rec_obs, rec_logits, rec_values, rec_actions, rec_teams = [],[],[],[],[],[]

        buckets = self.registry.build_buckets(alive_idx)
        for bucket in buckets:
            loc = torch.searchsorted(alive_idx, bucket.indices)
            o, m = obs[loc], mask[loc]
            
            dist, vals = ensemble_forward(bucket.models, o)
            logits = dist.logits

            neg_inf = torch.finfo(logits.dtype).min
            masked_logits = torch.where(m, logits, neg_inf)
            dist = torch.distributions.Categorical(logits=masked_logits)
            a = dist.sample()
            
            if self._ppo:
                rec_agent_ids.append(bucket.indices.clone())
                rec_obs.append(o.clone())
                rec_logits.append(masked_logits.detach().clone())
                rec_values.append(vals.detach().clone())
                rec_actions.append(a.detach().clone())
                rec_teams.append(teams[loc].detach().clone())
                
            actions[loc] = a

        metrics.alive = int(alive_idx.numel())

        # ... (rest of the logic: movement, attacks, etc.)
        is_move = (actions >= 1) & (actions <= 8)
        if torch.any(is_move):
            move_idx = alive_idx[is_move]
            dir_idx = (actions[is_move] - 1)
            dxy = self.DIRS8_dev[dir_idx]
            x0, y0 = pos_xy[is_move].T
            nx, ny = (x0 + dxy[:, 0]).clamp(0, self.W-1), (y0 + dxy[:, 1]).clamp(0, self.H-1)

            # Prevent collisions
            target_occ = self.grid[0][ny, nx]
            can_move = (target_occ == 0.0)

            if can_move.any():
                move_idx_final = move_idx[can_move]
                x0_final, y0_final = x0[can_move], y0[can_move]
                nx_final, ny_final = nx[can_move], ny[can_move]

                # Clear old pos
                self.grid[0][y0_final, x0_final] = 0.0
                self.grid[1][y0_final, x0_final] = 0.0
                self.grid[2][y0_final, x0_final] = -1.0

                # Set new pos in registry and grid
                data[move_idx_final, COL_X] = nx_final.float()
                data[move_idx_final, COL_Y] = ny_final.float()
                self.grid[0][ny_final, nx_final] = data[move_idx_final, COL_TEAM]
                self.grid[1][ny_final, nx_final] = data[move_idx_final, COL_HP]
                self.grid[2][ny_final, nx_final] = move_idx_final.float()
                metrics.moved = can_move.sum().item()

        # Attacks
        combat_red_deaths, combat_blue_deaths = 0, 0
        is_attack = (actions >= 9) & (actions < self._ACTIONS)
        if torch.any(is_attack):
            atk_idx = alive_idx[is_attack]
            atk_act = actions[is_attack]
            
            rel = atk_act - 9
            dir_idx, r = rel // 4, (rel % 4) + 1
            dxy = self.DIRS8_dev[dir_idx] * r.unsqueeze(1)
            
            ax, ay = pos_xy[is_attack].T
            tx, ty = (ax + dxy[:, 0]).clamp(0, self.W-1), (ay + dxy[:, 1]).clamp(0, self.H-1)

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
                    self.grid[1][data[victims_enemy, COL_Y].long(), data[victims_enemy, COL_X].long()] = data[victims_enemy, COL_HP]
                    
                    metrics.attacks += is_enemy.sum().item()
                    was_alive = data[:, COL_ALIVE] > 0.5
                    now_dead = (data[:, COL_HP] <= 0.0) & was_alive
                    rD, bD = self._apply_deaths(now_dead, metrics)
                    combat_red_deaths += rD
                    combat_blue_deaths += bD
        
        # --- Start of Post-Action Phase ---
        # Re-evaluate who is still alive after combat
        alive_idx = self._recompute_alive_idx()
        if alive_idx.numel() == 0:
            self.stats.on_tick_advanced(1)
            metrics.tick = int(self.stats.tick)
            self.respawner.step(self.stats.tick, self.registry, self.grid)
            return vars(metrics)

        # Get the current positions of agents who survived combat
        pos_xy = self.registry.positions_xy(alive_idx)

        # Healing, Metabolism, CP Scoring...
        if self._z_heal is not None:
            on_heal_mask = self._z_heal[pos_xy[:, 1], pos_xy[:, 0]]
            if on_heal_mask.any():
                heal_idx = alive_idx[on_heal_mask]
                hp_max = data[heal_idx, COL_HP_MAX]
                data[heal_idx, COL_HP] = (data[heal_idx, COL_HP] + self._HEAL_RATE).clamp_max(hp_max)
                
                heal_pos = pos_xy[on_heal_mask]
                self.grid[1][heal_pos[:, 1], heal_pos[:, 0]] = data[heal_idx, COL_HP]
        
        if self._META_ON:
            units_alive = data[alive_idx, COL_UNIT]
            drain = torch.where(units_alive == 1.0, self._META_SOLDIER, self._META_ARCHER)
            data[alive_idx, COL_HP] -= drain
            self.grid[1][pos_xy[:,1], pos_xy[:,0]] = data[alive_idx, COL_HP] # Update grid HP after drain
            
            now_dead_from_meta_mask = (data[alive_idx, COL_HP] <= 0.0)
            if now_dead_from_meta_mask.any():
                dead_from_meta_idx = alive_idx[now_dead_from_meta_mask]
                rD, bD = self._apply_deaths(dead_from_meta_idx, metrics)
                combat_red_deaths += rD
                combat_blue_deaths += bD
        
        # PPO record and final updates
        if self._ppo and rec_agent_ids:
            # ... PPO reward calculation and recording ...
            R_kill = float(getattr(config, "PPO_REWARD_KILL", 1.0))
            R_death = float(getattr(config, "PPO_REWARD_DEATH", -0.3))
            
            team_reward_red  = (combat_blue_deaths * R_kill) + (combat_red_deaths * R_death)
            team_reward_blue = (combat_red_deaths * R_kill) + (combat_blue_deaths * R_death)
            
            agent_ids = torch.cat(rec_agent_ids)
            done_step = (self.registry.agent_data[agent_ids, COL_ALIVE] <= 0.5)

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

        self.stats.on_tick_advanced(1)
        metrics.tick = int(self.stats.tick)
        self.respawner.step(self.stats.tick, self.registry, self.grid)

        return vars(metrics)
