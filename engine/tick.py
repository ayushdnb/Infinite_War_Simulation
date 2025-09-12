# final_war_sim/engine/tick.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch

from .. import config
from ..simulation.stats import SimulationStats, TEAM_RED, TEAM_BLUE
from .agent_registry import (
    AgentsRegistry, COL_ALIVE, COL_TEAM, COL_X, COL_Y, COL_HP, COL_ATK
)
from .ray_engine.raycaster2d import raycast8
from .game.move_mask import build_mask, ACTION
from .respawn import RespawnController

from ..agent.ensemble import ensemble_forward
from ..agent.mutation import pick_mutants, mutate_model_inplace


@dataclass
class TickMetrics:
    alive: int = 0
    moved: int = 0
    attacks: int = 0
    deaths: int = 0
    tick: int = 0


class TickEngine:
    def __init__(self, registry: AgentsRegistry, grid: torch.Tensor, stats: SimulationStats) -> None:
        self.registry = registry
        self.grid = grid
        self.stats = stats
        self.device = grid.device
        self.H, self.W = grid.size(1), grid.size(2)
        self.respawner = RespawnController()  # unified respawn controller

    @torch.no_grad()
    def run_tick(self) -> Dict[str, int]:
        data = self.registry.agent_data
        assert data.device == self.device

        metrics = TickMetrics()

        # ---- alive set
        alive_idx = (data[:, COL_ALIVE] > 0.5).nonzero(as_tuple=False).squeeze(1)
        if alive_idx.numel() == 0:
            # advance time even if empty (respawner will refill)
            self.stats.on_tick_advanced(1)
            metrics.tick = int(self.stats.tick)
            self.respawner.step(self.stats.tick, self.registry, self.grid)
            return {"alive": 0, "moved": 0, "attacks": 0, "deaths": 0, "tick": metrics.tick}

        teams = data[alive_idx, COL_TEAM]
        pos_xy = self.registry.positions_xy(alive_idx)

        # ---- obs + mask
        obs = raycast8(pos_xy, self.grid)
        mask = build_mask(pos_xy, teams, self.grid)

        # ---- per-architecture ensemble
        actions = torch.zeros((alive_idx.size(0),), dtype=torch.long, device=self.device)
        buckets = self.registry.build_buckets(alive_idx)
        for bucket in buckets:
            # map bucket absolute indices -> positions inside alive_idx
            loc = torch.searchsorted(alive_idx, bucket.indices)
            o = obs[loc]
            m = mask[loc]
            with torch.cuda.amp.autocast(enabled=config.amp_enabled()):
                dist, _ = ensemble_forward(bucket.models, o)
                logits = dist.logits
                neg_inf = torch.finfo(logits.dtype).min
                masked_logits = torch.where(m, logits, torch.full_like(logits, neg_inf))
                dist = torch.distributions.Categorical(logits=masked_logits)
                a = dist.sample()
            actions[loc] = a

        metrics.alive = int(alive_idx.numel())

        # ---- movement (conflict suppression via scatter counts)
        MOVE = {
            ACTION["MOVE_UP"]:    (0, -1),
            ACTION["MOVE_RIGHT"]: (1,  0),
            ACTION["MOVE_DOWN"]:  (0,  1),
            ACTION["MOVE_LEFT"]:  (-1, 0),
        }
        is_move = (
            (actions == ACTION["MOVE_UP"]) |
            (actions == ACTION["MOVE_RIGHT"]) |
            (actions == ACTION["MOVE_DOWN"]) |
            (actions == ACTION["MOVE_LEFT"])
        )
        if torch.any(is_move):
            move_idx = alive_idx[is_move]
            move_act = actions[is_move]
            dx = torch.zeros_like(move_act, dtype=torch.long)
            dy = torch.zeros_like(move_act, dtype=torch.long)
            for k, (ddx, ddy) in MOVE.items():
                sel = (move_act == k)
                if torch.any(sel):
                    dx[sel] = ddx
                    dy[sel] = ddy

            x0 = data[move_idx, COL_X].to(torch.long)
            y0 = data[move_idx, COL_Y].to(torch.long)
            nx = (x0 + dx).clamp(0, self.W - 1)
            ny = (y0 + dy).clamp(0, self.H - 1)

            occ = self.grid[0]
            target = occ[ny, nx]
            free = (target == 0.0)

            flat = ny * self.W + nx
            counts = torch.zeros((self.H * self.W,), device=self.device, dtype=torch.int32)
            counts.scatter_add_(0, flat, free.to(torch.int32))

            ok = free & (counts[flat] == 1)
            if torch.any(ok):
                ok_idx = move_idx[ok]
                x_from, y_from = x0[ok], y0[ok]
                x_to, y_to = nx[ok], ny[ok]

                # clear from
                self.grid[0][y_from, x_from] = 0.0
                self.grid[1][y_from, x_from] = 0.0
                self.grid[2][y_from, x_from] = -1.0

                # write to
                data[ok_idx, COL_X] = x_to.to(data.dtype)
                data[ok_idx, COL_Y] = y_to.to(data.dtype)
                team_at = data[ok_idx, COL_TEAM]
                hp_at = data[ok_idx, COL_HP]
                self.grid[0][y_to, x_to] = team_at
                self.grid[1][y_to, x_to] = hp_at
                self.grid[2][y_to, x_to] = ok_idx.to(self.grid[2].dtype)

                metrics.moved = int(ok.sum().item())

        # ---- attacks (vectorized)
        ATTACK = {
            ACTION["ATTACK_UP"]:    (0, -1),
            ACTION["ATTACK_RIGHT"]: (1,  0),
            ACTION["ATTACK_DOWN"]:  (0,  1),
            ACTION["ATTACK_LEFT"]:  (-1, 0),
        }
        is_attack = (
            (actions == ACTION["ATTACK_UP"]) |
            (actions == ACTION["ATTACK_RIGHT"]) |
            (actions == ACTION["ATTACK_DOWN"]) |
            (actions == ACTION["ATTACK_LEFT"])
        )
        if torch.any(is_attack):
            atk_idx = alive_idx[is_attack]
            atk_act = actions[is_attack]
            dx = torch.zeros_like(atk_act, dtype=torch.long)
            dy = torch.zeros_like(atk_act, dtype=torch.long)
            for k, (ddx, ddy) in ATTACK.items():
                sel = (atk_act == k)
                if torch.any(sel):
                    dx[sel] = ddx
                    dy[sel] = ddy

            ax = data[atk_idx, COL_X].to(torch.long)
            ay = data[atk_idx, COL_Y].to(torch.long)
            tx = (ax + dx).clamp(0, self.W - 1)
            ty = (ay + dy).clamp(0, self.H - 1)

            victims = self.grid[2][ty, tx].to(torch.long)  # -1 => empty
            valid = victims >= 0
            if torch.any(valid):
                atk_idx = atk_idx[valid]
                victims = victims[valid]
                atk_team = data[atk_idx, COL_TEAM]
                vic_team = data[victims, COL_TEAM]
                enemy = (atk_team != vic_team)

                if torch.any(enemy):
                    atk_idx = atk_idx[enemy]
                    victims = victims[enemy]
                    atk_team = atk_team[enemy]
                    vic_team = vic_team[enemy]

                    dmg = data[atk_idx, COL_ATK]

                    # apply damage via scatter_add
                    hp_view = data[:, COL_HP]
                    delta = torch.zeros_like(hp_view)
                    delta.scatter_add_(0, victims, -dmg)
                    data[:, COL_HP] = hp_view + delta

                    # team damage accounting
                    red_dmg    = dmg[atk_team == 2.0].sum().item()
                    blue_dmg   = dmg[atk_team == 3.0].sum().item()
                    red_taken  = dmg[vic_team == 2.0].sum().item()
                    blue_taken = dmg[vic_team == 3.0].sum().item()
                    if red_dmg:
                        self.stats.add_damage_dealt(TEAM_RED, red_dmg)
                    if blue_dmg:
                        self.stats.add_damage_dealt(TEAM_BLUE, blue_dmg)
                    if red_taken:
                        self.stats.add_damage_taken(TEAM_RED, red_taken)
                    if blue_taken:
                        self.stats.add_damage_taken(TEAM_BLUE, blue_taken)

                    # deaths (single, clean attribution)
                    was_alive = data[:, COL_ALIVE] > 0.5
                    now_dead = (data[:, COL_HP] <= 0.0) & was_alive
                    if torch.any(now_dead):
                        dead_idx = now_dead.nonzero(as_tuple=False).squeeze(1)
                        dead_team = data[dead_idx, COL_TEAM]

                        red_deaths = (dead_team == 2.0).sum().item()
                        blue_deaths = (dead_team == 3.0).sum().item()

                        if red_deaths:
                            self.stats.add_death(TEAM_RED, red_deaths)
                            self.stats.add_kill(TEAM_BLUE, red_deaths)
                        if blue_deaths:
                            self.stats.add_death(TEAM_BLUE, blue_deaths)
                            self.stats.add_kill(TEAM_RED, blue_deaths)

                        # clear grid & flags
                        gx = data[dead_idx, COL_X].to(torch.long)
                        gy = data[dead_idx, COL_Y].to(torch.long)
                        self.grid[0][gy, gx] = 0.0
                        self.grid[1][gy, gx] = 0.0
                        self.grid[2][gy, gx] = -1.0
                        data[dead_idx, COL_ALIVE] = 0.0

                        # structured log entries
                        # team ids in grid are 2.0 (red) and 3.0 (blue); killer is the opposite
                        for i, tid, x_, y_ in zip(dead_idx.tolist(), dead_team.tolist(), gx.tolist(), gy.tolist()):
                            killer_id_val = 3.0 if tid == 2.0 else 2.0
                            self.stats.record_death_entry(
                                agent_id=i,
                                team_id_val=tid,
                                x=int(x_), y=int(y_),
                                killer_team_id_val=killer_id_val,
                            )

                        metrics.deaths = int(dead_idx.numel())

                    metrics.attacks = int(dmg.numel())

        # ---- tick++
        self.stats.on_tick_advanced(1)
        metrics.tick = int(self.stats.tick)

        # ---- periodic mutation
        if config.PER_AGENT_BRAINS and (self.stats.tick % config.MUTATION_PERIOD_TICKS == 0):
            alive_now = (data[:, COL_ALIVE] > 0.5).nonzero(as_tuple=False).squeeze(1)
            if alive_now.numel() > 0:
                chosen = pick_mutants(alive_now, fraction=config.MUTATION_FRACTION_ALIVE)
                if chosen.numel() > 0:
                    self.registry.apply_mutations(chosen, mutate_model_inplace)

        # ---- unified respawn
        self.respawner.step(self.stats.tick, self.registry, self.grid)

        return {
            "alive": metrics.alive,
            "moved": metrics.moved,
            "attacks": metrics.attacks,
            "deaths": metrics.deaths,
            "tick": metrics.tick,
        }
