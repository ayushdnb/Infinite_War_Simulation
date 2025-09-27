from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import random
import copy

import torch
import torch.jit

import config
from .agent_registry import (
    AgentsRegistry,
    COL_ALIVE, COL_TEAM, COL_X, COL_Y, COL_HP, COL_ATK, COL_UNIT,
    TEAM_RED_ID, TEAM_BLUE_ID
)
# Import the new brain
from agent.transformer_brain import TransformerBrain, scripted_transformer_brain

# Respawn counter for rare mutations
_respawn_counter = 0

# ----------------------------
# Config knobs (safe fallbacks)
# ----------------------------
RESP_FLOOR_PER_TEAM = int(getattr(config, "RESP_FLOOR_PER_TEAM", 50))
RESP_MAX_PER_TICK = int(getattr(config, "RESP_MAX_PER_TICK", 15))
RESP_PERIOD_TICKS = int(getattr(config, "RESP_PERIOD_TICKS", 200))
RESP_PERIOD_BUDGET = int(getattr(config, "RESP_PERIOD_BUDGET", 20))
RESP_HYST_COOLDOWN_TICKS = int(getattr(config, "RESP_HYST_COOLDOWN_TICKS", 30))
RESPAWN_MODE = str(getattr(config, "RESPAWN_MODE", "uniform")).lower()
RESP_WALL_MARGIN = int(getattr(config, "RESP_WALL_MARGIN", 2))
RESP_CLONE_PROB = float(getattr(config, "RESP_CLONE_PROB", 0.50))
RESP_NOISE_STD = float(getattr(config, "RESP_NOISE_STD", 0.02))
UNIT_SOLDIER = int(getattr(config, "UNIT_SOLDIER", 1))
UNIT_ARCHER = int(getattr(config, "UNIT_ARCHER", 2))
SPAWN_ARCHER_RATIO = float(getattr(config, "SPAWN_ARCHER_RATIO", 0.40))
SOLDIER_HP = float(getattr(config, "SOLDIER_HP", 1.0))
SOLDIER_ATK = float(getattr(config, "SOLDIER_ATK", 0.05))
ARCHER_HP = float(getattr(config, "ARCHER_HP", 1.0))
ARCHER_ATK = float(getattr(config, "ARCHER_ATK", 0.02))
VISION_SOLDIER = int(getattr(config, "VISION_RANGE_BY_UNIT", {}).get(1, 10))
VISION_ARCHER = int(getattr(config, "VISION_RANGE_BY_UNIT", {}).get(2, 15))
_MAX_TRIES_PER_AGENT = 32


@dataclass
class TeamCounts:
    red: int
    blue: int

def _team_counts(reg: AgentsRegistry) -> TeamCounts:
    d = reg.agent_data
    alive = (d[:, COL_ALIVE] > 0.5)
    red = int((alive & (d[:, COL_TEAM] == TEAM_RED_ID)).sum().item())
    blue = int((alive & (d[:, COL_TEAM] == TEAM_BLUE_ID)).sum().item())
    return TeamCounts(red=red, blue=blue)

def _inverse_split(a: int, b: int, budget: int) -> Tuple[int, int]:
    a = max(1, a); b = max(1, b)
    s = 1.0 / a + 1.0 / b
    qa = int(round(budget * (1.0 / a) / s))
    return qa, budget - qa

def _cap(n: int) -> int:
    return max(0, min(n, RESP_MAX_PER_TICK))

def _new_brain(device: torch.device):
    """Creates a new TransformerBrain, scripted or eager based on config."""
    obs_dim = (64 * 8) + 21  # Hardcoded for the new architecture
    act_dim = int(getattr(config, "NUM_ACTIONS", 41))
    if bool(getattr(config, "PPO_ENABLED", False)):
        return TransformerBrain(obs_dim, act_dim).to(device)
    else:
        return scripted_transformer_brain(obs_dim, act_dim).to(device)

def _clone_brain(parent, device: torch.device):
    """Clones a parent brain, lifting from script to eager if PPO is on."""
    if parent is None:
        return _new_brain(device)
    
    obs_dim = (64 * 8) + 21
    act_dim = int(getattr(config, "NUM_ACTIONS", 41))
    
    is_script = isinstance(parent, torch.jit.ScriptModule)
    is_ppo = bool(getattr(config, "PPO_ENABLED", False))

    if is_ppo and is_script:
        child = TransformerBrain(obs_dim, act_dim).to(device)
        child.load_state_dict(parent.state_dict())
        return child
    try:
        return copy.deepcopy(parent)
    except Exception:
        return _new_brain(device)

@torch.no_grad()
def _perturb_brain_(m: torch.nn.Module) -> None:
    """Injects small Gaussian noise into brain parameters."""
    if m is None or RESP_NOISE_STD <= 0.0:
        return
    for p in m.parameters():
        if p.is_floating_point():
            noise = torch.randn_like(p) * RESP_NOISE_STD
            p.add_(noise)

def _cell_free(grid: torch.Tensor, x: int, y: int) -> bool:
    H, W = grid.size(1), grid.size(2)
    return (RESP_WALL_MARGIN <= x < W - RESP_WALL_MARGIN and
            RESP_WALL_MARGIN <= y < H - RESP_WALL_MARGIN and
            float(grid[0, y, x].item()) == 0.0)

def _pick_uniform(grid: torch.Tensor) -> Tuple[int, int]:
    H, W = grid.size(1), grid.size(2)
    for _ in range(_MAX_TRIES_PER_AGENT):
        x = random.randint(RESP_WALL_MARGIN, W - RESP_WALL_MARGIN - 1)
        y = random.randint(RESP_WALL_MARGIN, H - RESP_WALL_MARGIN - 1)
        if _cell_free(grid, x, y):
            return x, y
    return -1, -1

def _pick_location(grid: torch.Tensor) -> Tuple[int, int]:
    return _pick_uniform(grid)

def _choose_unit() -> int:
    return UNIT_ARCHER if random.random() < SPAWN_ARCHER_RATIO else UNIT_SOLDIER

def _unit_stats(unit_id: int) -> Tuple[float, float, int]:
    if int(unit_id) == UNIT_ARCHER:
        return ARCHER_HP, ARCHER_ATK, VISION_ARCHER
    return SOLDIER_HP, SOLDIER_ATK, VISION_SOLDIER

def _write_agent_to_registry(
    reg: AgentsRegistry, slot: int, team_id: float, x: int, y: int,
    unit_id: int, hp: float, atk: float, vision: int, brain: torch.nn.Module
) -> None:
    reg.register(
        slot, team_is_red=(team_id == TEAM_RED_ID), x=x, y=y,
        hp=hp, atk=atk, brain=brain, unit=unit_id,
        hp_max=hp, vision_range=vision, generation=0
    )

@torch.no_grad()
def _respawn_some(reg: AgentsRegistry, grid: torch.Tensor, team_id: float, count: int) -> int:
    global _respawn_counter
    if count <= 0:
        return 0

    device = grid.device
    data = reg.agent_data
    alive = (data[:, COL_ALIVE] > 0.5)
    dead_slots = (~alive).nonzero(as_tuple=False).squeeze(1)
    if dead_slots.numel() == 0:
        return 0

    parents = (alive & (data[:, COL_TEAM] == team_id)).nonzero(as_tuple=False).squeeze(1)
    spawned = 0
    to_spawn = min(count, dead_slots.numel())

    for k in range(to_spawn):
        slot = int(dead_slots[k].item())
        x, y = _pick_location(grid)
        if x < 0:
            break

        unit_id = _choose_unit()
        hp0, atk0, vision0 = _unit_stats(unit_id)

        _respawn_counter += 1
        is_rare_mutation = (_respawn_counter % 1000 == 0)

        if is_rare_mutation:
            hp0 *= (1.0 + random.uniform(0.5, 2.0))
            atk0 *= (1.0 + random.uniform(0.5, 2.0))
            vision0 = int(vision0 * (1.0 + random.uniform(0.5, 2.0)))
            print(f"** Rare Mutation on agent {slot}! HP:{hp0:.2f}, ATK:{atk0:.2f}, VIS:{vision0} **")

        use_clone = (parents.numel() > 0) and (random.random() < RESP_CLONE_PROB)
        if use_clone:
            pj = int(parents[random.randrange(parents.numel())].item())
            brain = _clone_brain(reg.brains[pj], device)
            _perturb_brain_(brain)
        else:
            brain = _new_brain(device)

        _write_agent_to_registry(reg, slot, team_id, x, y, unit_id, hp0, atk0, vision0, brain)

        # Also update the grid
        grid[0, y, x] = team_id
        grid[1, y, x] = hp0
        grid[2, y, x] = float(slot)
        spawned += 1

    return spawned

class RespawnController:
    def __init__(self, cooldown_ticks: int = 50):
        self.cooldown_ticks = int(getattr(config, "RESPAWN_COOLDOWN_TICKS", cooldown_ticks))
        self._cooldown_red_until = 0
        self._cooldown_blue_until = 0
        self._last_period_tick = 0

    def step(self, tick: int, reg: AgentsRegistry, grid: torch.Tensor) -> Tuple[int, int]:
        counts = _team_counts(reg)
        spawned_r, spawned_b = 0, 0

        if counts.red < RESP_FLOOR_PER_TEAM and tick >= self._cooldown_red_until:
            need = RESP_FLOOR_PER_TEAM - counts.red
            spawned = _respawn_some(reg, grid, TEAM_RED_ID, _cap(need))
            spawned_r += spawned
            if counts.red + spawned >= RESP_FLOOR_PER_TEAM:
                self._cooldown_red_until = tick + RESP_HYST_COOLDOWN_TICKS

        if counts.blue < RESP_FLOOR_PER_TEAM and tick >= self._cooldown_blue_until:
            need = RESP_FLOOR_PER_TEAM - counts.blue
            spawned = _respawn_some(reg, grid, TEAM_BLUE_ID, _cap(need))
            spawned_b += spawned
            if counts.blue + spawned >= RESP_FLOOR_PER_TEAM:
                self._cooldown_blue_until = tick + RESP_HYST_COOLDOWN_TICKS

        if tick - self._last_period_tick >= RESP_PERIOD_TICKS:
            self._last_period_tick = tick
            if counts.red + counts.blue > 0:
                q_r, q_b = _inverse_split(counts.red, counts.blue, RESP_PERIOD_BUDGET)
                spawned_r += _respawn_some(reg, grid, TEAM_RED_ID, _cap(q_r))
                spawned_b += _respawn_some(reg, grid, TEAM_BLUE_ID, _cap(q_b))

        return spawned_r, spawned_b

