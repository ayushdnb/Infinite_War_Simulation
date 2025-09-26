# codex_bellum/engine/respawn.py
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
)
# Team id constants: try to import if provided, else fall back.
try:
    from .agent_registry import TEAM_RED_ID, TEAM_BLUE_ID
except Exception:
    TEAM_RED_ID = 2.0
    TEAM_BLUE_ID = 3.0

from agent.brain import ActorCriticBrain, scripted_brain

# ----------------------------
# Config knobs (safe fallbacks)
# ----------------------------
RESP_FLOOR_PER_TEAM       = int(getattr(config, "RESP_FLOOR_PER_TEAM", 50))
RESP_MAX_PER_TICK         = int(getattr(config, "RESP_MAX_PER_TICK", 15))

# Wave-style top-ups (fronts form, fewer corner trickles)
RESP_PERIOD_TICKS         = int(getattr(config, "RESP_PERIOD_TICKS", 200))
RESP_PERIOD_BUDGET        = int(getattr(config, "RESP_PERIOD_BUDGET", 20))
RESP_HYST_COOLDOWN_TICKS  = int(getattr(config, "RESP_HYST_COOLDOWN_TICKS", 30))

# Spawn policy
#   'uniform'   : uniform interior with wall margin (recommended)
#   'frontline' : near enemy centroid but still within walls
#   'ally_ring' : legacy (near random ally) — routed to uniform to avoid camping loops
RESPAWN_MODE              = str(getattr(config, "RESPAWN_MODE", "uniform")).lower()
RESP_WALL_MARGIN          = int(getattr(config, "RESP_WALL_MARGIN", 2))

# Brain mixing on respawn (helps diversity)
RESP_CLONE_PROB           = float(getattr(config, "RESP_CLONE_PROB", 0.50))  # 0..1
RESP_NOISE_STD            = float(getattr(config, "RESP_NOISE_STD", 0.02))   # Gaussian std on cloned params
RESP_NOISE_MODE           = str(getattr(config, "RESP_NOISE_MODE", "relative"))
RESP_NOISE_PROB           = float(getattr(config, "RESP_NOISE_PROB", 1.0))
RESP_NOISE_CLAMP          = getattr(config, "RESP_NOISE_CLAMP", None)        # None or float

# Unit mix
UNIT_SOLDIER              = int(getattr(config, "UNIT_SOLDIER", 1))
UNIT_ARCHER               = int(getattr(config, "UNIT_ARCHER", 2))
SPAWN_ARCHER_RATIO        = float(getattr(config, "SPAWN_ARCHER_RATIO", 0.40))

SOLDIER_HP                = float(getattr(config, "SOLDIER_HP", 1.0))
SOLDIER_ATK               = float(getattr(config, "SOLDIER_ATK", 0.05))
ARCHER_HP                 = float(getattr(config, "ARCHER_HP", 1.0))
ARCHER_ATK                = float(getattr(config, "ARCHER_ATK", 0.02))

# Other caps
_MAX_TRIES_PER_AGENT      = 32  # to find a free cell


@dataclass
class TeamCounts:
    red: int
    blue: int


def _team_counts(reg: AgentsRegistry) -> TeamCounts:
    d = reg.agent_data
    alive = (d[:, COL_ALIVE] > 0.5)
    red  = int((alive & (d[:, COL_TEAM] == TEAM_RED_ID)).sum().item())
    blue = int((alive & (d[:, COL_TEAM] == TEAM_BLUE_ID)).sum().item())
    return TeamCounts(red=red, blue=blue)


def _inverse_split(a: int, b: int, budget: int) -> Tuple[int, int]:
    a = max(1, a); b = max(1, b)
    inv_a, inv_b = 1.0 / a, 1.0 / b
    s = inv_a + inv_b
    qa = int(round(budget * (inv_a / s)))
    qb = budget - qa
    return qa, qb


def _cap(n: int) -> int:
    return max(0, min(int(n), RESP_MAX_PER_TICK))


def _new_brain(device: torch.device):
    """
    When PPO is ON: return an eager, trainable model.
    Otherwise: return scripted model for faster inference.
    """
    if bool(getattr(config, "PPO_ENABLED", False)):
        return ActorCriticBrain(config.OBS_DIM, config.NUM_ACTIONS, hidden=64).to(device)
    else:
        return scripted_brain(config.OBS_DIM, config.NUM_ACTIONS, hidden=64).to(device)


def _clone_brain(parent, device: torch.device):
    """
    Clone a parent brain. If PPO is ON and parent is ScriptModule, lift to eager.
    """
    if parent is None:
        return _new_brain(device)

    # TorchScript detection
    is_script = isinstance(parent, torch.jit.ScriptModule) or \
                ("script" in parent.__class__.__name__.lower())

    if bool(getattr(config, "PPO_ENABLED", False)) and is_script:
        # Lift to eager + copy weights
        child = ActorCriticBrain(config.OBS_DIM, config.NUM_ACTIONS, hidden=64).to(device)
        child.load_state_dict(parent.state_dict())
        return child

    # Eager clone or keep as scripted clone (works with param-level requires_grad)
    try:
        return copy.deepcopy(parent)
    except Exception:
        # Fallback: new brain
        return _new_brain(device)


@torch.no_grad()
def _perturb_brain_(m: torch.nn.Module) -> None:
    """Add small Gaussian noise to floating-point params (in-place)."""
    if m is None or RESP_NOISE_STD <= 0.0:
        return
    # Probabilistic apply
    if RESP_NOISE_PROB < 1.0:
        # Use CPU RNG to avoid syncing CUDA
        if random.random() > RESP_NOISE_PROB:
            return

    for p in m.parameters():
        if not p.is_floating_point():
            continue
        eps = torch.randn_like(p)
        if RESP_NOISE_MODE == "relative":
            # Mean abs as scale proxy (avoid zero)
            scale = max(p.detach().abs().mean().item(), 1e-3)
            delta = eps * (RESP_NOISE_STD * scale)
        else:
            delta = eps * RESP_NOISE_STD
        p.add_(delta)
        if RESP_NOISE_CLAMP is not None:
            c = float(RESP_NOISE_CLAMP)
            p.clamp_(-c, c)


# ----------------------------
# Spawn location selection
# ----------------------------
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


def _pick_frontline(grid: torch.Tensor, team_id: float) -> Tuple[int, int]:
    """Spawn biased toward the side facing the enemy centroid (still within margins)."""
    H, W = grid.size(1), grid.size(2)
    enemy_id = TEAM_BLUE_ID if team_id == TEAM_RED_ID else TEAM_RED_ID
    occ = grid[0]
    ys, xs = torch.nonzero(occ == enemy_id, as_tuple=True)
    if xs.numel() == 0:
        return _pick_uniform(grid)
    cx = int(xs.float().mean().item())
    cy = int(ys.float().mean().item())

    for _ in range(_MAX_TRIES_PER_AGENT):
        jitter = 6
        if team_id == TEAM_RED_ID:
            x = cx - random.randint(3, 7) + random.randint(-jitter, jitter)
        else:
            x = cx + random.randint(3, 7) + random.randint(-jitter, jitter)
        y = cy + random.randint(-jitter, jitter)
        # clamp into margins
        x = max(RESP_WALL_MARGIN, min(W - RESP_WALL_MARGIN - 1, x))
        y = max(RESP_WALL_MARGIN, min(H - RESP_WALL_MARGIN - 1, y))
        if _cell_free(grid, x, y):
            return x, y
    return _pick_uniform(grid)


def _pick_location(grid: torch.Tensor, team_id: float) -> Tuple[int, int]:
    if RESPAWN_MODE == "frontline":
        return _pick_frontline(grid, team_id)
    # route "ally_ring" to uniform on purpose (avoid clustering/camping)
    return _pick_uniform(grid)


def _choose_unit() -> int:
    return UNIT_ARCHER if (random.random() < SPAWN_ARCHER_RATIO) else UNIT_SOLDIER


def _unit_stats(unit_id: int) -> Tuple[float, float]:
    if int(unit_id) == UNIT_ARCHER:
        return ARCHER_HP, ARCHER_ATK
    else:
        return SOLDIER_HP, SOLDIER_ATK


def _write_agent_row(
    reg: AgentsRegistry, grid: torch.Tensor, slot: int,
    team_id: float, x: int, y: int, unit_id: int, hp: float, atk: float, brain: torch.nn.Module
) -> None:
    # Grid
    grid[0, y, x] = float(team_id)
    grid[1, y, x] = float(hp)
    grid[2, y, x] = float(slot)

    # Table
    reg.agent_data[slot, COL_TEAM]  = float(team_id)
    reg.agent_data[slot, COL_X]     = float(x)
    reg.agent_data[slot, COL_Y]     = float(y)
    reg.agent_data[slot, COL_UNIT]  = float(unit_id)
    reg.agent_data[slot, COL_HP]    = float(hp)
    reg.agent_data[slot, COL_ATK]   = float(atk)
    reg.agent_data[slot, COL_ALIVE] = 1.0

    # Brain
    reg.brains[slot] = brain


@torch.no_grad()
def _respawn_some(reg: AgentsRegistry, grid: torch.Tensor, team_id: float, count: int) -> int:
    """
    Wave-friendly respawn:
      - uniform/frontline placement with wall margin
      - mix fresh brain and (lightly noised) clone brain per RESP_CLONE_PROB
      - unit mix Soldier/Archer via SPAWN_ARCHER_RATIO
    """
    if count <= 0:
        return 0

    device = grid.device
    data = reg.agent_data
    alive = (data[:, COL_ALIVE] > 0.5)
    dead_slots = (~alive).nonzero(as_tuple=False).squeeze(1)
    if dead_slots.numel() == 0:
        return 0

    # Potential clone parents (same team, alive). If none, all spawns are fresh.
    parents = (alive & (data[:, COL_TEAM] == team_id)).nonzero(as_tuple=False).squeeze(1)

    spawned = 0
    to_spawn = min(count, int(dead_slots.numel()))

    for k in range(to_spawn):
        slot = int(dead_slots[k].item())
        x, y = _pick_location(grid, team_id)
        if x < 0:
            break  # no space found

        # Unit & stats
        unit_id = _choose_unit()
        hp0, atk0 = _unit_stats(unit_id)

        # Brain
        use_clone = (parents.numel() > 0) and (random.random() < RESP_CLONE_PROB)
        if use_clone:
            pj = int(parents[random.randrange(parents.numel())].item())
            parent_brain = reg.brains[pj]
            brain = _clone_brain(parent_brain, device)
            _perturb_brain_(brain)
        else:
            brain = _new_brain(device)

        _write_agent_row(reg, grid, slot, team_id, x, y, unit_id, hp0, atk0, brain)
        spawned += 1

    return spawned


class RespawnController:
    """Single source of truth for respawn decisions (wave-style + team floors)."""

    def __init__(self, cooldown_ticks: int = 50):
        # We use team-level hysteresis/cooldowns (not per-agent) to keep fronts stable.
        self.cooldown_ticks = int(getattr(config, "RESPAWN_COOLDOWN_TICKS", cooldown_ticks))
        self._cooldown_red_until  = 0
        self._cooldown_blue_until = 0
        self._last_period_tick    = 0

    def _spawn_team(self, reg: AgentsRegistry, grid: torch.Tensor, team_id: float, k: int) -> int:
        return _respawn_some(reg, grid, team_id, k) if k > 0 else 0

    def step(self, tick: int, reg: AgentsRegistry, grid: torch.Tensor) -> Tuple[int, int]:
        counts = _team_counts(reg)
        spawned_r = 0
        spawned_b = 0

        # Floor keepers (with hysteresis per team)
        if counts.red < RESP_FLOOR_PER_TEAM and tick >= self._cooldown_red_until:
            need = RESP_FLOOR_PER_TEAM - counts.red
            spawned = self._spawn_team(reg, grid, TEAM_RED_ID, _cap(need))
            spawned_r += spawned
            if counts.red + spawned >= RESP_FLOOR_PER_TEAM:
                self._cooldown_red_until = tick + RESP_HYST_COOLDOWN_TICKS

        if counts.blue < RESP_FLOOR_PER_TEAM and tick >= self._cooldown_blue_until:
            need = RESP_FLOOR_PER_TEAM - counts.blue
            spawned = self._spawn_team(reg, grid, TEAM_BLUE_ID, _cap(need))
            spawned_b += spawned
            if counts.blue + spawned >= RESP_FLOOR_PER_TEAM:
                self._cooldown_blue_until = tick + RESP_HYST_COOLDOWN_TICKS

        # Periodic wave top-ups (inverse split → rubber-band toward parity)
        if tick - self._last_period_tick >= RESP_PERIOD_TICKS:
            self._last_period_tick = tick
            q_r, q_b = _inverse_split(counts.red, counts.blue, RESP_PERIOD_BUDGET)
            spawned_r += self._spawn_team(reg, grid, TEAM_RED_ID, _cap(q_r))
            spawned_b += self._spawn_team(reg, grid, TEAM_BLUE_ID, _cap(q_b))

        return spawned_r, spawned_b


# Back-compat simple API
def respawn_some(registry: AgentsRegistry, grid: torch.Tensor, team_is_red: bool, count: int) -> int:
    team_id = TEAM_RED_ID if team_is_red else TEAM_BLUE_ID
    return _respawn_some(registry, grid, team_id, count)
