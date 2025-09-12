# final_war_sim/engine/respawn.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import random
import copy
import torch

from .agent_registry import (
    AgentsRegistry,
    COL_ALIVE, COL_TEAM, COL_X, COL_Y, COL_HP, COL_ATK,
    TEAM_RED_ID, TEAM_BLUE_ID
)
from .. import config
from ..agent.brain import scripted_brain

# ----------------------------
# Config knobs (fallbacks)
# ----------------------------
RESP_FLOOR_PER_TEAM      = getattr(config, "RESP_FLOOR_PER_TEAM", 50)
RESP_MAX_PER_TICK        = getattr(config, "RESP_MAX_PER_TICK", 15)
RESP_PERIOD_TICKS        = getattr(config, "RESP_PERIOD_TICKS", 200)
RESP_PERIOD_BUDGET       = getattr(config, "RESP_PERIOD_BUDGET", 20)
RESP_HYST_COOLDOWN_TICKS = getattr(config, "RESP_HYST_COOLDOWN_TICKS", 30)

# Spawn attempt parameters
_SPAWN_NEAR_RADII = (1, 2, 3)  # try these radii around an ally
_MAX_TRIES_PER_AGENT = 20      # attempts to find a free cell

@dataclass
class TeamCounts:
    red: int
    blue: int


def _team_counts(reg: AgentsRegistry) -> TeamCounts:
    """Alive counts per team from registry SoA."""
    data = reg.agent_data
    alive = (data[:, COL_ALIVE] > 0.5)
    teams = data[:, COL_TEAM]
    red  = int((alive & (teams == TEAM_RED_ID)).sum().item())
    blue = int((alive & (teams == TEAM_BLUE_ID)).sum().item())
    return TeamCounts(red=red, blue=blue)


def _inverse_split(a: int, b: int, budget: int) -> Tuple[int, int]:
    """Inverse-proportional split of `budget` between counts a,b."""
    a = max(1, a); b = max(1, b)
    inv_a, inv_b = 1.0 / a, 1.0 / b
    s = inv_a + inv_b
    qa = int(round(budget * (inv_a / s)))
    qb = budget - qa
    return qa, qb


def _cap(n: int) -> int:
    return max(0, min(int(n), RESP_MAX_PER_TICK))


def _mk_default_brain(device: torch.device):
    """Small starting brain."""
    return scripted_brain(config.OBS_DIM, config.NUM_ACTIONS, hidden=64).to(device)


def _deepclone_brain(m):
    """Safe brain clone (no parameter sharing)."""
    return copy.deepcopy(m) if m is not None else None


@torch.no_grad()
def _respawn_some(reg: AgentsRegistry, grid: torch.Tensor, team_id: float, count: int) -> int:
    """
    Core respawn primitive:
      - fills DEAD slots,
      - chooses a free cell near alive allies (fallback: random interior),
      - clones nearest ally brain; if none, uses default tiny brain,
      - writes registry + grid.
    Returns number actually spawned.
    """
    if count <= 0:
        return 0

    device = reg.device
    data = reg.agent_data
    H, W = grid.size(1), grid.size(2)

    alive = (data[:, COL_ALIVE] > 0.5)
    dead  = ~alive
    dead_slots = dead.nonzero(as_tuple=False).squeeze(1)
    if dead_slots.numel() == 0:
        return 0

    # Allies currently alive
    allies = (alive & (data[:, COL_TEAM] == team_id)).nonzero(as_tuple=False).squeeze(1)

    # helper: free?
    def _cell_free(x: int, y: int) -> bool:
        # occupancy channel: 0 empty, 1 wall, 2 red, 3 blue
        return (0 <= x < W) and (0 <= y < H) and (grid[0, y, x].item() == 0.0)

    # helper: pick a location
    def _pick_location() -> Tuple[int, int]:
        # Prefer near an ally
        if allies.numel() > 0:
            ai = int(allies[random.randrange(allies.numel())].item())
            ax = int(data[ai, COL_X].item())
            ay = int(data[ai, COL_Y].item())

            # try expanding rings
            for r in _SPAWN_NEAR_RADII:
                # sample some offsets in the ring square
                for _ in range(8):
                    dx = random.randint(-r, r)
                    dy = random.randint(-r, r)
                    x = max(0, min(W - 1, ax + dx))
                    y = max(0, min(H - 1, ay + dy))
                    if _cell_free(x, y):
                        return x, y

        # Fallback: random interior search
        for _ in range(_MAX_TRIES_PER_AGENT):
            x = random.randint(1, W - 2)
            y = random.randint(1, H - 2)
            if _cell_free(x, y):
                return x, y
        return -1, -1  # no space

    spawned = 0
    to_spawn = min(count, int(dead_slots.numel()))
    hp0 = float(config.MAX_HP)
    atk0 = float(config.BASE_ATK)

    for k in range(to_spawn):
        slot = int(dead_slots[k].item())

        x, y = _pick_location()
        if x < 0:
            # map is jammed; stop early
            break

        # choose brain: clone ally if available else default
        if allies.numel() > 0:
            # pick nearest ally to (x, y)
            ax = data[allies, COL_X]
            ay = data[allies, COL_Y]
            dx = ax - float(x)
            dy = ay - float(y)
            d2 = dx * dx + dy * dy
            j = int(torch.argmin(d2).item())
            ally_idx = int(allies[j].item())
            brain = _deepclone_brain(reg.brains[ally_idx])
            if brain is None:
                brain = _mk_default_brain(device)
        else:
            brain = _mk_default_brain(device)

        # write registry & grid
        team_is_red = (team_id == TEAM_RED_ID)
        reg.register(slot, team_is_red=team_is_red, x=x, y=y, hp=hp0, atk=atk0, brain=brain)
        grid[0, y, x] = team_id
        grid[1, y, x] = hp0
        grid[2, y, x] = float(slot)

        spawned += 1

    return spawned


class RespawnController:
    """
    Single source of truth for respawn decisions.
    Call step(tick, registry, grid) once per tick, after deaths are applied.
    Uses local _respawn_some(...) primitive (no registry method required).
    """

    def __init__(self):
        self._last_period_tick = -10**9
        self._cooldown_red_until = -10**9
        self._cooldown_blue_until = -10**9

    def _spawn_near_friendly(self, reg: AgentsRegistry, grid: torch.Tensor, team_id: float, k: int) -> int:
        if k <= 0:
            return 0
        return _respawn_some(reg, grid, team_id, k)

    def step(self, tick: int, reg: AgentsRegistry, grid: torch.Tensor) -> Tuple[int, int]:
        counts = _team_counts(reg)
        spawned_r = 0
        spawned_b = 0

        # --- Floor: keep population above minimum per-team ---
        if counts.red < RESP_FLOOR_PER_TEAM and tick >= self._cooldown_red_until:
            need = RESP_FLOOR_PER_TEAM - counts.red
            spawned = self._spawn_near_friendly(reg, grid, TEAM_RED_ID, _cap(need))
            spawned_r += spawned
            if counts.red + spawned >= RESP_FLOOR_PER_TEAM:
                self._cooldown_red_until = tick + RESP_HYST_COOLDOWN_TICKS

        if counts.blue < RESP_FLOOR_PER_TEAM and tick >= self._cooldown_blue_until:
            need = RESP_FLOOR_PER_TEAM - counts.blue
            spawned = self._spawn_near_friendly(reg, grid, TEAM_BLUE_ID, _cap(need))
            spawned_b += spawned
            if counts.blue + spawned >= RESP_FLOOR_PER_TEAM:
                self._cooldown_blue_until = tick + RESP_HYST_COOLDOWN_TICKS

        # --- Periodic inverse top-up: gentle balancing ---
        if tick - self._last_period_tick >= RESP_PERIOD_TICKS:
            self._last_period_tick = tick
            q_r, q_b = _inverse_split(counts.red, counts.blue, RESP_PERIOD_BUDGET)
            if q_r > 0:
                spawned_r += self._spawn_near_friendly(reg, grid, TEAM_RED_ID, _cap(q_r))
            if q_b > 0:
                spawned_b += self._spawn_near_friendly(reg, grid, TEAM_BLUE_ID, _cap(q_b))

        return spawned_r, spawned_b


# -------------------------------------------------------------------------
# Back-compat wrapper: keep old API alive for main.py imports
# -------------------------------------------------------------------------
def respawn_some(registry: AgentsRegistry, grid: torch.Tensor, team_is_red: bool, count: int) -> int:
    team_id = TEAM_RED_ID if team_is_red else TEAM_BLUE_ID
    return _respawn_some(registry, grid, team_id, count)
