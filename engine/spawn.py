# final_war_sim/engine/spawn.py
from __future__ import annotations
import math
import random
from typing import Optional, Tuple

import torch

from .. import config
from .agent_registry import (
    AgentsRegistry,
    COL_TEAM, COL_X, COL_Y, COL_UNIT, COL_ALIVE,
)
from ..agent.brain import scripted_brain, ActorCriticBrain

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _rect_dims(n: int, max_cols: int, max_rows: int) -> Tuple[int, int, int]:
    """
    Choose (cols, rows, n_eff) for a compact rectangle within max_cols x max_rows
    that can place at least n agents if possible.
    """
    if max_cols <= 0 or max_rows <= 0 or n <= 0:
        return 0, 0, 0
    cols = min(max_cols, max(1, int(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    if rows > max_rows:
        rows = max_rows
        cols = min(max_cols, int(math.ceil(n / rows)))
    n_eff = min(n, cols * rows)
    return cols, rows, n_eff


def _mk_brain():
    """
    If PPO is enabled, return an eager nn.Module (trainable).
    Otherwise, return a TorchScript scripted brain (fast inference).
    """
    if bool(getattr(config, "PPO_ENABLED", False)):
        model = ActorCriticBrain(config.OBS_DIM, config.NUM_ACTIONS, hidden=64)
        return model.to(getattr(config, "TORCH_DEVICE", torch.device("cpu")))
    else:
        return scripted_brain(config.OBS_DIM, config.NUM_ACTIONS, hidden=64).to(getattr(config, "TORCH_DEVICE", torch.device("cpu")))



def _choose_unit(is_archer_prob: float) -> float:
    return float(config.UNIT_ARCHER if random.random() < is_archer_prob else config.UNIT_SOLDIER)


def _unit_stats(unit_val: float) -> Tuple[float, float]:
    """Return (hp, atk) for the given unit id (float)."""
    if int(unit_val) == int(config.UNIT_ARCHER):
        return float(config.ARCHER_HP), float(config.ARCHER_ATK)
    else:
        return float(config.SOLDIER_HP), float(config.SOLDIER_ATK)


# --- helpers ------------------------------------------------------------------
def _place_if_free(reg: AgentsRegistry, grid: torch.Tensor, slot: int, *, team_is_red: bool, x: int, y: int, unit_val: float) -> bool:
    occ = grid[0]
    if occ[y, x] != 0.0:
        return False
    # place on grid
    grid[0][y, x] = 2.0 if team_is_red else 3.0
    grid[1][y, x] = 1.0  # start HP
    grid[2][y, x] = float(slot)

    # write agent row
    reg.agent_data[slot, COL_TEAM] = 2.0 if team_is_red else 3.0
    reg.agent_data[slot, COL_X] = float(x)
    reg.agent_data[slot, COL_Y] = float(y)
    reg.agent_data[slot, COL_UNIT] = float(unit_val)
    reg.agent_data[slot, COL_ALIVE] = 1.0

    # attach a fresh brain (per-agent)
    brain = _mk_brain()
    reg.brains[slot] = brain
    return True


# ---------------------------------------------------------------------
# Public spawners
# ---------------------------------------------------------------------

def spawn_symmetric(reg: AgentsRegistry, grid: torch.Tensor, per_team: int) -> None:
    """
    Rectangular formations on opposite sides:
      - RED rectangle on left half
      - BLUE rectangle on right half
    Spawns a mix of Soldiers/Archers per team using SPAWN_ARCHER_RATIO.
    """
    H, W = grid.size(1), grid.size(2)
    margin = 1  # keep borders clear (outer walls)
    half_w = W // 2

    # Capacity guards
    per_team_cap = reg.capacity // 2
    per_team_grid_cap = max(0, (half_w - 2) * (H - 2))
    per_team_eff = max(0, min(per_team, per_team_cap, per_team_grid_cap))
    if per_team_eff <= 0:
        return

    # Archer mix
    try:
        ar_ratio = float(getattr(config, "SPAWN_ARCHER_RATIO", os.getenv("FWS_SPAWN_ARCHER_RATIO", 0.4)))
    except Exception:
        ar_ratio = 0.4
    ar_ratio = max(0.0, min(1.0, ar_ratio))

    # Red rectangle (left half)
    red_max_cols = max(1, half_w - 2)
    red_max_rows = max(1, H - 2)
    r_cols, r_rows, r_n = _rect_dims(per_team_eff, red_max_cols, red_max_rows)
    red_x0 = margin
    red_y0 = margin

    # Blue rectangle (right half)
    blue_max_cols = max(1, half_w - 2)
    blue_max_rows = max(1, H - 2)
    b_cols, b_rows, b_n = _rect_dims(per_team_eff, blue_max_cols, blue_max_rows)
    blue_x0 = W - margin - b_cols
    blue_y0 = margin

    slot = 0

    # RED (row-major)
    placed = 0
    for iy in range(r_rows):
        if placed >= r_n or slot >= reg.capacity:
            break
        for ix in range(r_cols):
            if placed >= r_n or slot >= reg.capacity:
                break
            x, y = red_x0 + ix, red_y0 + iy
            unit = _choose_unit(ar_ratio)
            if _place_if_free(reg, grid, slot, team_is_red=True, x=x, y=y, unit_val=unit):
                slot += 1
                placed += 1

    # BLUE (row-major)
    placed = 0
    for iy in range(b_rows):
        if placed >= b_n or slot >= reg.capacity:
            break
        for ix in range(b_cols):
            if placed >= b_n or slot >= reg.capacity:
                break
            x, y = blue_x0 + ix, blue_y0 + iy
            unit = _choose_unit(ar_ratio)
            if _place_if_free(reg, grid, slot, team_is_red=False, x=x, y=y, unit_val=unit):
                slot += 1
                placed += 1


# --- public API ---------------------------------------------------------------
def spawn_uniform_random(registry: AgentsRegistry, grid: torch.Tensor, per_team: int, *, unit: float = 1.0) -> None:
    """
    Spawn 'per_team' agents for each team at random free cells.
    """
    H, W = grid.size(1), grid.size(2)
    total = registry.capacity
    slot = 0

    # start red
    team_is_red = True
    for _ in range(per_team):
        for _try in range(1000):
            x = torch.randint(0, W//2, ()).item()
            y = torch.randint(0, H, ()).item()
            if _place_if_free(registry, grid, slot, team_is_red=team_is_red, x=x, y=y, unit_val=unit):
                slot += 1
                break

    # start blue
    team_is_red = False
    for _ in range(per_team):
        for _try in range(1000):
            x = torch.randint(W//2, W, ()).item()
            y = torch.randint(0, H, ()).item()
            if _place_if_free(registry, grid, slot, team_is_red=team_is_red, x=x, y=y, unit_val=unit):
                slot += 1
                break
