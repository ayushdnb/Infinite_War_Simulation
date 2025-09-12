# final_war_sim/engine/spawn.py
from __future__ import annotations
from typing import Tuple, List
import math
import torch
from .. import config
from .agent_registry import AgentsRegistry
from ..agent.brain import scripted_brain

def _rect_dims(n: int, max_cols: int, max_rows: int) -> Tuple[int, int, int]:
    """
    Choose (cols, rows, n_eff) for a tight rectangle that fits within
    max_cols x max_rows and can place at least n agents if possible.
    """
    if max_cols <= 0 or max_rows <= 0 or n <= 0:
        return 0, 0, 0
    # start near square
    cols = min(max_cols, max(1, int(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    # shrink rows if needed
    if rows > max_rows:
        rows = max_rows
        cols = min(max_cols, int(math.ceil(n / rows)))
    n_eff = min(n, cols * rows)
    return cols, rows, n_eff

def spawn_symmetric(reg: AgentsRegistry, grid: torch.Tensor, per_team: int) -> None:
    """
    Rectangular formations on opposite sides:
      - RED rectangle on left half
      - BLUE rectangle on right half
    Capacity-safe and grid-bounds-safe.
    """
    H, W = grid.size(1), grid.size(2)
    margin = 1  # keep walls clear
    half_w = W // 2

    # --- capacity guard: don't exceed registry capacity ---
    per_team_cap = reg.capacity // 2
    per_team_grid_cap = (half_w - 2) * (H - 2)  # rough half-map capacity (1-cell spacing)
    per_team_eff = max(0, min(per_team, per_team_cap, per_team_grid_cap))

    if per_team_eff <= 0:
        return

    # build a brain factory
    def mk_brain():
        return scripted_brain(config.OBS_DIM, config.NUM_ACTIONS, hidden=64).to(reg.device)

    # --- RED rectangle (left half) ---
    red_max_cols = max(1, half_w - 2)        # within [1 .. half_w-2]
    red_max_rows = max(1, H - 2)             # within [1 .. H-2]
    r_cols, r_rows, r_n = _rect_dims(per_team_eff, red_max_cols, red_max_rows)
    red_x0 = margin + 0                       # start near left wall
    red_y0 = margin

    # --- BLUE rectangle (right half) ---
    blue_max_cols = max(1, half_w - 2)
    blue_max_rows = max(1, H - 2)
    b_cols, b_rows, b_n = _rect_dims(per_team_eff, blue_max_cols, blue_max_rows)
    blue_x0 = W - margin - b_cols             # flush to right side
    blue_y0 = margin

    slot = 0
    hp0, atk0 = float(config.MAX_HP), float(config.BASE_ATK)

    # RED fill (row-major)
    placed = 0
    for iy in range(r_rows):
        if placed >= r_n or slot >= reg.capacity: break
        for ix in range(r_cols):
            if placed >= r_n or slot >= reg.capacity: break
            x = red_x0 + ix
            y = red_y0 + iy
            if not (0 <= x < W and 0 <= y < H): continue
            reg.register(slot, team_is_red=True, x=x, y=y, hp=hp0, atk=atk0, brain=mk_brain())
            grid[0, y, x] = 2.0; grid[1, y, x] = hp0; grid[2, y, x] = float(slot)
            slot += 1
            placed += 1

    # BLUE fill (row-major)
    placed = 0
    for iy in range(b_rows):
        if placed >= b_n or slot >= reg.capacity: break
        for ix in range(b_cols):
            if placed >= b_n or slot >= reg.capacity: break
            x = blue_x0 + ix
            y = blue_y0 + iy
            if not (0 <= x < W and 0 <= y < H): continue
            reg.register(slot, team_is_red=False, x=x, y=y, hp=hp0, atk=atk0, brain=mk_brain())
            grid[0, y, x] = 3.0; grid[1, y, x] = hp0; grid[2, y, x] = float(slot)
            slot += 1
            placed += 1
