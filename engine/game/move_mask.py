# final_war_sim/engine/game/move_mask.py
from __future__ import annotations
import torch

from ... import config  # uses MAX_HP, TORCH_DTYPE, etc.

# === Action indices (17 total) ===============================================
IDLE_IDX = 0
MOVE_START = 1          # 1..8  (8-way move)
MOVE_END   = 9
ATK_START  = 9          # 9..16 (8-way attack)
ATK_END    = 17

# 8 directions: (dx, dy) with y downwards, x rightwards
# order must be consistent across engine (viewer arrows etc.)
DIRS8 = torch.tensor([
    [ 0, -1],  # N
    [ 1, -1],  # NE
    [ 1,  0],  # E
    [ 1,  1],  # SE
    [ 0,  1],  # S
    [-1,  1],  # SW
    [-1,  0],  # W
    [-1, -1],  # NW
], dtype=torch.long)

# Occupancy channel encoding in grid[0]: 0 empty, 2 red, 3 blue
OCC_EMPTY = 0
OCC_RED   = 2
OCC_BLUE  = 3

TEAM_RED_ID  = 2.0
TEAM_BLUE_ID = 3.0


@torch.no_grad()
def build_mask(pos_xy: torch.Tensor, teams: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """
    Build a boolean mask of legal actions per agent.
    Inputs:
      pos_xy: (N,2) long tensor with [x,y]
      teams : (N,) float/long with 2.0 for red, 3.0 for blue
      grid  : (C,H,W) tensor where C>=1 and grid[0] is occupancy as ints {0,2,3}
    Output:
      mask  : (N,17) bool tensor, True = legal
              idx=0 idle
              idx=1..8  : move N,NE,E,SE,S,SW,W,NW if empty
              idx=9..16 : attack in those dirs if enemy present
    """
    device = grid.device
    occ = grid[0]  # (H,W)
    H, W = occ.shape[-2], occ.shape[-1]

    # Ensure types
    if pos_xy.dtype != torch.long:
        pos_xy = pos_xy.long()
    if teams.dtype.is_floating_point:
        teams_long = teams.long()
    else:
        teams_long = teams

    N = pos_xy.size(0)

    # (N,1,2) + (1,8,2) -> (N,8,2)
    base = pos_xy.view(N, 1, 2).to(device)
    d8   = DIRS8.to(device).view(1, 8, 2)
    neigh = base + d8  # (N,8,2)

    nx = neigh[..., 0].clamp_(0, W - 1)
    ny = neigh[..., 1].clamp_(0, H - 1)

    # Advanced indexing is safe & fast (no gather OOB asserts)
    n_occ = occ[ny, nx]  # (N,8)

    # Own occupancy id per agent
    own_occ = torch.where(
        teams_long.to(device) == int(TEAM_RED_ID),
        torch.tensor(OCC_RED, device=device, dtype=n_occ.dtype),
        torch.tensor(OCC_BLUE, device=device, dtype=n_occ.dtype),
    ).view(N, 1)  # (N,1) for broadcast

    # Legal moves: target cell must be empty
    move_ok = (n_occ == OCC_EMPTY)  # (N,8)

    # Legal attacks: target cell must be enemy (not empty, not own)
    atk_ok = (n_occ != OCC_EMPTY) & (n_occ != own_occ)  # (N,8)

    # Compose mask
    mask = torch.zeros((N, 17), dtype=torch.bool, device=device)
    mask[:, IDLE_IDX] = True
    mask[:, MOVE_START:MOVE_END] = move_ok
    mask[:, ATK_START:ATK_END]   = atk_ok
    return mask
ACTION = {
    "IDLE": IDLE_IDX,

    # === Moves (8-way) ===
    "MOVE_N": MOVE_START + 0,
    "MOVE_NE": MOVE_START + 1,
    "MOVE_E": MOVE_START + 2,
    "MOVE_SE": MOVE_START + 3,
    "MOVE_S": MOVE_START + 4,
    "MOVE_SW": MOVE_START + 5,
    "MOVE_W": MOVE_START + 6,
    "MOVE_NW": MOVE_START + 7,

    # Cardinal synonyms
    "MOVE_UP":    MOVE_START + 0,
    "MOVE_RIGHT": MOVE_START + 2,
    "MOVE_DOWN":  MOVE_START + 4,
    "MOVE_LEFT":  MOVE_START + 6,

    # === Attacks (8-way) ===
    "ATK_N": ATK_START + 0,
    "ATK_NE": ATK_START + 1,
    "ATK_E": ATK_START + 2,
    "ATK_SE": ATK_START + 3,
    "ATK_S": ATK_START + 4,
    "ATK_SW": ATK_START + 5,
    "ATK_W": ATK_START + 6,
    "ATK_NW": ATK_START + 7,

    # Synonyms (short)
    "ATK_UP":    ATK_START + 0,
    "ATK_RIGHT": ATK_START + 2,
    "ATK_DOWN":  ATK_START + 4,
    "ATK_LEFT":  ATK_START + 6,

    # Synonyms (long — tick.py uses these)
    "ATTACK_UP":    ATK_START + 0,
    "ATTACK_RIGHT": ATK_START + 2,
    "ATTACK_DOWN":  ATK_START + 4,
    "ATTACK_LEFT":  ATK_START + 6,
}

