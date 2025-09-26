# final_war_sim/engine/ray_engine/raycaster2d.py
from __future__ import annotations
import torch
import config

# Grid channels
# 0: occupancy (0 empty, 1 wall, 2 red, 3 blue)
# 1: hp (0..MAX_HP)
# 2: agent index (-1 if none)

# 8 directions (dx, dy): R, RU, U, LU, L, LD, D, RD (clockwise)
DIRS = torch.tensor([
    [ 1,  0],
    [ 1, -1],
    [ 0, -1],
    [-1, -1],
    [-1,  0],
    [-1,  1],
    [ 0,  1],
    [ 1,  1],
], dtype=torch.long)

@torch.no_grad()
def raycast8(pos_xy: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    device = grid.device
    N = pos_xy.size(0)
    H, W = grid.size(1), grid.size(2)
    steps = int(config.RAY_MAX_STEPS)

    dirs = DIRS.to(device)  # (8,2)
    base = pos_xy.view(N, 1, 1, 2)            # (N,1,1,2)
    d    = dirs.view(1, 8, 1, 2)              # (1,8,1,2)
    s    = torch.arange(1, steps + 1, device=device, dtype=torch.long).view(1, 1, steps, 1)  # (1,1,S,1)

    coords = base + d * s                     # (N,8,S,2)
    x = coords[..., 0].clamp(0, W - 1)        # (N,8,S)
    y = coords[..., 1].clamp(0, H - 1)        # (N,8,S)

    # ✅ advanced indexing instead of gather
    occ = grid[0][y, x]                       # (N,8,S)
    hp  = grid[1][y, x]                       # (N,8,S)

    occ_norm = occ / 3.0
    hp_norm  = hp / float(config.MAX_HP)

    feat = torch.stack((occ_norm, hp_norm), dim=-1)  # (N,8,S,2)
    return feat.reshape(N, 8 * steps * 2).to(config.TORCH_DTYPE)