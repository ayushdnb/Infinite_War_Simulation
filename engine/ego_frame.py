# final_war_sim/engine/ego_frame.py
from __future__ import annotations
from typing import Tuple, Optional, List
import torch

try:
    from .. import config
    _DTYPE = getattr(config, "TORCH_DTYPE", torch.float32)
except Exception:
    _DTYPE = torch.float32

# Global 8-neighborhood, consistent with move mask & action layout:
# N, NE, E, SE, S, SW, W, NW
DIRS8 = torch.tensor([
    [ 0, -1],
    [ 1, -1],
    [ 1,  0],
    [ 1,  1],
    [ 0,  1],
    [-1,  1],
    [-1,  0],
    [-1, -1],
], dtype=torch.long)


@torch.no_grad()
def compute_heading8(
    pos_xy: torch.Tensor,                 # (N,2) long
    last_pos_xy: torch.Tensor,            # (N,2) long — previous tick (sticky across ticks)
    prev_heading8: torch.Tensor,          # (N,) long in {0..7}; sticky
    *,
    team_ids: Optional[torch.Tensor] = None,   # (N,) float 2.0 red / 3.0 blue
    rays64: Optional[torch.Tensor] = None,     # (N,64) first-hit rays (onehot6 + dist + hp)*8
) -> torch.Tensor:
    """
    Determine per-agent heading index k ∈ {0..7} with this priority:
      1) If the agent moved this tick: nearest DIRS8 to its displacement.
      2) Else if rays64 given: use the ray dir of nearest visible ENEMY first-hit.
      3) Else: keep prev_heading8.
    """
    device = pos_xy.device
    N = pos_xy.size(0)
    if N == 0:
        return prev_heading8

    dxy = (pos_xy - last_pos_xy).to(torch.float32)   # (N,2)
    moved = (dxy.abs().sum(dim=1) > 0)

    # (1) movement-based heading (argmax dot with DIRS8)
    dirs_f = DIRS8.to(device=device, dtype=torch.float32)  # (8,2)
    # avoid zero norm
    norm = torch.clamp(torch.linalg.vector_norm(dxy, dim=1, keepdim=True), min=1.0)
    vec = dxy / norm
    cos = (vec.unsqueeze(1) * dirs_f.unsqueeze(0)).sum(dim=-1)  # (N,8)
    k_move = torch.argmax(cos, dim=1)                           # (N,)

    k = torch.where(moved, k_move, prev_heading8)

    # (2) enemy-ray fallback when stationary
    need = (~moved)
    if rays64 is not None and team_ids is not None and torch.any(need):
        # rays64 layout per ray: [onehot6 (0:none,1:wall,2:red-s,3:red-a,4:blue-s,5:blue-a), dist_norm, hp_norm]
        # reshape to (N,8,8)
        r = rays64.view(N, 8, 8)

        # Build "enemy seen?" mask per ray using team ids
        teams = team_ids.to(device=device)
        # class channels 2..5 are entity types by team/unit
        red_mask  = (r[..., 2] > 0.5) | (r[..., 3] > 0.5)
        blue_mask = (r[..., 4] > 0.5) | (r[..., 5] > 0.5)
        # For red agents, enemy is blue_*; for blue agents, enemy is red_*
        is_red  = (teams == 2.0).view(N, 1)
        enemy = torch.where(is_red, blue_mask, red_mask)  # (N,8)

        # use distance channel at dim index 6
        dist = r[..., 6]
        # large penalty if no enemy on that ray
        dist_w = torch.where(enemy, dist, torch.full_like(dist, 10.0))
        k_enemy = torch.argmin(dist_w, dim=1)  # nearest enemy direction (or arbitrary if none)

        k = torch.where(need, k_enemy, k)

    return k.to(dtype=torch.long)


@torch.no_grad()
def rotate_rays_to_heading(rays64: torch.Tensor, heading8: torch.Tensor) -> torch.Tensor:
    """
    Rotate 8 rays so that index 0 corresponds to the agent's 'ahead' direction.
    rays64:  (N,64)  as (N,8 rays, 8 feats)
    heading8:(N,) in {0..7}
    returns: (N,64) rotated left by heading (i.e., ego-centric)
    """
    if rays64.numel() == 0:
        return rays64
    device = rays64.device
    N = rays64.size(0)
    r = rays64.view(N, 8, 8)
    # roll each row by -k along ray axis using gather (no data movement)
    idx = (torch.arange(8, device=device).unsqueeze(0) - heading8.view(-1, 1)) % 8  # (N,8)
    r_rot = torch.gather(r, 1, idx.unsqueeze(-1).expand(-1, -1, 8))
    return r_rot.reshape(N, 64)


@torch.no_grad()
def _group_slices_for_actions(num_actions: int) -> List[slice]:
    """
    Returns list of 8-wide directional slices that must be un-rotated back to global.
      For 17 actions: [1..8] moves, [9..16] melee.
      For >=41 actions: plus 3 more 8-wide ranged rings, if present.
    """
    groups: List[slice] = []
    if num_actions <= 1:
        return groups
    # moves: 1..8
    groups.append(slice(1, min(9, num_actions)))
    if num_actions <= 9:
        return groups
    # then 8-wide blocks from 9 upward
    start = 9
    while start + 8 <= num_actions:
        end = start + 8
        groups.append(slice(start, end))
        start = end
    return groups


@torch.no_grad()
def unrotate_logits_by_heading(
    logits: torch.Tensor,      # (B,A) ego-centric directional layout in slices
    heading8: torch.Tensor,    # (B,)
) -> torch.Tensor:
    """
    Apply inverse rotation (+k) to each 8-wide directional group so that action indices
    remain GLOBAL (N, NE, E, …). Non-directional columns (e.g. idle=0) are untouched.
    """
    if logits.numel() == 0:
        return logits
    A = int(logits.size(1))
    out = logits.clone()
    groups = _group_slices_for_actions(A)
    if not groups:
        return out

    device = logits.device
    B = logits.size(0)
    roll_idx = (torch.arange(8, device=device).unsqueeze(0) + heading8.view(-1, 1)) % 8  # (B,8)

    for sl in groups:
        width = sl.stop - sl.start
        if width < 8:
            # partial groups at tail (rare) — skip rotation
            continue
        blk = logits[:, sl]  # (B,8)
        out[:, sl] = torch.gather(blk, 1, roll_idx)
    return out
