# final_war_sim/engine/ego_tick_adapter.py
from __future__ import annotations
from typing import Optional, Sequence, Tuple
import torch

try:
    from .. import config
    _DTYPE = getattr(config, "TORCH_DTYPE", torch.float32)
    _REL = bool(getattr(config, "RELATIVE_DIRS", True))
except Exception:
    _DTYPE = torch.float32
    _REL = True

# Reuse our proven, tiny, vectorized primitives
from .ego_frame import compute_heading8, rotate_rays_to_heading, unrotate_logits_by_heading


class EgoFrameRuntime:
    """
    Per-simulation runtime adapter for ego-centric obs & directional-logit unrotation.
    This is 100% per-agent. No parameter sharing, no PPO changes.

    Usage inside your tick loop (pseudocode):
        ego = EgoFrameRuntime(capacity=registry.capacity, device=self.device)

        # at the start of the tick:
        alive_idx = ... (LongTensor [K])
        pos_xy    = registry.positions_xy(alive_idx)        # Long [K,2]
        teams     = registry.teams_of(alive_idx).to(float)  # Float [K], e.g., 2.0 red, 3.0 blue

        # after you built V2 obs (K,85) => 64-ray block at [:, :64]
        obs = ego.rotate_obs64_inplace(alive_idx, pos_xy, teams, obs, rays64=None)
        # NOTE: if you don't have rays64 separately, pass obs (we slice [:, :64])

        # per-bucket forward -> logits
        logits = ...  # (K, A)
        logits = ego.unrotate_logits_inplace(alive_idx, logits)

        # then apply your existing mask & sample as usual
    """

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.device   = device

        self._last_pos_xy = torch.zeros((self.capacity, 2), dtype=torch.long, device=self.device)
        self._heading8    = torch.zeros((self.capacity,), dtype=torch.long, device=self.device)
        self._has_heading = torch.zeros((self.capacity,), dtype=torch.bool, device=self.device)

        # scratch buffers reused each tick to avoid allocs
        self._scratch_alive: Optional[torch.Tensor] = None

    @torch.no_grad()
    def _ensure_alive_scratch(self, n: int) -> None:
        if self._scratch_alive is None or self._scratch_alive.numel() < n:
            self._scratch_alive = torch.empty((n,), dtype=torch.long, device=self.device)

    @torch.no_grad()
    def update_headings(
        self,
        alive_idx: torch.Tensor,        # [K] long absolute indices into registry tables
        pos_xy: torch.Tensor,           # [K,2] long, current positions
        teams: Optional[torch.Tensor],  # [K] float team ids (2.0 red / 3.0 blue) or None
        rays64: Optional[torch.Tensor] = None,  # [K,64] first-hit rays in GLOBAL order (N..NW)
    ) -> torch.Tensor:
        """
        Compute per-agent heading8 for alive agents and cache sticky state.
        returns: heading8 for alive agents, shape [K], long in {0..7}
        """
        K = alive_idx.numel()
        if K == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)

        last_pos = self._last_pos_xy[alive_idx]   # [K,2]
        prev_h   = self._heading8[alive_idx]      # [K]

        h = compute_heading8(
            pos_xy=pos_xy,
            last_pos_xy=last_pos,
            prev_heading8=prev_h,
            team_ids=teams,
            rays64=rays64,
        )  # [K] long

        # write back caches
        self._heading8[alive_idx]    = h
        self._has_heading[alive_idx] = True
        self._last_pos_xy[alive_idx] = pos_xy
        return h

    @torch.no_grad()
    def rotate_obs64_inplace(
        self,
        alive_idx: torch.Tensor,      # [K]
        pos_xy: torch.Tensor,         # [K,2]
        teams: Optional[torch.Tensor],# [K] float or None
        obs: torch.Tensor,            # [K, D] (expects V2 if D>=85: first 64 dims are rays)
        rays64: Optional[torch.Tensor] = None,   # [K,64] (optional)
    ) -> torch.Tensor:
        """
        Rotate only the first 64 dims (8 rays x 8 feats) so index-0 = 'ahead'.
        Dims >=85 are assumed: 64 ray block + 21 rich features (your V2).

        Returns a *new* tensor (same storage if contiguous) with the first 64 dims rotated.
        """
        if not _REL:
            return obs  # feature disabled

        K, D = obs.shape
        if D < 64:
            return obs  # legacy/no-ray or not using V2

        h = self.update_headings(
            alive_idx=alive_idx,
            pos_xy=pos_xy,
            teams=teams,
            rays64=(rays64 if rays64 is not None else obs[:, :64]),
        )  # [K]
        # rotate first 64 dims
        rays = (rays64 if rays64 is not None else obs[:, :64])
        rays_ego = rotate_rays_to_heading(rays, h)      # [K,64]
        if rays64 is not None:
            # caller owns the concat
            return rays_ego
        # splice back into obs
        out = obs.clone()
        out[:, :64] = rays_ego
        return out

    @torch.no_grad()
    def unrotate_logits_inplace(
        self,
        alive_idx: torch.Tensor,  # [K]
        logits: torch.Tensor,     # [K, A] (directional 8-wide groups in ego order)
    ) -> torch.Tensor:
        """
        Inverse rotation on all 8-wide directional groups so action indices stay GLOBAL.
        Call this BEFORE masking/sampling.
        """
        if not _REL:
            return logits
        if logits.numel() == 0:
            return logits
        h = self._heading8[alive_idx]  # [K], already computed this tick
        return unrotate_logits_by_heading(logits, h)
