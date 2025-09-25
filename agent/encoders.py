# codex_bellum/agent/encoders.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# optional dtype hint from config; defaults to float32
try:
    from .. import config
    _DTYPE = getattr(config, "TORCH_DTYPE", torch.float32)
except Exception:
    _DTYPE = torch.float32


class RingPositionalEncoding(nn.Module):
    """
    Fixed sin/cos encoding for the 8 ray directions (0..7), concatenated to features.
    dim must be even; we pack sin/cos pairs; if odd, we pad one zero feature.
    """
    def __init__(self, dim: int = 4):
        super().__init__()
        dim = int(max(0, dim))
        self.dim = dim
        if dim <= 0:
            self.register_buffer("pe", torch.zeros(8, 0), persistent=False)
            return

        half = dim // 2  # number of sin/cos pairs
        # positions 0..7 around the ring
        pos = torch.arange(8, dtype=_DTYPE).view(8, 1)                 # (8,1)
        freqs = torch.arange(max(1, half), dtype=_DTYPE).view(1, -1)   # (1,half)
        # geometric frequency progression works well for tiny rings
        angles = 2.0 * math.pi * pos / 8.0 * (2.0 ** freqs)            # (8,half)
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # (8,2*half)
        if pe.size(1) < dim:  # odd target -> pad one column
            pe = F.pad(pe, (0, dim - pe.size(1)))
        self.register_buffer("pe", pe.to(dtype=_DTYPE), persistent=False)

    def forward(self, x8: torch.Tensor) -> torch.Tensor:
        """
        x8: (B, 8, F)
        returns: (B, 8, F + dim)
        """
        if self.dim <= 0:
            return x8
        B = x8.size(0)
        pe = self.pe.unsqueeze(0).expand(B, -1, -1)  # (B,8,dim)
        return torch.cat([x8, pe.to(dtype=x8.dtype, device=x8.device)], dim=-1)


class TinyRayAttention(nn.Module):
    """
    Super-light 1-head self-attention over the 8 ray slots.
    If attn_dim <= 0, skip this module entirely at call site.
    """
    def __init__(self, dim_in: int, attn_dim: int = 16):
        super().__init__()
        # bias=False on q,k,v is intentional; we'll guard bias init below.
        self.q = nn.Linear(dim_in, attn_dim, bias=False)
        self.k = nn.Linear(dim_in, attn_dim, bias=False)
        self.v = nn.Linear(dim_in, dim_in,   bias=False)
        self.o = nn.Linear(dim_in, dim_in,   bias=True)

        # stable init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 8, D)
        q, k, v = self.q(x), self.k(x), self.v(x)  # (B,8,d), (B,8,d), (B,8,D)
        scale = float(max(1.0, q.size(-1))) ** 0.5
        att = torch.softmax(q @ k.transpose(1, 2) / scale, dim=-1)  # (B,8,8)
        out = self.o(att @ v) + x
        return out


class RayEncoder(nn.Module):
    """
    Encode the 64-dim first-hit ray block (8 rays × 8 per-ray features) into a compact context.
      • per-ray Linear(8→16) + optional Ring PE (dim=0 disables)
      • optional TinyRayAttention over 8 rays (attn_dim=0 disables)
      • flatten → Linear(8*D → 32)

    Designed to be extremely small (<< 5k params) so 2k agents are easy.
    """
    def __init__(
        self,
        per_ray_in: int = 8,    # features per ray (e.g., onehot6 + dist + hp = 8)
        proj_dim: int = 16,     # per-ray hidden size
        pe_dim: int = 4,        # appended PE per ray (0 disables)
        attn_dim: int = 16,     # 0 disables attention
        out_dim: int = 32,      # final context size
    ):
        super().__init__()
        self.per_ray_in = int(per_ray_in)
        self.proj = nn.Linear(self.per_ray_in, proj_dim, bias=True)
        self.use_pe = pe_dim > 0
        self.pe = RingPositionalEncoding(pe_dim) if self.use_pe else None
        inner_dim = proj_dim + (pe_dim if self.use_pe else 0)
        self.use_attn = attn_dim > 0
        self.attn = TinyRayAttention(inner_dim, attn_dim) if self.use_attn else nn.Identity()
        self.out = nn.Linear(inner_dim * 8, out_dim, bias=True)

        # stable init (guard bias=None)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, rays64: torch.Tensor) -> torch.Tensor:
        """
        rays64: (B, 64) -> treat as (B, 8, 8)
        returns: (B, out_dim)
        """
        B = rays64.size(0)
        x = rays64.view(B, 8, self.per_ray_in)  # (B,8,8)
        x = self.proj(x)                         # (B,8,proj_dim)
        if self.use_pe:
            x = self.pe(x)                       # (B,8,proj_dim+pe)
        x = self.attn(x)                         # (B,8,D)
        x = x.reshape(B, -1)                     # (B,8*D)
        return self.out(x)                       # (B,out_dim)
