from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from .encoders import RayEncoder


class ActorCriticBrain(nn.Module):
    """
    Per-agent actor-critic.

    V2 obs (85): first 64 = rays (8×8 first-hit features), rest = rich self/env (21).
    Rays -> RayEncoder(32d), concat rich -> MLP trunk -> actor/critic.

    PPO/AMP safe: we cast inputs to float32 before any Linear/Encoder to avoid
    half/float matmul mismatches when inference used AMP.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        # Ray block present if obs >= 64
        self.has_ray_block = (self.obs_dim >= 64)

        # Ray encoder (configurable dims)
        pe_dim = int(getattr(config, "RAY_PE_DIM", 4))
        attn_dim = int(getattr(config, "RAY_ATTN_DIM", 16))
        self.ray_enc = RayEncoder(per_ray_in=8, proj_dim=16, pe_dim=pe_dim, attn_dim=attn_dim, out_dim=32)

        rich_in = max(0, self.obs_dim - 64)
        trunk_in = (32 + rich_in) if self.has_ray_block else self.obs_dim

        self.fc_in = nn.Linear(trunk_in, hidden, bias=True)
        self.fc1   = nn.Linear(hidden, hidden, bias=True)
        self.fc2   = nn.Linear(hidden, hidden, bias=True)
        self.actor  = nn.Linear(hidden, self.act_dim, bias=True)
        self.critic = nn.Linear(hidden, 1, bias=True)

        # -------- Stable, TorchScript-friendly init (no 'silu' gain) --------
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)   # neutral, works well with SiLU
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, A)
            value : (B, 1)
        """
        # Avoid AMP half/float mismatches during PPO training
        x = obs.float()

        if self.has_ray_block:
            rays = x[:, :64]    # (B, 64)
            rich = x[:, 64:]    # (B, D-64)
            e = self.ray_enc(rays)           # (B, 32)
            x = torch.cat([e, rich], dim=-1) # (B, 32 + rich)

        h = F.silu(self.fc_in(x))
        h = F.silu(self.fc1(h))
        h = F.silu(self.fc2(h))

        logits = self.actor(h)               # (B, A)
        value  = self.critic(h)              # (B, 1)
        return logits, value


# Backward-compat alias some modules import
class TinyActorCritic(ActorCriticBrain):
    pass


def scripted_brain(obs_dim: int, act_dim: int, hidden: int = 64) -> torch.jit.ScriptModule:
    """TorchScript brain for non-PPO runs."""
    model = ActorCriticBrain(obs_dim, act_dim, hidden)
    return torch.jit.script(model)
