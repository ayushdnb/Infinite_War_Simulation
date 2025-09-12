# final_war_sim/agent/brain.py
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config

class ActorCriticBrain(nn.Module):
    """
    Canonical actor-critic used across the project.
    Exposes fc1/fc2/actor/critic so mutation code can widen/insert layers.
    Forward returns (logits, value) to match ensemble expectations.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        self.fc1 = nn.Linear(self.obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.actor = nn.Linear(hidden, self.act_dim)
        self.critic = nn.Linear(hidden, 1)

        # Kaiming + zero bias init for stability (matches earlier drops)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(dtype=config.TORCH_DTYPE)
        h = F.silu(self.fc1(x))
        h = F.silu(self.fc2(h))
        logits = self.actor(h)                 # (B, A)
        value  = self.critic(h).squeeze(-1)    # (B,)
        return logits, value


# Backward-compat for mutation code that checks TinyActorCritic
class TinyActorCritic(ActorCriticBrain):
    pass


def scripted_brain(obs: int, act: int, hidden: int = 64) -> torch.jit.ScriptModule:
    model = ActorCriticBrain(obs, act, hidden)
    scripted = torch.jit.script(model)
    return scripted

