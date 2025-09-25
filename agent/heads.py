# final_war_sim/agent/heads.py
from __future__ import annotations
import torch
import torch.nn as nn

class FactorizedDirectionalHeads(nn.Module):
    """
    Per-agent, factorized action head that preserves a SINGLE categorical action space
    (same indices as your engine: idle(0), move(1..8), melee(9..16), range r1(17..24), r2(25..32), r3(33..40)).

    Why:
      • directional slices specialize cleanly
      • easier masking by 8-wide groups
      • no change to PPO, no centralization

    Use:
      trunk -> FactorizedDirectionalHeads(A=41)(h) -> logits [B,41]
    """
    def __init__(self, total_actions: int):
        super().__init__()
        self.A = int(total_actions)
        if self.A not in (17, 41):
            raise ValueError(f"Unsupported action count {self.A}; expected 17 or 41.")

        # Scalar head for idle (index 0)
        self.idle = nn.Linear(128, 1)

        # 8-wide directional groups
        self.move  = nn.Linear(128, 8)  # 1..8
        self.melee = nn.Linear(128, 8)  # 9..16

        if self.A == 41:
            self.r1 = nn.Linear(128, 8)  # 17..24
            self.r2 = nn.Linear(128, 8)  # 25..32
            self.r3 = nn.Linear(128, 8)  # 33..40
            # small learned biases for attack groups (helpful when rays are empty)
            self.b1 = nn.Parameter(torch.zeros(1))
            self.b2 = nn.Parameter(torch.zeros(1))
            self.b3 = nn.Parameter(torch.zeros(1))
        else:
            self.r1 = self.r2 = self.r3 = None
            self.b1 = self.b2 = self.b3 = None

        # value head (optional; keep your existing if you prefer)
        self.value = nn.Linear(128, 1)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor):
        """
        h: [B,128] trunk activations
        returns: logits [B,A], value [B]
        """
        idle = self.idle(h)           # (B,1)
        mv   = self.move(h)           # (B,8)
        ml   = self.melee(h)          # (B,8)

        if self.A == 41:
            r1 = self.r1(h) + self.b1
            r2 = self.r2(h) + self.b2
            r3 = self.r3(h) + self.b3
            logits = torch.cat([idle, mv, ml, r1, r2, r3], dim=1)  # (B,41)
        else:
            logits = torch.cat([idle, mv, ml], dim=1)              # (B,17)

        value = self.value(h).squeeze(-1)
        return logits, value
