# final_war_sim/agent/ensemble.py
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn

class _DistWrap:
    """Lightweight container to mimic a torch.distributions object with logits."""
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits

@torch.no_grad()
def ensemble_forward(models: List[nn.Module], obs: torch.Tensor) -> Tuple[_DistWrap, torch.Tensor]:
    """
    Fuses per-agent models for a bucket into one batched tensor of outputs.

    models: list of nn.Module, len K
    obs:    (K, F) observation batch aligned with models ordering
    returns:
      - dist-like object with .logits -> (K, A)
      - values tensor (K,)
    Expectation: each model.forward(x: (1,F)) -> (logits: (1,A), value: (1,))
    """
    device = obs.device
    K = obs.size(0)
    if K == 0:
        return _DistWrap(logits=torch.empty((0, 0), device=device)), torch.empty((0,), device=device)

    # We cannot strictly batch different parameter sets; do a tight Python loop
    # (still vectorized upstream by architecture bucket).
    logits_out = []
    values_out = []
    for i, model in enumerate(models):
        o = obs[i].unsqueeze(0)  # (1,F)
        # Model contract: returns (logits, value) or (dist, value)
        out = model(o)
        if isinstance(out, tuple) and len(out) == 2:
            head, val = out
            if hasattr(head, "logits"):
                logits = head.logits  # (1,A)
            else:
                logits = head  # assume raw logits
            values_out.append(val.view(-1))    # (1,)
            logits_out.append(logits)          # (1,A)
        else:
            raise RuntimeError("Brain.forward must return (logits_or_dist, value)")

    logits_cat = torch.cat(logits_out, dim=0)      # (K,A)
    values_cat = torch.cat(values_out, dim=0).squeeze(-1)  # (K,)
    return _DistWrap(logits=logits_cat), values_cat
