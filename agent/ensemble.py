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

    Args:
      models: list of nn.Module, length K
      obs:    (K, F) observation batch aligned with models ordering

    Returns:
      - dist-like object with .logits -> (K, A)
      - values tensor -> (K,)    (NEVER 0-dim)

    Contract:
      Each model.forward(x: (1,F)) -> (logits: (1,A)) or (dist_with_logits, value)
    """
    device = obs.device
    K = int(obs.size(0)) if obs.dim() > 0 else 0
    if K == 0:
        # Empty bucket: return empty, but well-shaped, tensors.
        return _DistWrap(logits=torch.empty((0, 0), device=device)), torch.empty((0,), device=device)

    logits_out: List[torch.Tensor] = []
    values_out: List[torch.Tensor] = []

    for i, model in enumerate(models):
        # (1, F) slice for this model
        o = obs[i].unsqueeze(0)

        # Forward API: (logits_or_dist, value)
        out = model(o)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise RuntimeError("Brain.forward must return (logits_or_dist, value)")

        head, val = out

        # Normalize logits handle: either a distribution-like object or raw tensor
        logits = head.logits if hasattr(head, "logits") else head  # (1, A)

        # Ensure value is 1-D length 1 so cat() never sees a 0-dim tensor
        v = val.view(-1)  # (1,)

        logits_out.append(logits)
        values_out.append(v)

    # Batch them up
    logits_cat = torch.cat(logits_out, dim=0)         # (K, A)
    values_cat = torch.cat(values_out, dim=0)         # (K,)  <-- no squeeze! stays 1-D even for K=1

    # Final safety: if some model ever returned a scalar, make sure we still return 1-D
    if values_cat.dim() == 0:
        values_cat = values_cat.unsqueeze(0)

    return _DistWrap(logits=logits_cat), values_cat
