from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

import config

# ---------------- Config (safe defaults; override in config.py) ----------------
MUTATION_TWEAK_PROB        = getattr(config, "MUTATION_TWEAK_PROB", 0.70)      # chance to add small noise
MUTATION_TWEAK_FRAC        = getattr(config, "MUTATION_TWEAK_FRAC", 0.003)     # ~0.3% weights
MUTATION_TWEAK_STD         = getattr(config, "MUTATION_TWEAK_STD", 0.01)       # N(0, std)

# When TickEngine calls pick_mutants(..., fraction=?), this is its fallback:
DEFAULT_MUTANT_FRACTION    = getattr(config, "MUTATION_FRACTION_ALIVE", 0.10)


# ---------------- Utilities ----------------
def _tweak_some_weights(m: nn.Module, frac: float, std: float) -> None:
    """Add small Gaussian noise to a tiny random subset of weights (and biases)."""
    with torch.no_grad():
        for p in m.parameters():
            if not p.is_floating_point() or p.numel() == 0:
                continue
            k = max(1, int(p.numel() * frac))
            idx = torch.randperm(p.numel(), device=p.device)[:k]
            flat = p.view(-1)
            flat[idx] += torch.randn_like(flat[idx]) * std


# ---------------- Selection (exported; used by tick.py) ----------------
@torch.no_grad()
def pick_mutants(alive_indices: torch.Tensor,
                 fraction: float = DEFAULT_MUTANT_FRACTION,
                 min_count: int = 1) -> torch.Tensor:
    """
    Choose a subset of alive indices for mutation.
    - fraction: % of alive to mutate (fallback to config.MUTATION_FRACTION_ALIVE)
    - min_count: at least this many if alive > 0
    """
    if alive_indices is None or alive_indices.numel() == 0:
        # empty LongTensor on the same device
        return torch.empty((0,), dtype=torch.long, device=alive_indices.device if alive_indices is not None else None)

    n = alive_indices.numel()
    k = max(min_count, int(n * max(0.0, min(1.0, fraction))))
    k = min(k, n)

    # random without replacement on the same device for speed
    perm = torch.randperm(n, device=alive_indices.device)
    return alive_indices[perm[:k]]


# ---------------- Mutation (exported; used by registry via TickEngine) ----------------
@torch.no_grad()
def mutate_model_inplace(model: nn.Module, now_tick: Optional[int] = None) -> nn.Module:
    """
    Applies a small amount of noise to a fraction of the model's weights.
    This is the only mutation type.
    """
    _tweak_some_weights(model, frac=MUTATION_TWEAK_FRAC, std=MUTATION_TWEAK_STD)
    return model
