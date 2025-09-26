# final_war_sim/agent/mutation.py
from __future__ import annotations
from typing import Optional
import math
import random
import torch
import torch.nn as nn

import config

# ---------------- Config (safe defaults; override in config.py) ----------------
MAX_PARAMS_PER_BRAIN       = getattr(config, "MAX_PARAMS_PER_BRAIN", 40_000)   # hard ceiling
MUTATION_WIDEN_PROB        = getattr(config, "MUTATION_WIDEN_PROB", 0.30)      # chance to widen a bit
MUTATION_TWEAK_PROB        = getattr(config, "MUTATION_TWEAK_PROB", 0.70)      # chance to add small noise
MUTATION_WIDEN_DELTA_MIN   = getattr(config, "MUTATION_WIDEN_DELTA_MIN", 1)    # add 1..2 neurons
MUTATION_WIDEN_DELTA_MAX   = getattr(config, "MUTATION_WIDEN_DELTA_MAX", 2)
MUTATION_TWEAK_FRAC        = getattr(config, "MUTATION_TWEAK_FRAC", 0.003)     # ~0.3% weights
MUTATION_TWEAK_STD         = getattr(config, "MUTATION_TWEAK_STD", 0.01)       # N(0, std)
PRUNE_SOFT_BUDGET          = getattr(config, "PRUNE_SOFT_BUDGET", 35_000)      # start pruning above this
PRUNE_NEURON_DROP_MIN      = getattr(config, "PRUNE_NEURON_DROP_MIN", 1)
PRUNE_NEURON_DROP_MAX      = getattr(config, "PRUNE_NEURON_DROP_MAX", 4)

# When TickEngine calls pick_mutants(..., fraction=?), this is its fallback:
DEFAULT_MUTANT_FRACTION    = getattr(config, "MUTATION_FRACTION_ALIVE", 0.10)


# ---------------- Utilities ----------------
def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _safe_linear_widen(linear: nn.Linear, add_neurons: int) -> Optional[nn.Linear]:
    """Return a widened copy of `linear` by add_neurons on out_features; keep old weights."""
    if add_neurons <= 0:
        return None
    in_f = linear.in_features
    out_old = linear.out_features
    out_new = out_old + add_neurons

    new = nn.Linear(in_f, out_new, bias=linear.bias is not None)

    with torch.no_grad():
        # copy existing rows
        new.weight[:out_old, :].copy_(linear.weight)
        if linear.bias is not None:
            new.bias[:out_old].copy_(linear.bias)

        # init new rows (Kaiming uniform)
        nn.init.kaiming_uniform_(new.weight[out_old:, :], a=math.sqrt(5))
        if linear.bias is not None:
            fan_in = in_f
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(new.bias[out_old:], -bound, bound)

    return new


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


def _drop_neurons_magnitude(fc: nn.Linear, drop: int) -> nn.Linear:
    """
    Drop `drop` output neurons in `fc` by smallest L2 row-norm (structured prune).
    Caller must immediately adjust next layer in_features (we do that below for actor/critic).
    """
    if drop <= 0 or drop >= fc.out_features:
        return fc

    with torch.no_grad():
        row_norms = torch.linalg.vector_norm(fc.weight, ord=2, dim=1)
        keep = torch.topk(row_norms, k=fc.out_features - drop, largest=True).indices

    new = nn.Linear(fc.in_features, fc.out_features - drop, bias=fc.bias is not None)
    with torch.no_grad():
        new.weight.copy_(fc.weight[keep])
        if fc.bias is not None:
            new.bias.copy_(fc.bias[keep])
    return new


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
    Gentle mutation to avoid parameter explosion:
      1) Usually: small weight noise on a tiny subset of params.
      2) Sometimes: widen fc2 by +1 or +2 neurons, then adjust actor/critic heads.
      3) If drifting large: small structured prune on fc2.
    Hard cap: never exceed MAX_PARAMS_PER_BRAIN.
    Assumes model has attributes: fc1, fc2, actor, critic (TinyActorCritic style).
    """
    changed = False

    # (1) Tiny weight tweak — dominant path
    if random.random() < MUTATION_TWEAK_PROB:
        _tweak_some_weights(model, frac=MUTATION_TWEAK_FRAC, std=MUTATION_TWEAK_STD)
        changed = True

    # (2) Micro-widen on fc2 — rare, small
    if hasattr(model, "fc2") and isinstance(model.fc2, nn.Linear) and (random.random() < MUTATION_WIDEN_PROB):
        add = random.randint(MUTATION_WIDEN_DELTA_MIN, MUTATION_WIDEN_DELTA_MAX)
        new_fc2 = _safe_linear_widen(model.fc2, add_neurons=add)
        if new_fc2 is not None:
            old_h = model.fc2.out_features
            new_h = new_fc2.out_features

            old_fc2 = model.fc2
            model.fc2 = new_fc2

            # Adjust actor head
            if hasattr(model, "actor") and isinstance(model.actor, nn.Linear) and model.actor.in_features == old_h:
                new_actor = nn.Linear(new_h, model.actor.out_features, bias=model.actor.bias is not None)
                with torch.no_grad():
                    new_actor.weight[:, :old_h].copy_(model.actor.weight[:, :old_h])
                    if model.actor.bias is not None:
                        new_actor.bias.copy_(model.actor.bias)
                model.actor = new_actor

            # Adjust critic head
            if hasattr(model, "critic") and isinstance(model.critic, nn.Linear) and model.critic.in_features == old_h:
                new_critic = nn.Linear(new_h, model.critic.out_features, bias=model.critic.bias is not None)
                with torch.no_grad():
                    new_critic.weight[:, :old_h].copy_(model.critic.weight[:, :old_h])
                    if model.critic.bias is not None:
                        new_critic.bias.copy_(model.critic.bias)
                model.critic = new_critic

            if param_count(model) > MAX_PARAMS_PER_BRAIN:
                model.fc2 = old_fc2  # revert if over budget
            else:
                changed = True

    # (3) Light prune if above soft budget
    if hasattr(model, "fc2") and isinstance(model.fc2, nn.Linear) and param_count(model) > PRUNE_SOFT_BUDGET:
        drop = random.randint(PRUNE_NEURON_DROP_MIN, PRUNE_NEURON_DROP_MAX)
        pruned_fc2 = _drop_neurons_magnitude(model.fc2, drop=drop)
        if pruned_fc2 is not model.fc2:
            old_h = model.fc2.out_features
            new_h = pruned_fc2.out_features
            model.fc2 = pruned_fc2

            # Rewire actor
            if hasattr(model, "actor") and isinstance(model.actor, nn.Linear) and model.actor.in_features == old_h:
                new_actor = nn.Linear(new_h, model.actor.out_features, bias=model.actor.bias is not None)
                with torch.no_grad():
                    copy_w = min(new_h, old_h)
                    new_actor.weight[:, :copy_w].copy_(model.actor.weight[:, :copy_w])
                    if model.actor.bias is not None:
                        new_actor.bias.copy_(model.actor.bias)
                model.actor = new_actor

            # Rewire critic
            if hasattr(model, "critic") and isinstance(model.critic, nn.Linear) and model.critic.in_features == old_h:
                new_critic = nn.Linear(new_h, model.critic.out_features, bias=model.critic.bias is not None)
                with torch.no_grad():
                    copy_w = min(new_h, old_h)
                    new_critic.weight[:, :copy_w].copy_(model.critic.weight[:, :old_h])
                    if model.critic.bias is not None:
                        new_critic.bias.copy_(model.critic.bias)
                model.critic = new_critic

            changed = True

    return model

