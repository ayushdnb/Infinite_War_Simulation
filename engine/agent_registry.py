# final_war_sim/engine/agent_registry.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import torch
import torch.nn as nn

from .. import config

# ================================================================
# Column layout (Struct-of-Arrays for GPU efficiency)
# ================================================================
COL_ALIVE = 0  # float: 1.0 alive, 0.0 dead
COL_TEAM  = 1  # float: 2.0 red, 3.0 blue
COL_X     = 2  # float: x coordinate
COL_Y     = 3  # float: y coordinate
COL_HP    = 4  # float: health points
COL_ATK   = 5  # float: attack power

NUM_COLS = config.AGENT_FEATURES  # always use config, no hardcoding

TEAM_RED_ID  = 2.0
TEAM_BLUE_ID = 3.0


# ================================================================
# Buckets: allow grouping agents with same NN architecture
# ================================================================
@dataclass
class Bucket:
    signature: str
    indices: torch.Tensor          # LongTensor [K] of agent indices
    models: List[nn.Module]        # length K (same order as indices)


# ================================================================
# Agents Registry
# ================================================================
class AgentsRegistry:
    """
    Stores all agents in the simulation as a big tensor (SoA layout).
    Mirrors: grid occupancy channel -> registry rows.

    Owns:
      - agent_data: (MAX_AGENTS, NUM_COLS) tensor [alive, team, x, y, hp, atk]
      - brains: Python list[nn.Module or None], length MAX_AGENTS
    """

    def __init__(self, grid: torch.Tensor) -> None:
        self.grid = grid
        self.device = grid.device
        self.capacity = int(config.MAX_AGENTS)

        # Main agent tensor (SoA layout)
        self.agent_data = torch.zeros(
            (config.MAX_AGENTS, config.AGENT_FEATURES),
            dtype=config.TORCH_DTYPE,
            device=config.TORCH_DEVICE
        )
        self.agent_data[:, COL_ALIVE] = 0.0

        # Brains are stored separately (not inside tensor, to keep TorchScript JIT-safe)
        self.brains: List[nn.Module | None] = [None] * self.capacity

    # ------------------------------------------------------------
    # Basic ops
    # ------------------------------------------------------------
    def clear(self) -> None:
        """Reset all agents."""
        self.agent_data.zero_()
        self.agent_data[:, COL_ALIVE] = 0.0
        self.brains = [None] * self.capacity

    def register(self, slot: int, *, team_is_red: bool, x: int, y: int,
                 hp: float, atk: float, brain: nn.Module) -> None:
        """Put an agent into a fixed slot (caller manages slot selection)."""
        assert 0 <= slot < self.capacity
        d = self.agent_data
        d[slot, COL_ALIVE] = 1.0
        d[slot, COL_TEAM]  = TEAM_RED_ID if team_is_red else TEAM_BLUE_ID
        d[slot, COL_X]     = float(x)
        d[slot, COL_Y]     = float(y)
        d[slot, COL_HP]    = float(hp)
        d[slot, COL_ATK]   = float(atk)
        self.brains[slot]  = brain.to(self.device)

    def kill(self, slots: torch.Tensor) -> None:
        """Mark agents dead (grid clearing happens in TickEngine)."""
        if slots.numel() > 0:
            self.agent_data[slots, COL_ALIVE] = 0.0

    # ------------------------------------------------------------
    # Views
    # ------------------------------------------------------------
    def positions_xy(self, indices: torch.Tensor) -> torch.Tensor:
        """Return LongTensor [(N,2)] of XY positions for given indices."""
        d = self.agent_data
        x = d[indices, COL_X].to(torch.long)
        y = d[indices, COL_Y].to(torch.long)
        return torch.stack((x, y), dim=1)

    # ------------------------------------------------------------
    # Bucketing by architecture
    # ------------------------------------------------------------
    @staticmethod
    def _signature(model: nn.Module) -> str:
        """Create a cheap architecture fingerprint."""
        sig_parts: List[str] = [model.__class__.__name__]
        for _, m in model.named_modules():
            if isinstance(m, nn.Linear):
                sig_parts.append(f"L({m.in_features},{m.out_features})")
            elif isinstance(m, nn.Conv2d):
                sig_parts.append(f"C({m.in_channels},{m.out_channels},{m.kernel_size})")
        return "|".join(sig_parts)

    def build_buckets(self, alive_idx: torch.Tensor) -> List[Bucket]:
        """Group alive agents by model signature for batched inference."""
        buckets: Dict[str, List[int]] = {}
        for i in alive_idx.tolist():
            brain = self.brains[i]
            if brain is None:
                self.agent_data[i, COL_ALIVE] = 0.0
                continue
            key = self._signature(brain)
            buckets.setdefault(key, []).append(i)

        out: List[Bucket] = []
        for key, lst in buckets.items():
            idx = torch.tensor(lst, dtype=torch.long, device=self.device)
            models = [self.brains[j] for j in lst]
            out.append(Bucket(signature=key, indices=idx, models=models))
        return out

    # ------------------------------------------------------------
    # Mutation hook
    # ------------------------------------------------------------
    def apply_mutations(self, indices: torch.Tensor, mutate_fn) -> None:
        """
        Apply structural mutations in-place.
        mutate_fn(model: nn.Module) -> nn.Module
        """
        if indices.numel() == 0:
            return
        for i in indices.tolist():
            model = self.brains[i]
            if model is None:
                continue
            new_model = mutate_fn(model)
            self.brains[i] = new_model.to(self.device)
