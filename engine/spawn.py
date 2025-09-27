from __future__ import annotations
import math
import random
from typing import Optional, Tuple

import torch
import config
from .agent_registry import AgentsRegistry
# Import the new brain
from agent.transformer_brain import TransformerBrain, scripted_transformer_brain

def _rect_dims(n: int, max_cols: int, max_rows: int) -> Tuple[int, int, int]:
    """Calculates dimensions for a compact rectangle to place n agents."""
    if n <= 0: return 0, 0, 0
    cols = min(max_cols, max(1, int(math.sqrt(n))))
    rows = min(max_rows, int(math.ceil(n / cols)))
    n_eff = min(n, cols * rows)
    return cols, rows, n_eff

def _mk_brain(device: torch.device):
    """Creates a new TransformerBrain instance."""
    obs_dim = (64 * 8) + 21
    act_dim = int(getattr(config, "NUM_ACTIONS", 41))
    if bool(getattr(config, "PPO_ENABLED", False)):
        return TransformerBrain(obs_dim, act_dim).to(device)
    else:
        return scripted_transformer_brain(obs_dim, act_dim).to(device)

def _choose_unit(is_archer_prob: float) -> float:
    return float(config.UNIT_ARCHER if random.random() < is_archer_prob else config.UNIT_SOLDIER)

def _unit_stats(unit_val: float) -> Tuple[float, float, int]:
    """Returns (hp, atk, vision) for a given unit id."""
    vision_map = getattr(config, "VISION_RANGE_BY_UNIT", {})
    if int(unit_val) == int(config.UNIT_ARCHER):
        hp = float(config.ARCHER_HP)
        atk = float(config.ARCHER_ATK)
        vision = int(vision_map.get(config.UNIT_ARCHER, 15))
    else:
        hp = float(config.SOLDIER_HP)
        atk = float(config.SOLDIER_ATK)
        vision = int(vision_map.get(config.UNIT_SOLDIER, 10))
    return hp, atk, vision

def _place_if_free(
    reg: AgentsRegistry, grid: torch.Tensor, slot: int, *,
    team_is_red: bool, x: int, y: int, unit_val: float
) -> bool:
    """Places an agent if the cell is free and registers it."""
    if grid[0, y, x] != 0.0:
        return False
    
    hp, atk, vision = _unit_stats(unit_val)
    
    # Register agent with its new attributes
    reg.register(
        slot,
        team_is_red=team_is_red,
        x=x, y=y,
        hp=hp,
        atk=atk,
        brain=_mk_brain(reg.device),
        unit=unit_val,
        hp_max=hp,
        vision_range=vision,
        generation=1
    )
    
    # Update grid
    grid[0, y, x] = 2.0 if team_is_red else 3.0
    grid[1, y, x] = hp
    grid[2, y, x] = float(slot)
    return True

def spawn_symmetric(reg: AgentsRegistry, grid: torch.Tensor, per_team: int) -> None:
    """Spawns agents in symmetric rectangular formations on opposite sides."""
    H, W = grid.size(1), grid.size(2)
    margin = 2
    half_w = W // 2
    placeable_w = half_w - margin
    placeable_h = H - 2 * margin

    per_team_eff = min(per_team, reg.capacity // 2, placeable_w * placeable_h)
    if per_team_eff <= 0: return

    ar_ratio = float(getattr(config, "SPAWN_ARCHER_RATIO", 0.4))

    # Red team (left)
    r_cols, r_rows, r_n = _rect_dims(per_team_eff, placeable_w, placeable_h)
    red_x0, red_y0 = margin, (H - r_rows) // 2

    # Blue team (right)
    b_cols, b_rows, b_n = _rect_dims(per_team_eff, placeable_w, placeable_h)
    blue_x0, blue_y0 = W - margin - b_cols, (H - b_rows) // 2
    
    slot = 0
    # Place Red
    for iy in range(r_rows):
        for ix in range(r_cols):
            if slot >= r_n or slot >= reg.capacity: break
            x, y = red_x0 + ix, red_y0 + iy
            unit = _choose_unit(ar_ratio)
            if _place_if_free(reg, grid, slot, team_is_red=True, x=x, y=y, unit_val=unit):
                slot += 1
        if slot >= r_n or slot >= reg.capacity: break

    # Place Blue
    blue_start_slot = slot
    for iy in range(b_rows):
        for ix in range(b_cols):
            if slot >= blue_start_slot + b_n or slot >= reg.capacity: break
            x, y = blue_x0 + ix, blue_y0 + iy
            unit = _choose_unit(ar_ratio)
            if _place_if_free(reg, grid, slot, team_is_red=False, x=x, y=y, unit_val=unit):
                slot += 1
        if slot >= blue_start_slot + b_n or slot >= reg.capacity: break

def spawn_uniform_random(reg: AgentsRegistry, grid: torch.Tensor, per_team: int) -> None:
    """Spawns agents for both teams randomly across the entire map for maximum chaos."""
    H, W = grid.size(1), grid.size(2)
    ar_ratio = float(getattr(config, "SPAWN_ARCHER_RATIO", 0.4))
    
    total_to_spawn = per_team * 2
    red_to_spawn = per_team
    blue_to_spawn = per_team
    
    slot = 0
    attempts = 0
    # Give up after a reasonable number of tries to avoid infinite loops on full maps
    max_attempts = total_to_spawn * 50 

    while slot < total_to_spawn and attempts < max_attempts and slot < reg.capacity:
        # Pick a random location anywhere on the grid (respecting a 1-cell border)
        x = random.randint(1, W - 2)
        y = random.randint(1, H - 2)
        
        # Decide which team to try and spawn for this cell
        # This logic ensures we spawn teams randomly until one quota is met, then fill the other.
        spawn_red = (red_to_spawn > 0 and blue_to_spawn == 0) or \
                    (red_to_spawn > 0 and blue_to_spawn > 0 and random.random() < 0.5)
        
        # Check if the cell is actually free before attempting to place
        if grid[0, y, x] == 0.0:
            team_placed = False
            if spawn_red and red_to_spawn > 0:
                unit = _choose_unit(ar_ratio)
                if _place_if_free(reg, grid, slot, team_is_red=True, x=x, y=y, unit_val=unit):
                    slot += 1
                    red_to_spawn -= 1
                    team_placed = True
            # Check the other team if the first pick was invalid or its quota was full
            elif not spawn_red and blue_to_spawn > 0:
                unit = _choose_unit(ar_ratio)
                if _place_if_free(reg, grid, slot, team_is_red=False, x=x, y=y, unit_val=unit):
                    slot += 1
                    blue_to_spawn -= 1
                    team_placed = True
        
        attempts += 1
    
    if slot < total_to_spawn:
        print(f"[spawn] Warning: Could only spawn {slot}/{total_to_spawn} agents. The map might be too full.")

