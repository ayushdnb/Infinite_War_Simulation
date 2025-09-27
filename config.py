from __future__ import annotations
import os
import torch

# ================================================================
# Utility: env parsing
# ================================================================

def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    v = v.strip().lower()
    return v not in {"0", "false", "no", "off", ""}


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except Exception:
        return float(default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except Exception:
        return int(default)

# ================================================================
# Device / Precision
# ================================================================
AMP_ENABLED: bool = _env_bool("FWS_AMP", True)

def amp_enabled() -> bool:
    return AMP_ENABLED

TORCH_DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)
# Use fp16 on CUDA when AMP is on; otherwise fp32
TORCH_DTYPE: torch.dtype = torch.float16 if AMP_ENABLED and torch.cuda.is_available() else torch.float32

# ================================================================
# World / Runtime
# ================================================================
GRID_WIDTH  = _env_int("FWS_GRID_W", 128)
GRID_HEIGHT = _env_int("FWS_GRID_H", 128)
START_AGENTS_PER_TEAM = _env_int("FWS_START_PER_TEAM", 100)
MAX_AGENTS = _env_int("FWS_MAX_AGENTS", 500)
# SoA columns in agent_data (X,Y,HP,ALIVE,TEAM,ATK,UNIT) → 7
AGENT_FEATURES = _env_int("FWS_AGENT_FEATS", 7)

TICK_LIMIT = _env_int("FWS_TICK_LIMIT", 0)   # 0 = unlimited
TARGET_TPS = _env_int("FWS_TARGET_TPS", 60)  # engine tick rate; viewer caps FPS separately

# ================================================================
# Perception (V2 by default)
# ================================================================
# Unit IDs must match registry encoding
UNIT_SOLDIER_ID = 1
UNIT_ARCHER_ID  = 2

# Per-unit vision (in grid cells). Extend dict for new unit types.
VISION_RANGE_BY_UNIT = {
    UNIT_SOLDIER_ID: _env_int("FWS_VISION_SOLDIER", 10),
    UNIT_ARCHER_ID:  _env_int("FWS_VISION_ARCHER", 15),
}

# Raycaster global cap (must be >= any unit vision)
RAYCAST_MAX_STEPS = max(max(VISION_RANGE_BY_UNIT.values()), 1)

# --- Back-compat aliases (keep until all modules are migrated) ---
UNIT_SOLDIER = UNIT_SOLDIER_ID
UNIT_ARCHER  = UNIT_ARCHER_ID
RAY_MAX_STEPS = RAYCAST_MAX_STEPS
NUM_DIRECTIONS = 8  # legacy constants still referenced by some modules
OBS_DIM_LEGACY = NUM_DIRECTIONS * 2 * RAYCAST_MAX_STEPS

# V2 rays are 8 directions × (onehot6 + dist_norm + hp_norm = 8) → 64
# Rich self/env features that we append after rays (kept in sync with TickEngine)
SELF_FEATS_V2 = 21
OBS_DIM_V2 = 64 + SELF_FEATS_V2  # 85

# Effective observation dim expected by the policy
OBS_DIM = _env_int("FWS_OBS_DIM", OBS_DIM_V2)  # default 85

# ================================================================
# Actions
# ================================================================
# 17 = idle + 8 moves + 8 melee; 41 = +32 ranged (archer r=1..4 over 8 dirs)
NUM_ACTIONS_LEGACY = 17
NUM_ACTIONS_V2 = 41
NUM_ACTIONS = _env_int("FWS_NUM_ACTIONS", NUM_ACTIONS_V2)  # default 41

# ================================================================
# Units & Combat
# ================================================================
# Occupancy encoding in grid[0]: 0 empty, 1 wall, 2 red, 3 blue

# Base stats (normalized to MAX_HP units)
MAX_HP       = _env_float("FWS_MAX_HP", 1.0)
SOLDIER_HP   = _env_float("FWS_SOLDIER_HP", 1.00)
ARCHER_HP    = _env_float("FWS_ARCHER_HP", 0.85)

# Attack values
BASE_ATK     = _env_float("FWS_BASE_ATK", 0.40)
SOLDIER_ATK  = _env_float("FWS_SOLDIER_ATK", BASE_ATK)
ARCHER_ATK   = _env_float("FWS_ARCHER_ATK", 0.30)
# For feature normalization in TickEngine
MAX_ATK      = max(SOLDIER_ATK, ARCHER_ATK, BASE_ATK, 1e-6)

# Archer rules (engine/mask enforce these)
ARCHER_RANGE = _env_int("FWS_ARCHER_RANGE", 4)
ARCHER_LOS_BLOCKS_WALLS = _env_bool("FWS_ARCHER_BLOCK_LOS", False)

# ================================================================
# Metabolism (HP drain per tick)
# ================================================================
METABOLISM_ENABLED = _env_bool("FWS_META_ON", True)
META_SOLDIER_HP_PER_TICK = _env_float("FWS_META_SOLDIER", 0.001)  # ~500 ticks from 1.00
META_ARCHER_HP_PER_TICK  = _env_float("FWS_META_ARCHER",  0.0006)  # tuned vs 0.70 base

# ================================================================
# Respawn knobs
# ================================================================
RESPAWN_COOLDOWN_TICKS   = _env_int("FWS_RESPAWN_CD", 500)
RESPAWN_BATCH_PER_TEAM   = _env_int("FWS_RESPAWN_BATCH", 2)
RESPAWN_ARCHER_SHARE     = _env_float("FWS_RESPAWN_ARCHER_SHARE", 0.50)  # 35% archer
RESPAWN_INTERIOR_BIAS    = _env_float("FWS_RESPAWN_INTERIOR_BIAS", 0.75)
RESPAWN_JITTER_RADIUS    = _env_int("FWS_RESPAWN_JITTER", 5)

# ================================================================
# Random Walls (thin, gray)
# ================================================================
RANDOM_WALLS      = _env_int("FWS_RAND_WALLS", 25)   # segments
WALL_SEG_MIN      = _env_int("FWS_WALL_SEG_MIN", 12)
WALL_SEG_MAX      = _env_int("FWS_WALL_SEG_MAX", 100)
WALL_AVOID_MARGIN = _env_int("FWS_WALL_AVOID_MARGIN", 1)  # cells from border

# ================================================================
# Heal Zones (rects, scaled to grid)
# ================================================================
HEAL_ZONE_COUNT      = _env_int("FWS_HEAL_COUNT", 7)
HEAL_ZONE_SIZE_RATIO = _env_float("FWS_HEAL_SIZE_RATIO", 40/256)
HEAL_RATE            = _env_float("FWS_HEAL_RATE", 0.02)      # HP/tick, clamped

# ================================================================
# Capture Zones (rects, scaled to grid)
# ================================================================
CP_COUNT           = _env_int("FWS_CP_COUNT", 3)
CP_SIZE_RATIO      = _env_float("FWS_CP_SIZE_RATIO", 30/256)
CP_REWARD_PER_TICK = _env_float("FWS_CP_REWARD", 0.05)

# ================================================================
# Team Rewards (accounting handled in stats)
# ================================================================
TEAM_KILL_REWARD       = _env_float("FWS_REW_KILL",       1.0)
TEAM_DMG_DEALT_REWARD  = _env_float("FWS_REW_DMG_DEALT",  0.00)
TEAM_DEATH_PENALTY     = _env_float("FWS_REW_DEATH",     -0.4)
TEAM_DMG_TAKEN_PENALTY = _env_float("FWS_REW_DMG_TAKEN",  0.00)

# ================================================================
# Evolution / Mutation
# ================================================================
PER_AGENT_BRAINS        = _env_bool("FWS_PER_AGENT_BRAINS", True)
MUTATION_PERIOD_TICKS   = _env_int("FWS_MUTATE_EVERY", 2000)
MUTATION_FRACTION_ALIVE = _env_float("FWS_MUTATE_FRAC", 0.10)

# ================================================================
# PPO (used by rl/ppo.py and main._config_snapshot)
# ================================================================
PPO_UPDATE_TICKS  = _env_int("FWS_PPO_TICKS", 5)
PPO_LR            = _env_float("FWS_PPO_LR", 3e-4)
PPO_EPOCHS        = _env_int("FWS_PPO_EPOCHS", 3)
PPO_CLIP          = _env_float("FWS_PPO_CLIP", 0.2)
PPO_ENTROPY_BONUS = _env_float("FWS_PPO_ENTROPY", 0.01)
PPO_VALUE_COEF    = _env_float("FWS_PPO_VCOEF", 0.5)
PPO_MAX_GRAD_NORM = _env_float("FWS_PPO_MAXGN", 1.0)

# ================================================================
# UI / Viewer
# ================================================================
ENABLE_UI  = _env_bool("FWS_UI", True)
CELL_SIZE  = _env_int("FWS_CELL_SIZE", 6)
TARGET_FPS = _env_int("FWS_TARGET_FPS", 60)
# final_war_sim/config_addons.py
# Optional config knobs used by the ego adapter and ray encoder.
# Safe defaults are baked into the code via getattr(...), so this file is optional.

# Use ego-centric rotation of the 8 directional rays and inverse-rotation of logits.
RELATIVE_DIRS = True

# Tiny ray encoder tuning (used in brain.py if obs_dim >= 85)
RAY_PE_DIM   = 4    # 0 disables ring positional encoding
RAY_ATTN_DIM = 16   # 0 disables tiny attention block

# PPO ON
PPO_ENABLED = True

# Rewards
PPO_REWARD_KILL = 1.0
PPO_REWARD_DEATH = -0.4

# PPO knobs (safe defaults)
PPO_WINDOW_TICKS = 20
PPO_EPOCHS = 2
PPO_LR = 3e-4
PPO_CLIP = 0.2
PPO_ENTROPY_COEF = 0.01
PPO_VALUE_COEF = 0.5
PPO_GAMMA = 0.99
PPO_LAMBDA = 0.95

# dtype control (fp16 is fine if the rest of your pipeline is AMP-friendly)
# import torch
# TORCH_DTYPE = torch.float16
# ================================================================
# Summary / Snapshot helpers
# ================================================================

def summary_str() -> str:
    return (
        f"[final_war_sim] "
        f"dev={TORCH_DEVICE.type} "
        f"grid={GRID_WIDTH}x{GRID_HEIGHT} "
        f"start={START_AGENTS_PER_TEAM}/team "
        f"obs={OBS_DIM} acts={NUM_ACTIONS} "
        f"AMP={'on' if AMP_ENABLED else 'off'}"
    )
