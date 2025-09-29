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
        return float(os.getenv(key, str(default)))
    except Exception:
        return float(default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
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
TORCH_DTYPE: torch.dtype = torch.float16 if AMP_ENABLED and torch.cuda.is_available() else torch.float32

# ================================================================
# World / Runtime
# ================================================================
GRID_WIDTH  = _env_int("FWS_GRID_W", 64)
GRID_HEIGHT = _env_int("FWS_GRID_H", 64)
START_AGENTS_PER_TEAM = _env_int("FWS_START_PER_TEAM", 100)
MAX_AGENTS = _env_int("FWS_MAX_AGENTS", 500)
AGENT_FEATURES = _env_int("FWS_AGENT_FEATS", 9) # Updated to 9 for new registry

TICK_LIMIT = _env_int("FWS_TICK_LIMIT", 0)
TARGET_TPS = _env_int("FWS_TARGET_TPS", 60)

# ================================================================
# Perception
# ================================================================
UNIT_SOLDIER_ID = 1
UNIT_ARCHER_ID  = 2

VISION_RANGE_BY_UNIT = {
    UNIT_SOLDIER_ID: _env_int("FWS_VISION_SOLDIER", 8),
    UNIT_ARCHER_ID:  _env_int("FWS_VISION_ARCHER", 12),
}

RAYCAST_MAX_STEPS = max(max(VISION_RANGE_BY_UNIT.values()), 1)
UNIT_SOLDIER = UNIT_SOLDIER_ID
UNIT_ARCHER  = UNIT_ARCHER_ID
RAY_MAX_STEPS = RAYCAST_MAX_STEPS

OBS_DIM = (64 * 8) + 21 # 512 for rays, 21 for rich features

# ================================================================
# Actions
# ================================================================
NUM_ACTIONS = _env_int("FWS_NUM_ACTIONS", 41)

# ================================================================
# Units & Combat
# ================================================================
MAX_HP       = _env_float("FWS_MAX_HP", 1.0)
SOLDIER_HP   = _env_float("FWS_SOLDIER_HP", 1.00)
ARCHER_HP    = _env_float("FWS_ARCHER_HP", 0.85)

BASE_ATK     = _env_float("FWS_BASE_ATK", 0.40)
SOLDIER_ATK  = _env_float("FWS_SOLDIER_ATK", BASE_ATK)
ARCHER_ATK   = _env_float("FWS_ARCHER_ATK", 0.30)
MAX_ATK      = max(SOLDIER_ATK, ARCHER_ATK, BASE_ATK, 1e-6)

ARCHER_RANGE = _env_int("FWS_ARCHER_RANGE", 4)
ARCHER_LOS_BLOCKS_WALLS = _env_bool("FWS_ARCHER_BLOCK_LOS", False)

# ================================================================
# Metabolism
# ================================================================
METABOLISM_ENABLED = _env_bool("FWS_META_ON", True)
META_SOLDIER_HP_PER_TICK = _env_float("FWS_META_SOLDIER", 0.0001)
META_ARCHER_HP_PER_TICK  = _env_float("FWS_META_ARCHER",  0.00008)

# ================================================================
# Respawn knobs
# ================================================================
RESPAWN_COOLDOWN_TICKS   = _env_int("FWS_RESPAWN_CD", 2000)
RESPAWN_BATCH_PER_TEAM   = _env_int("FWS_RESPAWN_BATCH", 1)
RESPAWN_ARCHER_SHARE     = _env_float("FWS_RESPAWN_ARCHER_SHARE", 0.50)
RESPAWN_INTERIOR_BIAS    = _env_float("FWS_RESPAWN_INTERIOR_BIAS", 0.95)
RESPAWN_JITTER_RADIUS    = _env_int("FWS_RESPAWN_JITTER", 5)

# ================================================================
# Map Generation
# ================================================================
RANDOM_WALLS      = _env_int("FWS_RAND_WALLS", 10)
WALL_SEG_MIN      = _env_int("FWS_WALL_SEG_MIN", 7)
WALL_SEG_MAX      = _env_int("FWS_WALL_SEG_MAX", 15)
WALL_AVOID_MARGIN = _env_int("FWS_WALL_AVOID_MARGIN", 1)

HEAL_ZONE_COUNT      = _env_int("FWS_HEAL_COUNT", 1)
HEAL_ZONE_SIZE_RATIO = _env_float("FWS_HEAL_SIZE_RATIO", 40/256)
HEAL_RATE            = _env_float("FWS_HEAL_RATE", 0.02)

CP_COUNT           = _env_int("FWS_CP_COUNT", 1)
CP_SIZE_RATIO      = _env_float("FWS_CP_SIZE_RATIO", 30/256)
CP_REWARD_PER_TICK = _env_float("FWS_CP_REWARD", 0.05)

# ================================================================
# Team Rewards
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
# PPO
# ================================================================
PPO_UPDATE_TICKS  = _env_int("FWS_PPO_TICKS", 64)
PPO_LR            = _env_float("FWS_PPO_LR", 3e-4)
# --- NEW SCHEDULER PARAMS ---
PPO_LR_T_MAX      = _env_int("FWS_PPO_T_MAX", 500_000) # Decay over 500k training steps
PPO_LR_ETA_MIN    = _env_float("FWS_PPO_ETA_MIN", 1e-6) # Floor at a very small LR
# --- END NEW ---
PPO_EPOCHS        = _env_int("FWS_PPO_EPOCHS", 3)
PPO_CLIP          = _env_float("FWS_PPO_CLIP", 0.2)
PPO_ENTROPY_BONUS = _env_float("FWS_PPO_ENTROPY", 0.01)
PPO_VALUE_COEF    = _env_float("FWS_PPO_VCOEF", 0.5)
PPO_MAX_GRAD_NORM = _env_float("FWS_PPO_MAXGN", 1.0)

# ================================================================
# UI / Viewer
# ================================================================
ENABLE_UI  = _env_bool("FWS_UI", True)
CELL_SIZE  = _env_int("FWS_CELL_SIZE", 8) # Increased for better visibility on smaller grids
TARGET_FPS = _env_int("FWS_TARGET_FPS", 60)

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