from __future__ import annotations
import os
import torch

# ================================================================
# Device / Precision Setup (JIT-safe)
# ================================================================
_FWS_AMP = os.getenv("FWS_AMP", "1")
AMP_ENABLED: bool = not (_FWS_AMP == "0" or _FWS_AMP.lower() == "false")

def amp_enabled() -> bool:
    return AMP_ENABLED

TORCH_DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)
TORCH_DTYPE: torch.dtype = torch.float16 if AMP_ENABLED and torch.cuda.is_available() else torch.float32

# ================================================================
# Simulation World
# ================================================================
GRID_WIDTH  = int(os.getenv("FWS_GRID_W", 128))
GRID_HEIGHT = int(os.getenv("FWS_GRID_H", 128))
START_AGENTS_PER_TEAM = int(os.getenv("FWS_START_PER_TEAM", 900))
MAX_AGENTS = int(os.getenv("FWS_MAX_AGENTS", 3000))
AGENT_FEATURES = 6

TICK_LIMIT = int(os.getenv("FWS_TICK_LIMIT", 0))
TARGET_TPS = int(os.getenv("FWS_TARGET_TPS", 60))

# ================================================================
# Perception
# ================================================================
NUM_DIRECTIONS = 8
RAY_MAX_STEPS  = int(os.getenv("FWS_RAY_STEPS", 6))
OBS_DIM        = NUM_DIRECTIONS * 2 * RAY_MAX_STEPS

# ================================================================
# Actions
# ================================================================
NUM_ACTIONS = int(os.getenv("FWS_NUM_ACTIONS", 17))

# ================================================================
# Agents / Brains / Evolution
# ================================================================
PER_AGENT_BRAINS        = True
MUTATION_PERIOD_TICKS   = int(os.getenv("FWS_MUTATE_EVERY", 2000))
MUTATION_FRACTION_ALIVE = float(os.getenv("FWS_MUTATE_FRAC", 0.10))

# ================================================================
# Combat Parameters
# ================================================================
MAX_HP   = float(os.getenv("FWS_MAX_HP", 1.0))
BASE_ATK = float(os.getenv("FWS_BASE_ATK", 0.2))

# ================================================================
# Team Rewards
# ================================================================
TEAM_KILL_REWARD       = float(os.getenv("FWS_REW_KILL",       1.0))
TEAM_DMG_DEALT_REWARD  = float(os.getenv("FWS_REW_DMG_DEALT",  0.2))
TEAM_DEATH_PENALTY     = float(os.getenv("FWS_REW_DEATH",     -1.0))
TEAM_DMG_TAKEN_PENALTY = float(os.getenv("FWS_REW_DMG_TAKEN", -0.1))

# ================================================================
# PPO Hyperparameters
# ================================================================
PPO_UPDATE_TICKS  = int(os.getenv("FWS_PPO_TICKS", 20))
PPO_LR            = float(os.getenv("FWS_PPO_LR", 3e-4))
PPO_EPOCHS        = int(os.getenv("FWS_PPO_EPOCHS", 3))
PPO_CLIP          = float(os.getenv("FWS_PPO_CLIP", 0.2))
PPO_ENTROPY_BONUS = float(os.getenv("FWS_PPO_ENTROPY", 0.01))
PPO_VALUE_COEF    = float(os.getenv("FWS_PPO_VCOEF", 0.5))
PPO_MAX_GRAD_NORM = float(os.getenv("FWS_PPO_MAXGN", 1.0))

# ================================================================
# UI / Viewer
# ================================================================
ENABLE_UI  = os.getenv("FWS_UI", "1") not in {"0", "false", "False"}
CELL_SIZE  = int(os.getenv("FWS_CELL_SIZE", 6))
TARGET_FPS = int(os.getenv("FWS_TARGET_FPS", 60))

# ================================================================
# Summary String
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
