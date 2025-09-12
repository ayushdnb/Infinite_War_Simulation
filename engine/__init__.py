# final_war_sim/engine/__init__.py
from .grid import make_grid, assert_on_same_device

# Backward-compat alias (old code may import create_grid)
def create_grid(device):
    return make_grid(device)

from .agent_registry import AgentsRegistry
# Re-export subpackages for convenience
from .game import move_mask  # noqa: F401
from .ray_engine import raycaster2d  # noqa: F401
