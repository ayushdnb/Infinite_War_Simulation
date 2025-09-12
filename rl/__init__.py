"""
Reinforcement learning components for final_war_sim.

This package contains training code for learning agent policies. The
current implementation includes a simple PPO trainer skeleton. It is
not fully functional but demonstrates how to structure the RL logic.
"""

from .ppo import PerAgentPPO

__all__ = ["PPOTrainer"]
