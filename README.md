WarSim Pro – Transformer Brain Branch

Branch: transformer_brain

Overview

This experimental branch extends the baseline WarSim Pro simulator with a transformer‑based agent brain and provides a pre‑trained weight file for quick evaluation. All core engine, perception and contract definitions remain identical to main, ensuring drop‑in compatibility
raw.githubusercontent.com
.

What’s New?

Transformer‑style Ray Encoder – Instead of the tiny MLP‑based RayEncoder used in the baseline, this branch explores a deeper attention model. It still respects the input/output contract (B, F) → (logits, value)
raw.githubusercontent.com
, but internally uses multi‑head self‑attention layers to capture interactions between rays across multiple timesteps. The model is defined in agent/brain.py and can be swapped without changing the tick loop or mask logic.

Pre‑trained weights – A checkpoint file, brain_agent_80_t_354.pth, contains weights trained for 80 agents over 354 thousand ticks. Loading this file allows you to observe emergent tactics without training from scratch.

Configurable attention sizes – You can adjust RAY_PE_DIM and RAY_ATTN_DIM in config.py to control the size of positional encodings and attention layers
raw.githubusercontent.com
. Larger values improve expressiveness at the cost of throughput.

Loading the Pre‑Trained Brain

The brain_agent_80_t_354.pth file stores the state_dict of the transformer brain. To use it during simulation:

import torch
from war_simulation.agent.brain import ActorCriticBrain

# construct the model with matching dims
model = ActorCriticBrain(obs_dim=85, act_dim=41)
model.load_state_dict(torch.load('brain_agent_80_t_354.pth'))
model.eval()

# register this model for all agents before starting the tick loop
registry = AgentsRegistry(grid)
registry.brains = [model] * registry.capacity


Alternatively, when training with PPO you can initialise the policy weights from this checkpoint to accelerate convergence.

Usage Tips

Disable tiny attention in the baseline when comparing against the transformer brain. Set RAY_ATTN_DIM=0 on main for a fair FPS comparison
raw.githubusercontent.com
.

Increase MAX_AGENTS cautiously. Transformer layers are heavier than the default tiny MLP; ensure your GPU has sufficient memory
raw.githubusercontent.com
.

Experiment with longer observation windows. The transformer brain may benefit from richer perception (e.g. scanline rays)
raw.githubusercontent.com
.

Rationale

The baseline brain uses a minimal MLP with a one‑head attention over 8 rays
raw.githubusercontent.com
. While efficient, it may struggle to capture long‑range dependencies or complex patterns. A transformer can model relationships between rays more flexibly at the cost of compute. This branch serves as a sandbox to test such ideas while keeping the rest of the codebase unchanged.

Remaining Docs

For installation, configuration, performance tips and the general architecture, see the README for the main branch.
