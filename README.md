<div align="center">
<img src="https://www.google.com/search?q=https://placehold.co/600x300/121418/FFFFFF%3Ftext%3DCodex-Bellum%26font%3Dmontserrat" alt="Codex-Bellum Banner">
<h1>Codex-Bellum</h1>
<p><b>An Engine for Emergent Combat Doctrines</b></p>
<p><i>Survival of the fittest, rewritten in PyTorch. A high-performance digital crucible where thousands of neural agents evolve their own "Book of War."</i></p>

<p>
<a href="#"><img src="https://www.google.com/search?q=https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python Version"></a>
<a href="#"><img src="https://www.google.com/search?q=https://img.shields.io/badge/pytorch-2.0%2B-orange.svg" alt="PyTorch Version"></a>
<a href="#"><img src="https://www.google.com/search?q=https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
<a href="#"><img src="https://www.google.com/search?q=https://img.shields.io/badge/status-active-brightgreen.svg" alt="Project Status"></a>
</p>
</div>

Codex-Bellum is not a game. It is a massively multi-agent simulation framework where the doctrines of war are not programmed, but discovered. In this 2D world, thousands of autonomous agents, each guided by a unique and evolving neural network, discover sophisticated combat tactics from first principles.

This repository provides a research-grade, high-performance platform for studying neuroevolution, emergent complexity, and the genesis of intelligent, coordinated behavior. The "Codex" is the unwritten, ever-changing set of strategies that proves superior in the relentless crucible of digital conflict.

<p align="center">
<!-- TODO: Replace with an actual GIF of the simulation. -->
<img src="https://www.google.com/search?q=https://placehold.co/800x450/121418/80C0F0%3Ftext%3DSimulation%2Bin%2BAction%2B(Placeholder%2BGIF)" alt="Simulation Showcase">
<br>
<em>Thousands of agents in a dynamic, large-scale conflict.</em>
</p>

✨ Core Features
🧠 Neuroevolution at Scale: Every single agent possesses a unique, mutable ActorCriticBrain. The system doesn't train a single master policy, but rather a diverse ecosystem of thousands of competing, evolving minds.

🧬 Hybrid Learning Paradigm: Fuses state-of-the-art Per-Agent Proximal Policy Optimization (PPO) for short-term reward maximization with Genetic Algorithms for long-term structural evolution. Brains don't just learn—they grow, prune, and adapt their very architecture over generations.

🚀 High-Performance GPU Architecture: Engineered from the ground up for massive parallelism. Leverages a Struct-of-Arrays (SoA) memory layout, vectorized PyTorch operations, and AMP (float16) for maximum throughput, simulating thousands of agents at high TPS.

👁️ Dynamic Raycast Perception: Agents perceive their environment not through a simple grid view, but through an efficient 8-directional raycasting engine, providing rich, high-dimensional input about nearby allies, enemies, and obstacles.

📊 Extensive Data & Analytics: Every action, death, and state change can be recorded. The simulation features a background persistence writer for CSV and Parquet, capturing detailed logs for offline analysis, policy inspection, and academic research.

🖥️ Interactive Visualization Engine: A sophisticated UI built with Pygame allows for real-time observation, camera control (pan/zoom), and direct inspection of any agent's state, neural network topology, and vital statistics.

🏛️ Architecture Deep Dive
The engine's design prioritizes performance, scalability, and research flexibility. It's built on a foundation of highly optimized, decoupled components.

1. The Simulation Trinity
The core of the simulation is managed by three tightly integrated, GPU-native components:

AgentsRegistry: The central nervous system. It manages all agent state within a single, massive SoA tensor (MAX_AGENTS, AGENT_FEATURES). This layout is critical for coalesced memory access on the GPU, enabling vectorized operations across the entire agent population simultaneously. It also holds the list of individual brain models for each agent.

Grid: A multi-layered tensor (3, H, W) representing the world state:

Channel 0: Occupancy (empty, wall, red team, blue team)

Channel 1: Health Points

Channel 2: Agent ID at that location
This structure allows for instant, parallel lookups and modifications.

TickEngine: The heart of the simulation. It orchestrates each discrete timestep, from agent perception and action selection to movement, combat resolution, and state updates. All operations within the engine are designed to be vectorized and conflict-free.

2. The Agent Brain & Learning
ActorCriticBrain: A canonical Actor-Critic neural network serves as the universal mind for all agents. Its simplicity is a feature, providing a stable foundation for genetic mutation. The network exposes its layers (fc1, fc2, actor, critic) to allow the mutation engine to perform architectural modifications.

PerAgentPPO: A custom PPO implementation that operates on experience collected in windows of time (PPO_WINDOW_TICKS). It calculates team-level rewards and updates the brains of a subset of agents in each window, ensuring that learning remains computationally tractable even with thousands of agents.

MutationEngine: This is where long-term evolution occurs. It applies a suite of genetic operators to agent brains:

Weight Perturbation: Adds small Gaussian noise to a fraction of weights, enabling fine-tuning.

Network Widening: Occasionally adds new neurons to hidden layers, allowing for an increase in model capacity.

Network Pruning: Removes neurons with the lowest L2 magnitude if a brain exceeds a "soft" parameter budget, promoting efficiency.
This hybrid approach allows agents to learn effective short-term tactics via PPO while their underlying brain structures evolve over generations to better support more complex strategies.

3. Performance Engineering
Performance is not an afterthought; it is a core design principle.

Vectorization: All critical loops—perception, action masking, movement, and combat—are fully vectorized. There are no for loops over agents in the hot path.

Conflict-Free Parallelism: Movement conflicts (two agents wanting the same cell) are resolved in a single parallel step using scatter_add_ to count target destinations.

Automatic Mixed Precision (AMP): The entire simulation can run in torch.float16 on compatible GPUs, nearly doubling performance and halving memory usage with minimal loss in precision.

Asynchronous I/O: Simulation statistics and logs are written to disk by a separate process, ensuring that file I/O never blocks the main simulation loop.

🚀 Getting Started
Prerequisites
Python 3.9+

PyTorch 2.0+ (CUDA or MPS enabled for GPU acceleration)

pygame (for UI mode)

imageio (for video recording)

Installation
Clone the repository:

git clone [https://github.com/your-username/Codex-Bellum.git](https://github.com/your-username/Codex-Bellum.git)
cd Codex-Bellum

Install dependencies:

pip install -r requirements.txt

(Note: Ensure your PyTorch installation is correct for your specific hardware. See the official PyTorch website).

Running the Simulation
The simulation is launched via the main entry point.

python -m final_war_sim.main

Headless Mode (for training / servers)
By default, the simulation runs in headless mode if UI dependencies are not met or if explicitly disabled.

# Run a headless simulation, disabling the UI
FWS_UI=0 python -m final_war_sim.main

Results, logs, and statistics will be saved to a timestamped directory in results/.

UI Mode (for visualization)
If pygame is installed, the UI will run by default.

# Run with the interactive UI
FWS_UI=1 python -m final_war_sim.main

Controls:

Pan: WASD / Arrow Keys

Zoom: Mouse Wheel

Select Agent: Left Click

Mark Agent: M (when an agent is selected)

Copy Brain: C (saves selected agent's brain to copied_brain.pth)

Trigger Mutation: E (manually triggers mutation for ~10% of agents)

Quit: ESC

🛠️ Configuration
The simulation's behavior is deeply configurable via environment variables. This allows for rapid experimentation without code changes.

Variable

Default

Description

FWS_GRID_W

128

Width of the simulation world.

FWS_GRID_H

128

Height of the simulation world.

FWS_START_PER_TEAM

900

Number of agents to spawn for each team at the start.

FWS_MAX_AGENTS

3000

Hard capacity limit for the AgentsRegistry.

FWS_TICK_LIMIT

0

If > 0, simulation will stop after this many ticks.

FWS_AMP

1

Enable Automatic Mixed Precision (0 to disable).

FWS_SEED

None

Set a global seed for reproducibility.

FWS_PPO_TICKS

20

How often (in ticks) to run a PPO training update.

FWS_MUTATE_EVERY

2000

How often (in ticks) to run a genetic mutation cycle.

FWS_UI

1

Enable the Pygame UI (0 to disable).

FWS_TARGET_FPS

60

Target frames-per-second for the UI.

🔬 Research & Philosophy
Codex-Bellum is more than a codebase; it is a tool for inquiry. It can be used to explore fundamental questions in artificial intelligence and complex systems:

How do coordinated, strategic behaviors emerge from decentralized, individual learning?

What is the interplay between short-term optimization (RL) and long-term structural adaptation (EAs)?

Can diverse and specialized agent roles (e.g., scouts, defenders, attackers) evolve naturally within a homogenous population?

How do environmental pressures and map topologies shape the evolution of tactics?

This project is built with the hope that by creating a sufficiently complex and competitive digital ecosystem, we can observe the sparks of a truly emergent, artificial intelligence.

🤝 Contributing
Contributions are welcome. Please open an issue to discuss your ideas or submit a pull request.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

<div align="center">
<p><b>Forged in the fires of digital conflict.</b></p>
</div>