# final_war_sim

A GPU-friendly 2D grid war simulation with many agents, fast per-tick mechanics, and a transformer-based policy interface.

## What runs by default

`war_simulation/main.py` builds the simulation and runs either:
- an interactive UI viewer loop, or
- a headless loop that writes per-tick results.

Core wiring: `main.py` uses `AgentsRegistry`, `TickEngine`, `make_grid`, spawn helpers, zone/wall mapgen, `ResultsWriter`, and `Viewer`.

## Key concepts (from code)

### World representation
- Grid tensor with an occupancy channel.
- Agents stored in a Struct-of-Arrays tensor (`AgentsRegistry`) for GPU efficiency.

### Observation for the transformer
The per-agent observation includes:
- 64-ray sensing block (512 values total)
- additional “rich” features appended (team one-hot, unit type, hp/max_hp, vision, atk, + reserved slots)

### Policy / brain
Agents use a transformer brain at runtime.
Spawn logic instantiates `TransformerBrain` (or a scripted transformer brain depending on config flags).

### Output / logs
Runs write under `results/sim_<timestamp>/` and can optionally record a raw `.avi` video.

## Project layout

- `war_simulation/main.py` — entrypoint
- `war_simulation/config.py` — configuration + summary string
- `war_simulation/engine/` — tick loop, actions, raycasting, spawn/respawn, mapgen
- `war_simulation/agent/` — transformer brain + ensemble logic
- `war_simulation/ui/` — pygame viewer + camera

## Requirements

Minimum:
- Python
- `torch`
- `numpy`
- `pygame` (viewer)

Optional:
- `opencv-python` (only if `RECORD_VIDEO` is enabled)

## Run

### UI mode
```bash
cd war_simulation
python main.py
