# War Simulation (Multi‑Agent Gridworld)

🧭 **Summary**

A GPU‑first, tick‑based war simulation on a 2D grid. Two teams (red/blue) spawn large numbers of agents (soldier/archer), perceive the world via ray features + compact state features, choose actions via per‑agent neural policies, fight, die, and respawn with optional PPO training.

* **State**: `grid` tensor `(3, H, W)` + `AgentsRegistry` struct‑of‑arrays.
* **Loop**: build observations → mask actions → batched inference → apply movement/combat → rewards → PPO record/update → respawn.
* **UI**: optional real‑time viewer (pygame) with performance‑aware CPU caching.

---

## ✅ Quick start

### 1) Install

```bash
pip install torch numpy pygame-ce
# optional (recording):
pip install opencv-python
```

### 2) Run

From repo root:

```bash
python -m war_simulation.main
```

Or:

```bash
python war_simulation/main.py
```

Outputs are written under `results/sim_YYYY-MM-DD_HH-MM-SS/`.

---

## 🧩 High-level architecture

```text
+---------------------+     +---------------------+     +---------------------+
|  grid (3,H,W)       |     | AgentsRegistry      |     |  TickEngine         |
|  occ / hp / slot_id |<--->| agent_data (N,C)    |<--->|  run_tick()         |
|                     |     | brains[slot]        |     |                     |
+----------+----------+     +----------+----------+     +----------+----------+
           |                           |                           |
           |                           |                           v
           |                           |                 +-------------------+
           |                           |                 |  PPO runtime      |
           |                           |                 |  (per-agent)      |
           |                           |                 +-------------------+
           v                           v
+---------------------+     +---------------------+
| mapgen / zones      |     | respawn controller  |
| walls, heal, CP     |     | clone/noise/new     |
+---------------------+     +---------------------+

(Optional UI) Viewer reads cached CPU snapshots of grid + minimal agent columns.
```

---

## ⚙️ Configuration

All runtime parameters live in `config.py`. Many can be overridden via environment variables prefixed with `FWS_` (see `config.py` for the full list).

Common knobs:

* Grid: `GRID_WIDTH`, `GRID_HEIGHT`
* Population: `START_AGENTS_PER_TEAM`, `MAX_AGENTS`
* Observation/action: `OBS_DIM`, `NUM_ACTIONS`
* PPO: `PPO_ENABLED`, `PPO_UPDATE_TICKS`, `PPO_LR`, `PPO_EPOCHS`, `PPO_CLIP`
* Respawn: `RESPAWN_COOLDOWN_TICKS`, `RESP_FLOOR_PER_TEAM`, `RESP_MAX_PER_TICK`
* UI: `ENABLE_UI`, `TARGET_FPS`, `CELL_SIZE`

---

## 🗺️ World state

### Grid tensor

`grid` is a float tensor `(3, H, W)`:

* **Channel 0: occupancy**

  * `0` empty
  * `1` wall
  * `2` red team
  * `3` blue team
* **Channel 1: hp** (0..MAX_HP) — used for fast reads / UI
* **Channel 2: slot_id** (`-1` empty; else registry slot index)

### AgentsRegistry

A struct‑of‑arrays tensor `agent_data` with per‑slot columns (GPU‑friendly). Key columns:

* `alive` (0/1)
* `team` (2 red / 3 blue)
* `hp`, `hp_max`
* `x`, `y`
* `unit` (1 soldier / 2 archer)
* `vision` (per agent)
* `atk` (per agent)
* `agent_id` (unique persistent ID; changes on respawn)

Each slot also stores its own **brain module** (`brains[slot]`). This supports fully independent policies per agent.

---

## 🔁 Tick loop (workflow + data flow)

Each simulation step is one **tick**:

```text
1) Alive scan
   alive_idx = where(agent_data[:,alive] == 1)

2) Observation build (for alive only)
   obs: (K, OBS_DIM)

3) Action mask
   mask: (K, NUM_ACTIONS) bool

4) Batched inference (bucket by model signature)
   logits: (K, NUM_ACTIONS)
   value:  (K,)
   action ~ Categorical(masked_logits)

5) Apply actions
   - movement updates grid + registry
   - combat updates hp, deaths, grid clears

6) Rewards + bookkeeping
   team rewards + optional per-agent rewards

7) PPO record/update (if enabled)

8) Respawn
   maintain floors + periodic budget spawns

9) Metrics/logging
```

---

## 🗺️ Observations (what an agent sees)

Default observation size is:

* `RAY_TOKEN_COUNT * RAY_FEAT_DIM` + `RICH_BASE_DIM` + `INSTINCT_DIM`
* With defaults: `32*8 + 23 + 4 = 283`.

### Ray features (flattened)

Each ray encodes **first-hit** information:

* `onehot6`: none / wall / red-soldier / red-archer / blue-soldier / blue-archer
* `dist_norm`: distance (normalized)
* `hp_norm`: target HP (normalized; 0 if none/wall)

Per ray: `6 + 1 + 1 = 8` features.

### Rich base vector (23 dims)

A compact, non-token feature block used to represent “who am I” and “global context” (e.g., normalized HP, position, unit stats, tick progress, and a small set of aggregated counters). The exact ordering is defined in `config.py`.

### Instinct vector (4 dims)

A small heuristic context computed from local/aggregate signals:

* ally archer density
* ally soldier density
* noisy enemy density
* threat ratio

(Exact definition lives in the tick engine.)

---

## 🧠 Brains and tokens

The project supports multiple brain implementations (selected via `BRAIN_KIND`). Each agent owns its own instance.

### TransformerBrain (baseline)

* Treats the observation as token groups (ray tokens + a compact rich token).
* Outputs policy logits and value estimate.

### TronBrain (token‑partitioned)

Tron splits the same input into structured tokens and processes them in stages.

**Token types (conceptual):**

* **Ray tokens**: `32` tokens × `8` dims
* **Semantic tokens** (sliced from rich_base):

  * own context
  * world context
  * zone context
  * team context
  * combat context
* **Instinct token**: 1 token
* **Memory token**: 1 token
* **Decision tokens**: 3 tokens (readout)

**Processing sketch:**

```text
Stage 1: Local perception
  ray tokens -> self-attention (local)

Stage 2: Semantic context
  semantic + instinct (+ memory) -> attention

Stage 3: Fusion
  decision tokens attend over (ray + semantic + instinct)

Stage 4: Readout
  decision tokens -> policy logits + value
```

This layout makes it easy to add/remove semantic blocks while keeping ray features stable.

---

## 🎮 Action space

Default `NUM_ACTIONS = 41`:

|   Range | Meaning                                           |
| ------: | ------------------------------------------------- |
|     `0` | idle                                              |
|  `1..8` | move in 8 directions (N, NE, E, SE, S, SW, W, NW) |
| `9..40` | attack in 8 directions × 4 ranges                 |

Attack encoding:

* `dir = (a - 9) // 4` in `[0..7]`
* `r   = ((a - 9) % 4) + 1` in `[1..4]`

Unit gating:

* **Soldier**: `r=1` only (melee)
* **Archer**: `r=1..ARCHER_RANGE` (clipped to 4)

### Action masking

Before sampling, invalid actions are masked out:

* moves are valid only into in‑bounds empty cells
* attacks are valid only if the target cell contains an enemy, and the unit is allowed to use that range

---

## ⚔️ Combat, death, and rewards

### Combat resolution

* Attacks target a grid cell computed from `(dir, r)`.
* If the cell contains an enemy (`grid[2]` provides victim slot id), HP is reduced by attacker `atk`.
* If victim HP crosses `<= 0`, the victim is marked dead and removed from the grid.

### Metabolism and healing

* Per tick, agents can lose HP due to a “metabolism” drain.
* A **heal zone** mask can restore HP for agents standing on those tiles.

### Capture points

* One or more **capture point** masks exist.
* Each tick, points are credited to the team(s) with living units inside those masks.

### Reward hooks (team level)

Reward shaping (used by PPO) is configured via `config.py`, including:

* kill reward
* damage dealt reward
* death penalty
* damage taken penalty
* optional capture‑tick reward

---

## ♻️ Respawning and population dynamics

Respawning is handled by `RespawnController`:

* Maintains a minimum **floor** of living agents per team.
* Uses per-team cooldown (hysteresis) to avoid rapid oscillation.
* Every `RESP_PERIOD_TICKS`, spawns a budget split across teams.

### Brain assignment on respawn

For each new agent slot:

* A unique `agent_id` is allocated.
* Unit type is sampled (soldier/archer ratio).
* Brain is either:

  * **clone+noise** from a living parent of the same team (probability `RESP_CLONE_PROB`), or
  * a **new** brain instance.

Rare mutation: every 1000 respawns, unit stats may be randomly scaled (HP/ATK/vision).

---

## 🖥️ UI viewer (optional)

The viewer is designed to reduce GPU→CPU sync cost:

* Refreshes cached CPU snapshots of `grid[0]` (occ) and `grid[2]` (slot_id) only every N frames.
* Copies only a small set of agent columns for alive agents to support click‑inspect and rendering.

Controls include pause, step‑tick, and speed multiplier.

---

## 📁 Outputs

Each run writes a timestamped folder:

* `config.json` (light snapshot)
* per‑tick metrics
* death events
* optional model snapshots / metadata

---

## 🧰 Notes and troubleshooting

* **Device/dtype guards**: the engine includes checks to catch device mismatches early.
* **Performance**: disable UI (`ENABLE_UI=0`) and reduce population for headless training.
* **Profiling**: enable torch profiler via `FWS_PROFILE=1` (writes traces under `prof/`).

---

### Repository map (core modules)

* `war_simulation/main.py` — entrypoint (setup, run loop, results)
* `engine/tick.py` — tick loop (obs, masking, actions, rewards, respawn)
* `engine/agent_registry.py` — per-agent storage + bucketing
* `engine/grid.py` — grid construction + device assertions
* `engine/mapgen.py` — random walls + zone masks
* `engine/respawn.py` — respawn logic (clone/noise/new + rare mutation)
* `engine/game/move_mask.py` — action mask
* `engine/ray_engine/*` — ray first-hit features
* `agent/*` — brain implementations
* `rl/ppo_runtime.py` — PPO data collection and updates
* `ui/viewer.py` — pygame viewer
