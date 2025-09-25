# WarSim Pro — Production‑Grade 2‑D Multi‑Agent Battle Simulator 🧠

> **Status:** Research‑grade, modular, and performance‑tuned for single‑GPU laptops (Windows 11, CUDA 12.x, RTX 3060 6 GB, 16 GB RAM).
> **Goals:** Deterministic contracts, ultra‑loose coupling, high observability, 2k+ agents at ≥60 ticks/s on a 128×128 grid.

---

## 0. Executive Overview 📌

**WarSim Pro** is a ground‑up, contract‑first rewrite of a 2‑D multi‑agent combat simulator. It cleanly separates **perception → ego‑frame → policy → mask → sampling → engine → grid** and imposes hard **ABI contracts** for actions, directions, observations, and masks. All hot paths are vectorized and GPU‑friendly (Struct‑of‑Arrays), with numerically stable tiny networks per agent for scale.

### Design Tenets

* **Contracts over code:** Rigid schemas for action layout, directional groups, and observation blocks.
* **Strict modularity:** Each layer is independently swappable; tests pin invariants at the seams.
* **Performance discipline:** SoA tensors, batched per‑bucket inference, no dynamic Python in inner loops.
* **Observability & testability:** Action/mask stats, mutation logs, perf counters; unit & property tests for rotation invariants.
* **Determinism knobs:** Global seeds, fixed init, and Torch AMP safety (float32 casts at model inputs).

---

## 1. System Architecture 🏗️

```
                 ┌────────────────────────────────────────────────┐
                 │                 Tick Engine                    │
                 │  (time step, collisions, hp, deaths, scoring) │
                 └───────────────┬────────────────────────────────┘
                                 │
                                 │ alive indices (K)
                                 ▼
┌──────────────────────────┐   positions/teams  ┌──────────────────────────┐
│      Agents Registry     │ ─────────────────▶ │     Ego‑Frame Runtime    │
│   SoA tensor + brains    │                   │  heading8, rotate rays   │
└───────────┬──────────────┘                   └──────────┬───────────────┘
            │                                           rotated obs (K,85)
            │ buckets by NN signature                              │
            ▼                                                       ▼
┌──────────────────────────┐   per‑bucket obs    ┌──────────────────────────┐
│        Bucketer          │ ─────────────────▶ │       Policy Runner       │
│ groups alive agents by   │                    │ per‑bucket forward pass   │
│ identical brain topology │ ◀──────────────────│ (K_b, F) → (K_b, A)       │
└───────────┬──────────────┘  value/logits      └──────────┬───────────────┘
            │                                             unrotate by heading
            ▼                                                       │
┌──────────────────────────┐           logits (global dirs)         ▼
│      Mask Builder        │ ◀─────────────────────────────────┌──────────────┐
│  move/attack constraints │                                   │  Sampler     │
└───────────┬──────────────┘                                   └────┬─────────┘
            │ actions                                                 │
            ▼                                                         ▼
┌──────────────────────────┐                               ┌──────────────────┐
│         Engine           │ ───────────────────────────▶  │  Grid (3,H,W)    │
│ moves, attacks, respawns │                               │ occ/hp/agent_id  │
└───────────┬──────────────┘                               └────────┬─────────┘
            │ rays                                                      │
            ▼                                                           ▼
┌──────────────────────────┐                                   ┌────────────────┐
│         Raycaster        │ ◀──────────────────────────────── │  Map/Zones     │
│ first‑hit 8×8 features   │                                   │ walls/heal/CP  │
└──────────────────────────┘                                   └────────────────┘
```

**Key flows:**

* **Ego‑frame** rotates the 8 ray slots so index 0 = “ahead”; **unrotation** restores global action indices before masking/sampling.
* **Buckets** ensure batched inference per unique brain topology to minimize kernel launches.
* **Masks** enforce legal moves/attacks (17 or 41 action layout) with unit‑type gating.

---

## 2. Hard Contracts (ABI) 🔒

### 2.1 Direction Set (DIRS8)

Order: **N, NE, E, SE, S, SW, W, NW**.
Single source of truth is shared by **ego‑frame, masks, and raycaster**. All rotation/unrotation assumes this order.

### 2.2 Action Layout (A = 17 or 41)

* **Index 0**: idle
* **1..8**: move in DIRS8 (one bin per direction)
* If **A = 17**: **9..16** = melee (r=1) in DIRS8.
* If **A = 41**: **9..40** = ranged (DIRS8 × r=1..4) contiguous per direction; groups are exact 8‑wide blocks.

### 2.3 Observation V2 (F = 85)

* **0..63**: **rays (64)** = 8 rays × 8 features: `onehot6(type) + dist_norm + hp_norm`.

  * type classes: 0: none, 1: wall, 2: red‑soldier, 3: red‑archer, 4: blue‑soldier, 5: blue‑archer.
* **64..84**: **rich self/env (21)** (freeform features; stable ordering enforced by tests).
* **Ego‑frame** rotates *only* the first 64 dims.

### 2.4 Grid Channels (3, H, W)

0: occupancy (0 empty, 1 wall, 2 red, 3 blue) · 1: hp (0..MAX\_HP) · 2: agent\_id (−1 if empty).

### 2.5 Units

* 1: **Soldier** (melee; ranged r=1 only)
* 2: **Archer**  (ranged; r≤`ARCHER_RANGE` ≤ 4)

**Invariant tests** cover: direction roll consistency, ego‑rotation/unrotation idempotence on 8‑wide groups, and mask algebra for bounds/unit gating.

---

## 3. Repository Layout 📂

```
final_war_sim/
├─ agent/
│  ├─ brain.py                 # Tiny Actor‑Critic + RayEncoder (PE + tiny attention)
│  ├─ encoders.py              # RingPositionalEncoding, TinyRayAttention, RayEncoder
│  ├─ heads.py                 # FactorizedDirectionalHeads (idle + DIRS8 groups)
│  ├─ ensemble.py              # Batched per‑bucket forward (value kept 1‑D)
│  └─ mutation.py              # Gentle evolve: tiny noise, micro‑widen/prune, hard caps
├─ engine/
│  ├─ agent_registry.py        # SoA table, brain list, buckets, mutations
│  ├─ ego_frame.py             # heading8, rotate rays, unrotate logits
│  ├─ ego_tick_adapter.py      # runtime adapter API for ticks
│  ├─ game/move_mask.py        # action masks (A=17/41), bounds + unit gating
│  ├─ ray_engine/raycaster2d.py# fast 8‑dir raycast (alt: first‑hit features)
│  ├─ grid.py                  # grid init, device/dtype assertions
│  └─ mapgen.py                # walls, heal zones, capture points (CP)
├─ config.py                   # single source of knobs (dtype/device/ranges/sizes)
├─ scripts/                    # run/train utilities (optional launcher)
└─ tests/                      # unit/property tests for contracts & hot paths
```

> If you maintain a legacy branch (`codex_bellum`), keep **ABI contracts identical** so entry‑points/visualizers can be shared.

---

## 4. Installation & Environment 🧰

### 4.1 Prerequisites

* Windows 11, Python 3.10.x, CUDA 12.x
* NVIDIA RTX 3060 (6 GB) and Intel i7‑10750H
* PyTorch ≥ 2.1 (CUDA build)

### 4.2 Setup

```bash
# clone
git clone https://github.com/<you>/battle_simulation.git
cd battle_simulation

# (optional) create venv
python -m venv .venv && .\.venv\Scripts\activate

# install deps (edit requirements as needed)
pip install -r requirements.txt

# editable install
pip install -e .
```

### 4.3 Quick Run

```bash
# Run a headless short simulation (example; adapt to your package path)
python -m final_war_sim.scripts.run_sim --ticks 2000 --grid 128 128 --agents 1000

# Or, if on the legacy branch name
python -m codex_bellum.main
```

> If you use a Pygame viewer, ensure SDL video drivers are available; otherwise run headless with recording disabled.

---

## 5. Configuration (Knobs) ⚙️

All knobs reside in **`config.py`** and are read once at import time. **Do not** read config dynamically inside hot loops. Example baseline:

```python
# Device & numerics
TORCH_DEVICE      = torch.device("cuda")
TORCH_DTYPE       = torch.float16      # AMP‑friendly
AMP_ENABLED       = True               # autocast in PPO / eval

# Grid & agents
GRID_WIDTH        = 128
GRID_HEIGHT       = 128
MAX_AGENTS        = 2048
MAX_HP            = 1.0

# Actions & units
NUM_ACTIONS       = 41                 # 17 or 41
ARCHER_RANGE      = 4                  # 1..4
UNIT_SOLDIER      = 1
UNIT_ARCHER       = 2

# Rays (first‑hit features)
RAYCAST_MAX_STEPS = 10
RAY_PE_DIM        = 4                  # 0 disables positional encoding
RAY_ATTN_DIM      = 16                 # 0 disables tiny attention

# Map generation
RANDOM_WALLS      = 24
WALL_SEG_MIN      = 4
WALL_SEG_MAX      = 18
WALL_AVOID_MARGIN = 4
HEAL_ZONE_COUNT   = 2
HEAL_ZONE_SIZE_RATIO = 0.04

# Mutation / evolution (gentle)
MAX_PARAMS_PER_BRAIN     = 40_000
PRUNE_SOFT_BUDGET        = 35_000
MUTATION_WIDEN_PROB      = 0.30     # widen fc2 by +1..+2 occasionally
MUTATION_TWEAK_PROB      = 0.70     # tiny Gaussian noise on ~0.3% weights
MUTATION_TWEAK_FRAC      = 0.003
MUTATION_TWEAK_STD       = 0.01
MUTATION_FRACTION_ALIVE  = 0.10

# Ego‑frame
RELATIVE_DIRS     = True            # enable rotate/unrotate machinery
```

**How to adapt:**

* **Throughput first:** lower `RAY_ATTN_DIM` to 0 (disable attention) and/or `RAY_PE_DIM` to 0 to maximize FPS.
* **Smarter agents:** increase `RAY_PE_DIM` (ring PE richness) and small `RAY_ATTN_DIM` (e.g., 8..16) for better directional context.
* **Combat density:** increase `MAX_AGENTS` cautiously; budget per‑tick memory and ensure batch sizes (per bucket) remain ≥ 64.
* **Action richness:** switch to `NUM_ACTIONS=41` only if ranged combat matters; masking cost and sampling overhead increase.

---

## 6. Agent Brain & Heads 🧩

### 6.1 Actor‑Critic Brain

A tiny **MLP trunk** with stable Xavier/Kaiming init, explicit **float32 cast on inputs** (AMP safety), and a **RayEncoder** for the first 64 dims.

* **Input:** `(B, F=85)`
* **RayEncoder:** per‑ray Linear(8→16) → optional **RingPositionalEncoding** (dim=`RAY_PE_DIM`) → optional **TinyRayAttention** (dim=`RAY_ATTN_DIM`) → flatten → Linear → **32‑dim** ray context.
* **Rich features:** concatenated `(B, 21)`
* **Trunk:** Linear→SiLU×3
* **Heads:** `actor: Linear(h, A)` and `critic: Linear(h, 1)`

**Why tiny attention?** With only 8 slots, a single head (`attn_dim≤16`) captures symmetry and salient directions without blowing up latency.

### 6.2 Factorized Directional Heads

`idle` (scalar) + **8‑wide groups** for move/melee/ranged. This preserves a **single categorical action space** while keeping logits organized by direction. Enables **cheap unrotation** and mask application.

**Output:** `logits: (B, A)` and `value: (B,)` (value kept **1‑D** even for `B=1` to avoid cat() scalars in batching).

### 6.3 Ensemble Forward (Per‑Bucket)

`ensemble_forward(models, obs)` iterates over K models for a bucket, slices `(1,F)` per model, and concatenates outputs to `(K, A)` and `(K,)`. It normalizes "dist‑like" heads (object with `.logits`) and guards value shape to **never** be 0‑D.

**Tuning guidance:** When most agents share the same topology (typical), bucket sizes are large and throughput is high. Avoid heterogeneous topologies unless evolution is aggressive.

---

## 7. Ego‑Frame Runtime 🎯

**Heading selection** prioritizes: (1) actual displacement; else (2) nearest visible **enemy** ray; else (3) sticky previous heading.

* `rotate_obs64_inplace`: rotates only the **first 64 dims** so ray index 0 aligns with heading (ego‑centric).
* `unrotate_logits_inplace`: inverses rotation on **each 8‑wide group** (moves, melee, each ranged ring) to keep global action indices.

**Do not** rotate non‑directional columns (idle, scalar features). Tests enforce that partial groups (<8) are left unrotated.

---

## 8. Action Masks 🔐

`build_mask(pos_xy, teams, grid, unit)` outputs `(N, A)` bool masks:

* **Move**: in‑bounds and **free** cells only (occ=0).
* **Melee (A≥17)**: neighbor cell contains **enemy** (team mismatch; ignore walls/ally).
* **Ranged (A=41)**: DIRS8 × r=1..4; unit gating: **Soldier→r=1**, **Archer→r≤ARCHER\_RANGE**.
* **Idle** always permitted.

Keep masks cheap: integer math for indices, single channel reads, and bulk writes per 8‑wide block to avoid scatter overhead.

---

## 9. Raycasting 🔦

Two patterns are supported:

1. **First‑hit features** (recommended for V2): one‑hot class + normalized distance + hp at first collision per ray. Output **(N,64)**.
2. **Scanline sampling** (alt): densified `(N, 8×S×2)` of occ/hp per step where S=`RAYCAST_MAX_STEPS`. Use only for richer models; higher memory bandwidth.

Normalize distances by **per‑agent vision cap** to keep features comparable across units.

---

## 10. Agents Registry & Bucketing 🗂️

A single **SoA tensor** `(MAX_AGENTS, NUM_COLS)` holds all agents. Brains live in a parallel Python list (kept off the tensor for contiguity). The registry:

* Tracks **alive** state, `(x,y)`, `team`, `hp`, `atk`, and `unit id`.
* Builds **buckets** by architecture signature (e.g., concatenated Linear/Conv shapes).
* Applies **mutations** in‑place using pluggable `mutate_fn`.

**Why SoA?** Contiguous columnar slices vectorize better (fewer cache misses) and simplify batched dispatch.

---

## 11. Mutation & Evolution 🧬

**Gentle mutation** to avoid parameter explosion:

* (1) Small Gaussian noise to a tiny fraction of weights (`~0.3%`).
* (2) Rare micro‑widen of `fc2` by +1..+2 neurons, then rewire actor/critic inputs.
* (3) Structured prune (drop smallest‑norm rows) if over budget.
* **Hard cap:** `MAX_PARAMS_PER_BRAIN`.

**Why gentle?** Keeps per‑agent forward pass cheap and preserves bucket homogeneity, improving overall FPS.

---

## 12. Tick Loop Integration 🔄

Typical high‑level pseudocode:

```python
# Pre-allocate runtime helpers
registry = AgentsRegistry(grid)
ego = EgoFrameRuntime(capacity=registry.capacity, device=device)

for t in range(T):
    alive_idx = ...                        # Long[K]
    pos_xy    = registry.positions_xy(alive_idx)
    teams     = registry.agent_data[alive_idx, COL_TEAM]

    # Build obs V2: rays (64) + rich (21)
    rays64    = raycast8_firsthit(pos_xy, grid, unit_map)
    obs       = torch.cat([rays64, rich21], dim=1)

    # Ego rotation
    obs_ego   = ego.rotate_obs64_inplace(alive_idx, pos_xy, teams, obs)

    # Bucketing & per-bucket forward
    buckets   = registry.build_buckets(alive_idx)
    logits    = torch.empty((0, A), device=device)
    values    = torch.empty((0,), device=device)
    for B in buckets:
        oB = obs_ego.index_select(0, B.indices)
        distB, valB = ensemble_forward(B.models, oB)
        logB = ego.unrotate_logits_inplace(B.indices, distB.logits)
        logits = torch.cat([logits, logB], dim=0)
        values = torch.cat([values, valB], dim=0)

    # Mask → sample → step engine
    mask      = build_mask(pos_xy, teams, grid, unit=registry.units(alive_idx))
    actions   = sample(logits, mask)
    step_engine(actions, registry, grid)

    # Optional evolution
    mutants   = pick_mutants(alive_idx)
    registry.apply_mutations(mutants, mutate_model_inplace)
```

---

## 13. Performance Playbook ⚡

### 13.1 Fast Path Principles

* **Avoid Python control** in inner loops; precompute indices and use vector operations.
* **Batch by bucket** to amortize kernel launch overhead.
* **Keep tensors contiguous** and on the same `device`/`dtype`; assert via guards early.
* **Prefer float16 compute** under AMP; cast model inputs to float32 *once* at head.

### 13.2 Hot Knobs

* `RAY_ATTN_DIM = 0` → disable attention; boosts FPS.
* `RAY_PE_DIM = 0` → no ring PE; smallest RayEncoder.
* `NUM_ACTIONS = 17` → narrow head & mask; cheaper sampling.
* **Grid size** and **agent count** dominate memory traffic; tune first.

### 13.3 GPU Memory Budgeting

* Ray features `(K,64)` are small; avoid large stepwise scan tensors unless needed.
* Ensure per‑bucket batch size **≥ 64** to keep SMs busy.
* Track parameter counts; evolution must respect `MAX_PARAMS_PER_BRAIN`.

---

## 14. Observability & Logging 📈

Recommended metrics (per tick or window):

* Action distribution histograms (per group; detect degenerate policies).
* Mask hit‑rates (move feasibility, enemy‑in‑range %) per unit type.
* Entropy of policy outputs; KL vs. previous tick for stability.
* Evolution logs: brain param counts, widen/prune events, generation.
* Perf counters: ticks/s, forward ms, mask ms, raycast ms.

Emit JSONL/CSV with stable field names for dashboards.

---

## 15. Testing Strategy 🧪

* **Unit tests** for: DIRS8 order, rotation/unrotation, mask correctness on synthetic boards, RayEncoder shapes.
* **Property tests**: rotate→unrotate = identity for all 8‑wide groups; masks remain within bounds under random placements.
* **Numerics**: value head returns **1‑D** always; AMP path equivalence vs. float32 within tolerance.

Run: `pytest -q` in the project root.

---

## 16. Extending the Engine 🔧

### 16.1 Add a Unit Type

1. Define new unit id in `config.py`.
2. Extend **type encoding** in raycaster (add classes if visible distinction is needed).
3. Update **mask** rules for new unit’s ranged/melee constraints.
4. Add unit‑type column to registry (if new stats required).

### 16.2 Change Action Space

* For **A=49 (DIRS8 × r=1..5)**: keep **8‑wide group contract**. Update `_group_slices_for_actions()` and mask writer to emit 5‑column blocks per dir. Tests must pin layout.

### 16.3 Replace the Brain

* Maintain **I/O contract**: `(B,F) → (logits(B,A), value(B,1 or B))`. Keep `.logits` attribute if returning a dist wrapper.
* Reuse **FactorizedDirectionalHeads** to preserve unrotation logic and mask compatibility.

### 16.4 Richer Perception

* Swap first‑hit rays for **scanline** encoding; adjust `RayEncoder` to handle `(8×S×2)` reshaping and compress to 32‑64 dims.

---

## 17. PPO Training (Optional) 🎓

* Use centralized experience buffers; per‑agent brains can still be independent if weights are cloned per bucket per update.
* **AMP on** by default; cast observations to float16 in buffers and promote to float32 at model ingress.
* Clip ratios and entropy bonuses should be tracked by **unit type** to detect niche collapse (e.g., archers disappearing).

> If you train off‑policy or with evolutionary strategies, keep policy evaluation API identical for drop‑in replacement.

---

## 18. Determinism & Reproducibility 🧪

* Seed Python, NumPy, and Torch; set `torch.backends.cudnn.deterministic = True` when benchmarking algorithmic changes (note: may reduce raw FPS).
* Fix initialization schemes; avoid time‑based seeding in mutation.
* Persist **config + seed + git sha** in run metadata directories.

---

## 19. Troubleshooting 🛠️

* **`ModuleNotFoundError`**: ensure `pip install -e .` and correct package path (`final_war_sim` vs `codex_bellum`).
* **Viewer cannot open writer**: disable recording or install required codecs; run headless for profiling.
* **Agents not learning strategy**: verify mask correctness, action entropy, and that ranged groups are reachable; increase `ARCHER_RANGE` or map open spaces.
* **Extinction of unit types**: add mild curriculum (spawn balance), or lower mutation amplitude; track unit population over time.

---

## 20. Roadmap 🧭

* **Capture points** scoring every N ticks with area control masks.
* **Metabolism drain** to reduce camping; reward shaping for activity.
* **Bucket‑aware JIT** (TorchScript/torch.compile) for steady topologies.
* **Triton kernels** for mask/raycast inner loops (optional).

---

## 21. License & Citation 📜

Choose a permissive license (e.g., MIT) for community contributions. When publishing, please cite **WarSim Pro** and include the commit SHA and configuration snapshot used to produce results.

---

## 22. Quick Reference (Cheat Sheet) 🗂️

* **Directions:** DIRS8 = `[N, NE, E, SE, S, SW, W, NW]`
* **Actions (A=41):** `idle | move×8 | melee×8 | ranged(r=1..4)×8`
* **Obs (F=85):** `rays64 (8×[onehot6, dist, hp]) + rich21`
* **Grid:** `(3,H,W)` with `occ/hp/agent_id`
* **Heads:** factorized; 8‑wide groups must rotate/unrotate together
* **Mutation:** gentle; cap params; prefer noise→micro‑widen→prune
* **Knobs:** `RAY_PE_DIM`, `RAY_ATTN_DIM`, `NUM_ACTIONS`, `ARCHER_RANGE`, `RAYCAST_MAX_STEPS`

---

### Final Note

This codebase is intended to be **surgically modular**. Preserve the ABI contracts and the engine will remain swappable, testable, and fast under growth. Maintain small, stable brains for throughput; use evolution sparingly; measure everything.
