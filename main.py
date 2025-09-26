# final_war_sim/main.py
from __future__ import annotations

import os
import json
import signal
import sys
from typing import Optional

import torch
import numpy as np
try:
    import cv2  # optional; recording becomes no-op if not installed
except Exception:
    cv2 = None

# PPO (package-local)
from rl.ppo import PerAgentPPO

# Local modules
import config
from simulation.stats import SimulationStats
from engine.agent_registry import AgentsRegistry
from engine.tick import TickEngine
from engine.grid import make_grid, assert_on_same_device
from engine.spawn import spawn_uniform_random, spawn_symmetric
from engine.respawn import respawn_some
from engine.mapgen import add_random_walls, make_zones
from utils.sanitize import runtime_sanity_check
from utils.persistence import ResultsWriter
from utils.profiler import torch_profiler_ctx, nvidia_smi_summary
from ui.viewer import Viewer


# ------------------------------- helpers -------------------------------

def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes"}

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _config_snapshot() -> dict:
    """Light, serializable snapshot of current config for run metadata."""
    return {
        "summary": config.summary_str(),
        "GRID_W": config.GRID_WIDTH,
        "GRID_H": config.GRID_HEIGHT,
        "START_PER_TEAM": config.START_AGENTS_PER_TEAM,
        "MAX_AGENTS": getattr(config, "MAX_AGENTS", None),
        "OBS_DIM": config.OBS_DIM,
        "NUM_ACTIONS": config.NUM_ACTIONS,
        "MAX_HP": config.MAX_HP,
        "BASE_ATK": config.BASE_ATK,
        "AMP": config.amp_enabled() if hasattr(config, "amp_enabled") else False,
        "PPO": {
            "UPDATE_TICKS": config.PPO_UPDATE_TICKS,
            "LR": config.PPO_LR,
            "EPOCHS": config.PPO_EPOCHS,
            "CLIP": config.PPO_CLIP,
            "ENTROPY": config.PPO_ENTROPY_BONUS,
            "VCOEF": config.PPO_VALUE_COEF,
            "MAX_GN": config.PPO_MAX_GRAD_NORM,
        },
        "REWARDS": {
            "KILL": config.TEAM_KILL_REWARD,
            "DMG_DEALT": config.TEAM_DMG_DEALT_REWARD,
            "DEATH": config.TEAM_DEATH_PENALTY,
            "DMG_TAKEN": config.TEAM_DMG_TAKEN_PENALTY,
            "CAPTURE_TICK": getattr(config, "CP_REWARD_PER_TICK", None),
        },
        "UI": {
            "ENABLE_UI": config.ENABLE_UI,
            "CELL_SIZE": config.CELL_SIZE,
            "TARGET_FPS": config.TARGET_FPS,
        },
        "SPAWN": {
            "SPAWN_ARCHER_RATIO": float(getattr(config, "SPAWN_ARCHER_RATIO", os.getenv("FWS_SPAWN_ARCHER_RATIO", 0.4))),
        },
        "ARCHER_LOS_BLOCKS_WALLS": bool(getattr(config, "ARCHER_LOS_BLOCKS_WALLS", False)),
    }

def _seed_all_from_env() -> Optional[int]:
    raw = os.getenv("FWS_SEED", "").strip()
    if not raw:
        return None
    try:
        seed = int(raw)
    except Exception:
        return None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    return seed


# --------------------------- simple video recorder ---------------------------

RECORD_VIDEO   = getattr(config, "RECORD_VIDEO", True)
VIDEO_FPS      = getattr(config, "VIDEO_FPS", 30)
VIDEO_SCALE    = getattr(config, "VIDEO_SCALE", 4)
VIDEO_EVERY_TICKS = 1  # record every tick

class _SimpleRecorder:
    """
    Minimal recorder that always works on Windows.
    Uses MJPG codec (baked into OpenCV builds) → AVI file.
    """
    def __init__(self, run_dir: str, grid: torch.Tensor, fps: int = 30, scale: int = 4):
        self.enabled = False
        self.grid = grid
        self.path = None
        self.size = None
        self.writer = None

        if not RECORD_VIDEO or cv2 is None:
            return

        base = os.path.basename(run_dir.rstrip("\\/"))
        self.path = os.path.join(run_dir, f"{base}_raw.avi")

        H, W = int(grid.size(1)), int(grid.size(2))
        self.size = (W * scale, H * scale)

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(self.path, fourcc, fps, self.size)

        if writer.isOpened():
            self.writer = writer
            self.enabled = True
            print(f"[video] recording → {self.path} (codec=MJPG AVI, fps={fps}, scale={scale}x)")
        else:
            print(f"[video] ERROR: could not open writer for {self.path}; recording disabled.")

        self.palette = np.array([
            [30, 30, 30],   # empty
            [80, 80, 80],   # wall
            [50, 50, 220],  # red
            [220, 120, 50], # blue
        ], dtype=np.uint8)

    def write(self):
        if not self.enabled:
            return
        occ = self.grid[0].to("cpu").numpy().astype(np.uint8)
        frame = self.palette[occ]
        frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_NEAREST)
        self.writer.write(frame)

    def close(self):
        if self.enabled and self.writer is not None:
            self.writer.release()
            print(f"[video] saved → {self.path}")


# ------------------------------ run loops ------------------------------

def _headless_loop(engine, stats, reg, grid, rw, limit: int) -> None:
    from .utils.profiler import torch_profiler_ctx, nvidia_smi_summary
    with torch_profiler_ctx() as prof:
        try:
            while limit == 0 or stats.tick < limit:
                engine.run_tick()
                rw.write_tick(stats.as_row())
                deaths = stats.drain_dead_log()
                if deaths:
                    rw.write_deaths(deaths)
                if (stats.tick % 500) == 0:
                    from .utils.sanitize import runtime_sanity_check
                    runtime_sanity_check(grid, reg.agent_data)
                if prof is not None:
                    prof.step()
                if (stats.tick % 100) == 0:
                    gpu = nvidia_smi_summary() or "-"
                    print(
                        f"Tick {stats.tick:7d} | "
                        f"Red {stats.red.score:8.2f} | Blue {stats.blue.score:8.2f} | "
                        f"Elapsed {stats.elapsed_seconds:7.2f}s | GPU {gpu}"
                    )
        except KeyboardInterrupt:
            print("\n[main] Interrupted — shutting down gracefully.")


# --------------------------------- main --------------------------------

def main() -> None:
    torch.set_float32_matmul_precision("high")
    seed = _seed_all_from_env()
    if seed is not None:
        print(f"[main] Using deterministic seed: {seed}")

    print(config.summary_str())

    # Build world
    grid = make_grid(config.TORCH_DEVICE)
    registry = AgentsRegistry(grid)
    stats = SimulationStats()

    # Add thin gray walls BEFORE spawning
    add_random_walls(grid,
                     n_segments=config.RANDOM_WALLS,
                     seg_min=config.WALL_SEG_MIN,
                     seg_max=config.WALL_SEG_MAX,
                     avoid_margin=config.WALL_AVOID_MARGIN)

    # Zones (heal & capture)
    H, W = int(grid.size(1)), int(grid.size(2))
    zones = make_zones(H, W, device=config.TORCH_DEVICE)

    # Spawn both teams uniformly across map with Soldier/Archer mix
    spawn_mode = os.getenv("FWS_SPAWN_MODE", "uniform").lower()
    if spawn_mode.startswith("sym"):
        spawn_symmetric(registry, grid, per_team=config.START_AGENTS_PER_TEAM)
    else:
        spawn_uniform_random(registry, grid, per_team=config.START_AGENTS_PER_TEAM)

    assert_on_same_device(grid, registry.agent_data)

    # Engine (pass zones to enable heal & capture scoring)
    engine = TickEngine(registry, grid, stats, zones=zones)

    # PPO window scaffold (optional, preserved)
    ppo = PerAgentPPO(registry)
    ppo.begin_window(stats.snapshot())

    # Persistence
    rw = ResultsWriter()
    run_dir = rw.start(config_obj=_config_snapshot())
    print(f"[main] Results → {run_dir}")

    # Video recorder (optional)
    recorder = _SimpleRecorder(run_dir, grid, fps=VIDEO_FPS, scale=VIDEO_SCALE)
    _orig_run_tick = engine.run_tick

    def _run_tick_with_recording():
        _orig_run_tick()
        if recorder.enabled and (stats.tick % VIDEO_EVERY_TICKS) == 0:
            recorder.write()

    engine.run_tick = _run_tick_with_recording

    # Run (UI or headless)
    try:
        signal.signal(signal.SIGINT, signal.getsignal(signal.SIGINT))
    except Exception:
        pass

    try:
        if config.ENABLE_UI:
            viewer = Viewer(grid, cell_size=config.CELL_SIZE)
            # Some older Viewer builds have slightly different signatures; keep both paths safe.
            try:
                viewer.run(engine, registry, stats, tick_limit=config.TICK_LIMIT, target_fps=config.TARGET_FPS)
            except TypeError:
                viewer.run(engine, registry, stats, tick_limit=config.TICK_LIMIT)
        else:
            _headless_loop(engine, stats, registry, grid, rw, limit=config.TICK_LIMIT)
    except KeyboardInterrupt:
        print("\n[main] Interrupted — flushing logs…")
    finally:
        try:
            deaths = stats.drain_dead_log()
            if deaths:
                rw.write_deaths(deaths)
        except Exception:
            pass
        try:
            final_summary = {
                "final_tick": stats.tick,
                "elapsed_seconds": getattr(stats, "elapsed_seconds", None),
                "scores": {"red": getattr(stats.red, "score", None), "blue": getattr(stats.blue, "score", None)},
                "cp": {"red": getattr(stats.red, "cp_points", 0.0), "blue": getattr(stats.blue, "cp_points", 0.0)},
            }
            with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(final_summary, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        try:
            recorder.close()
        except Exception:
            pass
        try:
            rw.close()
        except Exception:
            pass
        print("[main] Shutdown complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[main] Fatal error: {type(e).__name__}: {e}", file=sys.stderr)
        raise
