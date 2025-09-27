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

# Local modules - changed to absolute imports
import config
from simulation.stats import SimulationStats
from engine.agent_registry import AgentsRegistry
from engine.tick import TickEngine
from engine.grid import make_grid, assert_on_same_device
from engine.spawn import spawn_symmetric, spawn_uniform_random
from engine.mapgen import add_random_walls, make_zones
from utils.persistence import ResultsWriter
from ui.viewer import Viewer

# ------------------------------- helpers -------------------------------

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
            "UPDATE_TICKS": getattr(config, "PPO_UPDATE_TICKS", 5),
            "LR": getattr(config, "PPO_LR", 3e-4),
            "EPOCHS": getattr(config, "PPO_EPOCHS", 3),
            "CLIP": getattr(config, "PPO_CLIP", 0.2),
            "ENTROPY": getattr(config, "PPO_ENTROPY_BONUS", 0.01),
            "VCOEF": getattr(config, "PPO_VALUE_COEF", 0.5),
            "MAX_GN": getattr(config, "PPO_MAX_GRAD_NORM", 1.0),
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
            "SPAWN_ARCHER_RATIO": float(getattr(config, "SPAWN_ARCHER_RATIO", 0.4)),
        },
        "ARCHER_LOS_BLOCKS_WALLS": bool(getattr(config, "ARCHER_LOS_BLOCKS_WALLS", False)),
    }

def _seed_all_from_env() -> Optional[int]:
    raw = os.getenv("FWS_SEED", "").strip()
    if not raw:
        return None
    try:
        seed = int(raw)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        return seed
    except (ValueError, TypeError):
        return None

# --------------------------- simple video recorder ---------------------------

class _SimpleRecorder:
    def __init__(self, run_dir: str, grid: torch.Tensor, fps: int, scale: int):
        self.enabled = False
        if not getattr(config, "RECORD_VIDEO", False) or cv2 is None:
            return

        H, W = int(grid.size(1)), int(grid.size(2))
        self.size = (W * scale, H * scale)
        self.path = os.path.join(run_dir, "simulation_raw.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.writer = cv2.VideoWriter(self.path, fourcc, float(fps), self.size)

        if self.writer.isOpened():
            self.enabled = True
            print(f"[video] recording → {self.path}")
            self.palette = np.array([
                [30, 30, 30], [80, 80, 80], [220, 80, 80], [80, 120, 240]
            ], dtype=np.uint8)
            self.grid = grid
        else:
            print(f"[video] ERROR: could not open writer for {self.path}.")

    def write(self):
        if not self.enabled: return
        occ = self.grid[0].cpu().numpy().astype(np.uint8)
        frame = self.palette[occ]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_resized = cv2.resize(frame_bgr, self.size, interpolation=cv2.INTER_NEAREST)
        self.writer.write(frame_resized)

    def close(self):
        if self.enabled and self.writer:
            self.writer.release()
            print(f"[video] saved → {self.path}")

# ------------------------------ run loops ------------------------------

def _headless_loop(engine, stats, reg, grid, rw, limit: int) -> None:
    # Import here to avoid circular dependency issues at top-level
    from utils.profiler import torch_profiler_ctx, nvidia_smi_summary
    from utils.sanitize import runtime_sanity_check

    with torch_profiler_ctx() as prof:
        try:
            while limit == 0 or stats.tick < limit:
                engine.run_tick()
                rw.write_tick(stats.as_row())
                deaths = stats.drain_dead_log()
                if deaths:
                    rw.write_deaths(deaths)
                
                if (stats.tick % 500) == 0:
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

    grid = make_grid(config.TORCH_DEVICE)
    registry = AgentsRegistry(grid)
    stats = SimulationStats()

    add_random_walls(grid)
    zones = make_zones(config.GRID_HEIGHT, config.GRID_WIDTH, device=config.TORCH_DEVICE)
    
    spawn_mode = os.getenv("FWS_SPAWN_MODE", "uniform").lower()
    if spawn_mode == "symmetric":
        spawn_symmetric(registry, grid, per_team=config.START_AGENTS_PER_TEAM)
    else:
        spawn_uniform_random(registry, grid, per_team=config.START_AGENTS_PER_TEAM)

    engine = TickEngine(registry, grid, stats, zones=zones)

    rw = ResultsWriter()
    run_dir = rw.start(config_obj=_config_snapshot())
    print(f"[main] Results → {run_dir}")

    recorder = _SimpleRecorder(
        run_dir, grid, 
        fps=getattr(config, "VIDEO_FPS", 30), 
        scale=getattr(config, "VIDEO_SCALE", 4)
    )
    
    _orig_run_tick = engine.run_tick
    def _run_tick_with_recording():
        _orig_run_tick()
        if recorder.enabled and (stats.tick % getattr(config, "VIDEO_EVERY_TICKS", 1) == 0):
            recorder.write()
    engine.run_tick = _run_tick_with_recording

    try:
        if config.ENABLE_UI:
            viewer = Viewer(grid, cell_size=config.CELL_SIZE)
            viewer.run(engine, registry, stats, tick_limit=config.TICK_LIMIT, target_fps=config.TARGET_FPS)
        else:
            _headless_loop(engine, stats, registry, grid, rw, limit=config.TICK_LIMIT)
    except KeyboardInterrupt:
        print("\n[main] Interrupted — flushing logs…")
    finally:
        # Graceful shutdown for persistence
        if rw:
            deaths = stats.drain_dead_log()
            if deaths: rw.write_deaths(deaths)
            summary = {
                "final_tick": stats.tick,
                "elapsed_seconds": stats.elapsed_seconds,
                "scores": {"red": stats.red.score, "blue": stats.blue.score},
            }
            with open(os.path.join(run_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            rw.close()
        if recorder:
            recorder.close()
        print("[main] Shutdown complete.")

if __name__ == "__main__":
    main()

