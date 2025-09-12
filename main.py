# final_war_sim/main.py
from __future__ import annotations

"""
Entry point for FINAL_WAR_SIM.

Design goals:
- Zero surprises: fail-fast on device/dtype mismatches, but keep the sim running
  even if optional features (profiling/respawn) are disabled.
- Headless + UI modes share the same core TickEngine/Registry/Stats objects.
- Clean shutdown: always flush writers, even on Ctrl+C.
"""

import os
import json
import signal
import sys
from typing import Optional

import torch

# PPO lives under package rl/
from final_war_sim.rl.ppo import PerAgentPPO

# Local modules
from . import config
from .simulation.stats import SimulationStats
from .engine.agent_registry import AgentsRegistry
from .engine.tick import TickEngine
from .engine.grid import make_grid, assert_on_same_device
from .engine.spawn import spawn_symmetric
from .engine.respawn import respawn_some
from .utils.sanitize import runtime_sanity_check
from .utils.persistence import ResultsWriter
from .utils.profiler import torch_profiler_ctx, nvidia_smi_summary
from .ui.viewer import Viewer


# ------------------------------- helpers -------------------------------

def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) in {"1", "true", "True", "YES", "yes"}

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _config_snapshot() -> dict:
    """Light, serializable snapshot of current config for run metadata."""
    return {
        "summary": config.summary_str(),
        "GRID_W": config.GRID_WIDTH,
        "GRID_H": config.GRID_HEIGHT,
        "START_PER_TEAM": config.START_AGENTS_PER_TEAM,
        "MAX_AGENTS": config.MAX_AGETS if hasattr(config, "MAX_AGETS") else getattr(config, "MAX_AGENTS", None),
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
        },
        "UI": {
            "ENABLE_UI": config.ENABLE_UI,
            "CELL_SIZE": config.CELL_SIZE,
            "TARGET_FPS": config.TARGET_FPS,
        },
    }

def _maybe_respawn(reg: AgentsRegistry, grid: torch.Tensor, tick: int) -> None:
    """
    Controlled by env:
      FWS_RESPAWN_EVERY (ticks, >0), FWS_RESPAWN_PER_TEAM (count, >0)
    No-ops if unset.
    """
    every = _env_int("FWS_RESPAWN_EVERY", 0)
    per_team = _env_int("FWS_RESPAWN_PER_TEAM", 0)
    if every > 0 and per_team > 0 and (tick % every) == 0 and tick > 0:
        respawn_some(reg, grid, team_is_red=True,  count=per_team)
        respawn_some(reg, grid, team_is_red=False, count=per_team)

def _seed_all_from_env() -> Optional[int]:
    """
    If FWS_SEED is set, seed torch (and CUDA if present) for reproducibility.
    """
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
    return seed


# ------------------------------ run loops ------------------------------

def _headless_loop(
    engine: TickEngine,
    stats: SimulationStats,
    reg: AgentsRegistry,
    grid: torch.Tensor,
    rw: ResultsWriter,
    limit: int,
) -> None:
    """
    Headless loop (no UI). Keeps stdout light; prints heartbeat every 100 ticks.
    """
    with torch_profiler_ctx() as prof:
        try:
            while limit == 0 or stats.tick < limit:
                _maybe_respawn(reg, grid, stats.tick)

                engine.run_tick()

                # stream stats + death log
                rw.write_tick(stats.as_row())
                deaths = stats.drain_dead_log()
                if deaths:
                    rw.write_deaths(deaths)

                # occasional sanity checks (cheap but not free)
                if (stats.tick % 500) == 0:
                    runtime_sanity_check(grid, reg.agent_data)

                # profiler step if enabled
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
    # Torch precision knobs (safe on RTX 3060)
    torch.set_float32_matmul_precision("high")

    # Optional seeding
    seed = _seed_all_from_env()
    if seed is not None:
        print(f"[main] Using deterministic seed: {seed}")

    # Device selection is owned by config
    dev = config.TORCH_DEVICE
    print(config.summary_str())

    # --- World/bootstrap ---
    grid = make_grid(dev)
    registry = AgentsRegistry(grid)
    stats = SimulationStats()
    spawn_symmetric(registry, grid, per_team=config.START_AGENTS_PER_TEAM)

    # Safety: same device & dtype (fast fail if user mixed CPU/CUDA tensors)
    assert_on_same_device(grid, registry.agent_data)

    # --- Engine ---
    engine = TickEngine(registry, grid, stats)

    # --- PPO init AFTER engine/registry/stats exist ---
    ppo = PerAgentPPO(registry)
    # first window snapshot (team-level PPO per user's design)
    ppo.begin_window(stats.snapshot())

    # --- Persistence / results directory ---
    rw = ResultsWriter()
    run_dir = rw.start(config_obj=_config_snapshot())
    print(f"[main] Results → {run_dir}")

    # --- Handle SIGINT cleanly on Windows too ---
    # (Python already converts Ctrl+C to KeyboardInterrupt; this keeps parity.)
    try:
        signal.signal(signal.SIGINT, signal.getsignal(signal.SIGINT))
    except Exception:
        pass

    # --- Run ---
    try:
        if config.ENABLE_UI:
            # Viewer owns the loop in UI mode
            viewer = Viewer(grid, cell_size=config.CELL_SIZE)

            # If newer Viewer supports 'run(..., target_fps=)', pass it; else fallback.
            try:
                viewer.run(
                    engine,
                    registry,
                    stats,
                    tick_limit=config.TICK_LIMIT,
                    target_fps=config.TARGET_FPS,
                )
            except TypeError:
                # Backward compatible signature
                viewer.run(engine, registry, stats, tick_limit=config.TICK_LIMIT)
        else:
            _headless_loop(engine, stats, registry, grid, rw, limit=config.TICK_LIMIT)

    except KeyboardInterrupt:
        print("\n[main] Interrupted — flushing logs…")

    finally:
        # Always drain & close writer
        try:
            deaths = stats.drain_dead_log()
            if deaths:
                rw.write_deaths(deaths)
        except Exception:
            pass

        try:
            # drop a final summary.json side-car next to ResultsWriter output
            final_summary = {
                "final_tick": stats.tick,
                "elapsed_seconds": getattr(stats, "elapsed_seconds", None),
                "scores": {
                    "red": getattr(stats.red, "score", None),
                    "blue": getattr(stats.blue, "score", None),
                },
            }
            with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(final_summary, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        try:
            rw.close()
        except Exception:
            pass

        print("[main] Shutdown complete.")


if __name__ == "__main__":
    # Allow running the module directly: `python -m final_war_sim.main`
    # or as a script: `python final_war_sim/main.py`
    try:
        main()
    except Exception as e:
        # Last-ditch guard so the user sees a clear error before the interpreter exits
        print(f"[main] Fatal error: {type(e).__name__}: {e}", file=sys.stderr)
        raise
