# codex_bellum/record_sim.py
import os
import cv2
import numpy as np
import torch
from datetime import datetime

from codex_bellum.engine.grid import make_grid
from codex_bellum.engine.agent_registry import AgentsRegistry
from codex_bellum.engine.tick import TickEngine
from codex_bellum.engine.spawn import spawn_uniform_random
from codex_bellum.simulation.stats import SimulationStats
from codex_bellum.utils.persistence import ResultsWriter
from codex_bellum import config


def record_sim(fps=30, sample_every=1, limit_ticks=10000, scale=4):
    """
    Run the sim headless and record raw video (not timelapse).
    Saves guaranteed-working .avi video into results/<sim_timestamp>/.

    Args:
        fps          : playback FPS for output video
        sample_every : record every N ticks (1 = every tick)
        limit_ticks  : stop after this many ticks (0 = infinite)
        scale        : upscale pixels for readability
    """
    dev = config.TORCH_DEVICE
    grid = make_grid(dev)
    registry = AgentsRegistry(grid)
    stats = SimulationStats()

    # Spawn agents
    spawn_uniform_random(registry, grid, config.START_AGENTS_PER_TEAM)
    engine = TickEngine(registry, grid, stats)

    # --- results directory (same style as main.py) ---
    rw = ResultsWriter()
    run_dir = rw.start(config_obj={"summary": config.summary_str()})
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- output path (force .avi) ---
    H, W = grid.size(1), grid.size(2)
    out_h, out_w = H * scale, W * scale
    base_name = f"sim_{ts}"
    output_file = os.path.join(run_dir, f"{base_name}.avi")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_file, fourcc, fps, (out_w, out_h))

    if not writer.isOpened():
        print("[recorder] ERROR: could not open VideoWriter. Recording disabled.")
        return

    print(f"[recorder] Recording → {os.path.abspath(output_file)}")

    # --- color palette ---
    colors = {
        0: (30, 30, 30),   # empty
        1: (80, 80, 80),   # wall
        2: (50, 50, 220),  # red
        3: (220, 120, 50), # blue
    }

    try:
        while limit_ticks == 0 or stats.tick < limit_ticks:
            engine.run_tick()

            if stats.tick % sample_every == 0:
                occ = grid[0].to("cpu").numpy().astype(np.uint8)

                frame = np.zeros((H, W, 3), dtype=np.uint8)
                for k, c in colors.items():
                    frame[occ == k] = c

                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                writer.write(frame)

            if limit_ticks > 0 and stats.tick >= limit_ticks:
                break

    except KeyboardInterrupt:
        print("[recorder] Interrupted manually")

    finally:
        writer.release()
        rw.close()
        print(f"[recorder] Saved video → {os.path.abspath(output_file)}")


if __name__ == "__main__":
    # Run until manually interrupted
    record_sim(fps=30, sample_every=1, limit_ticks=0)