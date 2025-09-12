# final_war_sim/recorder/video_writer.py
from __future__ import annotations

import os
from typing import Optional

import numpy as np


class VideoWriter:
    """
    Lazy MP4 writer. If ffmpeg is unavailable, falls back to PNG frames.
    Expected frame: HxWx3 uint8 (RGB).
    """
    def __init__(self, run_dir: str, fps: int = 60):
        self.run_dir = os.path.abspath(run_dir)
        self.fps = int(max(1, fps))
        self._mp4_path = os.path.join(self.run_dir, "simulation.mp4")
        self._frames_dir = os.path.join(self.run_dir, "frames_fallback")
        self._writer = None
        self._mode = None  # "mp4" or "pngs"

    def _ensure_writer(self, frame: np.ndarray):
        if self._writer is not None:
            return
        # Try MP4 first
        try:
            import imageio  # type: ignore
            # H.264 sometimes not bundled; let imageio pick available codec
            self._writer = imageio.get_writer(self._mp4_path, fps=self.fps)
            self._mode = "mp4"
            return
        except Exception:
            # Fallback: PNG frames
            os.makedirs(self._frames_dir, exist_ok=True)
            self._writer = True  # sentinel
            self._mode = "pngs"

    def add_frame(self, frame: np.ndarray):
        if frame is None:
            return
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"VideoWriter expects HxWx3 RGB uint8; got {frame.shape}")
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        self._ensure_writer(frame)

        if self._mode == "mp4":
            import imageio  # type: ignore
            self._writer.append_data(frame)
        else:
            # frames_fallback/frame_00000001.png
            idx = len(os.listdir(self._frames_dir)) + 1
            path = os.path.join(self._frames_dir, f"frame_{idx:08d}.png")
            try:
                import imageio  # type: ignore
                imageio.imwrite(path, frame)
            except Exception:
                # Last-ditch: numpy save (rare)
                np.save(path.replace(".png", ".npy"), frame)

    def close(self):
        if self._mode == "mp4" and self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
        self._writer = None
