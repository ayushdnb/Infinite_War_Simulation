# final_war_sim/recorder/schemas.py
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Optional

try:
    import pyarrow as pa  # type: ignore
except Exception:
    pa = None  # graceful fallback


# --------- Arrow schema (if available) ----------
def tick_arrow_schema():
    """
    Parquet/Arrow schema for per-tick agent rows.
    Columns are intentionally simple (analytics-friendly).
    """
    if pa is None:
        return None
    return pa.schema([
        pa.field("tick", pa.int64()),
        pa.field("agent_id", pa.int32()),
        pa.field("team_id", pa.int16()),     # 2 or 3
        pa.field("is_alive", pa.bool_()),
        pa.field("pos_x", pa.int16()),
        pa.field("pos_y", pa.int16()),
        pa.field("hp", pa.float32()),
        pa.field("atk", pa.float32()),
        pa.field("action", pa.int16()),
        pa.field("logits", pa.list_(pa.float32())),  # variable length
    ])


# --------- Lightweight dataclass (for clarity/tests) ----------
@dataclass(frozen=True)
class RunMeta:
    started_utc: str
    grid_h: int
    grid_w: int
    obs_dim: int
    num_actions: int
    commit: Optional[str] = None
    note: Optional[str] = None

    @staticmethod
    def new(grid_h: int, grid_w: int, obs_dim: int, num_actions: int,
            commit: Optional[str] = None, note: Optional[str] = None) -> "RunMeta":
        return RunMeta(
            started_utc=_dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            grid_h=int(grid_h), grid_w=int(grid_w),
            obs_dim=int(obs_dim), num_actions=int(num_actions),
            commit=commit, note=note,
        )
