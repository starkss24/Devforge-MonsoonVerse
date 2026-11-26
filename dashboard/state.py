"""Shared data passed from app.py to each tab. No UI imports, no server route name."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Controls:
    """What the sidebar (dashboard/sidebar.py) returns on every rerun."""
    server_url: str
    last_day: str                       # last input day, YYYY-MM-DD
    channel: str                        # one of common.CHANNELS (display channel for maps)
    day: int                            # 1..10 lead day for tabs that need a forecast day (Maps reads its own timeline)
    shock_label: str                    # "none" | "warm_ocean" | "monsoon_stall" | "custom: ..."
    perturbation: Optional[np.ndarray]  # [6,32,36] physical deltas, None for no shock
    run_clicked: bool
    show_wind: bool = True
    steps: int = 10                     # forecast length asked from the server (the judges score 10)


@dataclass
class RunResult:
    """Produced by app.py when Run is clicked: two live server calls (baseline + shocked)."""
    last_day: str
    window: dict                        # dashboard.windows.get_window(...) output
    shock_label: str
    perturbation: Optional[np.ndarray]
    base: np.ndarray                    # [10,6,32,36]
    shocked: np.ndarray                 # [10,6,32,36] (== base if no shock)
    timings_ms: dict = field(default_factory=dict)   # {"base_rt":..,"shock_rt":..,"base_server":..}


@dataclass
class TabContext:
    controls: Controls
    run: Optional[RunResult]            # None until the first Run
    archive: object                     # dashboard.windows.Archive
    health: Optional[dict]              # last /health payload or None if the server is down
