"""Real starting windows from the organisers' training file (data/india_climate_train.nc, not committed).

Pure numpy/xarray, no UI imports. The app caches load_archive() once per session.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import CHANNELS, doy_sincos  # noqa: E402

DATA_PATH = ROOT / "data" / "india_climate_train.nc"
STATS_PATH = ROOT / "data" / "stats.json"
DOWNLOAD_HINT = ("Download india_climate_train.nc and stats.json from "
                 "https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/releases/tag/v1.0.0 into data/")


class DataMissing(FileNotFoundError):
    pass


@dataclass
class Archive:
    data: np.ndarray        # [T, 6, 32, 36] float32, physical units
    dates: np.ndarray       # [T] datetime64[D]
    static: np.ndarray      # [2, 32, 36] land_sea_mask, orography_m
    lats: np.ndarray        # [32] ascending (south -> north)
    lons: np.ndarray        # [36]
    std: np.ndarray         # [6] per-channel std from stats.json (judge's scale)


def load_archive(path: Path | str = DATA_PATH, stats_path: Path | str = STATS_PATH) -> Archive:
    path, stats_path = Path(path), Path(stats_path)
    if not path.is_file():
        raise DataMissing(f"{path} not found. {DOWNLOAD_HINT}")
    import xarray as xr
    with xr.open_dataset(path) as ds:
        chans = [str(c) for c in ds["channel"].values]
        if tuple(chans) != CHANNELS:
            raise ValueError(f"channel order {chans} != {CHANNELS}")
        data = ds["data"].values.astype(np.float32)
        dates = ds["time"].values.astype("datetime64[D]")
        static = ds["static"].values.astype(np.float32)
        lats, lons = ds["lat"].values.astype(float), ds["lon"].values.astype(float)
    if stats_path.is_file():
        std = np.asarray(json.loads(stats_path.read_text(encoding="utf-8"))["std"], dtype=np.float64)
    else:
        std = data.std(axis=(0, 2, 3)).astype(np.float64)
    return Archive(data, dates, static, lats, lons, std)


def valid_last_days(arc: Archive) -> np.ndarray:
    """Dates that can be the LAST input day (need 2 earlier days)."""
    return arc.dates[2:]


def get_window(arc: Archive, last_day) -> dict:
    """init_states [3,6,32,36] ending on last_day, its doy_sincos, and up to 10 days of truth after it."""
    d = np.datetime64(str(last_day), "D")
    idx = int(np.searchsorted(arc.dates, d))
    if idx >= len(arc.dates) or arc.dates[idx] != d or idx < 2:
        raise ValueError(f"{d} is not a valid last input day")
    init = arc.data[idx - 2: idx + 1].astype(np.float64)
    doy = int((d - d.astype("datetime64[Y]")).astype(int)) + 1
    truth = arc.data[idx + 1: idx + 11].astype(np.float64)   # may be shorter than 10 at the archive end
    return {"init_states": init, "doy": doy, "doy_sincos": doy_sincos(doy), "truth": truth, "last_day": str(d)}
