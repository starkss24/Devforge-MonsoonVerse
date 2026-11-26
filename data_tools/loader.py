"""Load the training archive, check it against stats.json, split by year and build 3-day windows.

Reusable by the model owner for training. Needs numpy + xarray (netCDF4 backend); torch is optional.

    from data_tools.loader import load_archive, load_stats, split_years, WindowDataset, normalize
    arc = load_archive("data/india_climate_train.nc")
    tr, va = split_years(arc.dates)
    ds = WindowDataset(normalize(arc.data, load_stats("data/stats.json")), arc.dates, tr, k=4)
    init, doy_sc, target = ds[0]      # [3,6,32,36], [2], [4,6,32,36]

CLI:  .venv/Scripts/python -m data_tools.loader --check
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import CHANNELS, LATS, LONS, N_CH, N_IN_DAYS, NLAT, NLON, UNITS, doy_sincos  # noqa: E402

DEFAULT_NC = ROOT / "data" / "india_climate_train.nc"
DEFAULT_STATS = ROOT / "data" / "stats.json"
TRAIN_YEARS = (2005, 2018)
VAL_YEARS = (2019, 2021)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@dataclass
class Archive:
    data: np.ndarray          # [T, 6, 32, 36] float32, physical units, CHANNELS order, lat south->north
    dates: np.ndarray         # [T] datetime64[D]
    static: np.ndarray        # [2, 32, 36] float32: land_sea_mask (0..1 fraction), orography_m
    static_names: tuple
    channels: tuple
    lat: np.ndarray           # [32]
    lon: np.ndarray           # [36]


def load_archive(path=DEFAULT_NC) -> Archive:
    """Read india_climate_train.nc into numpy. Does not reorder or flip anything silently:
    run check_against_stats / check_grid to verify the layout."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        da = ds["data"].transpose("time", "channel", "lat", "lon")
        data = np.asarray(da.values, dtype=np.float32)
        channels = tuple(str(c) for c in ds["channel"].values)
        dates = np.asarray(ds["time"].values).astype("datetime64[D]")
        if "static" in ds:
            st = ds["static"].transpose("static_channel", "lat", "lon")
            static = np.asarray(st.values, dtype=np.float32)
            static_names = tuple(str(c) for c in ds["static_channel"].values)
        else:
            static, static_names = np.zeros((0, NLAT, NLON), np.float32), ()
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
    return Archive(data, dates, static, static_names, channels, lat, lon)


def load_stats(path=DEFAULT_STATS) -> dict:
    """stats.json -> {"channels": [...], "mean": np.array[6], "std": np.array[6]}."""
    s = json.loads(Path(path).read_text())
    return {"channels": list(s["channels"]), "mean": np.asarray(s["mean"], np.float64),
            "std": np.asarray(s["std"], np.float64)}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_grid(lat: np.ndarray, lon: np.ndarray, channels: Sequence[str]) -> list[str]:
    """Problems with grid coords / channel order (empty list = OK). Exact match to common.LATS/LONS."""
    bad = []
    if lat.shape != LATS.shape or not np.array_equal(lat, LATS):
        bad.append(f"lat {lat[:1]}..{lat[-1:]} (n={lat.size}) != 6.5..37.5 ascending (n=32)")
    if lon.shape != LONS.shape or not np.array_equal(lon, LONS):
        bad.append(f"lon {lon[:1]}..{lon[-1:]} (n={lon.size}) != 65.5..100.5 ascending (n=36)")
    if tuple(channels) != CHANNELS:
        bad.append(f"channel order {tuple(channels)} != {CHANNELS}")
    return bad


def channel_summary(arr: np.ndarray) -> dict:
    """Per-channel min/max/mean/std over (time, lat, lon) for [T,6,32,36]; float64 accumulation."""
    a = np.asarray(arr)
    return {"min": a.min(axis=(0, 2, 3)).astype(np.float64), "max": a.max(axis=(0, 2, 3)).astype(np.float64),
            "mean": a.mean(axis=(0, 2, 3), dtype=np.float64), "std": a.std(axis=(0, 2, 3), dtype=np.float64)}


def check_against_stats(arr: np.ndarray, stats: dict, tol: float = 0.02, verbose: bool = True) -> list[str]:
    """Compare the archive to stats.json. Flags: NaN/Inf, channel-order mismatch, mean off by more than
    tol * std, std off by more than tol relative, negative rain, t2m outside -40..60 C.
    Returns the list of problems (empty = OK) and prints a table when verbose."""
    bad = []
    a = np.asarray(arr)
    if a.ndim != 4 or a.shape[1:] != (N_CH, NLAT, NLON):
        return [f"shape {a.shape} != [T, {N_CH}, {NLAT}, {NLON}]"]
    n_bad = int((~np.isfinite(a)).sum())
    if n_bad:
        bad.append(f"{n_bad} non-finite values")
    if list(stats["channels"]) != list(CHANNELS):
        bad.append(f"stats.json channel order {stats['channels']} != {list(CHANNELS)}")
    s = channel_summary(a)
    mean, std = np.asarray(stats["mean"], np.float64), np.asarray(stats["std"], np.float64)
    if verbose:
        print(f"{'ch':5s} {'units':7s} {'min':>11s} {'max':>11s} {'mean':>11s} {'std':>9s} "
              f"{'json mean':>11s} {'json std':>9s}  dmean/std  dstd/std")
    for c, name in enumerate(CHANNELS):
        dm = (s["mean"][c] - mean[c]) / std[c]
        ds_ = (s["std"][c] - std[c]) / std[c]
        flag = ""
        if abs(dm) > tol or abs(ds_) > tol:
            flag = "  <-- MISMATCH"
            bad.append(f"{name}: mean/std differ from stats.json (dmean {dm:+.3f} std, dstd {ds_:+.3f})")
        if verbose:
            print(f"{name:5s} {UNITS[name]:7s} {s['min'][c]:11.3f} {s['max'][c]:11.3f} {s['mean'][c]:11.3f} "
                  f"{s['std'][c]:9.3f} {mean[c]:11.3f} {std[c]:9.3f}  {dm:+9.4f} {ds_:+9.4f}{flag}")
    tp, t2m = CHANNELS.index("tp"), CHANNELS.index("t2m")
    if s["min"][tp] < 0:
        bad.append(f"negative rain (min {s['min'][tp]:.4f})")
    if s["min"][t2m] < 233.15 or s["max"][t2m] > 333.15:
        bad.append("t2m outside -40..60 C")
    return bad


# ---------------------------------------------------------------------------
# Dates, split
# ---------------------------------------------------------------------------
def year_of(dates: np.ndarray) -> np.ndarray:
    return np.asarray(dates).astype("datetime64[D]").astype("datetime64[Y]").astype(np.int64) + 1970


def day_of_year(dates: np.ndarray) -> np.ndarray:
    """1-based day of year (Jan 1 = 1), the convention eval/score.py and the judge feed to doy_sincos."""
    t = np.asarray(dates).astype("datetime64[D]")
    return (t - t.astype("datetime64[Y]")).astype(np.int64) + 1


def split_years(dates: np.ndarray, train=TRAIN_YEARS, val=VAL_YEARS) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (train, val) over dates, by calendar year (inclusive ranges). Disjoint by construction
    as long as the ranges do not overlap (asserted)."""
    if not (train[1] < val[0] or val[1] < train[0]):
        raise ValueError(f"train years {train} and val years {val} overlap")
    y = year_of(dates)
    return (y >= train[0]) & (y <= train[1]), (y >= val[0]) & (y <= val[1])


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalize(x: np.ndarray, stats: dict) -> np.ndarray:
    """(x - mean) / std per channel; channel axis is the one of size 6 at position -3 ([...,6,32,36]).

    Note: tp is heavily skewed (many zeros, long tail). Models usually train rain in log1p space
    (log1p(tp) then standardise with its own mean/std) - that is the model owner's choice and is NOT done
    here; this function is the plain stats.json z-score the judge's std also uses."""
    m = np.asarray(stats["mean"], np.float64)[:, None, None]
    s = np.asarray(stats["std"], np.float64)[:, None, None]
    return ((np.asarray(x, np.float64) - m) / s).astype(np.float32)


def denormalize(z: np.ndarray, stats: dict) -> np.ndarray:
    """Inverse of normalize."""
    m = np.asarray(stats["mean"], np.float64)[:, None, None]
    s = np.asarray(stats["std"], np.float64)[:, None, None]
    return (np.asarray(z, np.float64) * s + m).astype(np.float32)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
def window_starts(dates: np.ndarray, mask: np.ndarray, n_in: int = N_IN_DAYS, k: int = 1) -> np.ndarray:
    """Indices t of the LAST input day such that days t-n_in+1 .. t+k are consecutive calendar days and all
    inside mask. So a window never crosses the train/val boundary (or a gap in the file)."""
    days = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    mask = np.asarray(mask, bool)
    span = n_in + k
    T = days.size
    if T < span:
        return np.zeros(0, np.int64)
    # vectorised: all days in [t-n_in+1, t+k] inside mask, and calendar-contiguous
    inside = np.lib.stride_tricks.sliding_window_view(mask, span).all(axis=1)
    contiguous = (days[span - 1:] - days[:T - span + 1]) == span - 1
    a = np.nonzero(inside & contiguous)[0]           # a = first input day
    return (a + n_in - 1).astype(np.int64)


class WindowDataset:
    """Numpy window dataset. item i -> (init [n_in,6,32,36], doy_sincos [2] of the LAST input day,
    target [k,6,32,36] = the k days right after the last input day, target[j] = day t+1+j).

    data can be physical or normalised (it is indexed as given). mask selects the allowed days (e.g. the
    train mask from split_years); a window is kept only if every input and target day is in the mask."""

    def __init__(self, data: np.ndarray, dates: np.ndarray, mask: np.ndarray, k: int = 1, n_in: int = N_IN_DAYS):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.data, self.dates, self.k, self.n_in = data, np.asarray(dates).astype("datetime64[D]"), k, n_in
        self.index = window_starts(self.dates, mask, n_in, k)
        self.doy = day_of_year(self.dates)

    def __len__(self) -> int:
        return int(self.index.size)

    def last_input_date(self, i: int):
        return self.dates[self.index[i]]

    def __getitem__(self, i: int):
        t = int(self.index[i])
        init = np.asarray(self.data[t - self.n_in + 1: t + 1], dtype=np.float32)
        target = np.asarray(self.data[t + 1: t + 1 + self.k], dtype=np.float32)
        return init, doy_sincos(self.doy[t]).astype(np.float32), target


try:  # optional torch wrapper; the numpy dataset works without torch
    import torch
    from torch.utils.data import Dataset as _TorchDataset

    class TorchWindowDataset(_TorchDataset):
        """torch Dataset over a WindowDataset; returns float32 tensors (init, doy_sincos, target)."""

        def __init__(self, windows: WindowDataset):
            self.w = windows

        def __len__(self):
            return len(self.w)

        def __getitem__(self, i):
            init, sc, tgt = self.w[i]
            return torch.from_numpy(init), torch.from_numpy(sc), torch.from_numpy(tgt)
except ImportError:  # torch not installed: only the numpy dataset is available
    TorchWindowDataset = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_checks(nc=DEFAULT_NC, stats_path=DEFAULT_STATS) -> int:
    arc = load_archive(nc)
    stats = load_stats(stats_path)
    print(f"file      {nc}")
    print(f"data      {arc.data.shape} {arc.data.dtype}  (time, channel, lat, lon)")
    print(f"dates     {arc.dates[0]} .. {arc.dates[-1]}  ({arc.dates.size} days)")
    gaps = int((np.diff(arc.dates.astype(np.int64)) != 1).sum())
    print(f"gaps      {gaps} non-consecutive steps")
    print(f"channels  {arc.channels}")
    print(f"lat       {arc.lat[0]}..{arc.lat[-1]} step {np.diff(arc.lat).mean():.2f} (south->north)")
    print(f"lon       {arc.lon[0]}..{arc.lon[-1]} step {np.diff(arc.lon).mean():.2f}")
    for name, f in zip(arc.static_names, arc.static):
        print(f"static    {name:14s} min {f.min():9.3f} max {f.max():9.3f}")
    bad = check_grid(arc.lat, arc.lon, arc.channels) + (["date gaps"] if gaps else [])
    print("\n-- full archive vs stats.json (tolerance 2% of std) --")
    bad += check_against_stats(arc.data, stats)
    tr, va = split_years(arc.dates)
    s_tr = channel_summary(arc.data[tr])
    print("\n-- train years only (2005-2018), for reference: stats.json was computed on the full 2005-2021 archive --")
    for c, name in enumerate(CHANNELS):
        print(f"{name:5s} mean {s_tr['mean'][c]:11.3f} std {s_tr['std'][c]:9.3f}")
    for k in (1, 10):
        print(f"windows k={k:2d}: train {window_starts(arc.dates, tr, k=k).size}, "
              f"val {window_starts(arc.dates, va, k=k).size}")
    print(f"split     train {int(tr.sum())} days, val {int(va.sum())} days, other {int((~tr & ~va).sum())}")
    print("\nRESULT: " + ("OK" if not bad else "PROBLEMS:\n  " + "\n  ".join(bad)))
    return 0 if not bad else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true", help="print the dataset checks")
    p.add_argument("--data", default=str(DEFAULT_NC))
    p.add_argument("--stats", default=str(DEFAULT_STATS))
    a = p.parse_args(argv)
    if not a.check:
        p.print_help()
        return 0
    return run_checks(a.data, a.stats)


if __name__ == "__main__":
    raise SystemExit(main())
