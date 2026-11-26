"""Shared constants for the Multiverse of Monsoon twin.

Single source of truth for channel order, grid, validity bounds and the
grader's shock boxes. Every module (twin, serve, eval, dashboard, impact)
imports from here instead of re-typing numbers.

Only numpy is imported so serve.py can use this module inside the sandbox.
"""
from __future__ import annotations

import numpy as np

# Channel order is fixed by R1_CONTRACT.md.
CHANNELS = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
UNITS = {"t2m": "K", "tp": "mm/day", "u10": "m/s", "v10": "m/s", "msl": "Pa", "tcwv": "kg/m2"}
CH = {name: i for i, name in enumerate(CHANNELS)}  # e.g. CH["tp"] == 1

N_IN_DAYS = 3
N_CH = 6
NLAT, NLON = 32, 36
STATE_SHAPE = (N_CH, NLAT, NLON)            # one day
INIT_SHAPE = (N_IN_DAYS, N_CH, NLAT, NLON)  # init_states

# 1 degree grid, cell centres. ASSUMPTION: ascending latitude (south -> north).
# Confirm against the kit's .nc file; if the data is north -> south, flip LATS here
# and nothing else needs to change because every box lookup goes through box_mask().
LATS = np.arange(6.5, 37.5 + 0.5, 1.0)    # 32 values
LONS = np.arange(65.5, 100.5 + 0.5, 1.0)  # 36 values
assert LATS.size == NLAT and LONS.size == NLON

# Validity rules from R1_CONTRACT.md.
T2M_MIN_K = -40.0 + 273.15
T2M_MAX_K = 60.0 + 273.15
TP_MIN = 0.0
NULL_SHOCK_TOL_FRAC_STD = 1e-3   # null shock must match unshocked within 1e-3 * channel std
MAX_SIMULATE_SECONDS = 5.0
TARGET_SIMULATE_SECONDS = 1.0    # our own target (10 steps, 4 threads)
N_THREADS = 4

# Grader shock boxes (lat_min, lat_max, lon_min, lon_max), inclusive on cell centres.
BOXES = {
    "warm_ocean": (8.0, 20.0, 82.0, 94.0),
    "monsoon_stall": (8.0, 20.0, 65.0, 75.0),
}


def box_mask(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> np.ndarray:
    """Boolean [NLAT, NLON] mask of cells whose centres fall inside the box."""
    la = (LATS >= lat_min) & (LATS <= lat_max)
    lo = (LONS >= lon_min) & (LONS <= lon_max)
    return la[:, None] & lo[None, :]


def make_shock(name: str, last_day: np.ndarray) -> np.ndarray:
    """Physical-delta perturbation [6, 32, 36] for a named grader shock.

    last_day is init_states[-1] in physical units (needed for the relative shocks).
    "zero" returns all zeros. Delta convention (CONFIRM AGAINST KIT):
      warm_ocean:    +3 K on t2m, +15% of last-day tcwv, inside BOXES["warm_ocean"]
      monsoon_stall: -40% of last-day u10 and v10,     inside BOXES["monsoon_stall"]
    """
    last_day = np.asarray(last_day, dtype=np.float64)
    if last_day.shape != STATE_SHAPE:
        raise ValueError(f"last_day must be {STATE_SHAPE}, got {last_day.shape}")
    delta = np.zeros(STATE_SHAPE, dtype=np.float64)
    if name == "zero":
        return delta
    if name == "warm_ocean":
        m = box_mask(*BOXES["warm_ocean"])
        delta[CH["t2m"]][m] = 3.0
        delta[CH["tcwv"]][m] = 0.15 * last_day[CH["tcwv"]][m]
        return delta
    if name == "monsoon_stall":
        m = box_mask(*BOXES["monsoon_stall"])
        delta[CH["u10"]][m] = -0.40 * last_day[CH["u10"]][m]
        delta[CH["v10"]][m] = -0.40 * last_day[CH["v10"]][m]
        return delta
    raise ValueError(f"unknown shock {name!r}; expected zero, warm_ocean or monsoon_stall")


def doy_sincos(doy: float) -> np.ndarray:
    """sin and cos of 2*pi*doy/365 for the last input day, as the contract defines."""
    a = 2.0 * np.pi * float(doy) / 365.0
    return np.array([np.sin(a), np.cos(a)], dtype=np.float64)


def enforce_validity(rollout: np.ndarray) -> np.ndarray:
    """Clamp a [steps, 6, 32, 36] rollout to contract bounds. Does NOT hide NaN/Inf:
    callers must check np.isfinite first and fall back if it fails."""
    out = np.array(rollout, dtype=np.float64, copy=True)
    out[:, CH["tp"]] = np.maximum(out[:, CH["tp"]], TP_MIN)
    out[:, CH["t2m"]] = np.clip(out[:, CH["t2m"]], T2M_MIN_K, T2M_MAX_K)
    return out
