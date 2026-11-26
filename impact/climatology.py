"""Smoothed day-of-year climatology, numpy only.

Same definition as the organisers' monsoon_r1.stats.climatology (tests check equality):
  * every date maps to a slot 0..365 of the leap year 2000, so Feb 29 has its own slot (59) and
    Mar 1 is always slot 60;
  * the mean per slot is taken over all years given;
  * then a circular running mean over `smooth_days` slots (15 -> offsets -7..+7, wrapping Dec 31 -> Jan 1).

train_climatology() uses the TRAIN years 2005-2018 only, so the 2019-2021 test windows never see their own
truth in the climatology they are compared against.
"""
from __future__ import annotations

import numpy as np

TRAIN_YEARS = (2005, 2018)
# Day-of-year offset of the first of each month in a leap year (2000).
_MONTH_START = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])


def to_datetime64(dates) -> np.ndarray:
    """datetime64 / ISO strings / yyyymmdd ints -> datetime64[D] array."""
    d = np.asarray(dates)
    if np.issubdtype(d.dtype, np.integer):
        d = d.astype(np.int64)
        s = [f"{x // 10000:04d}-{x // 100 % 100:02d}-{x % 100:02d}" for x in d.ravel()]
        return np.array(s, dtype="datetime64[D]").reshape(d.shape)
    return d.astype("datetime64[D]")


def doy_index(dates) -> np.ndarray:
    """Slot 0..365 per date (leap-year calendar: Feb 29 = 59, Mar 1 = 60 in every year)."""
    d = to_datetime64(dates)
    month = (d.astype("datetime64[M]") - d.astype("datetime64[Y]")).astype(int)        # 0..11
    day = (d - d.astype("datetime64[M]")).astype(int)                                    # 0..30
    return _MONTH_START[month] + day


def daily_climatology(data, dates, smooth_days: int = 15) -> np.ndarray:
    """[N, ...] daily fields + their dates -> [366, ...] float32 smoothed climatology.

    Slots with no data count as zero before smoothing, exactly like the kit (never happens with full years).
    """
    data = np.asarray(data, dtype=np.float64)
    idx = doy_index(dates)
    if idx.shape != (data.shape[0],):
        raise ValueError(f"need one date per time step: {idx.shape} dates vs {data.shape[0]} steps")
    total = np.zeros((366,) + data.shape[1:], dtype=np.float64)
    np.add.at(total, idx, data)
    cnt = np.bincount(idx, minlength=366).astype(np.float64)
    clim = total / np.maximum(cnt, 1.0).reshape((366,) + (1,) * (data.ndim - 1))
    h = int(smooth_days) // 2
    out = np.zeros_like(clim)
    for o in range(-h, h + 1):
        out += np.roll(clim, -o, axis=0)          # np.roll(c, -o)[k] == c[(k + o) % 366]
    return (out / (2 * h + 1)).astype(np.float32)


def train_mask(dates, years: tuple[int, int] = TRAIN_YEARS) -> np.ndarray:
    y = to_datetime64(dates).astype("datetime64[Y]").astype(int) + 1970
    return (y >= years[0]) & (y <= years[1])


def train_climatology(data, dates, channels=(0, 1), years: tuple[int, int] = TRAIN_YEARS,
                      smooth_days: int = 15) -> np.ndarray:
    """[T, C, nlat, nlon] archive -> [366, len(channels), nlat, nlon] climatology of the chosen channels,
    from the train years only. Default channels (0, 1) = (t2m, tp) in common.CHANNELS order."""
    m = train_mask(dates, years)
    if not m.any():
        raise ValueError(f"no dates inside the train years {years}")
    sub = np.asarray(data)[m][:, list(channels)]
    return daily_climatology(sub, to_datetime64(dates)[m], smooth_days)


def forecast_dates(last_day, steps: int) -> np.ndarray:
    """The dates a rollout covers: last input day + 1 .. last input day + steps."""
    d0 = np.datetime64(str(last_day), "D")
    return d0 + np.arange(1, int(steps) + 1)


def clim_for_dates(clim: np.ndarray, dates) -> np.ndarray:
    """[366, ...] climatology -> [len(dates), ...] values for those calendar days."""
    return np.asarray(clim)[doy_index(dates)]
