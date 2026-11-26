"""Crop-stress index per region from a baseline and a shocked rollout. numpy only, pure and deterministic.

For each region r (a lat/lon box, optionally land cells only) and each scenario (baseline, shocked):

  weights      w_ij = cos(lat_i) on the region's cells (land_sea_mask > 0.5 if a mask is given), 0 elsewhere
  region mean  X_r(t) = sum_ij w_ij X_ij(t) / sum_ij w_ij
  rain         P = sum_t tp_r(t)                         forecast rain total over the horizon (mm)
               C = sum_t clim_tp_r(t)                    climatological total for the same dates (mm)
  deficit      D = 100 (C - P) / C   (%; > 0 = drier than normal). If C < min_clim_mm the region is in its
               dry season: D is set to 0 and clim_dry=True (a percentage of almost nothing means nothing).
  hot days     H = #{t : t2m_r(t) - clim_t2m_r(t) > hot_anom_k}   (strictly greater)
               t2m is a DAILY MEAN, so this counts warm daily-mean anomalies, not daily-maximum heatwaves.
  index        S = 100 [ w_rain * clip(D, 0, 100) / 100 + w_heat * H / steps ]  in [0, 100]
               (wetter than normal adds no rain stress; default w_rain = 0.6, w_heat = 0.4)
  change       shocked minus baseline for P, D, H, S and the mean t2m anomaly.
"""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

T2M, TP = 0, 1                      # channel indices (common.CHANNELS order)
DEFAULT_WEIGHTS = (0.6, 0.4)        # (rain deficit, hot days); must be >= 0 and sum to 1
MIN_CLIM_MM = 1.0                   # below this climatological horizon total, the deficit % is not meaningful


def _box(spec) -> tuple:
    b = spec["box"] if isinstance(spec, Mapping) else spec
    b = tuple(float(x) for x in b)
    if len(b) != 4 or b[0] > b[1] or b[2] > b[3]:
        raise ValueError(f"box must be (lat_min, lat_max, lon_min, lon_max), got {b}")
    return b


def region_weights(regions: Mapping, lats, lons, land_mask=None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Names, normalised cos-lat weights [R, nlat, nlon] (each sums to 1) and cell counts [R].

    Raises ValueError for a region with no (land) cell, instead of returning NaN later.
    """
    lats, lons = np.asarray(lats, dtype=np.float64), np.asarray(lons, dtype=np.float64)
    land = None
    if land_mask is not None:
        land = np.asarray(land_mask, dtype=np.float64)
        if land.shape != (lats.size, lons.size):
            raise ValueError(f"land_mask must be {(lats.size, lons.size)}, got {land.shape}")
        land = land > 0.5
    names = list(regions)
    if not names:
        raise ValueError("regions is empty")
    coslat = np.cos(np.deg2rad(lats))[:, None] * np.ones((1, lons.size))
    W = np.zeros((len(names), lats.size, lons.size))
    n = np.zeros(len(names), dtype=int)
    for k, name in enumerate(names):
        la0, la1, lo0, lo1 = _box(regions[name])
        m = ((lats >= la0) & (lats <= la1))[:, None] & ((lons >= lo0) & (lons <= lo1))[None, :]
        if land is not None:
            m &= land
        n[k] = int(m.sum())
        if n[k] == 0:
            raise ValueError(f"region {name!r} has no {'land ' if land is not None else ''}cell on this grid")
        w = np.where(m, coslat, 0.0)
        W[k] = w / w.sum()
    return names, W, n


def _scenario(roll, clim_tp_r, clim_t2m_r, W, hot_anom_k, w_rain, w_heat, min_clim_mm) -> dict:
    steps = roll.shape[0]
    tp_r = np.einsum("tij,rij->rt", roll[:, TP], W)            # [R, steps] region-mean rain
    t2m_r = np.einsum("tij,rij->rt", roll[:, T2M], W)          # [R, steps] region-mean t2m
    P = tp_r.sum(axis=1)
    C = clim_tp_r.sum(axis=1)
    dry = C < min_clim_mm
    D = np.where(dry, 0.0, 100.0 * (C - P) / np.where(dry, 1.0, C))
    anom = t2m_r - clim_t2m_r
    H = (anom > hot_anom_k).sum(axis=1)
    S = 100.0 * (w_rain * np.clip(D, 0.0, 100.0) / 100.0 + w_heat * H / steps)
    return {"rain_mm": P, "clim_rain_mm": C, "clim_dry": dry, "deficit_pct": D, "hot_days": H,
            "t2m_anom_k": anom.mean(axis=1), "index": S}


def crop_stress(base, shocked, clim_tp, clim_t2m, regions: Mapping, land_mask=None, hot_anom_k: float = 2.0,
                weights: tuple[float, float] = DEFAULT_WEIGHTS, lats=None, lons=None,
                min_clim_mm: float = MIN_CLIM_MM) -> dict[str, dict]:
    """Per-region rain deficit, hot-day count and crop-stress index for baseline and shocked rollouts.

    base, shocked      [steps, 6, nlat, nlon] physical units (t2m K in channel 0, tp mm/day in channel 1)
    clim_tp, clim_t2m  [steps, nlat, nlon] climatology for the SAME forecast dates (impact.climatology)
    regions            name -> (lat_min, lat_max, lon_min, lon_max) or {"box": (...), ...}
    land_mask          optional [nlat, nlon] land_sea_mask; only cells > 0.5 are used
    hot_anom_k         a day is hot when the region-mean DAILY-MEAN t2m anomaly is strictly above this (K)
    weights            (w_rain, w_heat) of the index, >= 0 and summing to 1
    lats, lons         grid centres; default common.LATS / common.LONS (32 x 36 grid)

    Returns {name: {"n_cells", "box", "baseline": {...}, "shocked": {...}, "change": {...}}} where each
    scenario holds rain_mm, clim_rain_mm, clim_dry, deficit_pct, hot_days, t2m_anom_k, index (python floats/
    ints) and change holds shocked - baseline for rain_mm, deficit_pct, hot_days, t2m_anom_k, index.
    """
    base = np.asarray(base, dtype=np.float64)
    shocked = np.asarray(shocked, dtype=np.float64)
    if base.ndim != 4 or base.shape[1] < 2:
        raise ValueError(f"base must be [steps, channels>=2, nlat, nlon], got {base.shape}")
    if shocked.shape != base.shape:
        raise ValueError(f"shocked {shocked.shape} must match base {base.shape}")
    steps, _, nlat, nlon = base.shape
    if steps < 1:
        raise ValueError("need at least one forecast step")
    clim_tp = np.asarray(clim_tp, dtype=np.float64)
    clim_t2m = np.asarray(clim_t2m, dtype=np.float64)
    for nm, c in (("clim_tp", clim_tp), ("clim_t2m", clim_t2m)):
        if c.shape != (steps, nlat, nlon):
            raise ValueError(f"{nm} must be {(steps, nlat, nlon)}, got {c.shape}")
    for nm, a in (("base", base[:, :2]), ("shocked", shocked[:, :2]), ("clim_tp", clim_tp), ("clim_t2m", clim_t2m)):
        if not np.isfinite(a).all():
            raise ValueError(f"{nm} contains NaN or Inf")
    w_rain, w_heat = (float(x) for x in weights)
    if w_rain < 0 or w_heat < 0 or abs(w_rain + w_heat - 1.0) > 1e-9:
        raise ValueError(f"weights must be >= 0 and sum to 1, got {weights}")
    if lats is None or lons is None:
        from common import LATS, LONS  # numpy-only module; default 1 degree India grid
        lats = LATS if lats is None else lats
        lons = LONS if lons is None else lons
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    if (lats.size, lons.size) != (nlat, nlon):
        raise ValueError(f"lats/lons {(lats.size, lons.size)} do not match the grid {(nlat, nlon)}")

    names, W, n = region_weights(regions, lats, lons, land_mask)
    ctp = np.einsum("tij,rij->rt", clim_tp, W)
    ct2 = np.einsum("tij,rij->rt", clim_t2m, W)
    args = (ctp, ct2, W, float(hot_anom_k), w_rain, w_heat, float(min_clim_mm))
    sb = _scenario(base, *args)
    ss = _scenario(shocked, *args)

    out: dict[str, dict] = {}
    for k, name in enumerate(names):
        def pick(s):
            return {"rain_mm": float(s["rain_mm"][k]), "clim_rain_mm": float(s["clim_rain_mm"][k]),
                    "clim_dry": bool(s["clim_dry"][k]), "deficit_pct": float(s["deficit_pct"][k]),
                    "hot_days": int(s["hot_days"][k]), "t2m_anom_k": float(s["t2m_anom_k"][k]),
                    "index": float(s["index"][k])}
        b, s = pick(sb), pick(ss)
        out[name] = {
            "n_cells": int(n[k]), "box": _box(regions[name]), "steps": int(steps), "baseline": b, "shocked": s,
            "change": {key: s[key] - b[key] for key in ("rain_mm", "deficit_pct", "hot_days", "t2m_anom_k", "index")},
        }
    return out
