"""Forecast-skill scorer: decides which checkpoint ships.

Pure metric functions (numpy only, importable from training code):
    rmse_per_channel, persistence_forecast, daily_climatology, climatology_forecast,
    heavy_rain_hit_rate, sharpness_ratio, validity_counts

CLI (run from the repo root):
    python -m eval.score --data india_climate_train.nc --stats stats.json --twin
    python -m eval.score --data ... --stats ... --server http://localhost:8000
    python -m eval.score --synthetic --twin            # no kit files needed

For every validation-year 3-day window the twin is rolled out 10 steps; per-channel
RMSE at the requested leads is compared with persistence and with a smoothed (+-7 day)
day-of-year climatology built from TRAIN years only. ratio < 1 means the model beats
that baseline.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    CH, CHANNELS, LATS, LONS, N_CH, N_IN_DAYS, NLAT, NLON, T2M_MAX_K, T2M_MIN_K, TP_MIN,
    UNITS, doy_sincos,
)

# The only hardcoded dates: the year split.
TRAIN_YEARS = (2005, 2018)
VAL_YEARS = (2019, 2021)
DEFAULT_STEPS = 10
CLIM_HALF_WINDOW = 7
HEAVY_RAIN_PCTL = 95.0

# Fallback names if the kit's .nc does not use the CHANNELS names directly.
VAR_ALIASES = {
    "t2m": ("2t", "t2", "2m_temperature", "temperature_2m", "tas"),
    "tp": ("total_precipitation", "precip", "precipitation", "pr", "rain"),
    "u10": ("10u", "u", "10m_u_component_of_wind", "uas"),
    "v10": ("10v", "v", "10m_v_component_of_wind", "vas"),
    "msl": ("mslp", "mean_sea_level_pressure", "psl", "slp"),
    "tcwv": ("total_column_water_vapour", "pwat", "prw", "tcw"),
}
TIME_NAMES = ("time", "valid_time", "date", "day")
LAT_NAMES = ("latitude", "lat", "y")
LON_NAMES = ("longitude", "lon", "x")


# ---------------------------------------------------------------------------
# Pure metric functions
# ---------------------------------------------------------------------------
def rmse_per_channel(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """RMSE per channel for arrays shaped [..., C, H, W] -> [C].
    Mean is taken over every axis except the channel axis (-3), unweighted."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs truth {t.shape}")
    if p.ndim < 3:
        raise ValueError(f"need [..., C, H, W], got {p.shape}")
    err = np.moveaxis((p - t) ** 2, -3, 0).reshape(p.shape[-3], -1)
    return np.sqrt(err.mean(axis=1))


def persistence_forecast(init_states: np.ndarray, steps: int = DEFAULT_STEPS) -> np.ndarray:
    """Last input day repeated: [n_in, C, H, W] -> [steps, C, H, W]
    (or batched [N, n_in, C, H, W] -> [N, steps, C, H, W])."""
    x = np.asarray(init_states)
    last = x[..., -1:, :, :, :]
    return np.repeat(last, int(steps), axis=-4)


def daily_climatology(data: np.ndarray, doys: np.ndarray,
                      half_window: int = CLIM_HALF_WINDOW) -> np.ndarray:
    """Smoothed day-of-year climatology [366, C, H, W] from daily data [T, C, H, W].

    Sums and counts are binned per day-of-year (1..366) and smoothed with a circular
    +-half_window box, so the sparse leap day borrows its neighbours. Pass TRAIN days only."""
    data = np.asarray(data, dtype=np.float64)
    doys = np.asarray(doys, dtype=np.int64)
    if data.shape[0] != doys.shape[0]:
        raise ValueError("data and doys must have the same length")
    if doys.min() < 1 or doys.max() > 366:
        raise ValueError("doys must be in 1..366")
    sums = np.zeros((366,) + data.shape[1:], dtype=np.float64)
    counts = np.zeros(366, dtype=np.float64)
    np.add.at(sums, doys - 1, data)
    np.add.at(counts, doys - 1, 1.0)
    ssum = np.zeros_like(sums)
    scnt = np.zeros_like(counts)
    for k in range(-half_window, half_window + 1):
        ssum += np.roll(sums, k, axis=0)
        scnt += np.roll(counts, k)
    if np.any(scnt == 0):
        raise ValueError("climatology has empty day-of-year windows; need more training days")
    return ssum / scnt.reshape((-1,) + (1,) * (data.ndim - 1))


def climatology_forecast(clim: np.ndarray, target_doys: Sequence[int]) -> np.ndarray:
    """Climatology for each target day-of-year: -> [len(target_doys), C, H, W]."""
    idx = np.asarray(target_doys, dtype=np.int64) - 1
    return np.asarray(clim)[idx]


def heavy_rain_hit_rate(pred_tp: np.ndarray, truth_tp: np.ndarray, threshold: float) -> float:
    """Fraction of observed heavy-rain cells (truth >= threshold) that the forecast also
    puts at or above threshold. NaN if there are no observed events."""
    p = np.asarray(pred_tp, dtype=np.float64)
    t = np.asarray(truth_tp, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs truth {t.shape}")
    events = t >= threshold
    n = int(events.sum())
    if n == 0:
        return float("nan")
    return float((p[events] >= threshold).sum() / n)


def sharpness_ratio(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Spatial std of forecast / spatial std of truth, per channel, averaged over samples.
    Inputs [..., C, H, W] -> [C]. ~1 is good; << 1 means the forecast is blurry.
    Samples whose truth field is spatially constant are skipped."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs truth {t.shape}")
    if p.ndim == 3:
        p, t = p[None], t[None]
    sp = p.std(axis=(-2, -1)).reshape(-1, p.shape[-3])
    st = t.std(axis=(-2, -1)).reshape(-1, t.shape[-3])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(st > 0, sp / np.where(st > 0, st, 1.0), np.nan)
    out = np.full(r.shape[1], np.nan)
    for c in range(r.shape[1]):
        col = r[:, c][np.isfinite(r[:, c])]
        if col.size:
            out[c] = col.mean()
    return out


def validity_counts(rollout: np.ndarray) -> dict:
    """Counts of contract violations in a [..., 6, H, W] rollout.
    nan / inf over all channels; neg_tp = tp < 0; t2m_oob = finite t2m outside -40..60 C."""
    r = np.asarray(rollout, dtype=np.float64)
    if r.ndim < 3 or r.shape[-3] != N_CH:
        raise ValueError(f"need [..., {N_CH}, H, W], got {r.shape}")
    tp = r[..., CH["tp"], :, :]
    t2m = r[..., CH["t2m"], :, :]
    with np.errstate(invalid="ignore"):
        return {
            "nan": int(np.isnan(r).sum()),
            "inf": int(np.isinf(r).sum()),
            "neg_tp": int((tp < TP_MIN).sum()),
            "t2m_oob": int((np.isfinite(t2m) & ((t2m < T2M_MIN_K) | (t2m > T2M_MAX_K))).sum()),
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _pick(names: Sequence[str], available: Sequence[str]) -> Optional[str]:
    low = {a.lower(): a for a in available}
    for n in names:
        if n in available:
            return n
        if n.lower() in low:
            return low[n.lower()]
    return None


def dataset_to_arrays(ds, var_map: Optional[dict] = None):
    """xarray Dataset -> (data [T, 6, 32, 36] float32 in CHANNELS order, times datetime64[D]).
    Variables are matched by CHANNELS names (then VAR_ALIASES, then --var overrides)."""
    var_map = dict(var_map or {})
    available = list(ds.data_vars)
    chosen, missing = {}, []
    for ch in CHANNELS:
        name = var_map.get(ch) or _pick((ch,) + VAR_ALIASES.get(ch, ()), available)
        if name is None or name not in ds.data_vars:
            missing.append(ch)
        else:
            chosen[ch] = name
    if missing:
        raise KeyError(
            f"could not find variables for channels {missing}. Available data vars: {available}. "
            f"Pass e.g. --var {missing[0]}=<name_in_file>."
        )
    dims = list(ds.dims)
    tname = _pick(TIME_NAMES, dims)
    laname = _pick(LAT_NAMES, dims)
    loname = _pick(LON_NAMES, dims)
    if None in (tname, laname, loname):
        raise KeyError(f"need time/lat/lon dims; found dims {dims}")

    lat_vals = np.asarray(ds[laname].values, dtype=np.float64)
    lon_vals = np.asarray(ds[loname].values, dtype=np.float64)
    if lat_vals.size != NLAT or lon_vals.size != NLON:
        raise ValueError(f"grid is {lat_vals.size}x{lon_vals.size}, expected {NLAT}x{NLON}")
    flip_lat = lat_vals.size > 1 and (lat_vals[1] - lat_vals[0]) * (LATS[1] - LATS[0]) < 0
    if flip_lat:
        ds = ds.isel({laname: slice(None, None, -1)})
        lat_vals = lat_vals[::-1]
        print(f"[score] latitude order in file is opposite to common.LATS; flipped to match.")
    if not (np.allclose(lat_vals, LATS, atol=1e-3) and np.allclose(lon_vals, LONS, atol=1e-3)):
        print(f"[score] WARNING grid coords differ from common.LATS/LONS "
              f"(file lat {lat_vals[0]}..{lat_vals[-1]}, lon {lon_vals[0]}..{lon_vals[-1]})")

    arrs = []
    for ch in CHANNELS:
        da = ds[chosen[ch]]
        extra = [d for d in da.dims if d not in (tname, laname, loname)]
        for d in extra:  # squeeze singleton dims like 'number' or 'surface'
            if da.sizes[d] != 1:
                raise ValueError(f"variable {chosen[ch]!r} has non-singleton extra dim {d!r}")
            da = da.isel({d: 0})
        arrs.append(np.asarray(da.transpose(tname, laname, loname).values, dtype=np.float32))
    data = np.stack(arrs, axis=1)
    times = np.asarray(ds[tname].values).astype("datetime64[D]")
    order = np.argsort(times)
    if not np.all(order == np.arange(order.size)):
        data, times = data[order], times[order]
    renamed = {ch: n for ch, n in chosen.items() if n != ch}
    if renamed:
        print(f"[score] variable mapping (channel -> file var): {renamed}")
    return data, times


def load_stats_std(path: Optional[str]) -> Optional[np.ndarray]:
    """Per-channel std from stats.json in CHANNELS order, or None if not found.
    Accepts {"t2m": {"std": x}, ...}, {"std": {"t2m": x, ...}} or {"std": [6 values]}."""
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        s = json.load(f)
    try:
        if isinstance(s.get("std"), list):
            v = np.asarray(s["std"], dtype=np.float64)
            if v.size == N_CH:
                return v
        if isinstance(s.get("std"), dict):
            return np.array([float(s["std"][c]) for c in CHANNELS])
        if all(isinstance(s.get(c), dict) and "std" in s[c] for c in CHANNELS):
            return np.array([float(s[c]["std"]) for c in CHANNELS])
    except (KeyError, TypeError, ValueError) as e:
        print(f"[score] WARNING could not parse std from {path}: {e!r}")
        return None
    print(f"[score] WARNING no per-channel std found in {path} (top-level keys: {list(s)[:10]}); "
          f"using training-set std for normalised RMSE")
    return None


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def synthetic_arrays(year0: int = TRAIN_YEARS[0], year1: int = VAL_YEARS[1], seed: int = 0):
    """Smooth seasonal fake fields [T, 6, 32, 36] float32 in contract units, daily."""
    rng = np.random.default_rng(seed)
    times = np.arange(np.datetime64(f"{year0}-01-01"), np.datetime64(f"{year1 + 1}-01-01"),
                      dtype="datetime64[D]")
    T = times.size
    doy = (times - times.astype("datetime64[Y]")).astype(np.int64) + 1
    ang = 2 * np.pi * doy / 365.25
    monsoon = np.clip(np.cos(ang - 2 * np.pi * 200 / 365.25), 0, None) ** 2      # peaks ~mid-July
    summer = np.cos(ang - 2 * np.pi * 150 / 365.25)                               # peaks ~late May
    lat = (LATS - LATS.mean()) / np.ptp(LATS)        # -0.5..0.5 south->north
    lon = (LONS - LONS.mean()) / np.ptp(LONS)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    wet = np.exp(-((LO - 0.1) ** 2 + (LA + 0.1) ** 2) / 0.08) + 0.6 * np.exp(-((LO + 0.4) ** 2) / 0.02)

    # AR(1) noise on a coarse 8x9 grid, upsampled 4x -> smooth in space and time.
    noise = np.empty((T, N_CH, NLAT // 4, NLON // 4))
    z = rng.standard_normal((N_CH, NLAT // 4, NLON // 4))
    for i in range(T):
        z = 0.8 * z + 0.6 * rng.standard_normal(z.shape)
        noise[i] = z
    noise = np.repeat(np.repeat(noise, 4, axis=2), 4, axis=3)

    m = monsoon[:, None, None]
    s = summer[:, None, None]
    data = np.empty((T, N_CH, NLAT, NLON), dtype=np.float32)
    data[:, CH["t2m"]] = 299.0 - 14.0 * LA + (4.0 + 8.0 * (LA + 0.5)) * s - 3.0 * m + 1.5 * noise[:, 0]
    data[:, CH["tp"]] = np.maximum(0.0, (1.0 + 18.0 * m * wet) * np.exp(0.5 * noise[:, 1]) - 1.0)
    data[:, CH["u10"]] = -3.0 + 9.0 * m * (0.5 - LA) + 1.5 * noise[:, 2]
    data[:, CH["v10"]] = -1.5 + 5.0 * m + 1.5 * noise[:, 3]
    data[:, CH["msl"]] = 101200.0 - 700.0 * m - 300.0 * s * (LA + 0.5) + 150.0 * noise[:, 4]
    data[:, CH["tcwv"]] = np.maximum(1.0, 18.0 + 35.0 * m * (0.3 + wet) - 8.0 * LA + 3.0 * noise[:, 5])
    return data, times


def synthetic_dataset(seed: int = 0):
    """synthetic_arrays wrapped as an xarray Dataset named like the CHANNELS."""
    import xarray as xr

    data, times = synthetic_arrays(seed=seed)
    return xr.Dataset(
        {ch: (("time", "latitude", "longitude"), data[:, i], {"units": UNITS[ch]})
         for i, ch in enumerate(CHANNELS)},
        coords={"time": times.astype("datetime64[ns]"), "latitude": LATS, "longitude": LONS},
    )


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------
Forecaster = Callable[[np.ndarray, np.ndarray, int], np.ndarray]


def twin_forecaster() -> tuple[Forecaster, str]:
    from twin import Twin

    tw = Twin()
    return (lambda x, d, steps: tw.rollout(x, d, steps=steps)), f"twin:{tw.backend}"


def server_forecaster(url: str, timeout: float = 30.0) -> tuple[Forecaster, str]:
    base = url.rstrip("/")

    def post(path: str, body: dict) -> dict:
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    health = post("/health", {})
    if health.get("status") != "ready":
        raise RuntimeError(f"server not ready: {health}")

    def f(x, d, steps):
        out = post("/simulate", {"init_states": np.asarray(x).tolist(),
                                 "doy_sincos": np.asarray(d).tolist(),
                                 "steps": int(steps), "perturbation": None})
        return np.asarray(out["rollout"], dtype=np.float64)

    return f, f"server:{health.get('backend', '?')}"


def day_of_year(times: np.ndarray) -> np.ndarray:
    t = np.asarray(times).astype("datetime64[D]")
    return (t - t.astype("datetime64[Y]")).astype(np.int64) + 1


def year_of(times: np.ndarray) -> np.ndarray:
    return np.asarray(times).astype("datetime64[Y]").astype(np.int64) + 1970


def build_windows(times: np.ndarray, in_mask: np.ndarray, n_in: int = N_IN_DAYS,
                  steps: int = DEFAULT_STEPS, stride: int = 1) -> np.ndarray:
    """Indices t of the last input day such that days t-n_in+1 .. t+steps are consecutive
    calendar days and all inside in_mask."""
    days = np.asarray(times).astype("datetime64[D]").astype(np.int64)
    span = n_in + steps
    out = []
    for t in range(n_in - 1, days.size - steps):
        a, b = t - n_in + 1, t + steps + 1
        if in_mask[a:b].all() and days[b - 1] - days[a] == span - 1:
            out.append(t)
    return np.asarray(out[::max(1, int(stride))], dtype=np.int64)


def evaluate(data: np.ndarray, times: np.ndarray, forecaster: Forecaster,
             leads: Sequence[int] = (1, 4, 8), steps: int = DEFAULT_STEPS,
             train_years=TRAIN_YEARS, val_years=VAL_YEARS, stride: int = 1,
             std: Optional[np.ndarray] = None, max_windows: Optional[int] = None,
             verbose: bool = False) -> dict:
    """Roll out every validation window and score model vs persistence vs climatology."""
    leads = sorted({int(l) for l in leads})
    if not leads or leads[0] < 1 or leads[-1] > steps:
        raise ValueError(f"leads must be within 1..{steps}, got {leads}")
    years = year_of(times)
    doys = day_of_year(times)
    train = (years >= train_years[0]) & (years <= train_years[1])
    val = (years >= val_years[0]) & (years <= val_years[1])
    if not train.any():
        raise ValueError(f"no training days in {train_years}")

    clim = daily_climatology(data[train], doys[train])
    tp_threshold = float(np.percentile(data[train, CH["tp"]], HEAVY_RAIN_PCTL))
    if std is None:
        std = data[train].std(axis=(0, 2, 3)).astype(np.float64)

    win = build_windows(times, val, steps=steps, stride=stride)
    if max_windows:
        win = win[:max_windows]
    if win.size == 0:
        raise ValueError(f"no complete validation windows in {val_years}")

    L = np.asarray(leads) - 1
    shape = (win.size, len(leads), N_CH, NLAT, NLON)
    model = np.empty(shape, np.float32)
    truth = np.empty(shape, np.float32)
    pers = np.empty(shape, np.float32)
    climf = np.empty(shape, np.float32)
    valid = {"nan": 0, "inf": 0, "neg_tp": 0, "t2m_oob": 0}
    t0 = time.perf_counter()
    for i, t in enumerate(win):
        x = data[t - N_IN_DAYS + 1: t + 1]
        out = np.asarray(forecaster(x, doy_sincos(doys[t]), steps))
        if out.shape != (steps, N_CH, NLAT, NLON):
            raise ValueError(f"forecaster returned {out.shape}, expected {(steps, N_CH, NLAT, NLON)}")
        for k, v in validity_counts(out).items():
            valid[k] += v
        tgt = t + 1 + L
        model[i] = out[L]
        truth[i] = data[tgt]
        pers[i] = persistence_forecast(x, steps)[L]
        climf[i] = climatology_forecast(clim, doys[tgt])
        if verbose and (i + 1) % 200 == 0:
            print(f"[score] {i + 1}/{win.size} windows")
    secs = time.perf_counter() - t0

    rows = []
    for j, lead in enumerate(leads):
        r_m = rmse_per_channel(model[:, j], truth[:, j])
        r_p = rmse_per_channel(pers[:, j], truth[:, j])
        r_c = rmse_per_channel(climf[:, j], truth[:, j])
        sh_m = sharpness_ratio(model[:, j], truth[:, j])
        sh_c = sharpness_ratio(climf[:, j], truth[:, j])
        tp_t = truth[:, j, CH["tp"]]
        n_obs = int((tp_t >= tp_threshold).sum())
        for c, ch in enumerate(CHANNELS):
            best = min(r_p[c], r_c[c])
            row = {
                "lead": lead, "channel": ch, "units": UNITS[ch],
                "rmse_model": r_m[c], "rmse_persistence": r_p[c], "rmse_climatology": r_c[c],
                "ratio_vs_persistence": _div(r_m[c], r_p[c]),
                "ratio_vs_climatology": _div(r_m[c], r_c[c]),
                "ratio_vs_best": _div(r_m[c], best),
                "best_baseline": "persistence" if r_p[c] <= r_c[c] else "climatology",
                "nrmse_model": _div(r_m[c], std[c]),
                "sharpness_model": sh_m[c], "sharpness_climatology": sh_c[c],
                "hit_rate_model": np.nan, "hit_rate_persistence": np.nan,
                "hit_rate_climatology": np.nan, "rain_event_bias_model": np.nan,
                "n_windows": int(win.size),
            }
            if ch == "tp":
                row["hit_rate_model"] = heavy_rain_hit_rate(model[:, j, c], tp_t, tp_threshold)
                row["hit_rate_persistence"] = heavy_rain_hit_rate(pers[:, j, c], tp_t, tp_threshold)
                row["hit_rate_climatology"] = heavy_rain_hit_rate(climf[:, j, c], tp_t, tp_threshold)
                n_pred = int((model[:, j, c] >= tp_threshold).sum())
                row["rain_event_bias_model"] = _div(n_pred, n_obs)
            rows.append(row)

    return {"rows": rows, "leads": leads, "tp_threshold": tp_threshold, "validity": valid,
            "n_windows": int(win.size), "seconds": secs}


def _div(a, b) -> float:
    return float(a) / float(b) if b else float("nan")


def summarize(res: dict, backend: str) -> str:
    rows = res["rows"]
    per_lead = []
    for lead in res["leads"]:
        r = [x["ratio_vs_best"] for x in rows if x["lead"] == lead]
        per_lead.append((lead, float(np.nanmean(r))))
    overall = float(np.nanmean([v for _, v in per_lead]))
    tp1 = next(x for x in rows if x["channel"] == "tp" and x["lead"] == res["leads"][0])
    v = res["validity"]
    leads_s = " ".join(f"lead{l}={r:.3f}" for l, r in per_lead)
    return (f"SUMMARY backend={backend} windows={res['n_windows']} "
            f"mean_ratio_vs_best={overall:.3f} ({leads_s}) "
            f"tp_hit@p{HEAVY_RAIN_PCTL:g}(lead{res['leads'][0]})={tp1['hit_rate_model']:.3f} "
            f"invalid(nan={v['nan']} inf={v['inf']} neg_tp={v['neg_tp']} t2m_oob={v['t2m_oob']}) "
            f"[ratio<1 beats the better of persistence/climatology]")


def print_table(res: dict) -> None:
    print(f"{'lead':>4} {'chan':>5} {'model':>10} {'persist':>10} {'clim':>10} "
          f"{'r/pers':>7} {'r/clim':>7} {'r/best':>7} {'sharp':>6}")
    for x in res["rows"]:
        print(f"{x['lead']:>4} {x['channel']:>5} {x['rmse_model']:>10.4g} {x['rmse_persistence']:>10.4g} "
              f"{x['rmse_climatology']:>10.4g} {x['ratio_vs_persistence']:>7.3f} "
              f"{x['ratio_vs_climatology']:>7.3f} {x['ratio_vs_best']:>7.3f} {x['sharpness_model']:>6.2f}")
    for lead in res["leads"]:
        x = next(r for r in res["rows"] if r["lead"] == lead and r["channel"] == "tp")
        print(f"[score] lead {lead} heavy rain hit rate: model={x['hit_rate_model']:.3f} "
              f"persistence={x['hit_rate_persistence']:.3f} climatology={x['hit_rate_climatology']:.3f} "
              f"model event bias={x['rain_event_bias_model']:.2f}")


def write_csv(rows: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="kit .nc file (daily, 2005-2021)")
    ap.add_argument("--stats", help="kit stats.json (per-channel std used for normalised RMSE)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--server", help="score a running serve.py, e.g. http://localhost:8000")
    src.add_argument("--twin", action="store_true", help="score twin.Twin in-process (default)")
    ap.add_argument("--synthetic", action="store_true", help="use generated fake data instead of --data")
    ap.add_argument("--leads", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth validation window")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--var", action="append", default=[], metavar="CH=NAME",
                    help="override variable name mapping, e.g. --var tp=total_precipitation")
    ap.add_argument("--out", default="eval/scores.csv")
    a = ap.parse_args(argv)

    if a.synthetic:
        ds = synthetic_dataset()
        print("[score] using SYNTHETIC data (years "
              f"{TRAIN_YEARS[0]}-{VAL_YEARS[1]}); numbers are meaningless for model choice")
    else:
        if not a.data:
            ap.error("--data is required unless --synthetic")
        import xarray as xr
        ds = xr.open_dataset(a.data)
    var_map = {}
    for kv in a.var:
        k, _, v = kv.partition("=")
        if k not in CHANNELS or not v:
            ap.error(f"--var must be CH=NAME with CH in {CHANNELS}, got {kv!r}")
        var_map[k] = v
    data, times = dataset_to_arrays(ds, var_map)

    forecaster, backend = server_forecaster(a.server) if a.server else twin_forecaster()
    std = load_stats_std(a.stats)
    train = (year_of(times) >= TRAIN_YEARS[0]) & (year_of(times) <= TRAIN_YEARS[1])
    tp_mean = float(data[train, CH["tp"]].mean())
    if tp_mean < 0.05:
        print(f"[score] WARNING training tp mean is {tp_mean:.4g}; contract says mm/day "
              f"- file may be in metres")

    res = evaluate(data, times, forecaster, leads=a.leads, steps=a.steps, stride=a.stride,
                   std=std, max_windows=a.max_windows, verbose=True)
    print(f"[score] train years {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]} (climatology, threshold), "
          f"val years {VAL_YEARS[0]}-{VAL_YEARS[1]}, {res['n_windows']} windows, "
          f"{res['seconds']:.1f}s of rollouts")
    print(f"[score] heavy-rain threshold = training-set p{HEAVY_RAIN_PCTL:g} of tp over all cells "
          f"and days = {res['tp_threshold']:.3f} {UNITS['tp']}")
    print_table(res)
    write_csv(res["rows"], a.out)
    print(f"[score] wrote {a.out}")
    print(summarize(res, backend))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
