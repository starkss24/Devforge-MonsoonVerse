"""Judge-aligned forecast scorer: decides which checkpoint ships.

Every metric is the organisers' own code (monsoon_r1.scoring / monsoon_r1.stats), never re-implemented:
  rmse_lat (cos-lat weighted RMSE), skill_vs_best_ref (SS vs the better of persistence / climatology),
  csi + csi_skill (heavy rain, tp > 10 mm/day), validity (NaN/Inf, rain < 0, t2m outside -40..60 C),
  variance_ratio + anti_blur_points (day-8 spatial variance), forecast_points (30 RMSE + 15 CSI),
  climatology (per-calendar-day mean, 15-day circular smoothing) + clim_for.

Leakage rule: climatology is built from TRAIN years 2005-2018 only; windows come from VAL years 2019-2021
(3 input days + 10 forecast days, all inside the val years).

How a "window" is scored (our reading of the judge; the grader's aggregation code is not in the kit):
  rollout[k] is day k+1 after the last input day; leads (1, 4, 8) -> rollout[[0, 3, 7]].
  window SS      = nanmean over the 6 channels of skill_vs_best_ref(pred[leads], truth, persist, clim)["ss"]
                   (rmse_lat pools the 3 leads, as its "[..., 6, 32, 36]" signature suggests)
  window CSI SS  = csi_skill(csi(pred tp over the 3 leads), csi(persist), csi(clim))  (NaN if no rain > 10 anywhere)
  day-8 ratio    = variance_ratio(rollout[7], truth day 8)  [6]
  An invalid / failed window scores SS 0 and CSI SS 0 (contract: "scores 0, never skipped") and, conservatively,
  variance ratio 0. Points: forecast_points(window SS, window CSI SS) -> rmse_pts/30, csi_pts/15;
  anti_blur_points(ratios) -> /10. The judge uses 5 windows; we score --windows N (default 40) for stability
  and also report the spread over random 5-window subsets ("judge5").

CLI (repo root):
    python -m eval.score --data data/india_climate_train.nc --twin
    python -m eval.score --data data/india_climate_train.nc --server http://localhost:8000
    python -m eval.score --synthetic --twin                  # no kit files needed
    options: --windows N (0 = every val window), --seed, --judge-windows 5, --out, --summary-out

Outputs
  eval/scores.csv, one row per (lead, channel), pooled over the scored windows:
    lead            forecast day (1, 4, 8)
    channel, units  t2m K | tp mm/day | u10 m/s | v10 m/s | msl Pa | tcwv kg/m2
    rmse_model      lat-weighted RMSE of the forecast (scoring.rmse_lat)
    rmse_persist    same for persistence (last input day repeated)
    rmse_clim       same for train-year climatology of the target calendar day
    best_ref        "persist" or "clim", whichever RMSE is lower
    ss              1 - rmse_model / min(rmse_persist, rmse_clim)   (> 0 beats the judge's reference)
    ss_persist      1 - rmse_model / rmse_persist
    ss_clim         1 - rmse_model / rmse_clim
    nrmse_model     rmse_model / channel std (stats.json if given, else train std)
    csi_model, csi_persist, csi_clim   CSI of tp > 10 mm/day at this lead (tp row only, else empty)
    csi_skill       scoring.csi_skill of the three (tp row only)
    var_ratio       mean over windows of scoring.variance_ratio at this lead (day 8 is the anti-blur one;
                    full marks need 0.40 <= ratio <= 2.5)
    n_windows       number of windows pooled
  eval/scores_summary.json: projected points (rmse_pts, csi_pts, anti_blur_pts, total of 55), mean window SS
    and CSI SS, judge5 spread, day-4 persistence gate, validity failures and the window dates.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
import urllib.request
import warnings
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monsoon_r1 import scoring as S  # noqa: E402
from monsoon_r1.contract import LAT, LON, LEAD_DAYS, SCORE_STEPS  # noqa: E402
from monsoon_r1.stats import clim_for, climatology  # noqa: E402

from common import CH, CHANNELS, N_CH, N_IN_DAYS, NLAT, NLON, UNITS, doy_sincos  # noqa: E402

TRAIN_YEARS = (2005, 2018)
VAL_YEARS = (2019, 2021)
DEFAULT_STEPS = SCORE_STEPS          # the judge always asks for 10 steps
DEFAULT_WINDOWS = 40
JUDGE_WINDOWS = 5
MAX_POINTS = {"rmse_pts": 30.0, "csi_pts": 15.0, "anti_blur_pts": 10.0}
CSV_COLUMNS = ["lead", "channel", "units", "rmse_model", "rmse_persist", "rmse_clim", "best_ref", "ss",
               "ss_persist", "ss_clim", "nrmse_model", "csi_model", "csi_persist", "csi_clim", "csi_skill",
               "var_ratio", "n_windows"]


# ---------------------------------------------------------------------------
# Dates and windows
# ---------------------------------------------------------------------------
def day_of_year(times: np.ndarray) -> np.ndarray:
    t = np.asarray(times).astype("datetime64[D]")
    return (t - t.astype("datetime64[Y]")).astype(np.int64) + 1


def year_of(times: np.ndarray) -> np.ndarray:
    return np.asarray(times).astype("datetime64[Y]").astype(np.int64) + 1970


def yyyymmdd(times: np.ndarray) -> np.ndarray:
    """datetime64 -> yyyymmdd ints (the format monsoon_r1.stats.climatology takes)."""
    t = np.asarray(times).astype("datetime64[D]")
    y = t.astype("datetime64[Y]")
    m = t.astype("datetime64[M]")
    return ((y.astype(np.int64) + 1970) * 10000 + (m - y).astype(np.int64) * 100 + 100
            + (t - m).astype(np.int64) + 1)


def to_date(t) -> dt.date:
    return np.datetime64(t, "D").astype(dt.date)


def build_windows(times: np.ndarray, in_mask: np.ndarray, n_in: int = N_IN_DAYS,
                  steps: int = DEFAULT_STEPS) -> np.ndarray:
    """Indices t of the last input day such that days t-n_in+1 .. t+steps are consecutive
    calendar days and all inside in_mask."""
    days = np.asarray(times).astype("datetime64[D]").astype(np.int64)
    span = n_in + steps
    out = []
    for t in range(n_in - 1, days.size - steps):
        a, b = t - n_in + 1, t + steps + 1
        if in_mask[a:b].all() and days[b - 1] - days[a] == span - 1:
            out.append(t)
    return np.asarray(out, dtype=np.int64)


def sample_windows(all_windows: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """n windows drawn without replacement (sorted); n <= 0 or n >= len -> all of them."""
    if n <= 0 or n >= all_windows.size:
        return all_windows.copy()
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(all_windows, size=n, replace=False))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def dataset_to_arrays(ds):
    """xarray Dataset -> (data [T, 6, 32, 36] float32 in CHANNELS order, times datetime64[D]).

    Kit layout (india_climate_train.nc): variable `data` (time, channel, lat, lon) with a `channel` coord, lat
    ascending 6.5..37.5. A dataset with one variable per channel name (t2m, tp, ...) is also accepted.
    A north->south file is flipped; any other grid mismatch raises."""
    if "data" in ds.data_vars and "channel" in ds["data"].dims:
        da = ds["data"]
        if tuple(str(c) for c in da["channel"].values) != CHANNELS:
            da = da.sel(channel=list(CHANNELS))
        latn = "lat" if "lat" in da.dims else "latitude"
        lonn = "lon" if "lon" in da.dims else "longitude"
        da = da.transpose("time", "channel", latn, lonn)
        data = np.asarray(da.values, dtype=np.float32)
    else:
        missing = [c for c in CHANNELS if c not in ds.data_vars]
        if missing:
            raise KeyError(f"no `data` variable and no per-channel variables {missing}; vars: {list(ds.data_vars)}")
        latn = "lat" if "lat" in ds.dims else "latitude"
        lonn = "lon" if "lon" in ds.dims else "longitude"
        data = np.stack([np.asarray(ds[c].transpose("time", latn, lonn).values, dtype=np.float32)
                         for c in CHANNELS], axis=1)
    lat = np.asarray(ds[latn].values, dtype=np.float64)
    lon = np.asarray(ds[lonn].values, dtype=np.float64)
    if lat.size == NLAT and lat[0] > lat[-1]:
        data, lat = data[:, :, ::-1].copy(), lat[::-1]
        print("[score] file latitude is north->south; flipped to the contract's south->north")
    if lat.shape != LAT.shape or lon.shape != LON.shape or not (
            np.allclose(lat, LAT, atol=1e-3) and np.allclose(lon, LON, atol=1e-3)):
        raise ValueError(f"grid lat {lat[:2]}..{lat[-1:]} lon {lon[:2]}..{lon[-1:]} != contract grid")
    times = np.asarray(ds["time"].values).astype("datetime64[D]")
    order = np.argsort(times, kind="stable")
    if not np.all(order == np.arange(order.size)):
        data, times = data[order], times[order]
    return data, times


def load_nc(path: str):
    import xarray as xr
    with xr.open_dataset(path) as ds:
        return dataset_to_arrays(ds)


def load_stats_std(path: Optional[str]) -> Optional[np.ndarray]:
    """Per-channel std from the kit's stats.json ({"channels": [...], "mean": [6], "std": [6]})."""
    if not path:
        return None
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    std = np.asarray(s["std"], dtype=np.float64)
    if "channels" in s and tuple(s["channels"]) != CHANNELS:
        raise ValueError(f"stats.json channel order {s['channels']} != {CHANNELS}")
    if std.shape != (N_CH,) or not np.all(std > 0):
        raise ValueError(f"stats.json std must be 6 positive numbers, got {std}")
    return std


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
    lat = (LAT - LAT.mean()) / np.ptp(LAT)        # -0.5..0.5 south->north
    lon = (LON - LON.mean()) / np.ptp(LON)
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


# ---------------------------------------------------------------------------
# Forecasters
# ---------------------------------------------------------------------------
Forecaster = Callable[[np.ndarray, np.ndarray, int], np.ndarray]


def persistence_forecast(init_states: np.ndarray, steps: int = DEFAULT_STEPS) -> np.ndarray:
    """Last input day repeated: [n_in, C, H, W] -> [steps, C, H, W]."""
    x = np.asarray(init_states)
    return np.repeat(x[-1][None], int(steps), axis=0)


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
        out = post("/simulate", {"init_states": np.asarray(x, dtype=np.float32).tolist(),
                                 "doy_sincos": np.asarray(d, dtype=np.float64).tolist(),
                                 "steps": int(steps), "perturbation": None})
        return np.asarray(out["rollout"], dtype=np.float32)

    return f, f"server:{health.get('model_name', health.get('backend', '?'))}"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def train_climatology(data: np.ndarray, times: np.ndarray, train_years=TRAIN_YEARS) -> np.ndarray:
    """Kit climatology [366, 6, 32, 36] (15-day smoothing) from train-year days only."""
    yrs = year_of(times)
    train = (yrs >= train_years[0]) & (yrs <= train_years[1])
    if not train.any():
        raise ValueError(f"no training days in {train_years}")
    return climatology(data[train], yyyymmdd(times[train]))


def evaluate(data: np.ndarray, times: np.ndarray, forecaster: Forecaster, n_windows: int = DEFAULT_WINDOWS,
             seed: int = 0, steps: int = DEFAULT_STEPS, train_years=TRAIN_YEARS, val_years=VAL_YEARS,
             std: Optional[np.ndarray] = None, judge_windows: int = JUDGE_WINDOWS, judge_draws: int = 500,
             clim: Optional[np.ndarray] = None, verbose: bool = False) -> dict:
    """Roll out sampled validation windows and score them exactly as the judge's functions do."""
    leads = list(LEAD_DAYS)
    if steps < max(leads):
        raise ValueError(f"steps must be >= {max(leads)}")
    L = np.asarray(leads) - 1
    yrs = year_of(times)
    train = (yrs >= train_years[0]) & (yrs <= train_years[1])
    val = (yrs >= val_years[0]) & (yrs <= val_years[1])
    if clim is None:
        clim = train_climatology(data, times, train_years)
    if std is None:
        std = data[train].std(axis=(0, 2, 3)).astype(np.float64)
    all_win = build_windows(times, val, steps=steps)
    if all_win.size == 0:
        raise ValueError(f"no complete validation windows in {val_years}")
    win = sample_windows(all_win, n_windows, seed)
    doys = day_of_year(times)

    n = win.size
    shape = (n, len(leads), N_CH, NLAT, NLON)
    model = np.full(shape, np.nan, np.float32)
    truth = np.empty(shape, np.float32)
    pers = np.empty(shape, np.float32)
    climf = np.empty(shape, np.float32)
    win_ss = np.zeros(n)
    win_ss_lead = np.zeros(n)      # sensitivity variant: per-lead SS averaged instead of leads pooled
    win_csi = np.zeros(n)
    ratios = np.zeros((n, N_CH))
    failures = []
    t0 = time.perf_counter()
    for i, t in enumerate(win):
        x = data[t - N_IN_DAYS + 1: t + 1]
        tgt = t + 1 + L
        truth[i] = data[tgt]
        pers[i] = np.repeat(x[-1][None], len(leads), axis=0)
        climf[i] = np.stack([clim_for(clim, to_date(times[k])) for k in tgt])
        try:
            out = np.asarray(forecaster(x, doy_sincos(doys[t]), steps), dtype=np.float32)
            bad = S.validity(out, steps)
        except Exception as e:  # a failed call scores 0 for the window, like the judge; it is reported, not hidden
            out, bad = None, [f"{type(e).__name__}: {e}"]
        if bad:
            failures.append({"date": str(times[t]), "why": bad})
            if out is not None and out.shape == (steps, N_CH, NLAT, NLON):
                model[i] = out[L]
            continue                                    # SS 0, CSI SS 0, ratios 0
        model[i] = out[L]
        sk = S.skill_vs_best_ref(model[i], truth[i], pers[i], climf[i])
        win_ss[i] = _nanmean(sk["ss"])
        win_ss_lead[i] = _nanmean([_nanmean(S.skill_vs_best_ref(model[i, j], truth[i, j], pers[i, j],
                                                                 climf[i, j])["ss"]) for j in range(len(leads))])
        tp = CH["tp"]
        win_csi[i] = S.csi_skill(S.csi(model[i, :, tp], truth[i, :, tp]), S.csi(pers[i, :, tp], truth[i, :, tp]),
                                 S.csi(climf[i, :, tp], truth[i, :, tp]))
        ratios[i] = S.variance_ratio(out[7], truth[i, leads.index(8)])
        if verbose and (i + 1) % 100 == 0:
            print(f"[score] {i + 1}/{n} windows")
    secs = time.perf_counter() - t0

    fp = S.forecast_points(list(win_ss), list(win_csi))
    anti = S.anti_blur_points(ratios)
    judge5 = judge_subsets(win_ss, win_csi, ratios, judge_windows, judge_draws, seed)

    rows = []
    for j, lead in enumerate(leads):
        sk = S.skill_vs_best_ref(model[:, j], truth[:, j], pers[:, j], climf[:, j])
        rp = S.rmse_lat(pers[:, j], truth[:, j])
        rc = S.rmse_lat(climf[:, j], truth[:, j])
        vr = np.array([S.variance_ratio(model[i, j], truth[i, j]) for i in range(n)])
        tp_t = truth[:, j, CH["tp"]]
        for c, ch in enumerate(CHANNELS):
            row = {"lead": lead, "channel": ch, "units": UNITS[ch],
                   "rmse_model": float(sk["rmse"][c]), "rmse_persist": float(rp[c]), "rmse_clim": float(rc[c]),
                   "best_ref": "persist" if rp[c] <= rc[c] else "clim",
                   "ss": float(sk["ss"][c]), "ss_persist": float(sk["ss_persist"][c]),
                   "ss_clim": float(sk["ss_clim"][c]), "nrmse_model": float(sk["rmse"][c] / std[c]) if std[c] > 0 else float("nan"),
                   "csi_model": None, "csi_persist": None, "csi_clim": None, "csi_skill": None,
                   "var_ratio": float(np.nanmean(vr[:, c])) if np.isfinite(vr[:, c]).any() else float("nan"),
                   "n_windows": int(n)}
            if ch == "tp":
                cm = S.csi(model[:, j, c], tp_t)
                cp = S.csi(pers[:, j, c], tp_t)
                cc = S.csi(climf[:, j, c], tp_t)
                row.update(csi_model=cm, csi_persist=cp, csi_clim=cc, csi_skill=S.csi_skill(cm, cp, cc))
            rows.append(row)

    lead4 = [r["ss_persist"] for r in rows if r["lead"] == 4]
    points = {"rmse_pts": fp["rmse_pts"], "csi_pts": fp["csi_pts"], "anti_blur_pts": anti}
    return {
        "rows": rows, "leads": leads, "n_windows": int(n), "n_val_windows": int(all_win.size), "seed": seed,
        "window_dates": [str(times[t]) for t in win], "seconds": secs,
        "mean_ss": fp["mean_ss"], "mean_csi_skill": fp["mean_csi_skill"],
        "mean_ss_per_lead_variant": float(np.mean(win_ss_lead)),
        "n_windows_csi_defined": int(np.isfinite(win_csi).sum()),
        "day8_var_ratio": {c: float(np.mean(ratios[:, k])) for k, c in enumerate(CHANNELS)},
        "points": points, "points_total": float(sum(points.values())), "points_max": float(sum(MAX_POINTS.values())),
        "judge5": judge5,
        "gate_day4_mean_ss_persist": float(np.mean(lead4)),
        "gate_day4_beats_persistence": bool(np.mean(lead4) > 0),
        "failures": failures,
        "window_ss": win_ss.tolist(), "window_csi_skill": win_csi.tolist(),
    }


def _nanmean(a) -> float:
    """nanmean that returns NaN quietly when every entry is NaN (e.g. a channel whose references are perfect)."""
    a = np.asarray(a, dtype=np.float64)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def judge_subsets(win_ss, win_csi, ratios, k: int = JUDGE_WINDOWS, draws: int = 500, seed: int = 0) -> dict:
    """Points the judge would give on random k-window subsets of the scored windows: mean, p10, p90."""
    n = len(win_ss)
    k = min(k, n)
    rng = np.random.default_rng(seed + 1)
    tot = []
    for _ in range(draws):
        idx = rng.choice(n, size=k, replace=False)
        fp = S.forecast_points([win_ss[i] for i in idx], [win_csi[i] for i in idx])
        tot.append((fp["rmse_pts"], fp["csi_pts"], S.anti_blur_points(ratios[idx])))
    a = np.array(tot)
    out = {"k": k, "draws": draws}
    for col, name in enumerate(("rmse_pts", "csi_pts", "anti_blur_pts")):
        out[name] = {"mean": float(a[:, col].mean()), "p10": float(np.percentile(a[:, col], 10)),
                     "p90": float(np.percentile(a[:, col], 90))}
    s = a.sum(axis=1)
    out["total"] = {"mean": float(s.mean()), "p10": float(np.percentile(s, 10)), "p90": float(np.percentile(s, 90))}
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _f(v, fmt="{:.3f}"):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)


def print_table(res: dict) -> None:
    print(f"{'lead':>4} {'chan':>5} {'model':>10} {'persist':>10} {'clim':>10} {'best':>7} "
          f"{'SS':>7} {'SS_pers':>7} {'SS_clim':>7} {'varR':>5}")
    for r in res["rows"]:
        print(f"{r['lead']:>4} {r['channel']:>5} {r['rmse_model']:>10.4g} {r['rmse_persist']:>10.4g} "
              f"{r['rmse_clim']:>10.4g} {r['best_ref']:>7} {_f(r['ss']):>7} {_f(r['ss_persist']):>7} "
              f"{_f(r['ss_clim']):>7} {_f(r['var_ratio'], '{:.2f}'):>5}")
    for r in res["rows"]:
        if r["channel"] == "tp":
            print(f"[score] lead {r['lead']} CSI(tp>{S.RAIN_THRESH:g}): model={_f(r['csi_model'])} "
                  f"persist={_f(r['csi_persist'])} clim={_f(r['csi_clim'])} skill={_f(r['csi_skill'])}")


def summarize(res: dict, backend: str) -> str:
    p, j = res["points"], res["judge5"]
    vr = " ".join(f"{c}={v:.2f}" for c, v in res["day8_var_ratio"].items())
    return "\n".join([
        f"[score] day-8 variance ratio (anti-blur needs 0.40..2.5): {vr}",
        f"[score] mean window SS={res['mean_ss']:.4f} (full 30 pts at {S.SS_REF}; per-lead variant "
        f"{res['mean_ss_per_lead_variant']:.4f})  mean CSI skill={_f(res['mean_csi_skill'], '{:.4f}')} "
        f"(full 15 at {S.CSI_SS_REF}; defined in {res['n_windows_csi_defined']}/{res['n_windows']} windows)",
        f"[score] gate (beat persistence at day 4): mean SS_persist(lead 4)={res['gate_day4_mean_ss_persist']:.4f} "
        f"-> {'PASS' if res['gate_day4_beats_persistence'] else 'FAIL'};  invalid/failed windows: {len(res['failures'])}",
        f"[score] judge-like {j['k']}-window estimate over {j['draws']} draws: total {j['total']['mean']:.2f} "
        f"(p10 {j['total']['p10']:.2f}, p90 {j['total']['p90']:.2f}) = rmse {j['rmse_pts']['mean']:.2f} "
        f"+ csi {j['csi_pts']['mean']:.2f} + anti-blur {j['anti_blur_pts']['mean']:.2f}",
        f"SUMMARY backend={backend} windows={res['n_windows']} rmse_pts={p['rmse_pts']:.2f}/30 "
        f"csi_pts={p['csi_pts']:.2f}/15 anti_blur_pts={p['anti_blur_pts']:.2f}/10 "
        f"total={res['points_total']:.2f}/{res['points_max']:g}",
    ])


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return v


def write_csv(rows: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(r[k]) for k in CSV_COLUMNS})


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def write_summary(res: dict, backend: str, path: str, data_src: str) -> None:
    keep = {k: v for k, v in res.items() if k != "rows"}
    keep.update(backend=backend, data=data_src, train_years=list(TRAIN_YEARS), val_years=list(VAL_YEARS),
                points_max_by_component=MAX_POINTS)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_json_safe(keep), indent=1, allow_nan=False), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="kit india_climate_train.nc (daily 2005-2021)")
    ap.add_argument("--stats", help="kit stats.json (only used for the nrmse column)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--server", help="score a running serve.py, e.g. http://localhost:8000")
    src.add_argument("--twin", action="store_true", help="score twin.Twin in-process (default)")
    ap.add_argument("--synthetic", action="store_true", help="generated fake data instead of --data")
    ap.add_argument("--windows", type=int, default=DEFAULT_WINDOWS, help="validation windows to sample (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge-windows", type=int, default=JUDGE_WINDOWS)
    ap.add_argument("--out", default=str(ROOT / "eval" / "scores.csv"))
    ap.add_argument("--summary-out", default=str(ROOT / "eval" / "scores_summary.json"))
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    if a.synthetic:
        data, times = synthetic_arrays()
        data_src = "synthetic"
        print("[score] SYNTHETIC data: numbers are meaningless for model choice")
    else:
        if not a.data:
            ap.error("--data is required unless --synthetic")
        data, times = load_nc(a.data)
        data_src = a.data
    std = load_stats_std(a.stats)
    forecaster, backend = server_forecaster(a.server) if a.server else twin_forecaster()
    res = evaluate(data, times, forecaster, n_windows=a.windows, seed=a.seed, std=std,
                   judge_windows=a.judge_windows, verbose=True)
    print(f"[score] climatology from {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]} (kit, 15-day smoothing); "
          f"{res['n_windows']}/{res['n_val_windows']} windows from {VAL_YEARS[0]}-{VAL_YEARS[1]} (seed {a.seed}); "
          f"{res['seconds']:.1f}s of rollouts, {time.perf_counter() - t0:.1f}s total")
    print_table(res)
    for fl in res["failures"][:5]:
        print(f"[score] INVALID window {fl['date']}: {fl['why']}")
    write_csv(res["rows"], a.out)
    write_summary(res, backend, a.summary_out, data_src)
    print(f"[score] wrote {a.out} and {a.summary_out}")
    print(summarize(res, backend))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
