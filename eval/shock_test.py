"""Perturbation-causality report: the judge's 15 dynamics points, plus the dashboard's Shock Lab profiles.

Judge part (organisers' code, monsoon_r1.scoring, never re-implemented):
  make_shock(warm_ocean | monsoon_stall | null), causality_checks (nonzero, localized, propagates incl. no_teleport,
  stable), null_consistent (max |null - base| / std <= 1e-3) and causality_points (/15, halved if null fails).
  std comes from stats.json (kit layout {"channels", "mean", "std"}); without it scoring.STD_DEFAULT is used.
  base = the unperturbed call (perturbation=None), as in the contract.

Our stricter checks (strict=True, the default; never loosened): perturbation=None and perturbation=zeros must be
bit-identical, and the named zero shock must change nothing (max |diff| == 0). Failing raises ShockCheckError;
--no-strict still writes the report so the kit points can be inspected.

Shock Lab profiles: per shock, channel and lead day, mean |diff| inside vs outside the box, a radial profile by
Chebyshev ring distance from the box (0 = inside, 1..8 = rings) and its centroid.

eval/shock_report.json keys (stable; changes vs the pre-kit version are marked NEW / CHANGED):
  steps, days, max_ring, std_source ("stats" | "kit_default"; CHANGED: no longer estimated from init_states),
  channel_std, determinism {max_abs_diff_none_vs_zeros, pass},
  shocks.warm_ocean / shocks.monsoon_stall: box, n_cells_inside, perturbed_channels, max_abs_delta,
      channels.<ch>.<day>: inside, outside, ratio, outside_is_zero, radial, centroid_ring
  shocks.zero: max_abs_diff, max_abs_diff_by_channel, tolerance_by_channel, pass   (kit name: "null")
  NEW judge: {null_ok, null_max_rel, points, max_points, shocks.<name>: causality_checks output (nonzero, stable,
      near_frac_day1, localized, far_frac_max, no_teleport, propagates, finite) + energy_near / energy_outside per
      day (response energy in std units inside / outside the kit's 3-cell neighbourhood) + points}
  NEW per_window: [{window, points, null_ok, checks per shock}] when several --date are given
  source, window

CLI:
  python -m eval.shock_test [--server URL] [--data data/india_climate_train.nc --date 2019-07-15 [--date ...]]
                            [--window synthetic|path.npz] [--stats data/stats.json] [--no-strict]
                            [--json-out eval/shock_report.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monsoon_r1 import scoring as S  # noqa: E402
from monsoon_r1.contract import BOX_ARABIAN_SEA, BOX_BAY_OF_BENGAL, box_mask  # noqa: E402

from common import CH, CHANNELS, INIT_SHAPE, NLAT, NLON, STATE_SHAPE  # noqa: E402
from common import doy_sincos as make_doy_sincos  # noqa: E402

RolloutFn = Callable[..., np.ndarray]

SHOCKS = ("warm_ocean", "monsoon_stall", "zero")
KIT_NAME = {"warm_ocean": "warm_ocean", "monsoon_stall": "monsoon_stall", "zero": "null"}
KIT_BOX = {"warm_ocean": BOX_BAY_OF_BENGAL, "monsoon_stall": BOX_ARABIAN_SEA}
REPORT_DAYS = (1, 4, 8)
MAX_RING = 8
DEFAULT_STATS = ROOT / "data" / "stats.json"


class ShockCheckError(AssertionError):
    """A strict dynamics check failed (zero shock not exact, or None vs zeros not identical)."""


# ---------------------------------------------------------------- windows
def synthetic_window(seed: int = 0, doy: float = 190.0) -> tuple[np.ndarray, float]:
    """Plausible physical init_states [3,6,32,36] (fixed seed) and day of year."""
    rng = np.random.default_rng(seed)
    lat = np.linspace(0.0, 1.0, NLAT)[:, None] * np.ones((1, NLON))
    lon = np.ones((NLAT, 1)) * np.linspace(0.0, 1.0, NLON)[None, :]
    x = np.empty(INIT_SHAPE, dtype=np.float64)
    for d in range(INIT_SHAPE[0]):
        n = lambda s: s * rng.standard_normal((NLAT, NLON))  # noqa: E731
        x[d, CH["t2m"]] = 302.0 - 6.0 * lat + n(1.5)
        x[d, CH["tp"]] = np.maximum(0.0, 6.0 + 8.0 * np.sin(3.0 * lon) * (1 - lat) + n(4.0))
        x[d, CH["u10"]] = 6.0 * (1.0 - 2.0 * lat) + n(2.0)
        x[d, CH["v10"]] = 3.0 * np.cos(2.0 * lon) + n(2.0)
        x[d, CH["u10"]] = np.clip(x[d, CH["u10"]], -10.0, 10.0)
        x[d, CH["v10"]] = np.clip(x[d, CH["v10"]], -10.0, 10.0)
        x[d, CH["msl"]] = 100_600.0 - 900.0 * lat + n(150.0)
        x[d, CH["tcwv"]] = np.maximum(1.0, 52.0 - 25.0 * lat + n(4.0))
    return x, float(doy)


def load_window(path: str | Path) -> tuple[np.ndarray, float]:
    """.npz with `init_states` [3,6,32,36] and `doy` (scalar, last input day)."""
    with np.load(path) as z:
        init = np.asarray(z["init_states"], dtype=np.float64)
        doy = float(np.asarray(z["doy"]).reshape(-1)[0])
    if init.shape != INIT_SHAPE:
        raise ValueError(f"init_states must be {INIT_SHAPE}, got {init.shape}")
    return init, doy


def real_windows(nc_path: str, dates: list[str]) -> list[tuple[np.ndarray, float, str]]:
    """init_states for each last-input-day date from the kit .nc: [(init [3,6,32,36], doy, date)]."""
    from eval.score import day_of_year, load_nc

    data, times = load_nc(nc_path)
    out = []
    for d in dates:
        hit = np.nonzero(times == np.datetime64(d, "D"))[0]
        if hit.size == 0 or hit[0] < 2:
            raise ValueError(f"date {d} not in {nc_path} (or fewer than 2 days before it)")
        t = int(hit[0])
        out.append((data[t - 2:t + 1].astype(np.float64), float(day_of_year(times[t:t + 1])[0]), d))
    return out


def load_channel_std(path: str | Path) -> np.ndarray:
    """Per-channel std from the kit's stats.json ({"channels": [...], "mean": [6], "std": [6]})."""
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    if "channels" in s and tuple(s["channels"]) != CHANNELS:
        raise ValueError(f"stats.json channel order {s['channels']} != {CHANNELS}")
    std = np.asarray(s["std"], dtype=np.float64)
    if std.shape != (len(CHANNELS),) or not np.all(std > 0):
        raise ValueError(f"stats.json std must be 6 positive numbers, got {std}")
    return std


# ---------------------------------------------------------------- HTTP client
def http_rollout_fn(url: str, timeout: float = 30.0) -> RolloutFn:
    """rollout_fn with Twin.rollout's signature that POSTs to {url}/simulate."""
    endpoint = url.rstrip("/") + "/simulate"

    def rollout(init_states, doy_sincos, steps: int = 10, perturbation=None) -> np.ndarray:
        body = {
            "init_states": np.asarray(init_states, dtype=np.float64).tolist(),
            "doy_sincos": np.asarray(doy_sincos, dtype=np.float64).reshape(-1).tolist(),
            "steps": int(steps),
            "perturbation": None if perturbation is None
            else np.asarray(perturbation, dtype=np.float64).tolist(),
        }
        req = urllib.request.Request(
            endpoint, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode("utf-8"))
        out = np.asarray(payload["rollout"], dtype=np.float64)
        expected = (int(steps),) + STATE_SHAPE
        if out.shape != expected:
            raise ValueError(f"/simulate returned {out.shape}, expected {expected}")
        return out

    return rollout


# ---------------------------------------------------------------- geometry
def ring_distance(mask: np.ndarray) -> np.ndarray:
    """Chebyshev distance in cells from the box: 0 inside, k for the k-th ring outside.
    Pure numpy (repeated 3x3 dilation); cells beyond the grid's reach get a large value."""
    dist = np.full(mask.shape, np.iinfo(np.int32).max, dtype=np.int64)
    dist[mask] = 0
    cur = mask.copy()
    for k in range(1, max(NLAT, NLON)):
        p = np.pad(cur, 1, constant_values=False)
        grown = np.zeros_like(cur)
        for di in (0, 1, 2):
            for dj in (0, 1, 2):
                grown |= p[di:di + NLAT, dj:dj + NLON]
        new = grown & ~cur
        if not new.any():
            break
        dist[new] = k
        cur = grown
    return dist


def _jsonable(v) -> Optional[float]:
    return None if (v is None or not math.isfinite(v)) else float(v)


def _shock_points(checks: dict, n_shocks: int = 2, max_pts: float = 15.0) -> float:
    """One shock's share of causality_points (null assumed ok), via the kit function itself."""
    return S.causality_points([checks], True, max_pts=max_pts / n_shocks)


# ---------------------------------------------------------------- report
def shock_report(
    rollout_fn: RolloutFn,
    init_states: np.ndarray,
    doy_sincos: np.ndarray,
    steps: int = 10,
    channel_std: Optional[np.ndarray] = None,
    shocks: tuple[str, ...] = SHOCKS,
    days: tuple[int, ...] = REPORT_DAYS,
    strict: bool = True,
) -> dict:
    """Run base, null and shocked rollouts; score them with the kit and build the locality report.

    channel_std: per-channel std (stats.json). None -> monsoon_r1.scoring.STD_DEFAULT.
    strict: raise ShockCheckError unless None vs zeros and the zero shock are bit-identical to the base.
    """
    init = np.asarray(init_states, dtype=np.float64)
    if init.shape != INIT_SHAPE:
        raise ValueError(f"init_states must be {INIT_SHAPE}, got {init.shape}")
    dsc = np.asarray(doy_sincos, dtype=np.float64).reshape(-1)
    steps = int(steps)
    days = tuple(d for d in days if 1 <= d <= steps)
    if channel_std is None:
        channel_std, std_source = S.STD_DEFAULT.astype(np.float64), "kit_default"
    else:
        std_source = "stats"
    std = np.asarray(channel_std, dtype=np.float64)
    tol = S.NULL_TOL * std
    last = init[-1].astype(np.float32)

    baseline = np.asarray(rollout_fn(init, dsc, steps=steps, perturbation=None), dtype=np.float64)
    if baseline.shape != (steps,) + STATE_SHAPE:
        raise ValueError(f"rollout returned {baseline.shape}")
    if not np.all(np.isfinite(baseline)):
        raise ShockCheckError("baseline rollout contains NaN/Inf")

    zeros_run = np.asarray(rollout_fn(init, dsc, steps=steps, perturbation=np.zeros(STATE_SHAPE)),
                           dtype=np.float64)
    det_max = float(np.max(np.abs(zeros_run - baseline)))
    if strict and det_max != 0.0:
        raise ShockCheckError(f"perturbation=None vs zeros differ: max |diff| = {det_max:g}")
    null_runs = [zeros_run]

    report = {
        "steps": steps,
        "days": list(days),
        "max_ring": MAX_RING,
        "std_source": std_source,
        "channel_std": {c: float(s) for c, s in zip(CHANNELS, std)},
        "determinism": {"max_abs_diff_none_vs_zeros": det_max, "pass": det_max == 0.0},
        "shocks": {},
    }
    judge_shocks = {}

    for name in shocks:
        delta = S.make_shock(KIT_NAME[name], last, std.astype(np.float32))
        shocked = np.asarray(rollout_fn(init, dsc, steps=steps, perturbation=delta), dtype=np.float64)
        diff = np.abs(shocked - baseline)  # [steps, 6, 32, 36]

        if name == "zero":
            null_runs.append(shocked)
            max_diff = float(diff.max()) if np.all(np.isfinite(diff)) else float("inf")
            if strict and max_diff != 0.0:
                raise ShockCheckError(f"zero shock changed the output: max |diff| = {max_diff:g}")
            per_ch = diff.max(axis=(0, 2, 3))
            report["shocks"][name] = {
                "max_abs_diff": _jsonable(max_diff),
                "max_abs_diff_by_channel": {c: _jsonable(float(m)) for c, m in zip(CHANNELS, per_ch)},
                "tolerance_by_channel": {c: float(t) for c, t in zip(CHANNELS, tol)},
                "pass": bool(max_diff == 0.0),
            }
            continue

        box = KIT_BOX[name]
        checks = S.causality_checks(baseline, shocked, box, std)
        if checks["finite"]:
            e = ((shocked - baseline) / std[None, :, None, None]) ** 2
            e = e.sum(axis=1)
            near = S.near_mask(box)
            checks["energy_near"] = [float(e[k][near].sum()) for k in range(steps)]
            checks["energy_outside"] = [float(e[k][~near].sum()) for k in range(steps)]
        checks["points"] = _shock_points(checks)
        judge_shocks[name] = checks
        if not checks["finite"]:
            report["shocks"][name] = {"box": list(box), "finite": False}
            continue

        mask = box_mask(box)
        dist = ring_distance(mask)
        perturbed = [c for c in CHANNELS if np.any(delta[CH[c]] != 0)]
        channels = perturbed + [c for c in ("tp",) if c not in perturbed]
        ch_out = {}
        for c in channels:
            per_day = {}
            for d in days:
                f = diff[d - 1, CH[c]]
                inside = float(f[mask].mean())
                outside = float(f[~mask].mean())
                ratio = inside / outside if outside > 0 else (math.inf if inside > 0 else math.nan)
                profile = [float(f[dist == k].mean()) if np.any(dist == k) else 0.0
                           for k in range(MAX_RING + 1)]
                tot = sum(profile)
                centroid = sum(k * p for k, p in enumerate(profile)) / tot if tot > 0 else 0.0
                per_day[str(d)] = {
                    "inside": inside,
                    "outside": outside,
                    "ratio": _jsonable(ratio),
                    "outside_is_zero": outside == 0.0,
                    "radial": profile,
                    "centroid_ring": centroid,
                }
            ch_out[c] = per_day
        report["shocks"][name] = {
            "box": list(box),
            "n_cells_inside": int(mask.sum()),
            "perturbed_channels": perturbed,
            "max_abs_delta": {c: float(np.abs(delta[CH[c]]).max()) for c in perturbed},
            "channels": ch_out,
        }

    null_ok = all(S.null_consistent(baseline, r, std) for r in null_runs)
    rel = max(float(np.max(np.abs(r - baseline) / std[None, :, None, None])) for r in null_runs)
    pts = S.causality_points(list(judge_shocks.values()), null_ok) if judge_shocks else 0.0
    report["judge"] = {"null_ok": bool(null_ok), "null_max_rel": _jsonable(rel), "points": pts,
                       "max_points": 15.0, "shocks": judge_shocks}
    return report


# ---------------------------------------------------------------- CLI
def format_report(rep: dict) -> str:
    j = rep["judge"]
    lines = [f"Shock report  steps={rep['steps']}  std from {rep['std_source']}",
             f"determinism (None vs zeros): max|diff| = {rep['determinism']['max_abs_diff_none_vs_zeros']:g}  "
             f"{'PASS' if rep['determinism']['pass'] else 'FAIL'}"]
    for name, s in rep["shocks"].items():
        if name == "zero":
            lines.append(f"\n[zero] max|diff| = {s['max_abs_diff']}  (must be exactly 0)  "
                         f"{'PASS' if s['pass'] else 'FAIL'}")
            continue
        if "channels" not in s:
            lines.append(f"\n[{name}] non-finite response")
            continue
        lines.append(f"\n[{name}] box={s['box']} cells={s['n_cells_inside']} "
                     f"perturbed={s['perturbed_channels']}")
        lines.append(f"  {'ch':5s} {'day':>3s} {'inside':>10s} {'outside':>10s} {'ratio':>9s} "
                     f"{'centroid':>8s}  radial 0..{rep['max_ring']}")
        for c, per_day in s["channels"].items():
            for d, v in per_day.items():
                r = "inf" if v["ratio"] is None and v["inside"] > 0 else (
                    "nan" if v["ratio"] is None else f"{v['ratio']:.2f}")
                rad = " ".join(f"{p:.2g}" for p in v["radial"])
                lines.append(f"  {c:5s} {d:>3s} {v['inside']:10.4g} {v['outside']:10.4g} {r:>9s} "
                             f"{v['centroid_ring']:8.2f}  {rad}")
    lines.append("\nJudge (monsoon_r1.scoring):")
    for name, c in j["shocks"].items():
        flags = " ".join(f"{k}={'Y' if c.get(k) else 'n'}"
                         for k in ("nonzero", "localized", "propagates", "no_teleport", "stable"))
        lines.append(f"  {name:13s} {flags}  near_frac_day1={c.get('near_frac_day1', 0):.3f} "
                     f"far_frac_max={c.get('far_frac_max', 0):.3f}  -> {c['points']:.2f}/7.5")
    lines.append(f"  null consistent (<= {S.NULL_TOL:g} std): {'Y' if j['null_ok'] else 'n'} "
                 f"(max {j['null_max_rel']:.3g} std)")
    lines.append(f"CAUSALITY points = {j['points']:.2f}/15")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--server", default=None, help="serve.py base URL; default in-process Twin")
    ap.add_argument("--window", default="synthetic", help="'synthetic' or .npz with init_states, doy")
    ap.add_argument("--data", default=None, help="kit .nc; use with --date for real windows")
    ap.add_argument("--date", action="append", default=[], help="last input day YYYY-MM-DD (repeatable)")
    ap.add_argument("--stats", default=None, help=f"stats.json (default {DEFAULT_STATS} if present)")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-strict", action="store_true", help="report even if the exact-zero checks fail")
    ap.add_argument("--json-out", default=str(ROOT / "eval" / "shock_report.json"))
    a = ap.parse_args(argv)

    if a.date:
        if not a.data:
            ap.error("--date needs --data")
        windows = real_windows(a.data, a.date)
    elif a.window == "synthetic":
        init, doy = synthetic_window(seed=a.seed)
        windows = [(init, doy, "synthetic")]
    else:
        init, doy = load_window(a.window)
        windows = [(init, doy, a.window)]
    if a.server:
        fn = http_rollout_fn(a.server)
    else:
        from twin import Twin
        fn = Twin().rollout
    stats = a.stats or (str(DEFAULT_STATS) if DEFAULT_STATS.exists() else None)
    std = load_channel_std(stats) if stats else None

    reps = [shock_report(fn, init, make_doy_sincos(doy), steps=a.steps, channel_std=std, strict=not a.no_strict)
            for init, doy, _ in windows]
    rep = reps[0]
    rep["source"] = a.server or "in-process Twin"
    rep["window"] = windows[0][2]
    if len(reps) > 1:
        rep["per_window"] = [{"window": w[2], "points": r["judge"]["points"], "null_ok": r["judge"]["null_ok"],
                              "checks": {k: {f: v[f] for f in ("nonzero", "localized", "propagates", "no_teleport",
                                                               "stable") if f in v}
                                         for k, v in r["judge"]["shocks"].items()}}
                             for w, r in zip(windows, reps)]
    print(format_report(rep))
    for pw in rep.get("per_window", []):
        print(f"  window {pw['window']}: {pw['points']:.2f}/15 null_ok={pw['null_ok']}")
    out = Path(a.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, allow_nan=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
