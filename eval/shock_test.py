"""Shock-locality report for the dynamics score (and the dashboard's Shock Lab).

For each grader shock (warm_ocean, monsoon_stall, zero) this runs a baseline and a
shocked rollout and reports, per channel and lead day:
  - mean |diff| inside vs outside the shock box, and their ratio
  - a radial profile: mean |diff| by Chebyshev distance (cells) from the box edge,
    0 = inside the box, 1..8 = rings outside it
  - the profile's centroid distance (grows as the shock spreads)

Hard checks (never loosen):
  - zero shock: max |diff| == 0 exactly
  - zero shock: max |diff| <= NULL_SHOCK_TOL_FRAC_STD * channel std (contract rule)
  - perturbation=None and perturbation=zeros give identical output (determinism)

Works with any rollout_fn that has Twin.rollout's signature, including
http_rollout_fn(url) which talks to serve.py's /simulate.

CLI:
  python -m eval.shock_test [--server URL] [--window synthetic|path.npz]
                            [--stats stats.json] [--json-out eval/shock_report.json]
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

from common import (  # noqa: E402
    BOXES,
    CH,
    CHANNELS,
    INIT_SHAPE,
    NLAT,
    NLON,
    NULL_SHOCK_TOL_FRAC_STD,
    STATE_SHAPE,
    box_mask,
    doy_sincos as make_doy_sincos,
    make_shock,
)

RolloutFn = Callable[..., np.ndarray]

SHOCKS = ("warm_ocean", "monsoon_stall", "zero")
REPORT_DAYS = (1, 4, 8)
MAX_RING = 8


class ShockCheckError(AssertionError):
    """A hard dynamics check failed (zero shock or determinism)."""


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


def load_channel_std(path: str | Path) -> np.ndarray:
    """Per-channel std from a stats.json. Accepts the same layouts as eval.score.load_stats_std:
    {"std": [6 values]}, {"std": {ch: v}} or {ch: {"std": v}}.
    The kit's exact layout is unknown until it arrives; fail loudly if none fits."""
    s = json.loads(Path(path).read_text())
    if isinstance(s.get("std"), list) and len(s["std"]) == len(CHANNELS):
        return np.asarray(s["std"], dtype=np.float64)
    if isinstance(s.get("std"), dict):
        return np.array([float(s["std"][c]) for c in CHANNELS])
    if all(isinstance(s.get(c), dict) and "std" in s[c] for c in CHANNELS):
        return np.array([float(s[c]["std"]) for c in CHANNELS])
    raise ValueError(f"cannot find per-channel std in {path}; update load_channel_std")


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


def _jsonable(v: float) -> Optional[float]:
    return None if (v is None or not math.isfinite(v)) else float(v)


# ---------------------------------------------------------------- report
def shock_report(
    rollout_fn: RolloutFn,
    init_states: np.ndarray,
    doy_sincos: np.ndarray,
    steps: int = 10,
    channel_std: Optional[np.ndarray] = None,
    shocks: tuple[str, ...] = SHOCKS,
    days: tuple[int, ...] = REPORT_DAYS,
) -> dict:
    """Run baseline + shocked rollouts and build the locality report.

    channel_std: per-channel std for the null-shock tolerance (from stats.json).
    If None it is estimated from init_states (spatial std over the 3 input days).
    Raises ShockCheckError if the zero-shock or determinism check fails.
    """
    init = np.asarray(init_states, dtype=np.float64)
    if init.shape != INIT_SHAPE:
        raise ValueError(f"init_states must be {INIT_SHAPE}, got {init.shape}")
    dsc = np.asarray(doy_sincos, dtype=np.float64).reshape(-1)
    steps = int(steps)
    days = tuple(d for d in days if 1 <= d <= steps)
    if channel_std is None:
        channel_std = init.transpose(1, 0, 2, 3).reshape(len(CHANNELS), -1).std(axis=1)
        std_source = "init_states"
    else:
        std_source = "stats"
    channel_std = np.asarray(channel_std, dtype=np.float64)
    tol = NULL_SHOCK_TOL_FRAC_STD * channel_std

    baseline = np.asarray(rollout_fn(init, dsc, steps=steps, perturbation=None), dtype=np.float64)
    if baseline.shape != (steps,) + STATE_SHAPE:
        raise ValueError(f"rollout returned {baseline.shape}")
    if not np.all(np.isfinite(baseline)):
        raise ShockCheckError("baseline rollout contains NaN/Inf")

    # Determinism: None vs explicit zeros must be bit-identical.
    zeros_run = np.asarray(rollout_fn(init, dsc, steps=steps,
                                      perturbation=np.zeros(STATE_SHAPE)), dtype=np.float64)
    det_max = float(np.max(np.abs(zeros_run - baseline)))
    if det_max != 0.0:
        raise ShockCheckError(f"perturbation=None vs zeros differ: max |diff| = {det_max:g}")

    report = {
        "steps": steps,
        "days": list(days),
        "max_ring": MAX_RING,
        "std_source": std_source,
        "channel_std": {c: float(s) for c, s in zip(CHANNELS, channel_std)},
        "determinism": {"max_abs_diff_none_vs_zeros": det_max, "pass": True},
        "shocks": {},
    }

    for name in shocks:
        delta = make_shock(name, init[-1])
        shocked = np.asarray(rollout_fn(init, dsc, steps=steps, perturbation=delta),
                             dtype=np.float64)
        if not np.all(np.isfinite(shocked)):
            raise ShockCheckError(f"{name}: shocked rollout contains NaN/Inf")
        diff = np.abs(shocked - baseline)  # [steps, 6, 32, 36]

        if name == "zero":
            max_diff = float(diff.max())
            per_ch = diff.max(axis=(0, 2, 3))
            if max_diff != 0.0:
                raise ShockCheckError(f"zero shock changed the output: max |diff| = {max_diff:g}")
            bad = [c for c, m, t in zip(CHANNELS, per_ch, tol) if m > t]
            if bad:
                raise ShockCheckError(f"zero shock exceeds 1e-3*std on {bad}")
            report["shocks"][name] = {
                "max_abs_diff": max_diff,
                "max_abs_diff_by_channel": {c: float(m) for c, m in zip(CHANNELS, per_ch)},
                "tolerance_by_channel": {c: float(t) for c, t in zip(CHANNELS, tol)},
                "pass": True,
            }
            continue

        mask = box_mask(*BOXES[name])
        dist = ring_distance(mask)
        perturbed = [c for c in CHANNELS if np.any(delta[CH[c]] != 0)]
        # Report perturbed channels plus tp (the rainfall response the story is about).
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
            "box": list(BOXES[name]),
            "n_cells_inside": int(mask.sum()),
            "perturbed_channels": perturbed,
            "max_abs_delta": {c: float(np.abs(delta[CH[c]]).max()) for c in perturbed},
            "channels": ch_out,
        }
    return report


# ---------------------------------------------------------------- CLI
def format_report(rep: dict) -> str:
    lines = [f"Shock report  steps={rep['steps']}  std from {rep['std_source']}",
             f"determinism (None vs zeros): max|diff| = {rep['determinism']['max_abs_diff_none_vs_zeros']:g}  PASS"]
    for name, s in rep["shocks"].items():
        if name == "zero":
            lines.append(f"\n[zero] max|diff| = {s['max_abs_diff']:g}  (must be exactly 0)  PASS")
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
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--server", default=None, help="serve.py base URL; default in-process Twin")
    ap.add_argument("--window", default="synthetic", help="'synthetic' or .npz with init_states, doy")
    ap.add_argument("--stats", default=None, help="stats.json with per-channel std")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=str(ROOT / "eval" / "shock_report.json"))
    a = ap.parse_args(argv)

    if a.window == "synthetic":
        init, doy = synthetic_window(seed=a.seed)
    else:
        init, doy = load_window(a.window)
    if a.server:
        fn = http_rollout_fn(a.server)
    else:
        from twin import Twin
        fn = Twin().rollout
    std = load_channel_std(a.stats) if a.stats else None

    rep = shock_report(fn, init, make_doy_sincos(doy), steps=a.steps, channel_std=std)
    rep["source"] = a.server or "in-process Twin"
    rep["window"] = a.window
    print(format_report(rep))
    out = Path(a.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, allow_nan=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
