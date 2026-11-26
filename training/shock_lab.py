"""Shock diagnostics for a checkpoint (model lane): the kit's causality checks plus the numbers behind them.

    python -m training.shock_lab runs/e5b_gradmag/best_post.pt [--raw] [--windows 6] [--gallery] [--json out.json]

For each shock and window: kit flags (nonzero/localized/propagates/stable/no_teleport), near_frac_day1,
far_frac_max (teleport), growth = mean outside-energy days 4-8 / day-1 outside energy (kit 'grows' needs > 1.5),
peak |response| in std (and which channel/day), and the fade day (first day the in-box response falls below 10%
of day 1). --raw disables the post-processing (blend + rain map) to separate model effects from post effects.
Windows: evenly spaced 2019-2021 held-out days, plus real post-2023 windows in data/eval2023/ if present.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
from monsoon_r1 import scoring as S  # noqa: E402
from monsoon_r1.contract import BOX_ARABIAN_SEA, BOX_BAY_OF_BENGAL, CHANNELS, SCORE_STEPS, box_mask, doy_sincos  # noqa: E402
from monsoon_r1.stats import channel_stats  # noqa: E402

from training.dataset import load_nc  # noqa: E402
from twin_model.twin import Twin  # noqa: E402

C = {c: i for i, c in enumerate(CHANNELS)}


def box_delta(ch, box, val=None, frac=None, last=None):
    p = np.zeros((6, 32, 36), np.float32); m = box_mask(box)
    p[C[ch]][m] = val if frac is None else frac * last[C[ch]][m]
    return p


# Gallery: physically meaningful shocks a judge might paint (box = lat0, lat1, lon0, lon1 on .5-degree centres).
GALLERY = [
    ("heatwave_NW_India_+4K", "t2m", (25, 32, 72, 82), {"val": 4.0}),
    ("westerly_burst_+5ms", "u10", (10, 16, 66, 74), {"val": 5.0}),
    ("bay_depression_-1000Pa", "msl", (15, 21, 86, 92), {"val": -1000.0}),
    ("bay_depression_deep_-1500Pa", "msl", (15, 21, 86, 92), {"val": -1500.0}),
    ("dry_spell_central_India_-30%tcwv", "tcwv", (18, 26, 76, 84), {"frac": -0.3}),
    ("moist_surge_west_coast_+20%tcwv", "tcwv", (10, 20, 70, 76), {"frac": 0.2}),
]


def kit_shocks(last, std):
    return [("warm_ocean", BOX_BAY_OF_BENGAL, S.make_shock("warm_ocean", last, std)),
            ("monsoon_stall", BOX_ARABIAN_SEA, S.make_shock("monsoon_stall", last, std))]


def gallery_shocks(last):
    return [(n, b, box_delta(ch, b, last=last, **kw)) for n, ch, b, kw in GALLERY]


def diagnose(base, shock, box, std):
    r = S.causality_checks(base, shock, box, std)
    d = shock - base
    e = S._energy(d, std); near = S.near_mask(box); inbox = box_mask(box)
    out_e = np.array([e[k][~near].sum() for k in range(len(e))])
    in_e = np.array([e[k][inbox].sum() for k in range(len(e))])
    z = np.abs(d / std[None, :, None, None])
    k, c = np.unravel_index(np.argmax(z.max(axis=(2, 3))), z.shape[:2])
    fade = next((i + 1 for i in range(len(in_e)) if in_e[i] < 0.1 * max(in_e[0], 1e-12)), None)
    return {"nonzero": r["nonzero"], "localized": r["localized"], "propagates": r["propagates"],
            "stable": r["stable"], "no_teleport": r["no_teleport"], "near_frac_day1": r["near_frac_day1"],
            "far_frac_max": r["far_frac_max"], "growth": float(out_e[3:8].mean() / max(out_e[0], 1e-12)),
            "peak_std": float(z.max()), "peak_at": f"{CHANNELS[c]} d{k + 1}", "fade_day": fade}


def windows(states, dates, n):
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    idx = np.where((years >= 2019) & (years <= 2021))[0]
    idx = idx[(idx >= 2)]
    pick = idx[np.linspace(0, len(idx) - 1, n).round().astype(int)]
    out = [(dates[i].astype(object), states[i - 2:i + 1]) for i in pick]
    for f in sorted(Path("data/eval2023").glob("*.npz")):
        x = np.load(f)["data"].astype(np.float32); x[:, 1] = np.maximum(x[:, 1], 0)
        out.append((dt.datetime.strptime(f.stem, "%Y%m%d").date(), x[:3]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--raw", action="store_true")
    ap.add_argument("--windows", type=int, default=6); ap.add_argument("--gallery", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    states, dates, _, _ = load_nc("data/india_climate_train.nc")
    std = np.array(channel_stats(states)["std"], "f4")
    twin = Twin(a.ckpt); assert not twin.fallback
    if a.raw:
        twin.post = None
    rows = []
    for day, init in windows(states, dates, a.windows):
        dsc = doy_sincos(day.timetuple().tm_yday)
        base = twin.rollout(init, dsc, SCORE_STEPS, None)
        shocks = kit_shocks(init[-1], std) + (gallery_shocks(init[-1]) if a.gallery else [])
        for name, box, p in shocks:
            rows.append({"window": str(day), "shock": name, **diagnose(base, twin.rollout(init, dsc, SCORE_STEPS, p), box, std)})
    print(f"{a.ckpt} {'(RAW, no post)' if a.raw else '(with post)'}: {len({r['window'] for r in rows})} windows")
    names = list(dict.fromkeys(r["shock"] for r in rows))
    print(f"{'shock':34s} local  prop  stable telep  near1  farmax  growth  peak_std (where)      fade")
    for n in names:
        rs = [r for r in rows if r["shock"] == n]; k = len(rs)
        f = lambda key: sum(bool(r[key]) for r in rs)
        fades = [r["fade_day"] for r in rs if r["fade_day"]]
        worst = max(rs, key=lambda r: r["peak_std"])
        print(f"{n:34s} {f('localized'):2d}/{k:<2d} {f('propagates'):2d}/{k:<2d} {f('stable'):2d}/{k:<2d} {f('no_teleport'):2d}/{k:<2d}"
              f"  {np.mean([r['near_frac_day1'] for r in rs]):.2f}  {max(r['far_frac_max'] for r in rs):.3f}"
              f"  {np.median([r['growth'] for r in rs]):6.2f}  {worst['peak_std']:5.2f} ({worst['peak_at']:8s})"
              f"  {('d' + str(int(np.median(fades)))) if fades else '>10'} ({len(fades)}/{k})")
    # expected kit causality points per window (2 kit shocks; null consistency is exact by construction)
    pts = []
    for w in dict.fromkeys(r["window"] for r in rows):
        ks = [r for r in rows if r["window"] == w and r["shock"] in ("warm_ocean", "monsoon_stall")]
        pts.append(S.causality_points(ks, True))
    print(f"kit causality points over {len(pts)} windows: mean {np.mean(pts):.2f}/15, min {min(pts):.2f}, max {max(pts):.2f}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
