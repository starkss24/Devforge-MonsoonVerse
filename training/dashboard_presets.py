"""Measure the dashboard's exact shock presets (dashboard/sidebar.py DEMO_SHOCKS + kit shocks) on a checkpoint.

    python -m training.dashboard_presets runs/ens2_e5b_e9b/pv/cap05_slope15.pt runs/e5b_gradmag/best_post.pt

Boxes and magnitudes are copied here (not imported: the model lane must not import dashboard code).
Per preset: local on day 1, spreads (kit check), stable, peak |response| in std and which channel, fade day of the
in-box response (< 10% of day 1), the in-box mean change of the shocked channel at days 1/4/8/10, and the largest
rain increase anywhere (mm/day).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
from monsoon_r1 import scoring as S  # noqa: E402
from monsoon_r1.contract import BOX_ARABIAN_SEA, BOX_BAY_OF_BENGAL, CHANNELS, SCORE_STEPS, box_mask, doy_sincos  # noqa: E402
from monsoon_r1.stats import channel_stats  # noqa: E402

from training.dataset import load_nc  # noqa: E402
from training.shock_lab import diagnose, windows  # noqa: E402
from twin_model.twin import Twin  # noqa: E402

C = {c: i for i, c in enumerate(CHANNELS)}
# dashboard/sidebar.py DEMO_SHOCKS (lat0, lat1, lon0, lon1 on cell centres)
DEMO = {
    "heatwave": ("t2m", 4.0, (24.5, 30.5, 73.5, 81.5)),
    "westerly_burst": ("u10", 5.0, (10.5, 18.5, 65.5, 73.5)),
    "bay_low": ("msl", -1500.0, (15.5, 20.5, 85.5, 90.5)),
}


def presets(last, std):
    out = [("warm_ocean", "t2m", BOX_BAY_OF_BENGAL, S.make_shock("warm_ocean", last, std)),
           ("monsoon_stall", "u10", BOX_ARABIAN_SEA, S.make_shock("monsoon_stall", last, std))]
    for name, (ch, mag, box) in DEMO.items():
        p = np.zeros((6, 32, 36), np.float32); p[C[ch]][box_mask(box)] = mag
        out.append((name, ch, box, p))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ckpts", nargs="+"); ap.add_argument("--windows", type=int, default=8)
    a = ap.parse_args()
    states, dates, _, _ = load_nc("data/india_climate_train.nc")
    std = np.array(channel_stats(states)["std"], "f4")
    wins = windows(states, dates, a.windows)
    for path in a.ckpts:
        twin = Twin(path); assert not twin.fallback
        rows = {}
        for day, init in wins:
            dsc = doy_sincos(day.timetuple().tm_yday)
            base = twin.rollout(init, dsc, SCORE_STEPS, None)
            for name, ch, box, p in presets(init[-1], std):
                sh = twin.rollout(init, dsc, SCORE_STEPS, p)
                d = diagnose(base, sh, box, std)
                m = box_mask(box)
                d["inbox"] = [float((sh[k - 1, C[ch]] - base[k - 1, C[ch]])[m].mean()) for k in (1, 4, 8, 10)]
                d["rain_up"] = float((sh[:, 1] - base[:, 1]).max())
                rows.setdefault(name, []).append(d)
        print(f"\n{path} ({len(wins)} windows)")
        print(f"{'preset':15s} local spreads stable  peak std (where)  fade    in-box mean change d1/d4/d8/d10      max rain up")
        for name, rs in rows.items():
            k = len(rs); f = lambda key: sum(bool(r[key]) for r in rs)
            fades = [r["fade_day"] for r in rs if r["fade_day"]]
            ib = np.mean([r["inbox"] for r in rs], axis=0)
            worst = max(rs, key=lambda r: r["peak_std"])
            print(f"{name:15s} {f('localized'):2d}/{k} {f('propagates'):3d}/{k} {f('stable'):3d}/{k}  {worst['peak_std']:5.2f} ({worst['peak_at']:7s})"
                  f"  {('d' + str(int(np.median(fades))) + f' ({len(fades)}/{k})') if fades else 'none   '}"
                  f"  {ib[0]:+8.2f} {ib[1]:+8.2f} {ib[2]:+8.2f} {ib[3]:+8.2f}   {max(r['rain_up'] for r in rs):6.1f}")


if __name__ == "__main__":
    main()
