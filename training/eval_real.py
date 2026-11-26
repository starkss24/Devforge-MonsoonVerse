"""Score checkpoints on REAL post-2023 windows (out of sample for every model we train), with the kit's scoring.

    python -m training.eval_real runs/e5b_gradmag/best_post.pt runs/<other>/best_post.pt [--fetch 2024-01-20,...]

Windows live in data/eval2023/<last-input-day YYYYMMDD>.npz, array "data" [6,6,32,36]:
3 input days (last input day = file name) + truth at lead days 1, 4, 8. data/ is git-ignored.
--fetch downloads extra windows with the kit's own ARCO-ERA5 pipeline (~700 MB each, public bucket).
Climatology for the reference = kit climatology of the whole training archive (2005-2021), like the kit's artifacts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
from monsoon_r1 import scoring as S  # noqa: E402
from monsoon_r1.contract import BOX_ARABIAN_SEA, BOX_BAY_OF_BENGAL, CHANNELS, LEAD_DAYS, SCORE_STEPS, doy_sincos  # noqa: E402
from monsoon_r1.stats import channel_stats, clim_for, climatology  # noqa: E402

from training.dataset import load_nc  # noqa: E402
from twin_model.twin import Twin  # noqa: E402

DIR = Path("data/eval2023")
SHOCKS = [("warm_ocean", BOX_BAY_OF_BENGAL), ("monsoon_stall", BOX_ARABIAN_SEA)]


def fetch(last: dt.date) -> None:
    """Download one window with the kit's ARCO pipeline (the store is zarr v2 on disk: open it explicitly)."""
    import fsspec, zarr
    import monsoon_r1.build_dataset as B
    B._open = lambda url: zarr.open_group(store=fsspec.get_mapper(url, token="anon"), mode="r", zarr_format=2)
    days = [last + dt.timedelta(d) for d in (-2, -1, 0, *LEAD_DAYS)]
    t = time.time(); x = B.arco_daily(days, workers=16)
    DIR.mkdir(parents=True, exist_ok=True); np.savez(DIR / f"{last:%Y%m%d}.npz", data=x)
    print(f"fetched {last} in {time.time() - t:.0f}s", flush=True)


def windows():
    out = []
    for f in sorted(DIR.glob("*.npz")):
        last = dt.datetime.strptime(f.stem, "%Y%m%d").date()
        x = np.load(f)["data"].astype(np.float32)
        assert np.isfinite(x).all(), f"non-finite data in {f}"
        x[:, 1] = np.maximum(x[:, 1], 0)                 # as the kit's merge step
        out.append((last, x[:3], x[3:]))
    return out


def score(path, wins, clim, std):
    twin = Twin(path); assert not twin.fallback, f"{path} fell back to persistence"
    ss_w, csi_w, ratios, bad, cpts, per = [], [], [], [], [], []
    for last, init, tru in wins:
        dsc = doy_sincos(last.timetuple().tm_yday)
        out = twin.rollout(init, dsc, SCORE_STEPS, None); bad += S.validity(out, SCORE_STEPS)
        pred = out[[k - 1 for k in LEAD_DAYS]]
        per_ = np.repeat(init[-1][None], len(LEAD_DAYS), 0)
        cl = np.stack([clim_for(clim, last + dt.timedelta(k)) for k in LEAD_DAYS])
        r = S.skill_vs_best_ref(pred, tru, per_, cl); ss_w.append(float(np.nanmean(r["ss"])))
        c = lambda a: S.csi(a[:, 1], tru[:, 1]); csi_w.append(S.csi_skill(c(pred), c(per_), c(cl)))
        ratios.append(S.variance_ratio(pred[-1], tru[-1]))
        null_ok = S.null_consistent(out, twin.rollout(init, dsc, SCORE_STEPS, S.make_shock("null", init[-1], std)), std)
        res = [S.causality_checks(out, twin.rollout(init, dsc, SCORE_STEPS, S.make_shock(n, init[-1], std)), b, std)
               for n, b in SHOCKS]
        cpts.append(S.causality_points(res, null_ok))
        per.append(f"{last} SS {ss_w[-1]:+.3f} CSI {csi_w[-1]:+.3f}")
    fp = S.forecast_points(ss_w, csi_w); blur = S.anti_blur_points(np.array(ratios)); caus = float(np.mean(cpts))
    return {"ckpt": path, "rmse": fp["rmse_pts"], "csi": fp["csi_pts"], "blur": blur, "caus": caus,
            "total": fp["points"] + blur + caus, "ss": fp["mean_ss"], "csi_skill": fp["mean_csi_skill"],
            "valid": not bad, "per": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="*"); ap.add_argument("--fetch", default="")
    a = ap.parse_args()
    for d in [s for s in a.fetch.split(",") if s]:
        if not (DIR / f"{dt.date.fromisoformat(d):%Y%m%d}.npz").exists():
            fetch(dt.date.fromisoformat(d))
    wins = windows(); print(f"{len(wins)} real windows: {', '.join(str(w[0]) for w in wins)}")
    if not a.ckpts:
        return
    states, dates, _, _ = load_nc("data/india_climate_train.nc")
    ymd = np.array([int(str(d).replace("-", "")) for d in dates.astype("datetime64[D]")])
    clim = climatology(states, ymd); std = np.array(channel_stats(states)["std"], "f4")
    rows = [score(p, wins, clim, std) for p in a.ckpts]
    for r in rows:
        print(f"\n{r['ckpt']}\n  " + "\n  ".join(r["per"]))
    print(f"\n{'checkpoint':44s} rmse/30 csi/15 blur/10 caus/15  total/70  SS     CSIsk  valid")
    for r in rows:
        print(f"{r['ckpt']:44s} {r['rmse']:7.2f} {r['csi']:6.2f} {r['blur']:7.2f} {r['caus']:7.2f}  {r['total']:7.2f}"
              f"  {r['ss']:+.3f} {r['csi_skill']:+.3f} {r['valid']}")


if __name__ == "__main__":
    main()
