"""Checkpoint evaluator that scores with the hackathon kit's OWN code (monsoon_r1.scoring / stats / contract).

    python -m training.kit_eval runs/<run>/best.pt [more ckpts...] [--windows N]

CPU only (CUDA hidden before torch is imported). Appends one row per checkpoint to kit_scores.csv in the repo root.

Places where the kit does not pin the procedure (guesses, see notes in the code marked GUESS):
  * RMSE/CSI skill is computed per window with the 3 scored leads (1,4,8) pooled, then averaged over windows.
  * Per-window RMSE skill scalar = mean over the 6 channels of skill_vs_best_ref()["ss"].
  * Anti-blur uses day 8 (rollout[7]) ratios from every window, averaged by anti_blur_points itself.
  * Causality: the kit scores ONE window; here it is scored on 3 windows and the points are averaged.
  * Climatology for a target date = clim_for(clim, that date); normalisation std = stats.channel_stats of train years.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # must precede the torch import

import argparse
import csv
import datetime as dt
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)

from monsoon_r1 import scoring as S
from monsoon_r1.contract import (BOX_ARABIAN_SEA, BOX_BAY_OF_BENGAL, CHANNELS, LEAD_DAYS, SCORE_STEPS, doy_sincos)
from monsoon_r1.stats import channel_stats, clim_for, climatology

from training.dataset import load_nc
from twin_model.twin import Twin

ROOT = Path(__file__).resolve().parent.parent
SHOCKS = (("warm_ocean", BOX_BAY_OF_BENGAL), ("monsoon_stall", BOX_ARABIAN_SEA))
CSV_PATH = ROOT / "kit_scores.csv"


def _d(x) -> dt.date:
    return x.astype("datetime64[D]").astype(object)


def prepare(data_path: str, n_windows: int):
    states, dates, _static, _names = load_nc(data_path)
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    tr = (years >= 2005) & (years <= 2018)
    ymd = np.array([int(str(d).replace("-", "")) for d in dates[tr].astype("datetime64[D]")])
    clim = climatology(states[tr], ymd)  # kit default smoothing (15 d)
    std = np.array(channel_stats(states[tr])["std"], "f4")
    # valid starts: 3 input + 10 truth consecutive days, all in validation years
    need = 3 + SCORE_STEPS
    val = (years >= 2019) & (years <= 2021)
    day = dates.astype("datetime64[D]").astype(int)
    starts = [s for s in range(len(dates) - need + 1)
              if val[s] and val[s + need - 1] and day[s + need - 1] - day[s] == need - 1]
    pick = np.unique(np.linspace(0, len(starts) - 1, n_windows).round().astype(int))
    return states, dates, clim, std, [starts[i] for i in pick]


def evaluate(path: str, ctx, n_windows: int) -> dict:
    states, dates, clim, std, starts = ctx
    twin = Twin(path)
    assert not twin.fallback, f"{path}: Twin fell back to persistence"
    leads = LEAD_DAYS
    nw = len(starts)
    ss_w, csi_w, ratios, bad_all = [], [], [], []
    ss_ch = np.zeros((nw, len(leads), 6)); ss_persist_ch = np.zeros((nw, len(leads), 6))
    for w, s in enumerate(starts):
        init = states[s:s + 3]
        last = _d(dates[s + 2])
        dsc = doy_sincos(last.timetuple().tm_yday)  # 1-based day of year, same as training/dataset.py
        out = twin.rollout(init, dsc, SCORE_STEPS, None)
        bad_all += S.validity(out, SCORE_STEPS)
        truth = states[s + 3:s + 3 + SCORE_STEPS]
        idx = [k - 1 for k in leads]
        pred = out[idx]; tru = truth[idx]
        persist = np.repeat(init[-1][None], len(leads), 0)
        cl = np.stack([clim_for(clim, _d(dates[s + 2 + k])) for k in leads])
        r = S.skill_vs_best_ref(pred, tru, persist, cl)
        ss_w.append(float(np.nanmean(r["ss"])))
        for li in range(len(leads)):  # per-lead per-channel detail
            rl = S.skill_vs_best_ref(pred[li:li + 1], tru[li:li + 1], persist[li:li + 1], cl[li:li + 1])
            ss_ch[w, li] = rl["ss"]; ss_persist_ch[w, li] = rl["ss_persist"]
        c = lambda a: S.csi(a[:, 1], tru[:, 1])
        csi_w.append(S.csi_skill(c(pred), c(persist), c(cl)))
        ratios.append(S.variance_ratio(out[7], truth[7]))
    fp = S.forecast_points(ss_w, csi_w)
    ratios = np.array(ratios)
    blur_pts = S.anti_blur_points(ratios)
    ratio_mean = np.nanmean(ratios, axis=0)

    # persistence gate at day 4: pooled over windows, per channel, then mean over channels
    li4 = list(leads).index(4)
    beats_persist4 = bool(np.nanmean(np.nanmean(ss_persist_ch[:, li4], axis=0)) > 0)

    # perturbation causality on 3 windows spread over the set
    cw = [starts[i] for i in sorted(set(np.linspace(0, nw - 1, 5).round().astype(int)[1:4]))]
    cpts, flags = [], {n: [] for n, _ in SHOCKS}
    nonzero_local = False
    for s in cw:
        init = states[s:s + 3]
        dsc = doy_sincos(_d(dates[s + 2]).timetuple().tm_yday)
        base = twin.rollout(init, dsc, SCORE_STEPS, None)
        nullr = twin.rollout(init, dsc, SCORE_STEPS, S.make_shock("null", init[-1], std))
        null_ok = S.null_consistent(base, nullr, std)
        res = []
        for name, box in SHOCKS:
            sh = twin.rollout(init, dsc, SCORE_STEPS, S.make_shock(name, init[-1], std))
            bad_all += S.validity(sh, SCORE_STEPS)
            cc = S.causality_checks(base, sh, box, std)
            res.append(cc)
            flags[name].append(cc)
            nonzero_local |= bool(cc["nonzero"] and cc["localized"])
        cpts.append(S.causality_points(res, null_ok))
        flags.setdefault("null_ok", []).append(null_ok)
    caus_pts = float(np.mean(cpts))
    flag_sum = {}
    for n, _ in SHOCKS:
        for k in ("nonzero", "localized", "propagates", "stable"):
            flag_sum[f"{n}_{k}"] = f"{sum(bool(x[k]) for x in flags[n])}/{len(cw)}"
    null_ok_frac = f"{sum(flags['null_ok'])}/{len(cw)}"

    # latency of the 10-step rollout (4 threads, CPU)
    init = states[starts[0]:starts[0] + 3]; dsc = doy_sincos(_d(dates[starts[0] + 2]).timetuple().tm_yday)
    twin.rollout(init, dsc, SCORE_STEPS, None)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); twin.rollout(init, dsc, SCORE_STEPS, None); ts.append((time.perf_counter() - t0) * 1000)
    lat = float(np.median(ts))

    return {
        "ckpt": path, "windows": nw, "causality_windows": len(cw),
        "rmse_pts": fp["rmse_pts"], "csi_pts": fp["csi_pts"], "mean_ss": fp["mean_ss"], "mean_csi_skill": fp["mean_csi_skill"],
        "blur_pts": blur_pts, "caus_pts": caus_pts,
        "total": fp["points"] + blur_pts + caus_pts,
        "beats_persist_d4": beats_persist4, "valid": not bad_all, "violations": ";".join(sorted(set(bad_all))),
        "nonzero_local_shock": nonzero_local,
        "var_ratio": dict(zip(CHANNELS, ratio_mean.tolist())),
        "ss_ch": {f"d{l}": dict(zip(CHANNELS, np.nanmean(ss_ch[:, i], axis=0).tolist())) for i, l in enumerate(leads)},
        "flags": flag_sum, "null_ok": null_ok_frac, "latency_ms": lat,
    }


def report(r: dict):
    print(f"\n=== {r['ckpt']}  ({r['windows']} windows, causality on {r['causality_windows']}) ===")
    print(f"forecast: RMSE {r['rmse_pts']:.2f}/30 (mean SS {r['mean_ss']:+.4f}, ref 0.12)  CSI {r['csi_pts']:.2f}/15 "
          f"(mean CSI skill {r['mean_csi_skill']:+.4f}, ref 0.10)")
    print(f"dynamics: anti-blur {r['blur_pts']:.2f}/10  causality {r['caus_pts']:.2f}/15 (null ok {r['null_ok']})")
    print("day-8 variance ratio: " + "  ".join(f"{k}={v:.2f}" for k, v in r["var_ratio"].items()))
    print("shock flags (windows passing): " + "  ".join(f"{k}={v}" for k, v in r["flags"].items()))
    print(f"TOTAL {r['total']:.2f}/70 | gates: beats_persist_d4={r['beats_persist_d4']} valid={r['valid']} "
          f"{r['violations']} nonzero_local_shock={r['nonzero_local_shock']} | latency {r['latency_ms']:.0f} ms")
    for l, d in r["ss_ch"].items():
        print(f"  SS {l}: " + "  ".join(f"{k}={v:+.3f}" for k, v in d.items()))


def append_csv(rows: list[dict]):
    cols = ["time", "ckpt", "windows", "rmse_pts", "csi_pts", "blur_pts", "caus_pts", "total", "mean_ss", "mean_csi_skill",
            "beats_persist_d4", "valid", "nonzero_local_shock", "latency_ms"] + \
           [f"vr_{c}" for c in CHANNELS] + [f"ss_{l}_{c}" for l in ("d1", "d4", "d8") for c in CHANNELS]
    new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(cols)
        for r in rows:
            d = {**r, "time": dt.datetime.now().isoformat(timespec="seconds")}
            d.update({f"vr_{c}": v for c, v in r["var_ratio"].items()})
            d.update({f"ss_{l}_{c}": v for l, dd in r["ss_ch"].items() for c, v in dd.items()})
            wr.writerow([f"{d[c]:.5g}" if isinstance(d[c], float) else d[c] for c in cols])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--data", default="data/india_climate_train.nc")
    a = ap.parse_args()
    ctx = prepare(a.data, a.windows)
    rows = []
    for c in a.ckpts:
        t0 = time.time(); r = evaluate(c, ctx, a.windows); r["secs"] = time.time() - t0
        report(r); rows.append(r)
    if len(rows) > 1:
        print("\n%-28s %7s %6s %6s %6s %7s  %s" % ("ckpt", "rmse/30", "csi/15", "blur", "caus", "total", "gates(p4,valid,shock) ms"))
        for r in rows:
            print("%-28s %7.2f %6.2f %6.2f %6.2f %7.2f  %s,%s,%s %.0f" % (
                r["ckpt"][-28:], r["rmse_pts"], r["csi_pts"], r["blur_pts"], r["caus_pts"], r["total"],
                int(r["beats_persist_d4"]), int(r["valid"]), int(r["nonzero_local_shock"]), r["latency_ms"]))
    append_csv(rows)


if __name__ == "__main__":
    main()
