"""Fit the optional output post-processing (twin_model/post.py) on TRAINING years and store it in a checkpoint copy.

    python -m training.fit_post runs/<run>/best.pt --out runs/<run>/best_post.pt [--no-blend] [--no-qmap]

CPU only (CUDA hidden before torch is imported). Steps:
  1. Roll the model 10 days from a deterministic subsample of 2005-2018 windows (3 input + 10 truth consecutive days).
  2. Climatology blend: per lead k and channel c, a = argmin_a sum w*((1-a)*m + a*clim - truth)^2 (lat-weighted,
     closed form), clipped to [0, --blend-cap]. Climatology = monsoon_r1.stats.climatology on 2005-2018 (366 slots),
     looked up exactly as twin_model/post.py does at inference (slot from the last-input doy_sincos + k).
  3. Rain quantile map per lead, fitted on the BLENDED rain (the order Twin applies them): forecast vs observed tp
     quantiles, 0 -> 0, monotone. Qmap last so the final rain distribution (heavy-rain frequency, spatial variance)
     matches the observed one; blending after the qmap would shrink it back.
A held-out 2019-2021 subsample is scored before/after for information only (never used for fitting).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # must precede the torch import

import argparse
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)

from monsoon_r1.stats import climatology

from training.dataset import load_nc
from twin_model.model import rollout_std
from twin_model.norm import CHANNELS, N_IN, RAIN, from_model, to_model
from twin_model.post import Post, target_slots
from twin_model.twin import load_checkpoint

STEPS = 10
LAT = np.arange(6.5, 37.5 + 1e-9, 1.0)


def lat_w(h: int) -> np.ndarray:
    if h != len(LAT):  # non-standard grid (synthetic tests): unweighted
        return np.ones(h)
    w = np.cos(np.deg2rad(LAT))
    return (w / w.mean()).astype(np.float64)


def window_starts(dates, years, lo, hi, need):
    day = dates.astype("datetime64[D]").astype(np.int64)
    ok = (years >= lo) & (years <= hi)
    return np.array([s for s in range(len(dates) - need + 1)
                     if ok[s] and ok[s + need - 1] and day[s + need - 1] - day[s] == need - 1], dtype=np.int64)


def doy_sincos_of(date) -> np.ndarray:
    d = date.astype("datetime64[D]")
    doy = int((d - d.astype("datetime64[Y]").astype("datetime64[D]")).astype(int)) + 1
    a = 2 * np.pi * doy / 365.0
    return np.array([np.sin(a), np.cos(a)], dtype=np.float32)


@torch.inference_mode()
def model_rollouts(ck, states, dates, starts, batch=32):
    """Physical [N,STEPS,6,H,W] raw model output (exactly Twin's pre-post pipeline, no perturbation), doy [N,2]."""
    model, norm = ck["model"], ck["norm"]
    st = torch.from_numpy(np.ascontiguousarray(norm.standardize_static(ck["static"]), dtype=np.float32))
    outs, doys = [], []
    for i in range(0, len(starts), batch):
        s = starts[i:i + batch]
        x = states[s[:, None] + np.arange(N_IN)[None]].astype(np.float32)          # [B,3,6,H,W]
        d = np.stack([doy_sincos_of(dates[j + N_IN - 1]) for j in s])
        win = torch.from_numpy(np.ascontiguousarray(norm.standardize(to_model(x)), dtype=np.float32))
        o = rollout_std(model, win, st[None].expand(len(s), -1, -1, -1), torch.from_numpy(d), STEPS, norm)
        outs.append(from_model(norm.unstandardize(o.numpy().astype(np.float32))))
        doys.append(d)
    return np.concatenate(outs).astype(np.float32), np.concatenate(doys)


def clim_targets(clim, doys):
    return np.stack([clim[target_slots(d, STEPS)] for d in doys]).astype(np.float32)  # [N,STEPS,6,H,W]


def fit_blend(pred, truth, cl, w, cap):
    a = np.zeros((STEPS, len(CHANNELS)), np.float64)
    for k in range(STEPS):
        for c in range(len(CHANNELS)):
            m = pred[:, k, c].astype(np.float64); d = cl[:, k, c] - m; e = truth[:, k, c] - m
            den = (w * d * d).sum()
            a[k, c] = (w * d * e).sum() / den if den > 0 else 0.0
    return np.clip(a, 0.0, cap).astype(np.float32)


def quantile_levels(q: int) -> np.ndarray:
    n_body = q // 3
    body = np.linspace(0.0, 0.95, n_body, endpoint=False)
    tail = 1.0 - np.geomspace(0.05, 1e-4, q - n_body)
    return np.concatenate([body, tail])


def fit_qmap(fc_tp, ob_tp, q, eps=1e-4):
    """fc_tp, ob_tp: [N,STEPS,H,W]. Returns fq, oq float32 [STEPS,q]; fq[:,0]=oq[:,0]=0, fq strictly increasing."""
    p = quantile_levels(q)
    fq = np.zeros((STEPS, q)); oq = np.zeros((STEPS, q))
    for k in range(STEPS):
        f = np.quantile(fc_tp[:, k].ravel().astype(np.float64), p)
        o = np.quantile(ob_tp[:, k].ravel().astype(np.float64), p)
        f[0] = 0.0; o[0] = 0.0
        o = np.maximum.accumulate(np.maximum(o, 0.0))
        for j in range(1, q):
            f[j] = max(f[j], f[j - 1] + eps)
        fq[k], oq[k] = f, o
    fq, oq = fq.astype(np.float32), oq.astype(np.float32)
    # float32 rounding must not create ties
    for k in range(STEPS):
        for j in range(1, q):
            if fq[k, j] <= fq[k, j - 1]:
                fq[k, j] = np.nextafter(fq[k, j - 1], np.float32(np.inf))
    return fq, oq


def apply_all(post: Post, pred, doys):
    return np.stack([post.apply(pred[i], doys[i]) for i in range(len(pred))])


def summarize(tag, pred, truth, w, thr=10.0):
    se = ((pred - truth) ** 2 * w).mean(axis=(0, 3, 4))          # [STEPS,6] lat-weighted MSE
    rm = np.sqrt(se)
    p, t = pred[:, :, RAIN] > thr, truth[:, :, RAIN] > thr
    h = (p & t).sum(); m = (~p & t).sum(); f = (p & ~t).sum()
    vr = (pred[:, 7].reshape(len(pred), 6, -1).var(-1) / np.maximum(truth[:, 7].reshape(len(pred), 6, -1).var(-1), 1e-30)).mean(0)
    print(f"  {tag:10s} RMSE d1/d4/d8 " + "  ".join(f"{c}={rm[0, i]:.3g}/{rm[3, i]:.3g}/{rm[7, i]:.3g}" for i, c in enumerate(CHANNELS)))
    print(f"  {tag:10s} CSI@10 (d1-10 pooled) {h / max(h + m + f, 1):.4f} | day-8 var ratio "
          + " ".join(f"{c}={v:.2f}" for c, v in zip(CHANNELS, vr)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/india_climate_train.nc")
    ap.add_argument("--no-blend", action="store_true")
    ap.add_argument("--no-qmap", action="store_true")
    ap.add_argument("--windows", type=int, default=600, help="training-year windows to fit on (evenly spaced)")
    ap.add_argument("--val-windows", type=int, default=120, help="held-out 2019-2021 windows for the report only")
    ap.add_argument("--q", type=int, default=160, help="quantile knots per lead")
    ap.add_argument("--blend-cap", type=float, default=1.0, help="upper clip for blend weights")
    ap.add_argument("--blend-channels", default=",".join(CHANNELS), help="channels the blend may touch")
    ap.add_argument("--blend-min-lead", type=int, default=1, help="leads < this get a = 0")
    ap.add_argument("--fit-end", type=int, default=2018, help="last fitting year (2021 for an all-years model)")
    a = ap.parse_args()
    if a.no_blend and a.no_qmap:
        ap.error("nothing to fit (--no-blend and --no-qmap)")
    t0 = time.time()

    ck = load_checkpoint(a.ckpt)
    states, dates, _static, _names = load_nc(a.data)
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    tr = (years >= 2005) & (years <= a.fit_end)
    ymd = np.array([int(str(d).replace("-", "")) for d in dates[tr].astype("datetime64[D]")])
    clim = climatology(states[tr], ymd).astype(np.float32)  # kit default (15-day smoothing), [366,6,H,W]
    w = lat_w(states.shape[-2])[:, None]

    need = N_IN + STEPS
    st_tr = window_starts(dates, years, 2005, a.fit_end, need)
    st_va = window_starts(dates, years, 2019, 2021, need)
    pick = lambda s, n: s[np.unique(np.linspace(0, len(s) - 1, min(n, len(s))).round().astype(int))]
    st_tr, st_va = pick(st_tr, a.windows), pick(st_va, a.val_windows)
    print(f"fit windows {len(st_tr)} (2005-{a.fit_end}), report windows {len(st_va)} (2019-2021)")

    pred, doys = model_rollouts(ck, states, dates, st_tr)
    truth = states[st_tr[:, None] + N_IN + np.arange(STEPS)[None]].astype(np.float32)
    cl = clim_targets(clim, doys)
    print(f"rollouts done ({time.time() - t0:.0f} s)")

    blend = None
    if not a.no_blend:
        blend = fit_blend(pred, truth, cl, w, a.blend_cap)
        allowed = [c.strip() for c in a.blend_channels.split(",") if c.strip()]
        for i, c in enumerate(CHANNELS):
            if c not in allowed:
                blend[:, i] = 0.0
        blend[: max(a.blend_min_lead - 1, 0)] = 0.0
        print("blend weights a[k,c] (rows = lead 1..10):")
        print("      " + " ".join(f"{c:>6s}" for c in CHANNELS))
        for k in range(STEPS):
            print(f"  d{k + 1:<3d}" + " ".join(f"{v:6.3f}" for v in blend[k]))
    post_b = Post(blend=blend, clim=clim if blend is not None else None)
    fq = oq = None
    if not a.no_qmap:
        fc = apply_all(post_b, pred, doys) if blend is not None else pred
        fq, oq = fit_qmap(fc[:, :, RAIN], truth[:, :, RAIN], a.q)
        print("qmap: forecast -> observed tp at p=0.99 / 0.999 per lead: "
              + " ".join(f"d{k + 1}:{np.interp(0.99, quantile_levels(a.q), fq[k]):.1f}->{np.interp(0.99, quantile_levels(a.q), oq[k]):.1f}"
                         f"/{np.interp(0.999, quantile_levels(a.q), fq[k]):.0f}->{np.interp(0.999, quantile_levels(a.q), oq[k]):.0f}"
                         for k in range(STEPS)))

    meta = {"fit_years": f"2005-{a.fit_end}", "fit_windows": int(len(st_tr)), "q": int(a.q), "blend_cap": float(a.blend_cap),
            "blend_channels": a.blend_channels, "blend_min_lead": int(a.blend_min_lead),
            "order": "blend then rain qmap", "source_ckpt": str(a.ckpt),
            "clim": f"monsoon_r1.stats.climatology(2005-{a.fit_end}, smooth_days=15)"}
    post = Post(blend=blend, clim=clim if blend is not None else None, qmap_fq=fq, qmap_oq=oq, meta=meta)

    print("training-year subsample:")
    summarize("raw", pred, truth, w); summarize("post", apply_all(post, pred, doys), truth, w)
    if len(st_va):
        pv, dv = model_rollouts(ck, states, dates, st_va)
        tv = states[st_va[:, None] + N_IN + np.arange(STEPS)[None]].astype(np.float32)
        print("held-out 2019-2021 subsample (not used for fitting):")
        summarize("raw", pv, tv, w); summarize("post", apply_all(post, pv, dv), tv, w)

    raw = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    raw["post"] = post.to_dict()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(raw, tmp); os.replace(tmp, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
