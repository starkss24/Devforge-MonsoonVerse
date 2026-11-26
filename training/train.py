"""Training CLI: `python -m training.train --synthetic --stage onestep --epochs 2 --run-name smoke1`.

Stage onestep: k=1, AdamW lr 1e-3 cosine.  Stage multistep: curriculum over --k, backprop through the
whole unrolled chain (rollout_std), lr 1e-4, resumed from --resume (its normalizer is reused).
best.pt is chosen on a validation ROLLOUT score, never on training loss.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import random
import time

import numpy as np
import torch

from monsoon_r1 import scoring as KS
from monsoon_r1.stats import clim_for, climatology
from twin_model.model import UNet, advance_doy, rollout_std
from twin_model.norm import CHANNELS, RAIN, Normalizer, from_model, to_model
from twin_model.twin import load_checkpoint, save_checkpoint

from .dataset import WindowDataset, load_nc, make_synthetic, year_split
from .losses import rollout_loss

VAL_LEADS = (1, 4, 8)          # lead days scored
VAL_STEPS = max(VAL_LEADS)
RMSE_CH = [CHANNELS.index(c) for c in ("t2m", "msl", "tcwv")]
SHARP_CH = [CHANNELS.index(c) for c in ("t2m", "msl", "tcwv", "u10", "v10")]
CSV_COLS = ["run", "change", "day1_ratio", "day4_ratio", "day8_ratio", "sharpness",
            "heavy_rain_hit", "zero_shock_ok", "notes"]


def _d(x) -> dt.date:
    """datetime64 -> datetime.date (same helper as training/kit_eval.py; not imported because kit_eval
    sets CUDA_VISIBLE_DEVICES='' at import time, which would hide the GPU)."""
    return x.astype("datetime64[D]").astype(object)


def train_climatology(states_train, dates_train):
    """Kit climatology (default 15 d smoothing) from TRAINING years only; same ymd construction as kit_eval.prepare."""
    ymd = np.array([int(str(d).replace("-", "")) for d in np.asarray(dates_train).astype("datetime64[D]")])
    return climatology(states_train, ymd)


def kit_metrics(P, T_, L, last_dates, clim):
    """Judge-aligned validation numbers, using the kit's own scoring functions.

    P, T_: [N, 8, 6, H, W] predicted / true physical rollouts (day k+1 at index k); L: [N, 6, H, W] last input day;
    last_dates: datetime64 date of the last input day of each window.  Mirrors kit_eval.evaluate: per window the
    scored leads (1,4,8) are pooled, SS = mean over channels of skill_vs_best_ref()['ss'], CSI skill pools leads on the
    rain channel at 10 mm/day, day-8 variance ratio per channel.  Returns uncapped means plus capped kit points.
    """
    idx = [l - 1 for l in VAL_LEADS]
    ss_w, csi_w, ratios = [], [], []
    for w in range(len(P)):
        last = _d(last_dates[w])
        pred, tru = P[w, idx], T_[w, idx]
        persist = np.repeat(L[w][None], len(idx), 0)
        cl = np.stack([clim_for(clim, last + dt.timedelta(days=l)) for l in VAL_LEADS])
        r = KS.skill_vs_best_ref(pred, tru, persist, cl)
        ss_w.append(float(np.nanmean(r["ss"])))
        c = lambda a: KS.csi(a[:, RAIN], tru[:, RAIN])
        csi_w.append(KS.csi_skill(c(pred), c(persist), c(cl)))
        ratios.append(KS.variance_ratio(P[w, VAL_STEPS - 1], T_[w, VAL_STEPS - 1]))
    fp = KS.forecast_points(ss_w, csi_w)
    ratios = np.array(ratios)
    blur = KS.anti_blur_points(ratios)
    ss, cs = fp["mean_ss"], fp["mean_csi_skill"]
    ss = ss if np.isfinite(ss) else 0.0
    cs = cs if np.isfinite(cs) else 0.0
    # selection: spec formula (SS has no lower clip so bad models still rank; CSI skill floored at 0)
    sel = -(min(ss, KS.SS_REF) / KS.SS_REF * 30.0 + min(max(cs, 0.0), KS.CSI_SS_REF) / KS.CSI_SS_REF * 15.0 + blur)
    return {"kit_ss": ss, "kit_csi_skill": cs, "kit_blur_pts": float(blur), "kit_sel": float(sel),
            "kit_rmse_pts": fp["rmse_pts"], "kit_csi_pts": fp["csi_pts"],
            "kit_var_ratio": np.nanmean(ratios, axis=0).tolist()}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------ validation
@torch.no_grad()
def validate(model, val_ds, norm, static_np, device, tp_thresh, max_windows=48, clim=None, select="kit"):
    """Rollout score on an evenly spaced subset of val windows (ds built with k=VAL_STEPS)."""
    was_training = model.training
    model.eval()
    n = len(val_ds)
    idx = np.unique(np.linspace(0, n - 1, min(max_windows, n)).round().astype(int))
    preds_phys, truth, last, last_dates = [], [], [], []
    shock_ok = True
    for b0 in range(0, len(idx), 16):
        bi = idx[b0:b0 + 16]
        win, stat, doy, _ = [t.to(device) for t in val_ds.get_batch(bi)]
        ph = val_ds.phys_batch(bi)
        pred = rollout_std(model, win, stat, doy, VAL_STEPS, norm)
        p = from_model(norm.unstandardize(pred.float().cpu().numpy()))
        preds_phys.append(p)
        truth.append(ph[:, 3:])
        last.append(ph[:, 2])
        last_dates.append(val_ds.dates[val_ds.starts[bi] + 2])
        if b0 == 0:  # zero-shock: adding an all-zero physical perturbation must change nothing
            lastp = ph[:, 2] + np.zeros_like(ph[:, 2])
            w2 = win.clone()
            w2[:, -1] = torch.as_tensor(np.asarray(norm.standardize(to_model(lastp)), dtype=np.float32), device=device)
            pred2 = rollout_std(model, w2, stat, doy, VAL_STEPS, norm)
            shock_ok = bool((pred2 - pred).abs().max().item() < 1e-3)
    P, T_, L = np.concatenate(preds_phys), np.concatenate(truth), np.concatenate(last)
    last_dates = np.concatenate(last_dates)
    ok = np.isfinite(P).all()

    def ratio(lead):
        r = []
        for c in RMSE_CH:
            e = np.sqrt(np.mean((P[:, lead - 1, c] - T_[:, lead - 1, c]) ** 2))
            ep = np.sqrt(np.mean((L[:, c] - T_[:, lead - 1, c]) ** 2)) + 1e-12
            r.append(e / ep)
        return float(np.mean(r))

    ratios = {l: ratio(l) for l in VAL_LEADS}
    sf = P[:, :, SHARP_CH].std(axis=(-2, -1)).mean()
    st = T_[:, :, SHARP_CH].std(axis=(-2, -1)).mean()
    sharp = float(sf / (st + 1e-12))
    heavy = T_[:, :, RAIN] >= tp_thresh
    hit = float((P[:, :, RAIN][heavy] >= tp_thresh).mean()) if heavy.any() else float("nan")
    score = (ratios[4] + ratios[8]) / 2 + abs(1 - sharp) * 0.5
    km = kit_metrics(P, T_, L, last_dates, clim) if (clim is not None and ok) else {}
    old_score = score
    if select == "kit" and clim is not None:
        score = km["kit_sel"] if ok else float("inf")
    if not ok:
        score, old_score = float("inf"), float("inf")
    model.train(was_training)
    return {**km, "old_score": float(old_score), "day1_ratio": ratios[1], "day4_ratio": ratios[4], "day8_ratio": ratios[8], "sharpness": sharp,
            "heavy_rain_hit": hit, "zero_shock_ok": shock_ok and bool(ok), "score": float(score)}


def append_csv(path, row):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in CSV_COLS})


def main(argv=None):
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--data", default=None)
    src.add_argument("--synthetic", action="store_true")
    ap.add_argument("--tp-scale", type=float, default=1.0, help="multiply tp on load (e.g. 1000 if metres)")
    ap.add_argument("--stage", choices=["onestep", "multistep"], default="onestep")
    ap.add_argument("--k", default="2,4,8", help="curriculum for multistep (epochs apply per k)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--input-noise", type=float, default=0.0)
    ap.add_argument("--rain-weight", type=float, default=2.0)
    ap.add_argument("--grad-weight", type=float, default=0.1)
    ap.add_argument("--grad-mode", choices=["match", "mag"], default="match",
                    help="anti-blur term: match gradients (MSE) or match gradient magnitude")
    ap.add_argument("--select", choices=["old", "kit"], default="kit",
                    help="best.pt criterion: old = persistence-ratio score; kit = -(projected kit forecast+blur points)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--val-windows", type=int, default=48)
    ap.add_argument("--synthetic-T", type=int, default=1360)
    ap.add_argument("--change", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--csv", default="experiments.csv")
    ap.add_argument("--runs-dir", default="runs")
    a = ap.parse_args(argv)
    assert a.data or a.synthetic, "give --data PATH or --synthetic"
    run = a.run_name or f"{a.stage}_{time.strftime('%Y%m%d_%H%M%S')}"
    seed_all(a.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[train] device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("!" * 80 + "\n[train] WARNING: CUDA NOT AVAILABLE, TRAINING ON CPU (slow)\n" + "!" * 80)
    use_amp = bool(a.amp and device.type == "cuda")

    # ---- data
    if a.synthetic:
        states, dates, static, names = make_synthetic(T=a.synthetic_T, seed=a.seed)
        print(f"[train] synthetic data T={len(dates)} {dates[0]}..{dates[-1]}")
    else:
        states, dates, static, names = load_nc(a.data, tp_scale=a.tp_scale)
    tr_idx, va_idx = year_split(dates)
    assert len(tr_idx) and len(va_idx), "empty train/val split"
    print(f"[train] train days={len(tr_idx)} val days={len(va_idx)} (year split 2005-2018 / 2019-2021)")

    # ---- model + normalizer
    if a.resume:
        ck = load_checkpoint(a.resume, device="cpu")
        norm, model, cfg = ck["norm"], ck["model"], ck["config"]
        print(f"[train] resumed {a.resume}; reusing its normalizer and config {cfg}")
        model.train()
    else:
        norm = Normalizer.fit(states[tr_idx], static)  # TRAIN years only
        model = UNet(in_ch=22, out_ch=6, base=a.base, depth=a.depth)
        cfg = model.config()
    model.to(device)
    print(f"[train] params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    s_tr, d_tr = states[tr_idx], dates[tr_idx]
    s_va, d_va = states[va_idx], dates[va_idx]
    tp_thresh = float(np.percentile(s_tr[:, RAIN], 95))
    print(f"[train] heavy-rain threshold (train p95 tp) = {tp_thresh:.3f} mm/day")
    clim = train_climatology(s_tr, d_tr)  # once, train years only
    val_ds = WindowDataset(s_va, d_va, static, norm, VAL_STEPS)
    assert len(val_ds) > 0, f"no validation windows with {3 + VAL_STEPS} consecutive days"

    if a.stage == "onestep":
        stages = [(1, a.epochs)]
        lr = a.lr or 1e-3
    else:
        stages = [(int(k), a.epochs) for k in a.k.split(",")]
        lr = a.lr or 1e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gen = torch.Generator().manual_seed(a.seed)

    outdir = os.path.join(a.runs_dir, run)
    os.makedirs(outdir, exist_ok=True)
    best, best_m = float("inf"), None
    t0 = time.time()
    epoch_no = 0
    for k, n_ep in stages:
        ds = WindowDataset(s_tr, d_tr, static, norm, k)
        assert len(ds) > 0, f"no training windows for k={k}"
        steps_per_ep = math.ceil(len(ds) / a.batch)
        total = steps_per_ep * n_ep
        for g_ in opt.param_groups:
            g_["lr"] = lr
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total), eta_min=lr * 0.01)
        for ep in range(n_ep):
            model.train()
            perm = torch.randperm(len(ds), generator=gen).numpy()
            tl, nb, te = 0.0, 0, time.time()
            for b0 in range(0, len(perm), a.batch):
                win, stat, doy, tgt = [t.to(device, non_blocking=True) for t in ds.get_batch(perm[b0:b0 + a.batch])]
                if a.input_noise > 0:
                    win = win + a.input_noise * torch.randn_like(win)
                with torch.autocast(device.type, enabled=use_amp):
                    pred = rollout_std(model, win, stat, doy, k, norm)
                loss = rollout_loss(pred.float(), tgt, rain_weight=a.rain_weight, grad_weight=a.grad_weight,
                                    grad_mode=a.grad_mode)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                tl += float(loss.item())
                nb += 1
            epoch_no += 1
            m = validate(model, val_ds, norm, static, device, tp_thresh, a.val_windows, clim, a.select)
            meta = {"run": run, "stage": a.stage, "k": int(k), "epoch": epoch_no, "tp_p95": tp_thresh,
                    "score": m["score"], "select": a.select,
                    **{k_: v for k_, v in m.items() if k_.startswith("kit_") or k_ == "old_score"},
                    "day1_ratio": m["day1_ratio"], "day4_ratio": m["day4_ratio"],
                    "day8_ratio": m["day8_ratio"], "sharpness": m["sharpness"],
                    "seed": int(a.seed), "rain_weight": float(a.rain_weight),
                    "grad_weight": float(a.grad_weight), "grad_mode": a.grad_mode,"input_noise": float(a.input_noise)}
            print(f"[ep {epoch_no:02d} k={k}] loss={tl / nb:.4f} score={m['score']:.4f} "
                  f"d1={m['day1_ratio']:.3f} d4={m['day4_ratio']:.3f} d8={m['day8_ratio']:.3f} "
                  f"sharp={m['sharpness']:.3f} heavy_hit={m['heavy_rain_hit']:.3f} "
                  f"zero_shock={m['zero_shock_ok']} ({time.time() - te:.1f}s)")
            print(f"    old_score={m['old_score']:.4f} | kit: sel={m['kit_sel']:.2f} SS={m['kit_ss']:+.4f} "
                  f"(rmse {m['kit_rmse_pts']:.2f}/30) CSIskill={m['kit_csi_skill']:+.4f} (csi {m['kit_csi_pts']:.2f}/15) "
                  f"blur={m['kit_blur_pts']:.2f}/10 vr8=" + ",".join(f"{v:.2f}" for v in m["kit_var_ratio"])
                  if "kit_sel" in m else "    (kit metrics unavailable: non-finite predictions)")
            save_checkpoint(os.path.join(outdir, "last.pt"), model, cfg, norm, static, names, meta)
            if m["score"] < best and m["zero_shock_ok"]:
                best, best_m = m["score"], m
                save_checkpoint(os.path.join(outdir, "best.pt"), model, cfg, norm, static, names, meta)
                print(f"  -> new best.pt (score {best:.4f})")
    if best_m is None:  # nothing passed the zero-shock gate: still keep the last model, flagged
        print("!" * 80 + "\n[train] WARNING: no epoch passed zero-shock/finite check; best.pt = last epoch\n" + "!" * 80)
        best_m = m
        save_checkpoint(os.path.join(outdir, "best.pt"), model, cfg, norm, static, names, meta)
    row = {"run": run, "change": a.change or f"{a.stage} k={[s[0] for s in stages]} lr={lr} rw={a.rain_weight} "
                                              f"gw={a.grad_weight} noise={a.input_noise}",
           "day1_ratio": f"{best_m['day1_ratio']:.4f}", "day4_ratio": f"{best_m['day4_ratio']:.4f}",
           "day8_ratio": f"{best_m['day8_ratio']:.4f}", "sharpness": f"{best_m['sharpness']:.4f}",
           "heavy_rain_hit": f"{best_m['heavy_rain_hit']:.4f}", "zero_shock_ok": best_m["zero_shock_ok"],
           "notes": " | ".join(x for x in [a.notes or ("synthetic" if a.synthetic else os.path.basename(a.data)),
                                          f"select={a.select}",
                                          (f"kit_sel={best_m['kit_sel']:.2f} kit_ss={best_m['kit_ss']:+.4f} "
                                           f"kit_csi_skill={best_m['kit_csi_skill']:+.4f} kit_blur={best_m['kit_blur_pts']:.2f}"
                                           if "kit_sel" in best_m else "")] if x)}
    append_csv(a.csv, row)
    print(f"[train] done in {time.time() - t0:.1f}s; best score {best:.4f}; logged to {a.csv}; ckpts in {outdir}")
    return best_m


if __name__ == "__main__":
    main()
