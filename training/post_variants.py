"""Make post-processing variants of a checkpoint by editing its "post" arrays (format unchanged: blend, clim,
qmap_fq, qmap_oq, qmap_strength), for shock-behaviour vs skill trade-offs.

    python -m training.post_variants runs/ens2_e5b_e9b/best_post.pt --out runs/ens2_e5b_e9b/post_variants

Variants (all deterministic functions of the rollout and doy, as before):
    cap05      blend weights capped at 0.5
    nowind     no blend on u10/v10 (keeps wind-driven shock signal)
    slope15    rain map slope capped at 1.5 between knots (limits how much rain *differences* are stretched)
    slope12    rain map slope capped at 1.25
    nowind_slope15, cap05_slope15   combinations
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

U10, V10 = 2, 3


def cap_slope(fq: np.ndarray, oq: np.ndarray, smax: float) -> np.ndarray:
    """Rebuild observed knots so every segment slope is <= smax; keeps 0 -> 0 and monotonicity."""
    out = oq.copy()
    for k in range(len(fq)):
        o = [0.0]
        for i in range(1, fq.shape[1]):
            step = max(0.0, float(oq[k, i] - oq[k, i - 1]))
            o.append(o[-1] + min(step, smax * float(fq[k, i] - fq[k, i - 1])))
        out[k] = np.asarray(o, np.float32)
    return out


def variants(post: dict) -> dict:
    b, fq, oq = post["blend"].numpy(), post["qmap_fq"].numpy(), post["qmap_oq"].numpy()
    def mk(blend=b, oqx=oq):
        p = dict(post); p["blend"] = torch.tensor(blend, dtype=torch.float32)
        p["qmap_oq"] = torch.tensor(oqx, dtype=torch.float32); return p
    nowind = b.copy(); nowind[:, [U10, V10]] = 0.0
    s15, s12 = cap_slope(fq, oq, 1.5), cap_slope(fq, oq, 1.25)
    return {"cap05": mk(np.minimum(b, 0.5)), "nowind": mk(nowind), "slope15": mk(oqx=s15), "slope12": mk(oqx=s12),
            "nowind_slope15": mk(nowind, s15), "cap05_slope15": mk(np.minimum(b, 0.5), s15)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, p in variants(ck["post"]).items():
        c = dict(ck); c["post"] = {**p, "meta": {**ck["post"].get("meta", {}), "variant": name}}
        torch.save(c, out / f"{name}.pt"); print("wrote", out / f"{name}.pt")


if __name__ == "__main__":
    main()
