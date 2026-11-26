"""Combine same-config checkpoints into one ensemble checkpoint (averaged daily change per step).

    python -m training.make_ensemble runs/e5b_gradmag/best.pt runs/e9b_seed1_multi/best.pt --out runs/ens2/best.pt

Members must share normalisation stats, static maps and config (checked; trained on the same years).
Post-processing is NOT copied: fit it for the ensemble with training/fit_post.py.
"""
from __future__ import annotations

import argparse

import numpy as np

from twin_model.model import EnsembleUNet
from twin_model.twin import load_checkpoint, save_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cks = [load_checkpoint(p) for p in a.ckpts]
    ref = cks[0]
    for p, c in zip(a.ckpts[1:], cks[1:]):
        for k in ("state_mean", "state_std", "delta_std", "static_mean", "static_std"):
            if not np.array_equal(getattr(ref["norm"], k), getattr(c["norm"], k)):
                raise SystemExit(f"{p}: normaliser '{k}' differs from {a.ckpts[0]} (train on the same years)")
        if not np.array_equal(ref["static"], c["static"]):
            raise SystemExit(f"{p}: static maps differ from {a.ckpts[0]}")
    members = []
    for c in cks:
        m = c["model"]
        members.extend(m.members if isinstance(m, EnsembleUNet) else [m])
    ens = EnsembleUNet(members).eval()
    meta = {"ensemble_of": [str(p).replace("\\", "/") for p in a.ckpts],
            "member_meta": [c["meta"] for c in cks]}
    save_checkpoint(a.out, ens, None, ref["norm"], ref["static"], ref["static_names"], meta)
    print(f"wrote {a.out}: {len(members)} members, config {ens.config()}")


if __name__ == "__main__":
    main()
