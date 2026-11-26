"""Output post-processing, numpy + stdlib only (safe for the serving path; no torch import).

Applied to a FINISHED rollout (never fed back into the autoregressive loop):
1. Climatology blend per lead k and channel c:
       y[k,c] <- (1 - a[k,c]) * y[k,c] + a[k,c] * clim[slot(last input day + k + 1), c]
2. Rain quantile map per lead k (after the blend): monotone piecewise-linear map through
   knots (fq[k], oq[k]) with fq[k,0] = oq[k,0] = 0 (dry stays dry), slope-1 continuation past
   the last knot; optional per-lead strength s[k] in [0,1]: tp <- tp + s[k] * (map(tp) - tp).

Leads beyond the fitted ones reuse the last fitted lead. ``doy_sincos`` is that of the LAST
INPUT day, exactly as in the /simulate request. Arrays come from ``weights/post.npz``
(written by ``training/export.py``): blend [L,6], clim [366,6,H,W], qmap_fq [L,Q], qmap_oq [L,Q],
optional qmap_strength [L].
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

RAIN = 1
N_CH = 6
N_SLOTS = 366
_MAR1_DOY = 60  # 1-based day of year of 1 March in a non-leap year


def doy_from_sincos(doy_sincos) -> int:
    """Inverse of sin/cos(2*pi*doy/365): 1-based day of year in 1..365."""
    s, c = (float(v) for v in np.asarray(doy_sincos, dtype=np.float64).reshape(2))
    a = math.atan2(s, c) % (2 * math.pi)
    d = int(round(a * 365.0 / (2 * math.pi))) % 365
    return 365 if d == 0 else d


def clim_slot(doy: int) -> int:
    """Kit climatology slot (0..365, leap-year calendar, Feb 29 = slot 59) of a non-leap 1-based doy."""
    doy = (int(doy) - 1) % 365 + 1
    return doy - 1 if doy < _MAR1_DOY else doy


def target_slots(doy_sincos, steps: int) -> np.ndarray:
    """Climatology slot of each forecast day k = 1..steps after the last input day."""
    d0 = doy_from_sincos(doy_sincos)
    return np.array([clim_slot(d0 + k) for k in range(1, steps + 1)], dtype=np.int64)


def qmap(x: np.ndarray, fq: np.ndarray, oq: np.ndarray) -> np.ndarray:
    """Monotone piecewise-linear map through (fq, oq); slope-1 continuation past the last knot."""
    y = np.interp(x, fq, oq)
    hi = x > fq[-1]
    if hi.any():
        y = np.where(hi, oq[-1] + (x - fq[-1]), y)
    return y.astype(np.float32, copy=False)


def validate(p: dict) -> dict:
    """Check shapes/values of a post-processing dict; returns a float32 copy. Raises ValueError."""
    q = {k: np.ascontiguousarray(np.asarray(v, dtype=np.float32)) for k, v in p.items() if v is not None}
    if ("blend" in q) != ("clim" in q):
        raise ValueError("post: 'blend' and 'clim' must be given together")
    if ("qmap_fq" in q) != ("qmap_oq" in q):
        raise ValueError("post: 'qmap_fq' and 'qmap_oq' must be given together")
    if "blend" in q:
        b, c = q["blend"], q["clim"]
        if b.ndim != 2 or b.shape[1] != N_CH or len(b) < 1:
            raise ValueError(f"post: blend must be [L,{N_CH}], got {b.shape}")
        if c.ndim != 4 or c.shape[:2] != (N_SLOTS, N_CH):
            raise ValueError(f"post: clim must be [{N_SLOTS},{N_CH},H,W], got {c.shape}")
        if not (np.isfinite(b).all() and np.isfinite(c).all()):
            raise ValueError("post: non-finite blend/clim")
        q["blend"] = np.clip(b, 0.0, 1.0)
    if "qmap_fq" in q:
        fq, oq = q["qmap_fq"], q["qmap_oq"]
        if fq.ndim != 2 or fq.shape != oq.shape or fq.shape[1] < 2:
            raise ValueError(f"post: qmap knots must be [L,Q>=2] x2, got {fq.shape}, {oq.shape}")
        if not (np.isfinite(fq).all() and np.isfinite(oq).all()):
            raise ValueError("post: non-finite qmap knots")
        if (np.diff(fq, axis=1) <= 0).any() or (np.diff(oq, axis=1) < 0).any():
            raise ValueError("post: qmap knots must be increasing (fq strictly, oq non-decreasing)")
        if (fq[:, 0] != 0).any() or (oq[:, 0] != 0).any():
            raise ValueError("post: qmap must map 0 -> 0")
    if "qmap_strength" in q:
        s = q["qmap_strength"].reshape(-1)
        if "qmap_fq" not in q:
            raise ValueError("post: 'qmap_strength' needs qmap_fq/qmap_oq")
        if len(s) < 1 or not np.isfinite(s).all() or (s < 0).any() or (s > 1).any():
            raise ValueError(f"post: qmap_strength must be finite values in [0,1], got {s}")
        q["qmap_strength"] = s
    return q


def load(path) -> dict:
    """Load and validate weights/post.npz."""
    with np.load(Path(path)) as z:
        return validate({k: z[k] for k in z.files})


def apply(y: np.ndarray, doy_sincos, p: dict) -> np.ndarray:
    """y: physical rollout [steps,6,H,W] (finite). Returns a new float32 array (blend, then rain map)."""
    y = np.array(y, dtype=np.float32, copy=True)
    steps = y.shape[0]
    blend, clim = p.get("blend"), p.get("clim")
    if blend is not None and clim.shape[-2:] == y.shape[-2:]:
        slots = target_slots(doy_sincos, steps)
        L = len(blend)
        for k in range(steps):
            a = blend[min(k, L - 1)]
            if (a > 0).any():
                a3 = a.reshape(-1, 1, 1)
                y[k] = (1.0 - a3) * y[k] + a3 * clim[slots[k]]
    fq, oq, s = p.get("qmap_fq"), p.get("qmap_oq"), p.get("qmap_strength")
    if fq is not None:
        L = len(fq)
        for k in range(steps):
            i = min(k, L - 1)
            mapped = qmap(y[k, RAIN], fq[i], oq[i])
            if s is not None:
                sk = float(s[min(k, len(s) - 1)])
                if sk < 1.0:  # sk == 1 keeps the mapped value bit-exact
                    mapped = (y[k, RAIN] + sk * (mapped - y[k, RAIN])).astype(np.float32, copy=False)
            y[k, RAIN] = mapped
    return y
