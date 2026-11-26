"""Optional output post-processing applied by Twin.rollout (stdlib + numpy + torch only).

Two deterministic, exact functions of the model's physical output, fitted offline on
training years by ``training/fit_post.py`` and stored in the checkpoint under ``"post"``:

1. Climatology blend per lead k and channel c:
       y[k,c] <- (1 - a[k,c]) * y[k,c] + a[k,c] * clim[slot(target day), c]
2. Rain quantile mapping per lead k (applied after the blend, so it is fitted on blended
   rain and fixes the final rain distribution):
       tp <- piecewise-linear map through knots (fq[k], oq[k]); fq[k,0] = oq[k,0] = 0 so
       dry stays dry; both knot rows are non-decreasing (fq strictly increasing), so the
       map is monotone; beyond the last knot it continues with slope 1.

   Optional per-lead strength s[k] in [0,1] (absent = 1 = full map):
       tp <- tp + s[k] * (qmap(tp) - tp)
   s < 1 backs off the stretch where the model is unsure of rain location (double penalty).
   0 -> 0 and monotonicity are preserved (convex mix of two monotone maps fixing 0).

Leads beyond the fitted ones reuse the last fitted lead. Checkpoint layout (all float32
torch tensors so the file loads with ``weights_only=True``):
    "post": {"blend": [L,6], "clim": [366,6,H,W],          # optional pair
             "qmap_fq": [L,Q], "qmap_oq": [L,Q],           # optional pair
             "qmap_strength": [L],                          # optional, needs qmap
             "meta": {...}}
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .norm import N_CH, RAIN

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


def _np(v) -> np.ndarray:
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(v, dtype=np.float32))


class Post:
    def __init__(self, blend=None, clim=None, qmap_fq=None, qmap_oq=None, meta=None, qmap_strength=None):
        self.blend = None if blend is None else np.clip(_np(blend), 0.0, 1.0)
        self.clim = None if clim is None else _np(clim)
        self.fq = None if qmap_fq is None else _np(qmap_fq)
        self.oq = None if qmap_oq is None else _np(qmap_oq)
        self.strength = None if qmap_strength is None else _np(qmap_strength).reshape(-1)
        self.meta = dict(meta or {})
        if self.strength is not None:
            if self.fq is None:
                raise ValueError("post: 'qmap_strength' needs qmap_fq/qmap_oq")
            if len(self.strength) < 1 or not np.isfinite(self.strength).all() \
                    or (self.strength < 0).any() or (self.strength > 1).any():
                raise ValueError(f"post: qmap_strength must be finite values in [0,1], got {self.strength}")
        if (self.blend is None) != (self.clim is None):
            raise ValueError("post: 'blend' and 'clim' must be given together")
        if (self.fq is None) != (self.oq is None):
            raise ValueError("post: 'qmap_fq' and 'qmap_oq' must be given together")
        if self.blend is not None:
            if self.blend.ndim != 2 or self.blend.shape[1] != N_CH or len(self.blend) < 1:
                raise ValueError(f"post: blend must be [L,{N_CH}], got {self.blend.shape}")
            if self.clim.ndim != 4 or self.clim.shape[:2] != (N_SLOTS, N_CH):
                raise ValueError(f"post: clim must be [{N_SLOTS},{N_CH},H,W], got {self.clim.shape}")
            if not np.isfinite(self.clim).all() or not np.isfinite(self.blend).all():
                raise ValueError("post: non-finite blend/clim")
        if self.fq is not None:
            if self.fq.ndim != 2 or self.fq.shape != self.oq.shape or self.fq.shape[1] < 2:
                raise ValueError(f"post: qmap knots must be [L,Q>=2] x2, got {self.fq.shape}, {self.oq.shape}")
            if not (np.isfinite(self.fq).all() and np.isfinite(self.oq).all()):
                raise ValueError("post: non-finite qmap knots")
            if (np.diff(self.fq, axis=1) <= 0).any() or (np.diff(self.oq, axis=1) < 0).any():
                raise ValueError("post: qmap knots must be increasing (fq strictly, oq non-decreasing)")
            if (self.fq[:, 0] != 0).any() or (self.oq[:, 0] != 0).any():
                raise ValueError("post: qmap must map 0 -> 0")

    @property
    def active(self) -> bool:
        return self.blend is not None or self.fq is not None

    @classmethod
    def from_dict(cls, d: dict) -> "Post":
        return cls(d.get("blend"), d.get("clim"), d.get("qmap_fq"), d.get("qmap_oq"), d.get("meta"),
                   d.get("qmap_strength"))

    def to_dict(self) -> dict:
        t = lambda a: None if a is None else torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).clone()
        d = {"meta": dict(self.meta)}
        if self.blend is not None:
            d["blend"], d["clim"] = t(self.blend), t(self.clim)
        if self.fq is not None:
            d["qmap_fq"], d["qmap_oq"] = t(self.fq), t(self.oq)
        if self.strength is not None:
            d["qmap_strength"] = t(self.strength)
        return d

    def apply(self, y: np.ndarray, doy_sincos) -> np.ndarray:
        """y: physical [steps,6,H,W] (finite). Returns a new float32 array; blend, then rain qmap."""
        y = np.array(y, dtype=np.float32, copy=True)
        steps = y.shape[0]
        if self.blend is not None and self.clim.shape[-2:] == y.shape[-2:]:
            slots = target_slots(doy_sincos, steps)
            L = len(self.blend)
            for k in range(steps):
                a = self.blend[min(k, L - 1)]
                if (a > 0).any():
                    a3 = a.reshape(-1, 1, 1)
                    y[k] = (1.0 - a3) * y[k] + a3 * self.clim[slots[k]]
        if self.fq is not None:
            L = len(self.fq)
            for k in range(steps):
                i = min(k, L - 1)
                mapped = qmap(y[k, RAIN], self.fq[i], self.oq[i])
                if self.strength is not None:
                    s = float(self.strength[min(k, len(self.strength) - 1)])
                    if s < 1.0:  # s == 1 keeps the mapped value bit-exact
                        mapped = (y[k, RAIN] + s * (mapped - y[k, RAIN])).astype(np.float32, copy=False)
                y[k, RAIN] = mapped
        return y
