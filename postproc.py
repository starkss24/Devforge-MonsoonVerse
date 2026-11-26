"""Output post-processing for the served rollout: climatology blend + rain quantile map (numpy only).

Inference file (imported by twin.py -> serve.py): stdlib + numpy only.

Pure-numpy port of the model lane's ``twin_model/post.py::Post.apply`` (the reference; tests/test_postproc.py
checks bit-level agreement). The numbers are exported by the model lane to ``weights/post.npz``; the format is
pinned in docs/WEIGHTS.md ("Post-processing: weights/post.npz"). It is applied once, to the FINISHED model
rollout [steps,6,32,36], and never fed back into the autoregressive loop:

1. Climatology blend per lead k and channel c (target day = last input day + k + 1, rollout index k):
       y[k,c] <- (1 - a[k,c]) * y[k,c] + a[k,c] * clim[slot(target day), c]
   clim is a ONE-YEAR climatology on the kit's 366-slot calendar (Feb 29 = slot 59).
2. Rain quantile map per lead (after the blend): piecewise-linear through knots (fq[k], oq[k]),
   0 -> 0, monotone, slope-1 continuation past the last knot; optional per-lead strength s[k] in [0,1]:
       tp <- tp + s[k] * (qmap(tp) - tp)      (s == 1 keeps the mapped value bit-exact)

Leads beyond the fitted L rows reuse the last row (same as Post). Deterministic: the same rollout and
doy_sincos always give bit-identical output, so a zero perturbation still equals no perturbation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

POST_FILE = "post.npz"
FORMAT_VERSION = 1
N_SLOTS = 366
N_CH = 6
GRID = (32, 36)
RAIN = 1
_MAR1_DOY = 60  # 1-based day of year of 1 March in a non-leap year
ARRAY_KEYS = ("blend", "clim", "qmap_fq", "qmap_oq", "qmap_strength")
KNOWN_KEYS = set(ARRAY_KEYS) | {"version", "meta"}
_LFS_MAGIC = b"version https://git-lfs"


# ---- calendar (identical to twin_model/post.py) ----------------------------------------------------------
def doy_from_sincos(doy_sincos) -> int:
    """Inverse of sin/cos(2*pi*doy/365): 1-based day of year in 1..365.

    Same convention as the model lane: a leap-year doy 366 (angle 2*pi*366/365) reads back as doy 1."""
    s, c = (float(v) for v in np.asarray(doy_sincos, dtype=np.float64).reshape(2))
    a = math.atan2(s, c) % (2 * math.pi)
    d = int(round(a * 365.0 / (2 * math.pi))) % 365
    return 365 if d == 0 else d


def clim_slot(doy: int) -> int:
    """Kit climatology slot (0..365, leap calendar, Feb 29 = slot 59) of a non-leap 1-based doy."""
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


# ---- the post-processor --------------------------------------------------------------------------------
def _f32(name: str, v) -> np.ndarray:
    a = np.asarray(v)
    if a.dtype.kind not in "fiu":
        raise ValueError(f"'{name}' must be a numeric array, got dtype {a.dtype}")
    return np.ascontiguousarray(a, dtype=np.float32)


class PostProc:
    """Validated post-processing parameters. Raises ValueError on anything malformed (never half-applies)."""

    def __init__(self, blend=None, clim=None, qmap_fq=None, qmap_oq=None, qmap_strength=None, meta=None):
        # Post clips blend to [0, 1] on load; do the same so the output matches it exactly.
        self.blend = None if blend is None else np.clip(_f32("blend", blend), 0.0, 1.0)
        self.clim = None if clim is None else _f32("clim", clim)
        self.fq = None if qmap_fq is None else _f32("qmap_fq", qmap_fq)
        self.oq = None if qmap_oq is None else _f32("qmap_oq", qmap_oq)
        self.strength = None if qmap_strength is None else _f32("qmap_strength", qmap_strength).reshape(-1)
        self.meta = dict(meta or {})
        if (self.blend is None) != (self.clim is None):
            raise ValueError("'blend' and 'clim' must be given together")
        if (self.fq is None) != (self.oq is None):
            raise ValueError("'qmap_fq' and 'qmap_oq' must be given together")
        if self.blend is None and self.fq is None:
            raise ValueError("no post-processing arrays (need blend+clim and/or qmap_fq+qmap_oq)")
        if self.blend is not None:
            if self.blend.ndim != 2 or self.blend.shape[1] != N_CH or len(self.blend) < 1:
                raise ValueError(f"blend must be [L,{N_CH}], got {list(self.blend.shape)}")
            if self.clim.shape != (N_SLOTS, N_CH, *GRID):
                raise ValueError(f"clim must be [{N_SLOTS},{N_CH},{GRID[0]},{GRID[1]}], got {list(self.clim.shape)}")
            if not (np.isfinite(self.clim).all() and np.isfinite(self.blend).all()):
                raise ValueError("non-finite blend/clim")
        if self.fq is not None:
            if self.fq.ndim != 2 or self.fq.shape != self.oq.shape or self.fq.shape[1] < 2 or len(self.fq) < 1:
                raise ValueError(f"qmap knots must be two [L,Q>=2] arrays, got {list(self.fq.shape)}, "
                                 f"{list(self.oq.shape)}")
            if not (np.isfinite(self.fq).all() and np.isfinite(self.oq).all()):
                raise ValueError("non-finite qmap knots")
            if (np.diff(self.fq, axis=1) <= 0).any() or (np.diff(self.oq, axis=1) < 0).any():
                raise ValueError("qmap knots must be increasing (qmap_fq strictly, qmap_oq non-decreasing)")
            if (self.fq[:, 0] != 0).any() or (self.oq[:, 0] != 0).any():
                raise ValueError("qmap must map 0 -> 0 (first knot of qmap_fq and qmap_oq must be 0)")
        if self.strength is not None:
            if self.fq is None:
                raise ValueError("'qmap_strength' needs qmap_fq/qmap_oq")
            if len(self.strength) < 1 or not np.isfinite(self.strength).all() \
                    or (self.strength < 0).any() or (self.strength > 1).any():
                raise ValueError("qmap_strength must be finite values in [0, 1]")

    def describe(self) -> str:
        parts = []
        if self.blend is not None:
            parts.append(f"blend L={len(self.blend)}")
        if self.fq is not None:
            parts.append(f"qmap L={len(self.fq)} Q={self.fq.shape[1]}")
        if self.strength is not None:
            parts.append(f"strength L={len(self.strength)}")
        return ", ".join(parts)

    @classmethod
    def load(cls, path) -> "PostProc":
        """Read and validate weights/post.npz (np.load with allow_pickle=False: no code is ever unpickled)."""
        p = Path(path)
        with open(p, "rb") as f:
            if f.read(64).startswith(_LFS_MAGIC):
                raise ValueError("file is a Git LFS pointer, not the arrays (commit post.npz as a plain git file)")
        with np.load(p, allow_pickle=False) as z:
            keys = set(z.files)
            unknown = keys - KNOWN_KEYS
            if unknown:
                raise ValueError(f"unknown keys {sorted(unknown)} (allowed: {sorted(KNOWN_KEYS)})")
            if "version" in keys:
                v = z["version"]
                if v.shape != () or v.dtype.kind not in "iu" or int(v) != FORMAT_VERSION:
                    raise ValueError(f"version must be integer scalar {FORMAT_VERSION}, got {v!r}")
            meta = {}
            if "meta" in keys:
                m = z["meta"]
                if m.shape != () or m.dtype.kind != "U":
                    raise ValueError("meta must be a JSON string scalar (np.array(json.dumps(...)))")
                meta = json.loads(str(m))
                if not isinstance(meta, dict):
                    raise ValueError("meta must be a JSON object")
            arrays = {k: z[k] for k in ARRAY_KEYS if k in keys}
        return cls(**arrays, meta=meta)

    def apply(self, y: np.ndarray, doy_sincos) -> np.ndarray:
        """y: physical rollout [steps,6,32,36] (finite); doy_sincos of the last input day.
        Returns a new float32 array: blend, then rain qmap. Op-for-op the same as Post.apply."""
        # Post receives doy_sincos as float32 from twin_model/twin.py; round the same way.
        d = np.asarray(doy_sincos, dtype=np.float32).reshape(2)
        y = np.array(y, dtype=np.float32, copy=True)
        steps = y.shape[0]
        if self.blend is not None:
            slots = target_slots(d, steps)
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


def save_npz(path, blend=None, clim=None, qmap_fq=None, qmap_oq=None, qmap_strength=None, meta=None) -> None:
    """Write post.npz in the pinned format (float32 arrays, version, meta as JSON). Validates first."""
    PostProc(blend, clim, qmap_fq, qmap_oq, qmap_strength, meta)  # raises if malformed
    out = {"version": np.array(FORMAT_VERSION, dtype=np.int64),
           "meta": np.array(json.dumps(dict(meta or {})))}
    for k, v in (("blend", blend), ("clim", clim), ("qmap_fq", qmap_fq), ("qmap_oq", qmap_oq),
                 ("qmap_strength", qmap_strength)):
        if v is not None:
            out[k] = np.ascontiguousarray(v, dtype=np.float32)
    np.savez(path, **out)
