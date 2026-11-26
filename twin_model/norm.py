"""Normalization: physical <-> model space <-> standardized model space.

Model space is identical to physical space except rain, which is log1p(max(tp, 0)).
Standardized space is (model - state_mean) / state_std, per channel.

All functions work on numpy arrays and torch tensors with any leading dims; the
channel axis is always -3 (``[..., C, H, W]``).
"""
from __future__ import annotations

import numpy as np
import torch

CHANNELS = ["t2m", "tp", "u10", "v10", "msl", "tcwv"]
RAIN = 1
H, W = 32, 36
N_IN = 3
N_CH = len(CHANNELS)

# Floors guard against division by ~0 for (near-)constant channels.
STD_FLOOR = 1e-6          # state/static std, in their own units
DELTA_STD_FLOOR = 1e-3    # std of the standardized daily change (dimensionless)


def _is_torch(x) -> bool:
    return isinstance(x, torch.Tensor)


def to_model(x_phys):
    """Physical -> model space (rain -> log1p(max(tp, 0))). Returns a new array."""
    if _is_torch(x_phys):
        out = x_phys.clone()
        out[..., RAIN, :, :] = torch.log1p(torch.clamp(x_phys[..., RAIN, :, :], min=0.0))
        return out
    x = np.asarray(x_phys)
    out = np.array(x, copy=True)
    out[..., RAIN, :, :] = np.log1p(np.maximum(x[..., RAIN, :, :], 0.0))
    return out


def from_model(x_mod):
    """Model space -> physical (rain -> expm1, clamped >= 0). Returns a new array."""
    if _is_torch(x_mod):
        out = x_mod.clone()
        out[..., RAIN, :, :] = torch.clamp(torch.expm1(x_mod[..., RAIN, :, :]), min=0.0)
        return out
    x = np.asarray(x_mod)
    out = np.array(x, copy=True)
    out[..., RAIN, :, :] = np.maximum(np.expm1(x[..., RAIN, :, :]), 0.0)
    return out


def _bcast(v: np.ndarray, like):
    """Per-channel vector [C] -> broadcastable [C,1,1] of the same kind/dtype/device as ``like``."""
    if _is_torch(like):
        return torch.as_tensor(v, dtype=like.dtype, device=like.device).view(-1, 1, 1)
    return np.asarray(v, dtype=np.result_type(like.dtype, np.float32)).reshape(-1, 1, 1)


class Normalizer:
    """Per-channel statistics, all float32 numpy.

    state_mean, state_std : [6]  stats of model-space states
    delta_std             : [6]  std of the day-to-day change of *standardized* states
    static_mean, static_std : [2]
    """

    def __init__(self, state_mean, state_std, delta_std, static_mean, static_std):
        self.state_mean = np.asarray(state_mean, dtype=np.float32).reshape(-1)
        self.state_std = np.asarray(state_std, dtype=np.float32).reshape(-1)
        self.delta_std = np.asarray(delta_std, dtype=np.float32).reshape(-1)
        self.static_mean = np.asarray(static_mean, dtype=np.float32).reshape(-1)
        self.static_std = np.asarray(static_std, dtype=np.float32).reshape(-1)
        self._torch_cache: dict = {}

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, states_phys, static) -> "Normalizer":
        """Fit on physical states [T,C,H,W] (training years only) and static [S,H,W].

        Assumes consecutive rows of ``states_phys`` are consecutive days (delta_std is
        computed from ``s[1:] - s[:-1]``). Computed per channel in float64.
        """
        states_phys = np.asarray(states_phys)
        static = np.asarray(static)
        if states_phys.ndim != 4 or states_phys.shape[0] < 2:
            raise ValueError(f"states_phys must be [T>=2,C,H,W], got {states_phys.shape}")
        C = states_phys.shape[1]
        mean = np.zeros(C)
        std = np.ones(C)
        dstd = np.ones(C)
        for c in range(C):
            x = states_phys[:, c].astype(np.float64)
            if c == RAIN:
                x = np.log1p(np.maximum(x, 0.0))
            m = np.nanmean(x)
            s = max(float(np.nanstd(x)), STD_FLOOR)
            z = (x - m) / s
            d = z[1:] - z[:-1]
            mean[c], std[c] = m, s
            dstd[c] = max(float(np.nanstd(d)), DELTA_STD_FLOOR)
        st = static.reshape(static.shape[0], -1).astype(np.float64)
        smean = np.nanmean(st, axis=1)
        sstd = np.maximum(np.nanstd(st, axis=1), STD_FLOOR)
        return cls(mean, std, dstd, smean, sstd)

    # ---------------------------------------------------------- transforms
    def standardize(self, x_mod):
        return (x_mod - _bcast(self.state_mean, x_mod)) / _bcast(self.state_std, x_mod)

    def unstandardize(self, x_std):
        return x_std * _bcast(self.state_std, x_std) + _bcast(self.state_mean, x_std)

    def standardize_static(self, static):
        return (static - _bcast(self.static_mean, static)) / _bcast(self.static_std, static)

    def unstandardize_static(self, static_std):
        return static_std * _bcast(self.static_std, static_std) + _bcast(self.static_mean, static_std)

    def torch_stats(self, device=None, dtype=torch.float32) -> dict:
        """Cached torch tensors of the state stats, shaped [C] (for model.step)."""
        key = (str(device), dtype)
        t = self._torch_cache.get(key)
        if t is None:
            t = {
                k: torch.as_tensor(getattr(self, k), dtype=dtype, device=device)
                for k in ("state_mean", "state_std", "delta_std")
            }
            self._torch_cache[key] = t
        return t

    # ----------------------------------------------------------- (de)serialize
    def to_dict(self) -> dict:
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "delta_std": self.delta_std.tolist(),
            "static_mean": self.static_mean.tolist(),
            "static_std": self.static_std.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(d["state_mean"], d["state_std"], d["delta_std"], d["static_mean"], d["static_std"])

    def __repr__(self) -> str:
        f = lambda a: np.array2string(a, precision=4, suppress_small=True)
        return (f"Normalizer(state_mean={f(self.state_mean)}, state_std={f(self.state_std)}, "
                f"delta_std={f(self.delta_std)})")
