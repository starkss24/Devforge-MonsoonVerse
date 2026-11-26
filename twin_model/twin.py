"""Twin: checkpoint I/O and the physical-units rollout used by serve.py.

Imports only stdlib, numpy and torch.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch

from .model import UNet, rollout_std
from .norm import CHANNELS, N_IN, RAIN, Normalizer, from_model, to_model
from .post import Post

log = logging.getLogger(__name__)

T2M = CHANNELS.index("t2m")
T2M_MIN, T2M_MAX = 233.15, 333.15
# Last-resort fill for non-finite *input* cells when no finite value exists in the
# window and no normalizer is available (plausible climatology, never zeros).
_DEFAULT_FILL = np.array([290.0, 0.0, 0.0, 0.0, 100800.0, 30.0], dtype=np.float32)
_REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_KEYS = ("base", "depth", "in_ch", "out_ch", "softplus_beta")


# ------------------------------------------------------------------ checkpoints
def save_checkpoint(path, model: UNet, config: dict | None, norm: Normalizer, static,
                    static_names, meta: dict | None = None) -> None:
    """Save the checkpoint dict described in docs/interfaces.md.

    ``config`` may be None/partial; missing keys are taken from ``model.config()``.
    ``static`` (physical, [2,H,W]) is stored as a float32 torch tensor so the file
    loads with ``torch.load(weights_only=True)``.
    """
    cfg = dict(model.config())
    if config:
        cfg.update({k: config[k] for k in CONFIG_KEYS if k in config})
    static_t = torch.as_tensor(np.asarray(static, dtype=np.float32)).clone()
    ckpt = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": cfg,
        "norm": norm.to_dict(),
        "static": static_t,
        "static_names": [str(n) for n in static_names],
        "meta": dict(meta or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device="cpu") -> dict:
    """Load a checkpoint. Returns a dict with keys:
    ``model`` (UNet, eval mode, on ``device``), ``norm`` (Normalizer), ``static``
    (np.float32 [2,H,W], physical), ``static_names``, ``config``, ``meta`` and
    ``raw`` (the dict as stored, e.g. for --resume) and ``post`` (the stored
    post-processing dict, or None when the checkpoint has no "post" key).
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # older files / meta with non-allowlisted types; our own weights
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = {k: ckpt["config"][k] for k in CONFIG_KEYS if k in ckpt["config"]}
    model = UNet(**cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    static = ckpt["static"]
    static = static.cpu().numpy() if isinstance(static, torch.Tensor) else np.asarray(static)
    return {
        "model": model,
        "norm": Normalizer.from_dict(ckpt["norm"]),
        "static": static.astype(np.float32),
        "static_names": list(ckpt.get("static_names", [])),
        "config": model.config(),
        "meta": dict(ckpt.get("meta", {})),
        "raw": ckpt,
        "post": ckpt.get("post"),
    }


def _resolve(ckpt_path) -> Path | None:
    if ckpt_path is None:
        return None
    p = Path(ckpt_path)
    if p.is_file():
        return p
    if not p.is_absolute() and (_REPO_ROOT / p).is_file():
        return _REPO_ROOT / p
    return None


# ------------------------------------------------------------------------ Twin
class Twin:
    def __init__(self, ckpt_path: str | None = "weights/model.pt"):
        torch.set_num_threads(4)
        self.fallback = True
        self.model: UNet | None = None
        self.norm: Normalizer | None = None
        self.static: np.ndarray | None = None
        self.static_names: list = []
        self.meta: dict = {}
        self._static_std_t: torch.Tensor | None = None
        self.post: Post | None = None

        path = _resolve(ckpt_path)
        if path is None:
            log.warning("Twin: checkpoint %r not found -> persistence fallback", ckpt_path)
            return
        try:
            ck = load_checkpoint(path)
            self.model, self.norm = ck["model"], ck["norm"]
            self.static, self.static_names, self.meta = ck["static"], ck["static_names"], ck["meta"]
            self._static_std_t = torch.from_numpy(
                np.ascontiguousarray(self.norm.standardize_static(self.static), dtype=np.float32)
            ).unsqueeze(0)
            self.model.eval()
            self.fallback = False
            log.info("Twin: loaded %s", path)
        except Exception as e:  # unloadable checkpoint: degrade, don't crash the server
            log.warning("Twin: failed to load %s (%s: %s) -> persistence fallback",
                        path, type(e).__name__, e)
            self.model = self.norm = self.static = self._static_std_t = None
            self.fallback = True
            return
        if ck.get("post") is not None:
            try:
                post = Post.from_dict(ck["post"])
                self.post = post if post.active else None
                log.info("Twin: post-processing %s", "on" if self.post else "empty")
            except Exception as e:  # bad post block: run the raw model, loudly
                log.warning("Twin: invalid 'post' in %s (%s: %s) -> raw model output",
                            path, type(e).__name__, e)
                self.post = None

    # ---------------------------------------------------------------- helpers
    def _fill_values(self, x: np.ndarray) -> np.ndarray:
        """Per-channel finite fill [6] for non-finite input cells: window mean, else climatology."""
        fill = _DEFAULT_FILL.copy()
        if self.norm is not None:
            fill = from_model(self.norm.state_mean.reshape(-1, 1, 1)).reshape(-1).astype(np.float32)
        for c in range(x.shape[1]):
            v = x[:, c][np.isfinite(x[:, c])]
            if v.size:
                fill[c] = v.mean()
        return fill

    @staticmethod
    def _clamp(y: np.ndarray) -> np.ndarray:
        y[..., RAIN, :, :] = np.maximum(y[..., RAIN, :, :], 0.0)
        y[..., T2M, :, :] = np.clip(y[..., T2M, :, :], T2M_MIN, T2M_MAX)
        return y

    # ---------------------------------------------------------------- rollout
    def rollout(self, init_states, doy_sincos, steps: int, perturbation=None) -> np.ndarray:
        """Physical-units rollout -> float32 [steps,6,H,W]; rollout[k] is day k+1."""
        x = np.array(init_states, dtype=np.float32, copy=True)
        if x.ndim != 4 or x.shape[0] != N_IN or x.shape[1] != len(CHANNELS):
            raise ValueError(f"init_states must be [{N_IN},{len(CHANNELS)},H,W], got {x.shape}")
        Hh, Ww = x.shape[-2:]
        steps = int(steps)
        if steps < 0:
            raise ValueError("steps must be >= 0")
        # None is treated as zeros and added the same way, so zero-shock == no-shock exactly.
        pert = (np.zeros(x.shape[1:], dtype=np.float32) if perturbation is None
                else np.asarray(perturbation, dtype=np.float32))
        if pert.shape != x.shape[1:]:
            raise ValueError(f"perturbation must be {list(x.shape[1:])}, got {list(pert.shape)}")
        x[-1] = x[-1] + pert

        bad = ~np.isfinite(x)
        if bad.any():
            fill = self._fill_values(x)
            x = np.where(bad, fill.reshape(1, -1, 1, 1), x).astype(np.float32)
        persist = self._clamp(x[-1].copy())  # [6,H,W]

        if steps == 0:
            return np.zeros((0, len(CHANNELS), Hh, Ww), dtype=np.float32)
        if self.fallback or (self.static is not None and self.static.shape[-2:] != (Hh, Ww)):
            if not self.fallback:
                log.warning("Twin: grid %s != static grid %s -> persistence", (Hh, Ww),
                            self.static.shape[-2:])
            return np.repeat(persist[None], steps, axis=0).astype(np.float32)

        d = np.asarray(doy_sincos, dtype=np.float32).reshape(2)
        win = torch.from_numpy(np.ascontiguousarray(self.norm.standardize(to_model(x)), dtype=np.float32))
        with torch.inference_mode():
            out = rollout_std(self.model, win.unsqueeze(0), self._static_std_t,
                              torch.from_numpy(d).unsqueeze(0), steps, self.norm)
        y = from_model(self.norm.unstandardize(out[0].numpy().astype(np.float32)))
        y = np.where(np.isfinite(y), y, persist[None]).astype(np.float32)
        if self.post is not None:  # optional: climatology blend + rain quantile map
            y = self.post.apply(y, d)
            y = np.where(np.isfinite(y), y, persist[None]).astype(np.float32)
        return self._clamp(y)
