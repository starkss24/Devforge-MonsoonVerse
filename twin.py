"""Twin.rollout(): the one function every lane codes against.

Backends
- "model": weights/model.pt is a TorchScript module (see docs/WEIGHTS.md) with
  forward(states: Tensor[3,6,32,36] physical float32, doy_sincos: Tensor[2]) -> Tensor[6,32,36]
  = the next day in PHYSICAL units. Normalisation, static maps, log1p rain etc. live inside it.
  Twin runs the autoregressive loop: slide the 3-day window, advance day-of-year by 1 per step.
- "persistence": used when no weights file exists, loading fails, or a rollout goes NaN/Inf.
  Every fallback is logged loudly and counted; nothing fails silently.

Signature of rollout() is frozen (docs/CONTRACT.md).
"""
from __future__ import annotations

import json
import logging
import math
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from common import CH, INIT_SHAPE, N_THREADS, STATE_SHAPE, T2M_MAX_K, T2M_MIN_K, TP_MIN, enforce_validity

log = logging.getLogger("twin")
WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "model.pt"
DEFAULT_MODEL_NAME = "persistence-stub"
MAX_STEPS = 60
DAYS_PER_YEAR = 365.0

# Plausible physical state used only to warm up / smoke-test a loaded module (K, mm/day, m/s, m/s, Pa, kg/m2).
_WARMUP_STATE = np.array([295.0, 4.0, 2.0, 1.0, 100500.0, 35.0], dtype=np.float32)


class Twin:
    def __init__(self, weights_path: Path | str | None = WEIGHTS_PATH, meta_path: Path | str | None = None):
        self.weights_path = Path(weights_path) if weights_path else None
        self.meta_path = Path(meta_path) if meta_path else (
            self.weights_path.with_name("model_meta.json") if self.weights_path else None)
        self.model = None
        self.backend = "persistence"
        self.model_name = DEFAULT_MODEL_NAME
        self.load_error: Optional[str] = None
        self.nonfinite_fallbacks = 0
        self._torch = None
        self._lock = threading.Lock()
        if self.weights_path is not None:
            self._load()

    # ---- loading -----------------------------------------------------------
    def _load(self) -> None:
        p = self.weights_path
        if not p.is_file():
            log.info("no weights at %s; backend=persistence", p)
            return
        try:
            with open(p, "rb") as f:
                if f.read(64).startswith(b"version https://git-lfs"):
                    raise ValueError("file is a Git LFS pointer, not the weights (run `git lfs pull`)")
            import torch

            torch.set_num_threads(N_THREADS)
            model = torch.jit.load(str(p), map_location="cpu")
            model.eval()
            self._torch, self.model = torch, model
            name = self._read_name(model)
            # Warm-up + smoke test inside construction, so cold start happens before /health says ready.
            warm = np.broadcast_to(_WARMUP_STATE[None, :, None, None], INIT_SHAPE).astype(np.float64)
            out = self._model_rollout(warm, np.array([0.0, 1.0]), 2)
            if not np.all(np.isfinite(out)):
                raise ValueError("warm-up rollout produced NaN/Inf")
            self.backend, self.model_name = "model", name
            log.info("loaded TorchScript model %r from %s; backend=model", name, p)
        except Exception as e:  # noqa: BLE001 - any load failure -> persistence, loudly
            self.model, self._torch = None, None
            self.load_error = f"{type(e).__name__}: {e}"
            log.error("FAILED to load weights %s (%s); STAYING ON PERSISTENCE", p, self.load_error)

    def _read_name(self, model) -> str:
        if self.meta_path is not None and self.meta_path.is_file():
            try:
                name = json.loads(self.meta_path.read_text(encoding="utf-8")).get("model_name")
                if isinstance(name, str) and name:
                    return name
            except (OSError, ValueError) as e:
                log.warning("could not read %s (%s)", self.meta_path, e)
        name = getattr(model, "model_name", None)
        if isinstance(name, str) and name:
            return name
        return f"torchscript:{self.weights_path.name}"

    # ---- public contract -------------------------------------------------
    def rollout(
        self,
        init_states: np.ndarray,
        doy_sincos: np.ndarray,
        steps: int = 10,
        perturbation: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """init_states [3,6,32,36] physical units, oldest day first.
        doy_sincos [2] for the last input day. perturbation None or [6,32,36]
        physical deltas added to init_states[-1]. Returns [steps,6,32,36]."""
        x = np.asarray(init_states, dtype=np.float64)
        if x.shape != INIT_SHAPE:
            raise ValueError(f"init_states must be {INIT_SHAPE}, got {x.shape}")
        d = np.asarray(doy_sincos, dtype=np.float64).reshape(-1)
        if d.shape != (2,):
            raise ValueError(f"doy_sincos must have 2 values, got {d.shape}")
        steps = int(steps)
        if not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"steps must be in 1..{MAX_STEPS}, got {steps}")
        if perturbation is not None:
            p = np.asarray(perturbation, dtype=np.float64)
            if p.shape != STATE_SHAPE:
                raise ValueError(f"perturbation must be {STATE_SHAPE}, got {p.shape}")
            x = x.copy()
            x[-1] = x[-1] + p

        if self.model is not None:
            out = self._model_rollout(x, d, steps)
            if np.all(np.isfinite(out)):
                return enforce_validity(out)
            # never return NaN/Inf: count, log loudly, fall through to persistence
            self.nonfinite_fallbacks += 1
            log.warning("model rollout produced NaN/Inf; persistence fallback (count=%d)",
                        self.nonfinite_fallbacks)
        return enforce_validity(self.persistence(x, steps))

    # ---- backends --------------------------------------------------------
    @staticmethod
    def persistence(x: np.ndarray, steps: int) -> np.ndarray:
        return np.repeat(x[-1][None], steps, axis=0)

    def _model_rollout(self, x: np.ndarray, d: np.ndarray, steps: int) -> np.ndarray:
        """Autoregressive loop. Returns float64 [steps,6,32,36]; NaN-filled if the module misbehaves
        (wrong shape / non-finite), which rollout() turns into a counted persistence fallback."""
        torch = self._torch
        theta = math.atan2(float(d[0]), float(d[1]))  # angle of the last input day
        out = np.full((steps, *STATE_SHAPE), np.nan, dtype=np.float64)
        with self._lock, torch.inference_mode():
            window = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
            for k in range(1, steps + 1):
                a = theta + 2.0 * math.pi * k / DAYS_PER_YEAR  # forecast step k is doy + k
                dsc = torch.tensor([math.sin(a), math.cos(a)], dtype=torch.float32)
                y = self.model(window, dsc)
                if not isinstance(y, torch.Tensor) or tuple(y.shape) != STATE_SHAPE:
                    log.error("model returned %s, expected tensor %s",
                              getattr(y, "shape", type(y)), STATE_SHAPE)
                    return out
                y = y.detach().to(torch.float32)
                if not bool(torch.isfinite(y).all()):
                    return out
                # Same clamps as enforce_validity, applied before feeding the day back in.
                y = y.clone()
                y[CH["tp"]].clamp_(min=TP_MIN)
                y[CH["t2m"]].clamp_(T2M_MIN_K, T2M_MAX_K)
                out[k - 1] = y.numpy()
                window = torch.cat([window[1:], y[None]], dim=0)
        return out
