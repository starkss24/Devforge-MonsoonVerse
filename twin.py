"""Twin.rollout(): the one function every lane codes against.

Hour-0 version is a PERSISTENCE STUB: it answers immediately with the
(perturbed) last input day repeated `steps` times. When Aether's weights land,
only the body of `_model_rollout` changes; the signature never does without
telling the team (docs/CONTRACT.md).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from common import INIT_SHAPE, STATE_SHAPE, enforce_validity

log = logging.getLogger("twin")
WEIGHTS_PATH =Path(__file__).resolve().parent / "weights" / "model.pt"


class Twin:
    def __init__(self, weights_path: Path | str | None = WEIGHTS_PATH):
        self.weights_path = Path(weights_path) if weights_path else None
        self.model = None  # set when real weights are loaded (hour 6)
        self.backend = "persistence"
        self.nonfinite_fallbacks = 0

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
        if not 1 <= steps <= 60:
            raise ValueError(f"steps must be in 1..60, got {steps}")
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
        raise NotImplementedError("real model wired in at hour 6 (Aether -> Vishwak)")
