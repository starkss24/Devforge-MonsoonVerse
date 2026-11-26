"""Twin.rollout(): the one function every lane codes against.

Backends (one-day step contract, docs/WEIGHTS.md):
    step(states: float32[3,6,32,36] physical, doy_sincos: float32[2]) -> float32[6,32,36] next day, physical.
  Normalisation, static maps, log1p rain etc. live inside the exported model.
  Twin runs the autoregressive loop: slide the 3-day window, advance day-of-year by 1 per step.
- "onnx": weights/model.onnx run with onnxruntime (CPU, 4 intra-op threads). Preferred, because
  TorchScript (torch.jit) is deprecated/unsupported on Python 3.14+.
- "torchscript": weights/model.pt TorchScript module (used if there is no model.onnx or it fails to load).
- "persistence": used when no weights load, or a rollout goes NaN/Inf.
  Every fallback is logged loudly and counted; nothing fails silently.
Override the preference with env TWIN_BACKEND=onnx|torchscript|persistence (testing).

Signature of rollout() is frozen (docs/CONTRACT.md).
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from common import CH, INIT_SHAPE, N_THREADS, STATE_SHAPE, T2M_MAX_K, T2M_MIN_K, TP_MIN, enforce_validity

log = logging.getLogger("twin")
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DEFAULT_MODEL_NAME = "persistence-stub"
MAX_STEPS = 60
DAYS_PER_YEAR = 365.0
BACKEND_FILES = {"onnx": "model.onnx", "torchscript": "model.pt"}  # in preference order
SUFFIX_BACKEND = {".onnx": "onnx", ".pt": "torchscript"}

# Plausible physical state used only to warm up / smoke-test a loaded model (K, mm/day, m/s, m/s, Pa, kg/m2).
_WARMUP_STATE = np.array([295.0, 4.0, 2.0, 1.0, 100500.0, 35.0], dtype=np.float32)


def _check_not_lfs_pointer(p: Path) -> None:
    with open(p, "rb") as f:
        if f.read(64).startswith(b"version https://git-lfs"):
            raise ValueError("file is a Git LFS pointer, not the weights (run `git lfs pull`)")


class Twin:
    def __init__(self, weights_path: Path | str | None = WEIGHTS_DIR, meta_path: Path | str | None = None):
        """weights_path: a directory (looks for model.onnx, then model.pt) or one weights file
        (.onnx -> onnx, anything else -> TorchScript). None -> persistence."""
        self.weights_path = Path(weights_path) if weights_path else None
        wdir = None
        if self.weights_path is not None:
            wdir = self.weights_path if self.weights_path.is_dir() else self.weights_path.parent
        self.meta_path = Path(meta_path) if meta_path else (wdir / "model_meta.json" if wdir else None)
        self.model = None                 # TorchScript module or onnxruntime InferenceSession
        self.backend = "persistence"
        self.model_name = DEFAULT_MODEL_NAME
        self.load_error: Optional[str] = None
        self.loaded_path: Optional[Path] = None
        self.nonfinite_fallbacks = 0
        self._step: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
        self._torch = None
        self._issue = ""
        self._lock = threading.Lock()
        if self.weights_path is not None:
            self._load()

    # ---- loading -----------------------------------------------------------
    def _candidates(self) -> list[tuple[str, Path]]:
        p = self.weights_path
        if not p.is_dir():
            return [(SUFFIX_BACKEND.get(p.suffix.lower(), "torchscript"), p)]
        cands = [(b, p / f) for b, f in BACKEND_FILES.items()]
        force = os.environ.get("TWIN_BACKEND", "").strip().lower()
        if force:
            if force == "persistence":
                log.warning("TWIN_BACKEND=persistence: not loading any weights")
                return []
            if force not in BACKEND_FILES:
                log.warning("ignoring unknown TWIN_BACKEND=%r (use onnx|torchscript|persistence)", force)
            else:
                log.info("TWIN_BACKEND=%s overrides the default preference (onnx first)", force)
                cands = [c for c in cands if c[0] == force]
        present = [c for c in cands if c[1].is_file()]
        if len(present) > 1:
            log.info("both %s found; preferring %s (set TWIN_BACKEND to override)",
                     " and ".join(c[1].name for c in present), present[0][1].name)
        return cands

    def _load(self) -> None:
        errors = []
        for backend, p in self._candidates():
            if not p.is_file():
                log.info("no %s weights at %s", backend, p)
                continue
            try:
                _check_not_lfs_pointer(p)
                if backend == "onnx":
                    self._load_onnx(p)
                else:
                    self._load_torchscript(p)
                # Warm-up + smoke test inside construction, so cold start happens before /health says ready.
                warm = np.broadcast_to(_WARMUP_STATE[None, :, None, None], INIT_SHAPE).astype(np.float64)
                out = self._model_rollout(warm, np.array([0.0, 1.0]), 2)
                if not np.all(np.isfinite(out)):
                    raise ValueError(f"warm-up rollout produced NaN/Inf ({self._issue or 'non-finite output'})")
                self.backend, self.loaded_path = backend, p
                self.model_name = self._read_name(backend, p)
                log.info("loaded %s model %r from %s; backend=%s", backend, self.model_name, p, backend)
                break
            except Exception as e:  # noqa: BLE001 - any load failure -> next candidate / persistence, loudly
                self.model, self._step, self._torch = None, None, None
                errors.append(f"{p.name}: {type(e).__name__}: {e}")
                log.error("FAILED to load %s weights %s (%s)", backend, p, errors[-1])
        if errors:
            self.load_error = "; ".join(errors)
        if self.model is None:
            if errors:
                log.error("no weights could be loaded; STAYING ON PERSISTENCE")
            else:
                log.info("no weights in %s; backend=persistence", self.weights_path)

    def _load_onnx(self, p: Path) -> None:
        import onnxruntime as ort  # optional dependency: only needed when model.onnx is present

        so = ort.SessionOptions()
        so.intra_op_num_threads = N_THREADS
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if hasattr(so, "use_deterministic_compute"):
            so.use_deterministic_compute = True
        so.log_severity_level = 3
        sess = ort.InferenceSession(str(p), sess_options=so, providers=["CPUExecutionProvider"])
        ins = {i.name: i for i in sess.get_inputs()}
        if set(ins) >= {"states", "doy_sincos"}:
            s_in, d_in = ins["states"], ins["doy_sincos"]
        elif len(ins) == 2:
            s_in, d_in = sess.get_inputs()
            log.warning("ONNX inputs are %s, expected states/doy_sincos; using them by position", list(ins))
        else:
            raise ValueError(f"ONNX model must have inputs (states, doy_sincos), got {list(ins)}")
        # Optional leading batch dim of 1: states [1,3,6,32,36], doy_sincos [1,2] -> output [1,6,32,36].
        batched = len(s_in.shape) == 5
        d_rank = len(d_in.shape)
        out_name = sess.get_outputs()[0].name
        s_name, d_name = s_in.name, d_in.name

        def step(window: np.ndarray, dsc: np.ndarray) -> np.ndarray:
            s = window[None] if batched else window
            d = dsc.reshape((1,) * (d_rank - 1) + (2,))
            y = sess.run([out_name], {s_name: s, d_name: d})[0]
            if y.ndim == 4 and y.shape[0] == 1:
                y = y[0]
            return y

        self.model, self._step = sess, step

    def _load_torchscript(self, p: Path) -> None:
        import torch

        torch.set_num_threads(N_THREADS)
        model = torch.jit.load(str(p), map_location="cpu")
        model.eval()

        def step(window: np.ndarray, dsc: np.ndarray):
            y = model(torch.from_numpy(window), torch.from_numpy(dsc))
            if not isinstance(y, torch.Tensor):
                return y
            return y.detach().to(torch.float32).numpy()

        self._torch, self.model, self._step = torch, model, step

    def _read_name(self, backend: str, p: Path) -> str:
        if self.meta_path is not None and self.meta_path.is_file():
            try:
                name = json.loads(self.meta_path.read_text(encoding="utf-8")).get("model_name")
                if isinstance(name, str) and name:
                    return name
            except (OSError, ValueError) as e:
                log.warning("could not read %s (%s)", self.meta_path, e)
        if backend == "onnx":
            name = self.model.get_modelmeta().custom_metadata_map.get("model_name")
        else:
            name = getattr(self.model, "model_name", None)
        if isinstance(name, str) and name:
            return name
        return f"{backend}:{p.name}"

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
            log.warning("%s rollout produced NaN/Inf (%s); persistence fallback (count=%d)",
                        self.backend, self._issue, self.nonfinite_fallbacks)
        return enforce_validity(self.persistence(x, steps))

    # ---- backends --------------------------------------------------------
    @staticmethod
    def persistence(x: np.ndarray, steps: int) -> np.ndarray:
        return np.repeat(x[-1][None], steps, axis=0)

    def _model_rollout(self, x: np.ndarray, d: np.ndarray, steps: int) -> np.ndarray:
        """Autoregressive loop (backend-independent). Returns float64 [steps,6,32,36]; NaN-filled if the
        model misbehaves (wrong shape / non-finite), which rollout() turns into a counted persistence fallback."""
        theta = math.atan2(float(d[0]), float(d[1]))  # angle of the last input day
        out = np.full((steps, *STATE_SHAPE), np.nan, dtype=np.float64)
        ctx = self._torch.inference_mode() if self._torch is not None else _NullCtx()
        with self._lock, ctx:
            self._issue = ""
            window = np.ascontiguousarray(x, dtype=np.float32)
            for k in range(1, steps + 1):
                a = theta + 2.0 * math.pi * k / DAYS_PER_YEAR  # forecast step k is doy + k
                dsc = np.array([math.sin(a), math.cos(a)], dtype=np.float32)
                y = self._step(window, dsc)
                if not isinstance(y, np.ndarray) or y.shape != STATE_SHAPE:
                    self._issue = f"model returned {getattr(y, 'shape', type(y))}, expected {STATE_SHAPE}"
                    log.error(self._issue)
                    return out
                y = np.array(y, dtype=np.float32, copy=True)
                if not np.all(np.isfinite(y)):
                    self._issue = f"non-finite output at step {k}"
                    return out
                # Same clamps as enforce_validity, applied before feeding the day back in.
                np.maximum(y[CH["tp"]], TP_MIN, out=y[CH["tp"]])
                np.clip(y[CH["t2m"]], T2M_MIN_K, T2M_MAX_K, out=y[CH["t2m"]])
                out[k - 1] = y
                window = np.concatenate([window[1:], y[None]], axis=0)
        return out


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
