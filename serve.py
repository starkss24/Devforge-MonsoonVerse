"""HTTP server for the grader: POST /health and POST /simulate (docs/CONTRACT.md).

Run:  python serve.py --port 8000     (listens on 0.0.0.0)

Allowed imports only: numpy, torch (optional), fastapi, uvicorn, stdlib, common, twin.
Never import monsoon_r1 (the organisers' kit is not in the judge sandbox).
Writes nothing to disk.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import logging
import math
import time
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from common import INIT_SHAPE, N_THREADS, STATE_SHAPE, enforce_validity
from twin import MAX_STEPS, Twin

try:  # torch is optional: persistence backend needs only numpy
    import torch

    torch.set_num_threads(N_THREADS)
except Exception:  # noqa: BLE001 - missing/broken torch must not stop the server
    torch = None

log = logging.getLogger("serve")


def _bad(msg: str) -> HTTPException:
    return HTTPException(status_code=422, detail=msg)


def _decode_b64(value: dict, name: str) -> np.ndarray:
    """Tolerated wire form {"b64": little-endian float32 bytes, "shape": [...]} (kit contract.decode_array)."""
    if "b64" not in value or "shape" not in value:
        raise _bad(f"{name}: array dict needs 'b64' and 'shape'")
    try:
        shp = tuple(int(s) for s in value["shape"])
        if any(s < 0 or s > 100000 for s in shp) or int(np.prod(shp)) > 20_000_000:
            raise ValueError("array too large")
        raw = base64.b64decode(value["b64"], validate=True)
        if len(raw) != int(np.prod(shp)) * 4:
            raise ValueError("b64 length does not match shape")
        return np.frombuffer(raw, dtype="<f4").reshape(shp).astype(np.float64)
    except (ValueError, TypeError, binascii.Error) as e:
        raise _bad(f"{name}: bad b64 array ({e})")


def _as_array(value: Any, name: str, shape: tuple) -> np.ndarray:
    """Convert a JSON list (canonical) or b64 dict (tolerated) to float64 with an exact shape
    and finite values, or raise 422."""
    if value is None:
        raise _bad(f"{name} is required")
    try:
        arr = _decode_b64(value, name) if isinstance(value, dict) else np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise _bad(f"{name} must be a numeric nested list of shape {list(shape)} ({e})")
    if arr.shape != tuple(shape):
        raise _bad(f"{name} must have shape {list(shape)}, got {list(arr.shape)}")
    if not np.all(np.isfinite(arr)):
        raise _bad(f"{name} contains NaN or Inf")
    return arr


def parse_request(body: Any) -> tuple[np.ndarray, np.ndarray, int, Optional[np.ndarray]]:
    if not isinstance(body, dict):
        raise _bad("request body must be a JSON object")
    x = _as_array(body.get("init_states"), "init_states", INIT_SHAPE)
    d = _as_array(body.get("doy_sincos"), "doy_sincos", (2,))
    steps = body.get("steps", 10)
    if (isinstance(steps, bool) or not isinstance(steps, (int, float))
            or not math.isfinite(steps) or float(steps) != int(steps)):
        raise _bad(f"steps must be an integer, got {steps!r}")
    steps = int(steps)
    if not 1 <= steps <= MAX_STEPS:
        raise _bad(f"steps must be in 1..{MAX_STEPS}, got {steps}")
    pert = body.get("perturbation")
    p = None if pert is None else _as_array(pert, "perturbation", STATE_SHAPE)
    return x, d, steps, p


def create_app(twin: Optional[Twin] = None) -> FastAPI:
    app = FastAPI(title="monsoon-twin", redirect_slashes=False)  # contract: no redirects
    # Twin() loads + warms up the model here, before uvicorn listens, so the cold start is
    # absorbed by the judge's /health polling and never by a scored /simulate call.
    app.state.twin = twin if twin is not None else Twin()
    app.state.fallbacks = 0
    log.info("twin loaded, backend=%s model_name=%s", app.state.twin.backend,
             getattr(app.state.twin, "model_name", "?"))

    @app.api_route("/health", methods=["GET", "POST"])
    def health() -> dict:
        t = app.state.twin
        return {
            "status": "ready",
            "model_name": getattr(t, "model_name", "persistence-stub"),
            "backend": t.backend,
            "load_error": getattr(t, "load_error", None),
            "nonfinite_fallbacks": getattr(t, "nonfinite_fallbacks", 0),
            "torch": torch is not None,
            "fallbacks": app.state.fallbacks,
        }

    @app.post("/simulate")
    async def simulate(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception as e:  # noqa: BLE001
            raise _bad(f"body is not valid JSON ({e})")
        x, d, steps, p = parse_request(body)

        t0 = time.perf_counter()
        try:
            out = await run_in_threadpool(app.state.twin.rollout, x, d, steps=steps, perturbation=p)
        except Exception:  # noqa: BLE001 - logged loudly, answered with persistence
            app.state.fallbacks += 1
            log.exception("twin.rollout failed; falling back to persistence (count=%d)",
                          app.state.fallbacks)
            xp = x.copy()
            if p is not None:
                xp[-1] = xp[-1] + p
            out = enforce_validity(Twin.persistence(xp, steps))
        ms = (time.perf_counter() - t0) * 1000.0

        out = np.asarray(out, dtype=np.float64)
        if out.shape != (steps, *STATE_SHAPE) or not np.all(np.isfinite(out)):
            # Twin guarantees this; if it ever breaks, say so instead of sending garbage.
            log.error("invalid rollout shape=%s finite=%s", out.shape, np.all(np.isfinite(out)))
            raise HTTPException(status_code=500, detail="internal rollout produced invalid output")
        return JSONResponse({"rollout": out.tolist(), "inference_ms": ms})

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
