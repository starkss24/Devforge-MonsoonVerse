"""HTTP server for the grader: POST /health and POST /simulate (docs/CONTRACT.md).

Run:  python serve.py --port 8000     (listens on 0.0.0.0)

Allowed imports only: numpy, torch (optional), fastapi, uvicorn, stdlib, common, twin.
Writes nothing to disk.
"""
from __future__ import annotations

import argparse
import logging
import math
import time
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from common import INIT_SHAPE, N_THREADS, STATE_SHAPE, enforce_validity
from twin import Twin

try:  # torch is optional: persistence backend needs only numpy
    import torch

    torch.set_num_threads(N_THREADS)
except Exception:  # noqa: BLE001 - missing/broken torch must not stop the server
    torch = None

log = logging.getLogger("serve")
MAX_STEPS = 60


def _bad(msg: str) -> HTTPException:
    return HTTPException(status_code=422, detail=msg)


def _as_array(value: Any, name: str, shape: tuple) -> np.ndarray:
    """Convert a JSON list to float64 with an exact shape and finite values, or raise 422."""
    if value is None:
        raise _bad(f"{name} is required")
    try:
        arr = np.asarray(value, dtype=np.float64)
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
    app = FastAPI(title="monsoon-twin")
    app.state.twin = twin if twin is not None else Twin()
    app.state.fallbacks = 0
    log.info("twin loaded, backend=%s", app.state.twin.backend)

    @app.api_route("/health", methods=["GET", "POST"])
    def health() -> dict:
        return {
            "status": "ready",
            "backend": app.state.twin.backend,
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
