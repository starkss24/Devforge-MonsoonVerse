"""Tests for serve.py via fastapi.testclient (requires httpx)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

import serve  # noqa: E402
from common import CH, INIT_SHAPE, STATE_SHAPE, T2M_MAX_K, T2M_MIN_K, doy_sincos  # noqa: E402
from twin import Twin  # noqa: E402


def _init(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(INIT_SHAPE)
    x[:, CH["t2m"]] = 300 + rng.normal(0, 2, x[:, 0].shape)
    x[:, CH["tp"]] = rng.gamma(0.8, 8.0, x[:, 0].shape)
    x[:, CH["u10"]] = rng.normal(5, 2, x[:, 0].shape)
    x[:, CH["v10"]] = rng.normal(3, 2, x[:, 0].shape)
    x[:, CH["msl"]] = rng.normal(100500, 100, x[:, 0].shape)
    x[:, CH["tcwv"]] = rng.uniform(20, 60, x[:, 0].shape)
    return x


def _req(pert=None, steps=10):
    return {"init_states": _init().tolist(), "doy_sincos": doy_sincos(200).tolist(),
            "steps": steps, "perturbation": pert}


@pytest.fixture(scope="module")
def client():
    return TestClient(serve.create_app())


@pytest.mark.parametrize("method", ["get", "post"])
def test_health(client, method):
    r = getattr(client, method)("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"
    h = r.json()
    assert h["backend"] in ("onnx", "torchscript", "persistence")     # twin.BACKEND_FILES + fallback
    assert isinstance(h["model_name"], str) and h["model_name"]
    # With the committed weights the server must really run the model (not silently fall back) and post-process.
    wdir = Path(__file__).resolve().parents[1] / "weights"
    if (wdir / "model.onnx").is_file() and "TWIN_BACKEND" not in os.environ:
        assert h["backend"] == "onnx" and not h["load_error"], h
        if (wdir / "post.npz").is_file():
            assert h["post"] == "loaded", h


def _b64(a):
    import base64
    a = np.ascontiguousarray(a, dtype="<f4")
    return {"b64": base64.b64encode(a.tobytes()).decode("ascii"), "shape": list(a.shape)}


def test_b64_input_tolerated(client):
    lists = np.asarray(client.post("/simulate", json=_req()).json()["rollout"])
    req = _req()
    req["init_states"] = _b64(_init())
    req["perturbation"] = _b64(np.zeros(STATE_SHAPE))
    r = client.post("/simulate", json=req)
    assert r.status_code == 200
    out = np.asarray(r.json()["rollout"])
    assert out.shape == lists.shape and np.allclose(out, lists, rtol=1e-6, atol=1e-2)


def test_b64_bad_length_422(client):
    req = _req()
    bad = _b64(_init())
    bad["shape"] = [3, 6, 32, 35]
    req["init_states"] = bad
    assert client.post("/simulate", json=req).status_code == 422


@pytest.mark.parametrize("steps", [1, 10, 14])
def test_kit_step_range(client, steps):
    out = np.asarray(client.post("/simulate", json=_req(steps=steps)).json()["rollout"])
    assert out.shape == (steps, *STATE_SHAPE)


def test_no_trailing_slash_redirect(client):
    assert client.post("/health/", follow_redirects=False).status_code != 307


def test_simulate_shape_and_validity(client):
    r = client.post("/simulate", json=_req())
    assert r.status_code == 200
    body = r.json()
    out = np.asarray(body["rollout"])
    assert out.shape == (10, *STATE_SHAPE)
    assert np.all(np.isfinite(out))
    assert out[:, CH["tp"]].min() >= 0
    assert T2M_MIN_K <= out[:, CH["t2m"]].min() and out[:, CH["t2m"]].max() <= T2M_MAX_K
    assert isinstance(body["inference_ms"], float)


def test_steps_default_and_custom(client):
    req = _req(steps=3)
    assert np.asarray(client.post("/simulate", json=req).json()["rollout"]).shape[0] == 3
    req.pop("steps")
    assert np.asarray(client.post("/simulate", json=req).json()["rollout"]).shape[0] == 10


def test_zero_perturbation_matches_null_exactly(client):
    a = np.asarray(client.post("/simulate", json=_req(None)).json()["rollout"])
    b = np.asarray(client.post("/simulate", json=_req(np.zeros(STATE_SHAPE).tolist())).json()["rollout"])
    assert np.array_equal(a, b)


def test_perturbation_changes_output(client):
    p = np.zeros(STATE_SHAPE)
    p[CH["t2m"]] = 3.0
    a = np.asarray(client.post("/simulate", json=_req(None)).json()["rollout"])
    b = np.asarray(client.post("/simulate", json=_req(p.tolist())).json()["rollout"])
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("mutate,needle", [
    (lambda r: r.update(init_states=r["init_states"][:2]), "init_states"),
    (lambda r: r.pop("init_states"), "init_states"),
    (lambda r: r.update(doy_sincos=[0.1]), "doy_sincos"),
    (lambda r: r.update(steps=0), "steps"),
    (lambda r: r.update(steps=1000), "steps"),
    (lambda r: r.update(steps="ten"), "steps"),
    (lambda r: r.update(perturbation=[[0.0]]), "perturbation"),
    (lambda r: r.update(init_states="abc"), "init_states"),
])
def test_bad_input_422(client, mutate, needle):
    req = _req()
    mutate(req)
    r = client.post("/simulate", json=req)
    assert r.status_code == 422
    assert needle in r.json()["detail"]


def test_non_object_body_422(client):
    assert client.post("/simulate", json=[1, 2]).status_code == 422
    assert client.post("/simulate", content=b"not json",
                       headers={"Content-Type": "application/json"}).status_code == 422


class _BrokenTwin(Twin):
    def rollout(self, *a, **k):
        raise RuntimeError("boom")


def test_backend_exception_falls_back_to_persistence(caplog):
    c = TestClient(serve.create_app(_BrokenTwin(weights_path=None)))
    p = np.zeros(STATE_SHAPE)
    p[CH["t2m"]] = 1.0
    r = c.post("/simulate", json=_req(p.tolist(), steps=4))
    assert r.status_code == 200
    out = np.asarray(r.json()["rollout"])
    expected = _init()[-1] + p
    assert np.allclose(out, np.repeat(expected[None], 4, axis=0))
    assert "falling back to persistence" in caplog.text
    assert c.post("/health").json()["fallbacks"] == 1
