"""Dynamics checks: shocks act locally, then spread; zero shock changes nothing."""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import BOXES, CH, STATE_SHAPE, T2M_MAX_K, T2M_MIN_K, box_mask, doy_sincos  # noqa: E402
from eval.shock_test import (  # noqa: E402
    ShockCheckError,
    http_rollout_fn,
    main,
    ring_distance,
    shock_report,
    synthetic_window,
)
from twin import Twin  # noqa: E402


@pytest.fixture(scope="module")
def window():
    init, doy = synthetic_window(seed=0)
    return init, doy_sincos(doy)


def blur3(f: np.ndarray) -> np.ndarray:
    """3x3 box blur on the last two axes, edge padding."""
    p = np.pad(f, [(0, 0)] * (f.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    h, w = f.shape[-2:]
    return sum(p[..., i:i + h, j:j + w] for i in range(3) for j in range(3)) / 9.0


def diffusing_rollout(init_states, doy_sincos, steps=10, perturbation=None):
    """Toy twin: start from the (perturbed) last day, blur once per step."""
    x = np.asarray(init_states, dtype=np.float64)[-1].copy()
    if perturbation is not None:
        x = x + np.asarray(perturbation, dtype=np.float64)
    out = []
    for _ in range(int(steps)):
        x = blur3(x)
        out.append(x)
    return np.stack(out)


def test_synthetic_window_is_physical(window):
    init, _ = window
    assert init.shape == (3,) + STATE_SHAPE and np.all(np.isfinite(init))
    assert np.all(init[:, CH["tp"]] >= 0)
    assert T2M_MIN_K < init[:, CH["t2m"]].min() and init[:, CH["t2m"]].max() < T2M_MAX_K
    assert 290 < init[:, CH["t2m"]].mean() < 310
    assert 9.5e4 < init[:, CH["msl"]].mean() < 1.05e5
    assert np.abs(init[:, CH["u10"]]).max() <= 10 and np.abs(init[:, CH["v10"]]).max() <= 10
    assert 20 < init[:, CH["tcwv"]].mean() < 60
    again, _ = synthetic_window(seed=0)
    assert np.array_equal(init, again)


def test_ring_distance():
    m = box_mask(*BOXES["warm_ocean"])
    d = ring_distance(m)
    assert np.all(d[m] == 0) and np.all(d[~m] >= 1)
    i, j = np.argwhere(m)[0]  # south-west corner of the box
    if i > 0 and j > 0:
        assert d[i - 1, j - 1] == 1
    if i > 1:
        assert d[i - 2, j] == 2


def test_persistence_twin_is_strictly_local(window):
    init, dsc = window
    rep = shock_report(Twin().rollout, init, dsc, steps=10)
    assert rep["determinism"]["pass"]
    assert rep["shocks"]["zero"]["max_abs_diff"] == 0.0
    for name, chans in (("warm_ocean", ("t2m", "tcwv")), ("monsoon_stall", ("u10", "v10"))):
        s = rep["shocks"][name]
        assert set(s["perturbed_channels"]) == set(chans)
        for c in chans:
            for d in ("1", "4", "8"):
                v = s["channels"][c][d]
                assert v["inside"] > 0
                assert v["outside"] == 0.0 and v["outside_is_zero"]
                assert v["radial"][0] > 0 and all(p == 0.0 for p in v["radial"][1:])
    # warm_ocean t2m is +3 K everywhere in the box (no clamp hit at ~300 K)
    assert rep["shocks"]["warm_ocean"]["channels"]["t2m"]["1"]["inside"] == pytest.approx(3.0)
    json.dumps(rep, allow_nan=False)  # dashboard-safe JSON


def test_diffusing_rollout_spreads_with_lead_day(window):
    init, dsc = window
    rep = shock_report(diffusing_rollout, init, dsc, steps=10)
    assert rep["shocks"]["zero"]["max_abs_diff"] == 0.0
    v = rep["shocks"]["warm_ocean"]["channels"]["t2m"]
    d1, d4, d8 = v["1"], v["4"], v["8"]
    # acts locally first: strongest in the box, and far rings untouched at day 1
    assert d1["inside"] > d1["outside"] > 0
    assert d1["radial"][0] > d1["radial"][1] > 0
    assert all(p == 0.0 for p in d1["radial"][2:])
    # then spreads: inside/outside ratio falls, far rings grow, centroid moves out
    assert d1["ratio"] > d4["ratio"] > d8["ratio"]
    assert d8["radial"][4] > d4["radial"][4] > d1["radial"][4] == 0.0
    assert d1["centroid_ring"] < d4["centroid_ring"] < d8["centroid_ring"]


def test_zero_shock_violation_is_caught(window):
    init, dsc = window

    def leaky(init_states, doy_sincos, steps=10, perturbation=None):
        # calls: 0 = baseline, 1 = determinism probe, 2 = the named zero shock
        out = Twin().rollout(init_states, doy_sincos, steps, perturbation)
        if leaky.calls >= 2:
            out = out + 1e-9  # tiny leak, far below the 1e-3*std tolerance
        leaky.calls += 1
        return out

    leaky.calls = 0
    with pytest.raises(ShockCheckError, match="zero shock changed"):
        shock_report(leaky, init, dsc, steps=4, shocks=("zero",))


def test_nondeterminism_is_caught(window):
    init, dsc = window

    def noisy(init_states, doy_sincos, steps=10, perturbation=None):
        out = Twin().rollout(init_states, doy_sincos, steps, perturbation)
        return out + (1e-7 if perturbation is not None else 0.0)

    with pytest.raises(ShockCheckError, match="None vs zeros"):
        shock_report(noisy, init, dsc, steps=4)


def test_none_vs_zeros_identical_on_twin(window):
    init, dsc = window
    t = Twin()
    a = t.rollout(init, dsc, 10, None)
    b = t.rollout(init, dsc, 10, np.zeros(STATE_SHAPE))
    assert np.array_equal(a, b)


@pytest.fixture()
def stub_server():
    """Stdlib HTTP server implementing /simulate per CONTRACT.md with the in-process Twin."""
    twin = Twin()

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            p = body.get("perturbation")
            out = twin.rollout(np.array(body["init_states"]), np.array(body["doy_sincos"]),
                               body.get("steps", 10), None if p is None else np.array(p))
            data = json.dumps({"rollout": out.tolist(), "inference_ms": 0.0}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_http_rollout_fn_matches_in_process(window, stub_server):
    init, dsc = window
    fn = http_rollout_fn(stub_server)
    delta = np.zeros(STATE_SHAPE)
    delta[CH["t2m"]] = 1.0
    assert np.array_equal(fn(init, dsc, steps=3, perturbation=delta),
                          Twin().rollout(init, dsc, 3, delta))
    rep = shock_report(fn, init, dsc, steps=8)
    assert rep["shocks"]["zero"]["max_abs_diff"] == 0.0


def test_cli_writes_json(tmp_path, capsys):
    out = tmp_path / "rep.json"
    assert main(["--json-out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert set(rep["shocks"]) == {"warm_ocean", "monsoon_stall", "zero"}
    assert "warm_ocean" in capsys.readouterr().out
