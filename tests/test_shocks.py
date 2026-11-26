"""Dynamics checks via the judge's own functions: shocks act locally, then spread without teleporting;
the zero shock changes nothing. Point totals are hand-computed from the contract's rules."""
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

from monsoon_r1 import scoring as S  # noqa: E402
from monsoon_r1.contract import BOX_BAY_OF_BENGAL, box_mask  # noqa: E402

from common import CH, STATE_SHAPE, T2M_MAX_K, T2M_MIN_K, doy_sincos  # noqa: E402
from eval.shock_test import (  # noqa: E402
    ShockCheckError,
    http_rollout_fn,
    load_channel_std,
    main,
    ring_distance,
    shock_report,
    synthetic_window,
)
from twin import Twin  # noqa: E402

STD = np.array([13.46, 7.55, 2.90, 2.43, 619.0, 20.5])   # roughly the kit's stats.json


@pytest.fixture(autouse=True)
def _persistence_backend(monkeypatch):
    """These tests check the shock-test harness against the persistence stub (exact expected points).
    Pin Twin() to persistence so they stay valid once real weights are committed in weights/."""
    monkeypatch.setenv("TWIN_BACKEND", "persistence")


@pytest.fixture(scope="module")
def window():
    init, doy = synthetic_window(seed=0)
    return init, doy_sincos(doy)


def blur3(f: np.ndarray) -> np.ndarray:
    """3x3 box blur on the last two axes, edge padding."""
    p = np.pad(f, [(0, 0)] * (f.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    h, w = f.shape[-2:]
    return sum(p[..., i:i + h, j:j + w] for i in range(3) for j in range(3)) / 9.0


def blur5(f: np.ndarray) -> np.ndarray:
    """5x5 box blur on the last two axes, edge padding."""
    p = np.pad(f, [(0, 0)] * (f.ndim - 2) + [(2, 2), (2, 2)], mode="edge")
    h, w = f.shape[-2:]
    return sum(p[..., i:i + h, j:j + w] for i in range(5) for j in range(5)) / 25.0


def _diffuser(blur, per_day):
    def rollout(init_states, doy_sincos, steps=10, perturbation=None):
        x = np.asarray(init_states, dtype=np.float64)[-1].copy()
        if perturbation is not None:
            x = x + np.asarray(perturbation, dtype=np.float64)
        out = []
        for _ in range(int(steps)):
            for _ in range(per_day):
                x = blur(x)
            out.append(x)
        return np.stack(out)
    return rollout


# slow: one 3x3 blur per day (1 cell/day reach); fast: two 5x5 blurs per day (4 cells/day, under the 8-cell limit)
diffusing_rollout = _diffuser(blur3, 1)
fast_diffusing_rollout = _diffuser(blur5, 2)


def _delta_model(transform):
    """Persistence base; the response to a perturbation is transform(delta) at every step."""
    def f(init_states, doy_sincos, steps=10, perturbation=None):
        base = np.repeat(np.asarray(init_states, dtype=np.float64)[-1][None], int(steps), axis=0)
        if perturbation is None:
            return base
        return base + transform(np.asarray(perturbation, dtype=np.float64))[None]
    return f


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
    m = box_mask(BOX_BAY_OF_BENGAL)
    assert m.sum() == 144                                   # 12 x 12 cell centres inside lat 8..20, lon 82..94
    d = ring_distance(m)
    assert np.all(d[m] == 0) and np.all(d[~m] >= 1)
    i, j = np.argwhere(m)[0]
    assert d[i - 1, j - 1] == 1 and d[i - 2, j] == 2


def test_shocks_are_the_kits(window):
    init, dsc = window
    rep = shock_report(Twin().rollout, init, dsc, steps=4, channel_std=STD)
    m = box_mask(BOX_BAY_OF_BENGAL)
    assert rep["shocks"]["warm_ocean"]["max_abs_delta"]["t2m"] == pytest.approx(3.0)
    assert rep["shocks"]["warm_ocean"]["max_abs_delta"]["tcwv"] == pytest.approx(
        0.15 * init[-1, CH["tcwv"]][m].max(), rel=1e-5)
    assert set(rep["shocks"]["monsoon_stall"]["perturbed_channels"]) == {"u10", "v10"}


def test_persistence_twin_is_local_but_does_not_propagate(window):
    init, dsc = window
    rep = shock_report(Twin().rollout, init, dsc, steps=10, channel_std=STD)
    assert rep["determinism"]["pass"] and rep["shocks"]["zero"]["max_abs_diff"] == 0.0
    for name, chans in (("warm_ocean", ("t2m", "tcwv")), ("monsoon_stall", ("u10", "v10"))):
        s = rep["shocks"][name]
        assert set(s["perturbed_channels"]) == set(chans)
        for c in chans:
            for d in ("1", "4", "8"):
                v = s["channels"][c][d]
                assert v["inside"] > 0 and v["outside"] == 0.0 and v["outside_is_zero"]
                assert v["radial"][0] > 0 and all(p == 0.0 for p in v["radial"][1:])
        j = rep["judge"]["shocks"][name]
        assert j["nonzero"] and j["localized"] and j["stable"] and j["no_teleport"]
        assert not j["propagates"] and j["near_frac_day1"] == 1.0
    # 2 shocks x 7.5 x (nonzero + localized + stable) / 4, null consistent -> 11.25
    assert rep["judge"]["null_ok"] and rep["judge"]["points"] == pytest.approx(11.25)
    json.dumps(rep, allow_nan=False)


def test_fast_diffusing_model_passes_every_kit_check(window):
    init, dsc = window
    rep = shock_report(fast_diffusing_rollout, init, dsc, steps=10, channel_std=STD)
    for name in ("warm_ocean", "monsoon_stall"):
        j = rep["judge"]["shocks"][name]
        assert j["nonzero"] and j["localized"] and j["stable"] and j["no_teleport"], j
        assert j["propagates"], j                          # energy beyond the 3-cell neighbourhood grows by day 4-8
        eo, tot = np.array(j["energy_outside"]), np.add(j["energy_near"], j["energy_outside"])
        assert eo[3:8].mean() > 1.5 * eo[0] and eo[3:8].mean() > 0.02 * tot.max()   # the kit's rule, by hand
        assert j["far_frac_max"] < 0.05
    assert rep["judge"]["points"] == pytest.approx(15.0)


def test_slow_diffusion_is_too_weak_to_count_as_propagation(window):
    """One 3x3 blur per day spreads, but days 4-8 energy outside the neighbourhood stays under 2% of the peak:
    the kit gives no propagation quarter (11.25/15). Radial profiles still show the spread for the Shock Lab."""
    init, dsc = window
    rep = shock_report(diffusing_rollout, init, dsc, steps=10, channel_std=STD)
    for name in ("warm_ocean", "monsoon_stall"):
        j = rep["judge"]["shocks"][name]
        eo, tot = np.array(j["energy_outside"]), np.add(j["energy_near"], j["energy_outside"])
        assert j["localized"] and j["no_teleport"] and not j["propagates"]
        assert eo[3:8].mean() > 1.5 * eo[0] and eo[3:8].mean() < 0.02 * tot.max()
    assert rep["judge"]["points"] == pytest.approx(11.25)
    v = rep["shocks"]["warm_ocean"]["channels"]["t2m"]
    d1, d4, d8 = v["1"], v["4"], v["8"]
    assert d1["inside"] > d1["outside"] > 0
    assert d1["radial"][0] > d1["radial"][1] > 0 and all(p == 0.0 for p in d1["radial"][2:])
    assert d1["ratio"] > d4["ratio"] > d8["ratio"]
    assert d8["radial"][4] > d4["radial"][4] > d1["radial"][4] == 0.0
    assert d1["centroid_ring"] < d4["centroid_ring"] < d8["centroid_ring"]


def test_teleporting_response_is_not_localized(window):
    """Response copied 15 cells away on day 1: half the energy is far -> not localised, teleport flagged.
    Only the 'non-zero' quarter counts: 2 x 7.5 / 4 = 3.75."""
    init, dsc = window

    def tele(d):
        return d + np.roll(d, 15, axis=-2) * (np.abs(np.roll(d, 15, axis=-2)) > 0)

    rep = shock_report(_delta_model(tele), init, dsc, steps=10, channel_std=STD)
    for name in ("warm_ocean", "monsoon_stall"):
        j = rep["judge"]["shocks"][name]
        assert j["nonzero"] and not j["localized"] and not j["no_teleport"] and not j["propagates"]
        assert j["near_frac_day1"] == pytest.approx(0.5, abs=1e-6)
    assert rep["judge"]["points"] == pytest.approx(3.75)


def test_global_flat_nudge_earns_only_nonzero(window):
    init, dsc = window
    rep = shock_report(_delta_model(lambda d: np.broadcast_to(d.mean(axis=(-2, -1), keepdims=True), d.shape)),
                       init, dsc, steps=10, channel_std=STD)
    assert all(not j["localized"] for j in rep["judge"]["shocks"].values())
    assert rep["judge"]["points"] == pytest.approx(3.75)


def test_exploding_response_is_unstable(window):
    init, dsc = window
    rep = shock_report(_delta_model(lambda d: 100.0 * d), init, dsc, steps=10, channel_std=STD)
    for j in rep["judge"]["shocks"].values():
        assert j["nonzero"] and j["localized"] and not j["stable"]
    assert rep["judge"]["points"] == pytest.approx(7.5)    # nonzero + localized per shock


def test_no_response_scores_zero(window):
    init, dsc = window
    rep = shock_report(_delta_model(lambda d: 0.0 * d), init, dsc, steps=10, channel_std=STD)
    assert rep["judge"]["points"] == 0.0


def test_zero_shock_tiny_leak_strict_vs_kit(window):
    init, dsc = window

    def leaky(init_states, doy_sincos, steps=10, perturbation=None):
        # calls: 0 = base, 1 = None-vs-zeros probe, 2 = the named zero shock
        out = Twin().rollout(init_states, doy_sincos, steps, perturbation)
        if leaky.calls >= 2:
            out = out + 1e-9          # far below the kit's 1e-3 * std tolerance
        leaky.calls += 1
        return out

    leaky.calls = 0
    with pytest.raises(ShockCheckError, match="zero shock changed"):
        shock_report(leaky, init, dsc, steps=4, channel_std=STD, shocks=("zero",))
    leaky.calls = 0
    rep = shock_report(leaky, init, dsc, steps=4, channel_std=STD, shocks=("zero",), strict=False)
    assert rep["judge"]["null_ok"] and not rep["shocks"]["zero"]["pass"]


def test_null_inconsistency_halves_points(window):
    init, dsc = window
    base = _delta_model(lambda d: d)

    def drift(init_states, doy_sincos, steps=10, perturbation=None):
        out = base(init_states, doy_sincos, steps, perturbation)
        if perturbation is not None and not np.any(perturbation):
            out = out + 2e-3 * STD[None, :, None, None]     # 2x the kit tolerance
        return out

    with pytest.raises(ShockCheckError, match="None vs zeros"):
        shock_report(drift, init, dsc, steps=10, channel_std=STD)
    rep = shock_report(drift, init, dsc, steps=10, channel_std=STD, strict=False)
    assert not rep["judge"]["null_ok"] and rep["judge"]["null_max_rel"] == pytest.approx(2e-3, rel=1e-3)
    assert rep["judge"]["points"] == pytest.approx(11.25 / 2)


def test_none_vs_zeros_identical_on_twin(window):
    init, dsc = window
    t = Twin()
    assert np.array_equal(t.rollout(init, dsc, 10, None), t.rollout(init, dsc, 10, np.zeros(STATE_SHAPE)))


def test_load_channel_std_kit_layout(tmp_path):
    p = tmp_path / "stats.json"
    p.write_text(json.dumps({"channels": ["t2m", "tp", "u10", "v10", "msl", "tcwv"], "mean": [0] * 6,
                             "std": list(STD)}))
    np.testing.assert_allclose(load_channel_std(p), STD)
    p.write_text(json.dumps({"std": [1, 2, 3]}))
    with pytest.raises(ValueError):
        load_channel_std(p)


def test_default_std_is_kit_default(window):
    init, dsc = window
    rep = shock_report(Twin().rollout, init, dsc, steps=4)
    assert rep["std_source"] == "kit_default"
    assert list(rep["channel_std"].values()) == pytest.approx(S.STD_DEFAULT.tolist())


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
    assert np.array_equal(fn(init, dsc, steps=3, perturbation=delta), Twin().rollout(init, dsc, 3, delta))
    rep = shock_report(fn, init, dsc, steps=10, channel_std=STD)
    assert rep["shocks"]["zero"]["max_abs_diff"] == 0.0 and rep["judge"]["points"] == pytest.approx(11.25)


def test_cli_writes_json(tmp_path, capsys):
    out = tmp_path / "rep.json"
    st = tmp_path / "stats.json"
    st.write_text(json.dumps({"channels": ["t2m", "tp", "u10", "v10", "msl", "tcwv"], "std": list(STD)}))
    assert main(["--json-out", str(out), "--stats", str(st)]) == 0
    rep = json.loads(out.read_text())
    assert set(rep["shocks"]) == {"warm_ocean", "monsoon_stall", "zero"}
    assert rep["std_source"] == "stats" and rep["judge"]["points"] == pytest.approx(11.25)
    assert "CAUSALITY points" in capsys.readouterr().out
