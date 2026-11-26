"""Tests for dashboard/sidebar.py, dashboard/plots.py, dashboard/tab_maps.py.

Unit tests need only numpy/plotly. AppTest tests need streamlit, the training archive and (for the
live test) a running twin server started here on a free port; each skips cleanly when unavailable.
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "dashboard", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

pytest.importorskip("plotly")
st = pytest.importorskip("streamlit")

import plots  # noqa: E402
import sidebar  # noqa: E402
from common import BOXES, CH, LATS, LONS, STATE_SHAPE, box_mask, make_shock  # noqa: E402

APP = str(ROOT / "dashboard" / "app.py")
HAVE_DATA = (ROOT / "data" / "india_climate_train.nc").is_file()


# ---------- custom shock building ----------
def test_custom_absolute_shock_inside_box_only():
    last = np.full(STATE_SHAPE, 300.0)
    d = sidebar.build_custom_shock(last, (8.5, 19.5), (82.5, 93.5), "t2m", 3.0, "absolute")
    m = box_mask(8.5, 19.5, 82.5, 93.5)
    assert d.shape == STATE_SHAPE
    assert np.all(d[CH["t2m"]][m] == 3.0) and np.all(d[CH["t2m"]][~m] == 0.0)
    assert np.count_nonzero(d) == m.sum() == 12 * 12
    others = [i for i in range(6) if i != CH["t2m"]]
    assert not d[others].any()


def test_custom_percent_shock_relative_to_current_value():
    rng = np.random.default_rng(0)
    last = rng.uniform(10, 60, STATE_SHAPE)
    d = sidebar.build_custom_shock(last, (8, 20), (65, 75), "tcwv", 15.0, "percent")
    m = box_mask(8, 20, 65, 75)
    c = CH["tcwv"]
    np.testing.assert_allclose(d[c][m], 0.15 * last[c][m])
    assert not d[c][~m].any()


def test_custom_matches_named_warm_ocean_t2m_part():
    last = np.full(STATE_SHAPE, 290.0)
    d = sidebar.build_custom_shock(last, BOXES["warm_ocean"][:2], BOXES["warm_ocean"][2:], "t2m", 3.0, "absolute")
    np.testing.assert_array_equal(d[CH["t2m"]], make_shock("warm_ocean", last)[CH["t2m"]])


def test_custom_shock_rejects_bad_input():
    last = np.zeros(STATE_SHAPE)
    with pytest.raises(ValueError):
        sidebar.build_custom_shock(last, (8, 20), (65, 75), "t2m", 1.0, "weird")
    with pytest.raises(ValueError):
        sidebar.build_custom_shock(np.zeros((6, 3, 3)), (8, 20), (65, 75), "t2m", 1.0)
    assert sidebar.preset_shock("none", last) is None
    lab = sidebar.custom_label((8.5, 19.5), (82.5, 93.5), "t2m", 3.0, "absolute")
    assert lab.startswith("custom:") and "+3 K" in lab


def test_snap_day():
    days = np.arange(np.datetime64("2019-07-01"), np.datetime64("2019-07-10")).astype("datetime64[D]")
    assert sidebar.snap_day(days, "2019-07-05") == "2019-07-05"
    assert sidebar.snap_day(days, "2025-01-01") == "2019-07-09"


# ---------- plots helpers ----------
def test_display_conversion_physical_to_display():
    np.testing.assert_allclose(plots.to_display("t2m", 300.0), 26.85)
    np.testing.assert_allclose(plots.to_display("t2m", 3.0, is_diff=True), 3.0)
    np.testing.assert_allclose(plots.to_display("msl", 101325.0), 1013.25)
    np.testing.assert_allclose(plots.to_display("tp", 5.0), 5.0)
    assert "°C" in plots.display_units("t2m") and "hPa" in plots.display_units("msl")


def test_box_from_perturbation_recovers_box():
    last = np.full(STATE_SHAPE, 50.0)
    pert = make_shock("warm_ocean", last)
    box = plots.box_from_perturbation(pert, LATS, LONS)
    assert box == (8.5, 19.5, 82.5, 93.5)
    assert plots.box_from_perturbation(np.zeros(STATE_SHAPE), LATS, LONS) is None
    assert plots.box_from_perturbation(None, LATS, LONS) is None


def test_field_map_box_wind_and_orientation():
    f = np.add.outer(LATS, np.zeros_like(LONS))  # increases northwards
    u = np.ones((32, 36)) * 5
    v = np.ones((32, 36)) * -2
    fig = plots.field_map(f, LATS, LONS, "t", "K", box=(8.5, 19.5, 82.5, 93.5), wind=(u, v))
    hm = fig.data[0]
    assert hm.type == "heatmap" and list(hm.y) == list(LATS) and hm.y[0] < hm.y[-1]
    assert len(fig.data) == 2 and fig.data[1].mode == "lines"
    assert len(fig.layout.shapes) == 1 and fig.layout.shapes[0].x0 == pytest.approx(82.0)
    assert fig.layout.yaxis.scaleanchor == "x"
    assert hm.colorbar.title.text == "K"
    fig2 = plots.field_map(f, LATS, LONS, "t", "K", wind=(np.zeros((32, 36)), np.zeros((32, 36))))
    assert len(fig2.data) == 1  # calm wind -> no arrow trace


def test_diff_map_symmetric_and_animation_frames():
    d = np.zeros((32, 36))
    d[5, 5] = -4.0
    d[6, 6] = 1.0
    fig = plots.diff_map(d, LATS, LONS, "d", "K")
    hm = fig.data[0]
    assert hm.zmin == -4.0 and hm.zmax == 4.0 and hm.zmid == 0.0
    z = np.zeros((32, 36))
    assert plots.diff_map(z, LATS, LONS, "d", "K").data[0].zmax > 0  # no zero-width range
    diffs = np.stack([d * (k + 1) for k in range(10)])
    anim = plots.diff_animation(diffs, LATS, LONS, "Δ", "K")
    assert len(anim.frames) == 10 and anim.data[0].zmax == pytest.approx(40.0)
    assert anim.layout.updatemenus and anim.layout.sliders


# ---------- AppTest (headless app) ----------
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server():
    import urllib.request
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve.py"), "--port", str(port)], cwd=str(ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(url + "/health", timeout=1).read()
                break
            except Exception:
                if proc.poll() is not None:
                    pytest.skip("twin server failed to start")
                time.sleep(0.5)
        else:
            pytest.skip("twin server did not become ready")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def _apptest():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(APP, default_timeout=120)


@pytest.mark.skipif(not HAVE_DATA, reason="data/india_climate_train.nc not present")
def test_app_live_warm_ocean_run(live_server):
    at = _apptest().run()
    assert not at.exception
    assert any("Run the twin" in i.value for i in at.info)  # instructions before the first run
    at.text_input(key="server_url").set_value(live_server).run()
    at.radio(key="shock_preset").set_value("warm_ocean").run()
    at.button(key="run_button").click().run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    run = at.session_state["run"]
    assert run.shock_label == "warm_ocean" and run.base.shape == (10, 6, 32, 36)
    assert np.abs(run.shocked - run.base).max() > 0
    charts = at.get("plotly_chart")
    assert len(charts) >= 3
    # channel/day change re-slices without a new server call
    at.selectbox(key="channel").set_value("t2m").run()
    at.slider(key="lead_day").set_value(5).run()
    assert not at.exception and at.session_state["run"] is run
    assert not any("Press Run to update" in w.value for w in at.warning)
    # shock change -> stale warning
    at.radio(key="shock_preset").set_value("monsoon_stall").run()
    assert any("Press Run to update" in w.value for w in at.warning)


@pytest.mark.skipif(not HAVE_DATA, reason="data/india_climate_train.nc not present")
def test_app_custom_shock_preview_renders():
    at = _apptest().run()
    at.radio(key="shock_preset").set_value("custom").run()
    assert not at.exception
    assert len(at.get("plotly_chart")) >= 1  # box preview in the sidebar


@pytest.mark.skipif(not HAVE_DATA, reason="data/india_climate_train.nc not present")
def test_app_server_down_shows_clear_error():
    at = _apptest().run()
    at.text_input(key="server_url").set_value(f"http://127.0.0.1:{_free_port()}").run()
    at.button(key="run_button").click().run()
    assert not at.exception
    assert any("not reachable" in e.value for e in at.error)
    assert "run" not in at.session_state
