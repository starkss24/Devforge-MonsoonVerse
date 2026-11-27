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
    strip = plots.diff_strip(diffs, LATS, LONS, "K", box=(8.5, 19.5, 82.5, 93.5), selected=3)
    hms = [t for t in strip.data if t.type == "heatmap"]
    assert len(hms) == 10                                            # one small map per day
    assert all(h.zmax == pytest.approx(40.0) and h.zmin == pytest.approx(-40.0) for h in hms)  # one shared scale
    assert sum(bool(h.showscale) for h in hms) == 1                  # one colourbar for the strip
    np.testing.assert_array_equal(hms[9].z, diffs[9])
    hits = [t for t in strip.data if t.type == "scatter" and t.customdata is not None]
    assert [int(t.customdata[0]) for t in hits] == list(range(1, 11))   # a click on panel k reads back day k
    sel = [s for s in strip.layout.shapes if s.name == "selected_day"]
    assert len(sel) == 1 and sel[0].line.color == plots.TURMERIC
    dom = strip.layout.xaxis3.domain
    assert sel[0].x0 < dom[0] < dom[1] < sel[0].x1                   # outline sits round day 3
    boxes = [s for s in strip.layout.shapes if s.name != "selected_day"]
    assert len(boxes) == 10 and all(s.line.color == plots.TURMERIC for s in boxes)


def test_diff_map_no_change_is_clean_and_annotated():
    z = np.zeros((32, 36))
    fig = plots.diff_map(z, LATS, LONS, "d", "Δ mm/day", empty_text="No change in tp on day 3")
    hm = fig.data[0]
    assert (hm.zmin, hm.zmax, hm.zmid) == (-1.0, 1.0, 0.0)          # fixed range, no "1n" colourbar
    assert any(a.text == "No change in tp on day 3" for a in fig.layout.annotations)
    small = z.copy()
    small[3, 3] = 0.5
    fig2 = plots.diff_map(small, LATS, LONS, "d", "K", empty_text="No change")
    assert fig2.data[0].zmax == 0.5 and not fig2.layout.annotations  # real change: no note, own range
    assert plots.diff_strip(np.zeros((10, 32, 36)), LATS, LONS, "K").data[0].zmax == 1.0


def test_day_from_selection():
    import tab_maps
    assert tab_maps.day_from_selection({"selection": {"points": [{"customdata": 8, "x": 70.5}]}}) == 8
    assert tab_maps.day_from_selection({"selection": {"points": [{"customdata": [4]}]}}) == 4
    assert tab_maps.day_from_selection({"selection": {"points": [{"x": 1}]}}) is None    # e.g. a coast point
    assert tab_maps.day_from_selection({"selection": {"points": []}}) is None
    assert tab_maps.day_from_selection(None) is None
    assert tab_maps.day_from_selection({"selection": {"points": [{"customdata": 11}]}}) is None


def test_field_map_coast_colorbar_and_places():
    mask = (np.add.outer(np.zeros_like(LATS), LONS) < 80).astype(float)   # land west of 80°E
    f = np.random.default_rng(1).normal(size=(32, 36))
    fig = plots.field_map(f, LATS, LONS, "t", "K", coast=mask, colorbar="bottom", places=True)
    if plots.OUTLINE_PATH.is_file():   # bundled Natural Earth outline replaces the grid contour
        outline = [t for t in fig.data if t.name == "outline"]
        assert len(outline) == 1 and outline[0].mode == "lines" and len(outline[0].x) > 100
        assert not any(t.type == "contour" for t in fig.data)
    else:
        coast = [t for t in fig.data if t.type == "contour"]
        assert len(coast) == 1 and coast[0].contours.start == coast[0].contours.end == 0.5
        assert coast[0].contours.coloring == "none" and coast[0].showscale is False
    c = plots.coastline_trace(mask, LATS, LONS)     # the grid-mask fallback itself
    assert c.contours.start == c.contours.end == 0.5 and c.contours.coloring == "none" and c.showscale is False
    cb = fig.data[0].colorbar
    assert cb.orientation == "h" and cb.title.text == "K" and cb.thickness <= 12
    texts = " ".join(a.text for a in fig.layout.annotations)
    assert "Arabian Sea" in texts and "Bay of Bengal" in texts
    assert fig.layout.yaxis.scaleanchor == "x" and fig.layout.xaxis.title.text is None
    # diff_map forwards the extras (other tabs pass coast= to it)
    d = plots.diff_map(f, LATS, LONS, "d", "K", coast=mask)
    assert any(t.type == "contour" or t.name == "outline" for t in d.data)
    assert plots.add_coast(plots.field_map(f, LATS, LONS, "t", "K"), None, LATS, LONS).data[-1].type == "heatmap"


def _arrow_extent(fig):
    tr = next(t for t in fig.data if t.name == "wind")
    xs = np.array([x for x in tr.x if x is not None], dtype=float)
    return xs.max() - xs.min(), tr


def test_wind_ref_fixed_scale_and_key():
    strong = (np.full((32, 36), 10.0), np.zeros((32, 36)))
    weak = (np.full((32, 36), 5.0), np.zeros((32, 36)))
    f = np.zeros((32, 36))
    # auto scale: weak and strong look the same; fixed scale: weak arrows are half as long
    a_s = plots.field_map(f, LATS, LONS, "t", "K", wind=strong)
    a_w = plots.field_map(f, LATS, LONS, "t", "K", wind=weak)
    tr_s = next(t for t in a_s.data if t.name == "wind")
    tr_w = next(t for t in a_w.data if t.name == "wind")
    assert np.allclose([x for x in tr_s.x if x is not None], [x for x in tr_w.x if x is not None])
    r_s = plots.field_map(f, LATS, LONS, "t", "K", wind=strong, wind_ref=10, wind_stride=4)
    r_w = plots.field_map(f, LATS, LONS, "t", "K", wind=weak, wind_ref=10, wind_stride=4)
    seg = lambda fig: next(t for t in fig.data if t.name == "wind").x[1] - next(t for t in fig.data if t.name == "wind").x[0]
    assert seg(r_s) == pytest.approx(2 * seg(r_w)) and seg(r_s) == pytest.approx(0.6 * 4)
    assert any(t.name == "wind_key" for t in r_s.data)
    assert any("10 m/s" in a.text for a in r_s.layout.annotations)
    tr = next(t for t in r_s.data if t.name == "wind")
    assert tr.line.width < 1.5 and "0.55" in tr.line.color      # thin, semi-transparent
    n_arrows = sum(1 for x in tr.x if x is None) // 3
    assert n_arrows == len(LATS[::4]) * len(LONS[::4])


def test_status_pill_and_server_url_env(monkeypatch):
    ok = sidebar.status_pill({"model_name": "twin-<v1>", "backend": "onnx"})
    assert "Server ready" in ok and "twin-&lt;v1&gt;" in ok and "mm-ok" in ok
    assert "·" not in ok and "·" not in sidebar.channel_label("t2m")
    assert not any("·" in t for t in sidebar.PRESET_TEXT.values())
    down = sidebar.status_pill(None, "connection refused")
    assert "Server down" in down and "mm-down" in down
    monkeypatch.delenv(sidebar.SERVER_ENV, raising=False)
    assert sidebar.default_server_url() == sidebar.client.DEFAULT_URL
    monkeypatch.setenv(sidebar.SERVER_ENV, "http://127.0.0.1:8071")
    assert sidebar.default_server_url() == "http://127.0.0.1:8071"


def test_shock_cells_and_preset_channels():
    import tab_maps
    last = np.full(STATE_SHAPE, 50.0)
    assert tab_maps.shock_cells(None) == 0
    assert tab_maps.shock_cells(make_shock("warm_ocean", last)) == box_mask(*BOXES["warm_ocean"]).sum()
    assert sidebar.SHOCK_CHANNEL["warm_ocean"] == "t2m" and sidebar.SHOCK_CHANNEL["monsoon_stall"] == "u10"
    assert set(sidebar.SHOCK_CHANNEL) == set(sidebar.PRESETS) - {"none", "custom"}   # every shock preset


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
    at = _apptest()
    at.session_state["live_mode"] = False     # this test covers the explicit Run button (Live has its own test)
    at.run()
    assert not at.exception
    assert any("Run the twin" in i.value for i in at.info)  # instructions before the first run
    assert at.radio(key="shock_preset").value == sidebar.DEFAULT_PRESET == "heatwave"   # first load: clearest demo
    assert at.segmented_control(key="channel").value == sidebar.DEFAULT_CHANNEL == "t2m"
    assert at.select_slider(key="lead_day").value == 1

    at.text_input(key="server_url").set_value(live_server).run()
    at.radio(key="shock_preset").set_value("warm_ocean").run()
    assert at.segmented_control(key="channel").value == "t2m"   # display follows the shocked channel
    at.button(key="run_button").click().run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    run = at.session_state["run"]
    assert run.shock_label == "warm_ocean" and run.base.shape == (10, 6, 32, 36)
    assert np.abs(run.shocked - run.base).max() > 0
    charts = at.get("plotly_chart")
    assert len(charts) >= 3
    assert any("How to read this" in c.value for c in at.caption)
    # channel/day change re-slices without a new server call; the user's channel choice sticks
    at.segmented_control(key="channel").set_value("tp").run()
    at.select_slider(key="lead_day").set_value(5).run()
    assert not at.exception and at.session_state["run"] is run
    assert any("Day 5" in m.value for m in at.markdown)          # the readout follows the chosen day
    assert at.segmented_control(key="channel").value == "tp"
    assert not any("Press Run to update" in w.value for w in at.warning)
    # shock change -> stale warning, display switches to the newly shocked channel
    at.radio(key="shock_preset").set_value("monsoon_stall").run()
    assert any("Press Run to update" in w.value for w in at.warning)
    assert at.segmented_control(key="channel").value == "u10"


@pytest.mark.skipif(not HAVE_DATA, reason="data/india_climate_train.nc not present")
def test_app_custom_shock_preview_renders():
    at = _apptest().run()
    at.radio(key="shock_preset").set_value("custom").run()
    assert not at.exception
    assert len(at.get("plotly_chart")) >= 1  # box preview in the sidebar


@pytest.mark.skipif(not HAVE_DATA, reason="data/india_climate_train.nc not present")
def test_app_server_down_shows_clear_error():
    at = _apptest()
    at.session_state["server_url"] = f"http://127.0.0.1:{_free_port()}"   # dead from the first run, whatever
    at.run()                                                             # is listening on the default port
    at.button(key="run_button").click().run()
    assert not at.exception
    assert any("not reachable" in e.value for e in at.error)
    assert "run" not in at.session_state
