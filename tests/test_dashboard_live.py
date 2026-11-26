"""Tests for the Maps-tab interaction features: Live mode, click-to-move shock, demo presets, place forecast,
spread narrative, Play / day buttons. Pure helpers need only numpy/plotly; the AppTest part needs streamlit, the
training archive and a twin server started here on a free port (each skips cleanly when unavailable).
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "dashboard", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

pytest.importorskip("plotly")
pytest.importorskip("streamlit")

import plots  # noqa: E402
import sidebar  # noqa: E402
import tab_maps  # noqa: E402
from common import CH, LATS, LONS, STATE_SHAPE, box_mask  # noqa: E402

APP = str(ROOT / "dashboard" / "app.py")
HAVE_DATA = (ROOT / "data" / "india_climate_train.nc").is_file()


# ---------- shock placement ----------
def test_box_from_center_snaps_and_clips():
    assert sidebar.box_from_center(14.2, 88.7, 11) == (9.5, 19.5, 83.5, 93.5)     # 11 cells, centred on 14.5/88.5
    b = sidebar.box_from_center(7.0, 66.0, 7)                                       # corner: clipped to the grid
    assert b == (float(LATS.min()), 9.5, float(LONS.min()), 68.5)
    assert box_mask(*b).sum() == 4 * 4


def test_click_turns_preset_into_custom_box_of_same_kind():
    u = sidebar.shock_state_for_click("warm_ocean", 25.2, 78.9)
    assert u["shock_preset"] == "custom" and (u["shock_lat"], u["shock_lon"]) == (25.5, 78.5)
    assert u["custom_channel"] == "t2m" and u["custom_mode"] == "absolute" and u["custom_mag_absolute_t2m"] == 3.0
    assert "channel" not in u                                        # a click never switches the shown layer
    stall = sidebar.shock_state_for_click("monsoon_stall", 15, 70)
    assert stall["custom_mode"] == "percent" and stall["custom_mag_percent_u10"] == -40.0
    moved = sidebar.shock_state_for_click("custom", 20.0, 80.0)      # a custom box only moves
    assert set(moved) == {"shock_lat", "shock_lon", "shock_preset"}


def test_demo_presets_are_local_single_channel_shocks():
    last = np.full(STATE_SHAPE, 290.0)
    last[CH["msl"]] = 100500.0
    for name, p in sidebar.DEMO_SHOCKS.items():
        assert name in sidebar.PRESETS and name in sidebar.PRESET_TEXT and name in sidebar.PRESET_EXPECT
        d = sidebar.build_custom_shock(last, p["box"][:2], p["box"][2:], p["channel"], p["mag"], p["mode"])
        m = box_mask(*p["box"])
        assert m.any() and np.all(d[CH[p["channel"]]][m] == p["mag"]) and not d[:, ~m].any()
        assert np.count_nonzero(d.any(axis=(1, 2))) == 1


# ---------- selections ----------
def test_latlon_from_selection():
    ev = {"selection": {"points": [{"customdata": [14.5, 88.5, 3.2]}]}}
    assert tab_maps.latlon_from_selection(ev) == (14.5, 88.5)
    real = {"selection": {"points": [{"curve_number": 5, "point_number": 516, "x": 77.5, "y": 20.5,   # as
                                      "customdata": {"0": 20.5, "1": 77.5, "2": 30.0}}]}}            # Streamlit
    assert tab_maps.latlon_from_selection(real) == (20.5, 77.5)                                     # sends it
    assert tab_maps.latlon_from_selection({"selection": {"points": [{"x": 66.5, "y": 9.5}]}}) == (9.5, 66.5)
    assert tab_maps.latlon_from_selection({"selection": {"points": [{"x": 1}]}}) is None
    assert tab_maps.latlon_from_selection(None) is None


def test_click_grid_covers_every_cell_and_takes_over_hover():
    f = np.random.default_rng(0).normal(size=(32, 36))
    fig = plots.add_click_grid(plots.field_map(f, LATS, LONS, "t", "K"), f, LATS, LONS, "K", "move the shock here")
    grid = next(t for t in fig.data if t.name == "click_grid")
    assert len(grid.x) == 32 * 36 and grid.marker.opacity == 0
    cd = np.asarray(grid.customdata)
    i = 5 * 36 + 7
    assert tuple(cd[i]) == (LATS[5], LONS[7], f[5, 7])
    assert fig.data[0].hoverinfo == "skip" and fig.layout.dragmode is False


# ---------- place forecast + story ----------
def test_cities_inside_grid_and_probe_location():
    for la, lo in tab_maps.CITIES.values():
        assert LATS.min() <= la <= LATS.max() and LONS.min() <= lo <= LONS.max()
    assert tab_maps.probe_location("Kolkata", None, None)[:2] == tab_maps.CITIES["Kolkata"]
    assert tab_maps.probe_location(tab_maps.BOX_CENTRE, (8.5, 19.5, 82.5, 93.5), None)[:2] == (14.0, 88.0)
    assert tab_maps.probe_location(tab_maps.CLICKED, None, (20.5, 75.5))[:2] == (20.5, 75.5)
    assert tab_maps.probe_location(tab_maps.CLICKED, None, None)[2] == "Delhi"      # no click yet: fallback
    assert tab_maps.nearest_cell(14.0, 88.0, LATS, LONS) == (int(np.argmin(abs(LATS - 14))), 22)


def test_point_series_figure():
    fig = plots.point_series(np.arange(10.0), np.arange(10.0) + 1, "K", "t", start=0.0, truth=np.ones(7),
                             selected=4)
    names = [t.name for t in fig.data]
    assert names == ["Baseline", "Shocked", "What really happened"]
    assert list(fig.data[0].x) == list(range(11)) and fig.data[1].fill == "tonexty"
    assert len(fig.data[2].x) == 7 and fig.layout.shapes[0].line.color == plots.TURMERIC


def test_describe_spread():
    diffs = np.zeros((10, 32, 36))
    box = (8.5, 19.5, 82.5, 93.5)
    diffs[0][box_mask(*box)] = 3.0
    diffs[3, 20:26, 25:31] = 2.0                       # day 4: moved north (lat ~26.5-31.5)
    assert "still over the shock box" in tab_maps.describe_spread(diffs, box, LATS, LONS, 1)
    s4 = tab_maps.describe_spread(diffs, box, LATS, LONS, 4)
    assert "36 cells (144 on day 1)" in s4 and "km north" in s4
    assert "faded" in tab_maps.describe_spread(diffs, box, LATS, LONS, 9)
    assert "no change" in tab_maps.describe_spread(np.zeros((10, 32, 36)), box, LATS, LONS, 2)


def _fake_run(steps=10, shock=True):
    rng = np.random.default_rng(1)
    base = rng.normal(size=(steps, 6, 32, 36))
    pert = np.zeros(STATE_SHAPE)
    pert[CH["t2m"], 5:10, 5:10] = 3.0
    shocked = base + (0.5 if shock else 0.0)
    win = {"init_states": rng.normal(size=(3, 6, 32, 36)), "truth": rng.normal(size=(10, 6, 32, 36))}
    return SimpleNamespace(base=base, shocked=shocked, window=win, perturbation=pert if shock else None,
                           last_day="2019-07-15", shock_label="warm_ocean" if shock else "none", timings_ms={})


def test_timeline_and_views():
    assert tab_maps.timeline_days(10) == list(range(-2, 11))
    assert tab_maps.day_date("2019-07-15", 3).isoformat() == "2019-07-18"
    assert tab_maps.day_tag(0) == "real data" and tab_maps.day_tag(4) == "forecast" and "experimental" in tab_maps.day_tag(12)
    r = _fake_run()
    c = CH["t2m"]
    np.testing.assert_array_equal(tab_maps.view_field(r, "t2m", -2, "shocked"), r.window["init_states"][0, c])
    np.testing.assert_array_equal(tab_maps.view_field(r, "t2m", 0, "shocked"),          # painted shock on day 0
                                  r.window["init_states"][2, c] + r.perturbation[c])
    np.testing.assert_array_equal(tab_maps.view_field(r, "t2m", 0, "base"), r.window["init_states"][2, c])
    np.testing.assert_array_equal(tab_maps.view_field(r, "t2m", 0, "effect"), r.perturbation[c])
    np.testing.assert_array_equal(tab_maps.view_field(r, "t2m", 4, "effect"), r.shocked[3, c] - r.base[3, c])
    np.testing.assert_array_equal(tab_maps.view_field(r, "t2m", 4, "observed"), r.window["truth"][3, c])
    assert not tab_maps.view_field(r, "t2m", -1, "effect").any()
    r15 = _fake_run(steps=15)
    assert tab_maps.view_field(r15, "t2m", 12, "observed") is None                     # beyond the archive
    np.testing.assert_array_equal(tab_maps.view_field(r15, "t2m", 12, "base"), r15.base[11, c])


def test_compare_observed():
    obs = np.zeros((32, 36))
    obs[10, 10] = 60.0
    same = tab_maps.compare_observed(obs, obs)
    assert same["peak_pred"] == same["peak_obs"] == 60.0 and same["detail_pct"] == 100.0 and same["rmse"] == 0.0
    blurred = np.full((32, 36), 60.0 / (32 * 36))                                      # same total, no peak
    b = tab_maps.compare_observed(blurred, obs)
    assert b["peak_pred"] < 1.0 and b["detail_pct"] < 1.0                              # blur shows as lost detail


def test_stale_when_forecast_length_changes():
    r = _fake_run(steps=10, shock=False)
    c = SimpleNamespace(run_clicked=False, last_day="2019-07-15", shock_label="none", perturbation=None, steps=10)
    assert not tab_maps.is_stale(c, r)
    c.steps = 15
    assert tab_maps.is_stale(c, r) and tab_maps.wants_run(c, r, {"status": "ready"}, live=True)


def test_wants_run_live_logic():
    c = SimpleNamespace(run_clicked=False, last_day="2019-07-15", shock_label="none", perturbation=None)
    run = SimpleNamespace(last_day="2019-07-15", shock_label="none", perturbation=None)
    assert tab_maps.wants_run(c, None, {"status": "ready"}, live=True)            # first run happens by itself
    assert not tab_maps.wants_run(c, run, {"status": "ready"}, live=True)         # nothing changed: no call
    assert not tab_maps.wants_run(c, None, None, live=True)                       # server down: no live run
    assert not tab_maps.wants_run(c, None, {"status": "ready"}, live=False)       # Live off: only the button
    c.shock_label = "warm_ocean"
    assert tab_maps.wants_run(c, run, {"status": "ready"}, live=True)             # shock changed
    assert tab_maps.wants_run(SimpleNamespace(run_clicked=True), run, None, live=False)


# ---------- live AppTest ----------
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
                urllib.request.urlopen(urllib.request.Request(url + "/health", method="POST"), timeout=1).read()
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


@pytest.mark.skipif(not HAVE_DATA, reason="data/india_climate_train.nc not present")
def test_app_live_mode_runs_without_the_button(live_server):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["server_url"] = "http://127.0.0.1:1"     # nothing listens: Live must not run or error
    at.run()
    assert not at.exception and "run" not in at.session_state
    assert [e.value for e in at.error] and all(e.value.startswith("Server down") for e in at.error)  # status only
    assert at.toggle(key="live_mode").value is True           # Live is on by default
    at.text_input(key="server_url").set_value(live_server).run()
    assert not at.exception and at.session_state["run"].shock_label == "none"   # first run, no click
    for preset in ("warm_ocean", "heatwave"):
        at.radio(key="shock_preset").set_value(preset).run()
        assert not at.exception and not at.error, [e.value for e in at.error]
        run = at.session_state["run"]
        assert run.shock_label == preset and np.abs(run.shocked - run.base).max() > 0
        assert not any("Press Run to update" in w.value for w in at.warning)
    # day buttons re-slice the run without a server call
    at.button(key="day_next").click().run()
    assert at.select_slider(key="lead_day").value == 2 and at.session_state["run"] is run
    at.button(key="day_prev").click().run()
    assert at.select_slider(key="lead_day").value == 1
    assert any("Day 1:" in m.value for m in at.markdown)       # the plain-language spread sentence
    # Play animates days 1..10 and ends on day 10
    at.button(key="play_button").click().run()
    assert not at.exception and at.select_slider(key="lead_day").value == 10
    # place forecast: a city switch re-charts without a server call
    at.selectbox(key="probe_place").set_value("Delhi").run()
    assert not at.exception and at.session_state["run"] is run
    assert len(at.get("plotly_chart")) >= 3                   # big map, day strip, place chart (+ side by side)
    # forecast length: 15 days in one live call, timeline grows to day 15
    at.segmented_control(key="horizon").set_value(15).run()
    assert not at.exception and at.session_state["run"].base.shape == (15, 6, 32, 36)
    assert at.select_slider(key="lead_day").options[-1] == "Day 15"
    # every view renders, including the real input days and "what happened"
    for view in ("base", "effect", "observed", "shocked"):
        at.segmented_control(key="map_view").set_value(view).run()
        at.select_slider(key="lead_day").set_value(-2).run()
        assert not at.exception, view
    at.select_slider(key="lead_day").set_value(14).run()      # beyond the archive's 10 observed days
    assert not at.exception
