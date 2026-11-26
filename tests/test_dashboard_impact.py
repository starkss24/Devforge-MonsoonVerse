"""Tests for dashboard/tab_impact.py: pure helpers, plus a headless AppTest run against a live twin server
(started here on its own port) with monsoon_stall and warm_ocean. Skips cleanly without streamlit or the data.
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

import tab_impact  # noqa: E402
from impact.regions import REGIONS  # noqa: E402

APP = str(ROOT / "dashboard" / "app.py")
HAVE_DATA = (ROOT / "data" / "india_climate_train.nc").is_file()
SERVER_PORT = 8082


def _fake_res(d_index=(0.0, 12.5)):
    out = {}
    for name, di in zip(list(REGIONS)[:2], d_index):
        b = {"rain_mm": 50.0, "clim_rain_mm": 60.0, "clim_dry": False, "deficit_pct": 16.7, "hot_days": 1,
             "t2m_anom_k": 0.5, "index": 14.0}
        s = dict(b, index=14.0 + di, hot_days=1 + int(di > 0))
        out[name] = {"n_cells": 10, "box": REGIONS[name]["box"], "steps": 10, "baseline": b, "shocked": s,
                     "change": {"rain_mm": 0.0, "deficit_pct": 0.0, "hot_days": int(di > 0), "t2m_anom_k": 0.0,
                                "index": di}}
    return out


def test_table_rows_and_headline():
    res = _fake_res()
    rows = tab_impact.table_rows(res)
    assert len(rows) == 2 and rows[1]["Δ index"] == 12.5
    assert rows[0]["Rain deficit"] == "+17% → +17%" and rows[1]["Hot days"] == "1 → 2"
    text, moved = tab_impact.headline(res, True)
    assert moved and "+12.5" in text
    text, moved = tab_impact.headline(_fake_res((0.0, 0.0)), True)
    assert not moved and "unchanged" in text
    assert not tab_impact.headline(res, False)[1]
    dry = _fake_res()
    first = next(iter(dry.values()))
    first["baseline"]["clim_dry"] = True
    assert tab_impact.table_rows(dry)[0]["Rain deficit"].startswith("dry season")


def test_delta_bars_colours():
    fig = tab_impact.delta_bars(_fake_res((-3.0, 12.5)))
    assert list(fig.data[0].marker.color) == [tab_impact.DOWN, tab_impact.UP]


# ---------------------------------------------------------------- live AppTest
def _port_free(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server():
    import urllib.request
    port = SERVER_PORT if _port_free(SERVER_PORT) else _free_port()
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


@pytest.mark.skipif(not HAVE_DATA, reason="data/india_climate_train.nc not present")
def test_app_impact_tab_live(live_server):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["live_mode"] = False     # explicit Run button; instructions must show before the first run
    at.run()
    assert not at.exception
    assert any("crop-stress" in i.value for i in at.info)          # Impact instructions before the first run
    at.text_input(key="server_url").set_value(live_server).run()
    for shock in ("monsoon_stall", "warm_ocean"):
        at.radio(key="shock_preset").set_value(shock).run()
        at.button(key="run_button").click().run()
        assert not at.exception, at.exception
        assert not at.error, [e.value for e in at.error]
        run = at.session_state["run"]
        assert run.shock_label == shock and run.perturbation is not None
        frames = [d for d in at.dataframe if "Δ index" in list(d.value.columns)]
        assert frames, "impact table missing"
        df = frames[0].value
        assert len(df) == len(REGIONS) and np.isfinite(df["Δ index"].to_numpy(float)).all()
        assert any("Honest limits" in c.value for c in at.caption)
    # the threshold slider recomputes without a new server call
    at.slider(key="impact_hot_k").set_value(1.0).run()
    assert not at.exception and at.session_state["run"] is run
