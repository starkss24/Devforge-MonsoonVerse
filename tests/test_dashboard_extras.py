"""Live UI test for the Shock Lab, Ensemble and Model card tabs (streamlit AppTest against a real server).

Starts serve.py on port 8062, runs the warm_ocean shock, clicks the Zero-shock check and the ensemble button,
and asserts no exceptions. Skips when streamlit, the data file or the server are unavailable.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PORT = 8062
URL = f"http://127.0.0.1:{PORT}"


def _ready() -> bool:
    try:
        req = urllib.request.Request(URL + "/health", data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=2) as r:
            return json.loads(r.read()).get("status") == "ready"
    except Exception:
        return False


@pytest.fixture(scope="module")
def server():
    if not (ROOT / "data" / "india_climate_train.nc").is_file():
        pytest.skip("training archive not present")
    proc = subprocess.Popen([sys.executable, "serve.py", "--port", str(PORT)], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            if _ready():
                break
            if proc.poll() is not None:
                pytest.skip("server exited during start-up")
            time.sleep(0.5)
        else:
            pytest.skip("server did not become ready")
        yield URL
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _all_text(at) -> str:
    parts = []
    for kind in ("subheader", "markdown", "caption", "success", "error", "warning", "info"):
        parts += [str(e.value) for e in getattr(at, kind)]
    return "\n".join(parts)


def test_extras_tabs_live(server):
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    at.text_input(key="server_url").set_value(server).run()
    at.radio(key="shock_preset").set_value("warm_ocean").run()
    at.button(key="run_button").click().run()
    assert not at.exception, at.exception
    assert at.session_state["run"].perturbation is not None

    text = _all_text(at)
    assert "Shock Lab" in text and "Known limits" in text
    assert len(at.dataframe) >= 1                       # causality table (and CSI table)

    at.button(key="zero_shock_button").click().run()
    assert not at.exception, at.exception
    zc = at.session_state["zero_check"]
    assert zc["max_abs"] == 0.0
    assert any("PASS" in str(s.value) for s in at.success)

    at.button(key="ens_run_button").click().run()
    assert not at.exception, at.exception
    ens = at.session_state["ensemble"]
    assert ens["members"].shape == (8, 10, 6, 32, 36)
    assert ens["members"].std(axis=0).max() > 0          # dashboard noise produced a spread
