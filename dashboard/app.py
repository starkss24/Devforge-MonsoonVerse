"""Multiverse of Monsoon dashboard. Run:  streamlit run dashboard/app.py

Every Run makes live calls to the twin server through dashboard/client.py. Nothing is pre-rendered or cached
across runs. Tabs live in their own modules and receive a TabContext.
Initial server URL: env DASHBOARD_SERVER_URL, else client.DEFAULT_URL (editable under Settings in the sidebar).
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import client  # noqa: E402
import sidebar  # noqa: E402
import tab_ensemble  # noqa: E402
import tab_maps  # noqa: E402
import tab_modelcard  # noqa: E402
import tab_shocklab  # noqa: E402
from state import RunResult, TabContext  # noqa: E402
from windows import DataMissing, get_window, load_archive  # noqa: E402

APP_NAME = "Multiverse of Monsoon"
SUBTITLE = "a live digital twin of Indian weather · paint a shock, watch 10 days unfold"

st.set_page_config(page_title=APP_NAME, page_icon=":material/thunderstorm:", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
html, body, [class*="css"] { font-size: 16px; }
.block-container { padding-top: 0.9rem; padding-bottom: 1rem; padding-left: 1.4rem; padding-right: 1.4rem;
                   max-width: 100%; }
header[data-testid="stHeader"] { height: 0; min-height: 0; background: transparent; }
div[data-testid="stToolbar"], div[data-testid="stDecoration"] { display: none; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 0; }
div[data-testid="stSidebarHeader"] { padding: 0.4rem 1rem 0 1rem; min-height: 0; height: 2rem; }
div[data-testid="stSidebarUserContent"] { padding-top: 0.2rem; }
section[data-testid="stSidebar"] label p { font-size: 15px; }
section[data-testid="stSidebar"] div[role="radiogroup"] label p { font-size: 14.5px; }
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] { gap: 0.55rem; }
button[kind="primary"] { font-size: 1.15rem; font-weight: 700; padding: 0.5rem 0; }
div[data-testid="stMetric"] { padding: 0; }
div[data-testid="stMetricLabel"] p { font-size: 0.78rem; color: #555; }
div[data-testid="stMetricValue"] { font-size: 1.25rem; line-height: 1.3; }
.mm-head { display: flex; align-items: baseline; gap: 0.8rem; flex-wrap: wrap; margin: 0 0 0.1rem 0; }
.mm-title { font-size: 1.55rem; font-weight: 800; letter-spacing: -0.01em; color: #10324a; }
.mm-sub { font-size: 0.95rem; color: #5b6770; }
.mm-pill { font-size: 0.8rem; font-weight: 600; padding: 0.12rem 0.6rem; border-radius: 999px;
           white-space: nowrap; }
.mm-ok { background: #e3f4ea; color: #146c37; border: 1px solid #b7e0c6; }
.mm-down { background: #fde8e8; color: #a11d1d; border: 1px solid #f3b9b9; }
.mm-head .mm-pill { margin-left: auto; }
button[data-baseweb="tab"] p { font-size: 1rem; font-weight: 600; }
div[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 1.2rem; }
</style>
"""


@st.cache_resource(show_spinner="Loading the training archive…")
def _archive():
    return load_archive()


def _header_html(pill: str) -> str:
    return (f'<div class="mm-head"><span class="mm-title">{APP_NAME}</span>'
            f'<span class="mm-sub">{SUBTITLE}</span>{pill}</div>')


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    header = st.empty()
    header.markdown(_header_html(""), unsafe_allow_html=True)
    try:
        arc = _archive()
    except DataMissing as e:
        st.error(str(e))
        st.stop()

    status_slot = st.sidebar.empty()
    display = st.container()
    controls = sidebar.render_sidebar(arc, status_slot=status_slot, display=display)

    health, err = None, None
    try:
        health = client.health(controls.server_url)
    except client.ServerDown as e:
        err = str(e)
    pill = sidebar.status_pill(health, err)
    header.markdown(_header_html(pill), unsafe_allow_html=True)
    if health is not None:
        status_slot.markdown(f"{pill}<br><span style='font-size:0.75rem;color:#666'>backend "
                             f"{health.get('backend', '?')} · {controls.server_url}</span>",
                             unsafe_allow_html=True)
    else:
        status_slot.error(f"Server down: {err}  \nStart it with `python serve.py --port 8000`, "
                          "or change the URL under Settings.")

    if controls.run_clicked:
        if health is None:
            st.error("The twin server is not reachable, so nothing can be computed. Start it with "
                     "`python serve.py --port 8000` and press Run again.")
        else:
            win = get_window(arc, controls.last_day)
            try:
                with st.spinner("Running the twin (live)…"):
                    base, base_srv, base_rt = client.run_twin(controls.server_url, win["init_states"], win["doy_sincos"])
                    if controls.perturbation is None:
                        shocked, sh_srv, sh_rt = base, base_srv, 0.0
                    else:
                        shocked, sh_srv, sh_rt = client.run_twin(controls.server_url, win["init_states"],
                                                                 win["doy_sincos"], perturbation=controls.perturbation)
                st.session_state["run"] = RunResult(
                    last_day=controls.last_day, window=win, shock_label=controls.shock_label,
                    perturbation=controls.perturbation, base=base, shocked=shocked,
                    timings_ms={"base_server": base_srv, "base_rt": base_rt, "shock_server": sh_srv, "shock_rt": sh_rt})
            except client.ServerDown as e:
                st.error(f"Run failed: {e}")

    ctx = TabContext(controls=controls, run=st.session_state.get("run"), archive=arc, health=health)
    t_maps, t_lab, t_ens, t_card, t_imp = st.tabs(["Maps", "Shock Lab", "Ensemble", "Model card", "Impact"])
    with t_maps:
        tab_maps.render(ctx)
    with t_lab:
        tab_shocklab.render(ctx)
    with t_ens:
        tab_ensemble.render(ctx)
    with t_card:
        tab_modelcard.render(ctx)
    with t_imp:
        st.info("Crop-stress impact panel: reserved for the impact module (impact/crop_stress.py), "
                "coming from Junaid's lane. No numbers are shown until it is connected.")


main()
