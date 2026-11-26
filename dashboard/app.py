"""Multiverse of Monsoon dashboard. Run:  streamlit run dashboard/app.py

Every Run makes live calls to the twin server through dashboard/client.py. Nothing is pre-rendered or cached
across runs. Tabs live in their own modules and receive a TabContext.
Initial server URL: env DASHBOARD_SERVER_URL, else client.DEFAULT_URL (editable under Settings in the sidebar).

Look: a meteorologist's synoptic chart desk. Theme colours and the Archivo type family come from
.streamlit/config.toml; the CSS below only sets widths of the one type family (expanded for the title and the
day readout, semi-condensed for chart lettering), the header, focus rings and the map frames.
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
import plots  # noqa: E402
import sidebar  # noqa: E402
import tab_ensemble  # noqa: E402
import tab_impact  # noqa: E402
import tab_maps  # noqa: E402
import tab_modelcard  # noqa: E402
import tab_shocklab  # noqa: E402
from state import RunResult, TabContext  # noqa: E402
from windows import DataMissing, get_window, load_archive  # noqa: E402

APP_NAME = "Multiverse of Monsoon"
SUBTITLE = "A live digital twin of India's daily weather. Paint a shock and watch it travel over 10 days."
START_HINT = "start it with python serve.py --port 8000, or change the URL under Settings"

st.set_page_config(page_title=APP_NAME, page_icon=":material/thunderstorm:", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
:root { --ink: #1B2A41; --mist: #EEF3F6; --blue: #2F6F9F; --turmeric: #D9A21B; --paddy: #3E7D4F;
        --signal: #B3261E; --hair: #C3CED6; --muted: #4F5D6B; }
.block-container { padding: 0.7rem 1.4rem 1rem 1.4rem; max-width: 100%; }
header[data-testid="stHeader"] { height: 0; min-height: 0; background: transparent; }
div[data-testid="stToolbar"], div[data-testid="stDecoration"] { display: none; }
.main div[data-testid="stVerticalBlock"], section.stMain div[data-testid="stVerticalBlock"] { gap: 0.55rem; }

/* sidebar: cloud ink, light text */
section[data-testid="stSidebar"] > div:first-child { padding-top: 0; }
div[data-testid="stSidebarHeader"] { padding: 0.4rem 1rem 0 1rem; min-height: 0; height: 2rem; }
div[data-testid="stSidebarUserContent"] { padding-top: 0.1rem; }
section[data-testid="stSidebar"] label p { font-size: 15px; }
section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p { font-weight: 600; }
section[data-testid="stSidebar"] div[role="radiogroup"] label p { font-size: 14.5px; }
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] { gap: 0.55rem; }
section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] p { color: #C9D5DF; }
.mm-side-meta { font-size: 0.8rem; color: #C9D5DF; line-height: 1.35; overflow-wrap: anywhere; }

/* primary action: monsoon blue, no glyphs */
button[kind="primary"] { font-size: 1.1rem; font-weight: 650; padding: 0.5rem 0; letter-spacing: 0.01em; }
section[data-testid="stSidebar"] button[kind="primary"] { border: 1px solid #6E9CC2; }

/* visible keyboard focus everywhere (blue on the page, pale blue on the ink sidebar) */
:focus-visible { outline: 3px solid var(--blue) !important; outline-offset: 2px; }
section[data-testid="stSidebar"] :focus-visible { outline-color: #A9CBE6 !important; }

/* header: expanded title left, twin status right, one plain subtitle */
.mm-head { display: flex; align-items: baseline; gap: 1rem; margin: 0; }
.mm-title { font-size: 1.75rem; font-weight: 720; font-stretch: 125%; letter-spacing: -0.005em;
            color: var(--ink); line-height: 1.1; }
.mm-head .mm-pill { margin-left: auto; }
.mm-sub { font-size: 0.98rem; color: var(--muted); margin: 0.1rem 0 0 0; }
.mm-pill { font-size: 0.9rem; font-weight: 600; white-space: nowrap; display: inline-flex; align-items: center;
           gap: 0.4rem; }
.mm-dot { width: 0.6rem; height: 0.6rem; border-radius: 50%; display: inline-block; }
.mm-ok { color: var(--paddy); } .mm-ok .mm-dot { background: var(--paddy); }
.mm-down { color: var(--signal); } .mm-down .mm-dot { background: var(--signal); }
section[data-testid="stSidebar"] .mm-pill { white-space: normal; }
section[data-testid="stSidebar"] .mm-ok { color: #9FD3AE; }

section[data-testid="stSidebar"] .mm-ok .mm-dot { background: #7CC191; }
section[data-testid="stSidebar"] .mm-down { color: #F2A7A1; }

/* tabs */
button[data-baseweb="tab"] p { font-size: 1rem; font-weight: 600; }
div[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 1.4rem; }

/* charts sit on white; each map's own hairline neatline (plots._frame) is its only frame.
   Chart lettering is semi-condensed, like a map legend. */
div[data-testid="stPlotlyChart"] { background: #fff; }

.js-plotly-plot svg text { font-stretch: 87.5%; }
.js-plotly-plot .gtitle, .js-plotly-plot .annotation-text-g .annotation-text { font-stretch: 100%; }
div[data-testid="stDataFrame"] { font-stretch: 87.5%; }

/* the day readout under the Maps panels */
.mm-readout { display: flex; align-items: baseline; flex-wrap: wrap; column-gap: 1.6rem; row-gap: 0.2rem;
              margin: 0.1rem 0 0 0; color: var(--ink); font-size: 1.02rem; }
.mm-readout .mm-day { font-size: 2.1rem; font-weight: 720; font-stretch: 125%; line-height: 1; }
.mm-readout b { font-stretch: 87.5%; font-size: 1.15rem; }
.mm-readout .mm-quiet { color: var(--muted); }
.mm-meta { font-size: 0.98rem; color: var(--ink); margin: 0; }
.mm-meta .mm-quiet { color: var(--muted); }

/* model card numbers: a plain row, no cards */
.mm-scores { display: flex; flex-wrap: wrap; column-gap: 2rem; row-gap: 0.4rem; margin: 0.1rem 0 0.2rem 0; }
.mm-score { max-width: 12.5rem; }

.mm-score .mm-num { font-size: 2rem; font-weight: 700; font-stretch: 87.5%; line-height: 1.1; }
.mm-score .mm-lab { font-size: 0.95rem; color: var(--muted); }
.mm-pass { color: var(--paddy); } .mm-fail { color: var(--signal); } .mm-even { color: var(--muted); }
</style>
"""


@st.cache_resource(show_spinner="Loading the training archive…")
def _archive():
    return load_archive()


def _header_html(pill: str) -> str:
    return (f'<div class="mm-head"><span class="mm-title">{APP_NAME}</span>{pill}</div>'
            f'<p class="mm-sub">{SUBTITLE}</p>')


def main() -> None:
    plots.apply_theme()
    st.markdown(CSS, unsafe_allow_html=True)
    header = st.empty()
    header.markdown(_header_html(""), unsafe_allow_html=True)
    try:
        arc = _archive()
    except DataMissing as e:
        st.error(str(e))
        st.stop()

    status_slot = st.sidebar.empty()
    controls = sidebar.render_sidebar(arc, status_slot=status_slot)

    health, err = None, None
    try:
        health = client.health(controls.server_url)
    except client.ServerDown as e:
        err = str(e)
    pill = sidebar.status_pill(health, err)
    header.markdown(_header_html(pill), unsafe_allow_html=True)
    if health is not None:
        status_slot.markdown(f"{pill}<div class='mm-side-meta'>Backend {health.get('backend', '?')} at "
                             f"{controls.server_url}</div>", unsafe_allow_html=True)
    else:
        status_slot.error(f"Server down: {START_HINT}.")

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
        tab_impact.render(ctx)


main()
