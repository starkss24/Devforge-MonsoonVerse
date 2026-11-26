"""Multiverse of Monsoon dashboard. Run:  streamlit run dashboard/app.py

Every Run makes live calls to the twin server through dashboard/client.py. Nothing is pre-rendered or cached
across runs. Tabs live in their own modules and receive a TabContext.
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

st.set_page_config(page_title="Multiverse of Monsoon", layout="wide")


@st.cache_resource(show_spinner="Loading the training archive…")
def _archive():
    return load_archive()


def main() -> None:
    st.title("Multiverse of Monsoon · digital twin")
    try:
        arc = _archive()
    except DataMissing as e:
        st.error(str(e))
        st.stop()

    controls = sidebar.render_sidebar(arc)

    health = None
    try:
        health = client.health(controls.server_url)
        st.sidebar.success(f"Server ready · {health.get('model_name', '?')} ({health.get('backend', '?')})")
    except client.ServerDown as e:
        st.sidebar.error(f"Server down: {e}")

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
