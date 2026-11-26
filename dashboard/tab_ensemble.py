"""Ensemble tab: 8 live server calls from slightly noisy starting states, to show forecast uncertainty.

The twin server is deterministic (same input -> same output). The spread here comes ONLY from small Gaussian
noise that this dashboard adds to init_states before each call (fixed seed per member). Server access goes
through client.run_twin only; results are kept for this session and never cached across runs.
"""
from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import client
import metrics as M
import plots
from common import CH, CHANNELS
from windows import get_window

N_MEMBERS = 8
COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17"]   # fixed categorical order


def _box(pert):
    return None if pert is None else M.box_from_perturbation(pert)


def _run_ensemble(ctx, frac: float, use_shock: bool) -> None:
    c = ctx.controls
    win = get_window(ctx.archive, c.last_day)
    pert = c.perturbation if use_shock else None
    std = np.asarray(ctx.archive.std, np.float64)
    members = []
    bar = st.progress(0.0, text="Starting the ensemble…")
    t0 = time.perf_counter()
    for m in range(N_MEMBERS):
        init = M.add_input_noise(win["init_states"], std, frac, seed=1000 + m)
        roll, _, _ = client.run_twin(c.server_url, init, win["doy_sincos"], perturbation=pert)
        members.append(roll)
        bar.progress((m + 1) / N_MEMBERS, text=f"Member {m + 1} of {N_MEMBERS} done (live call)")
    bar.empty()
    st.session_state["ensemble"] = {
        "members": np.stack(members), "last_day": win["last_day"], "frac": frac,
        "shock_label": c.shock_label if pert is not None else "none", "perturbation": pert,
        "seconds": time.perf_counter() - t0,
    }


def render(ctx) -> None:
    st.subheader("Ensemble · how sure is the forecast?")
    st.info("The twin server is **deterministic**: the same input always gives the same forecast. "
            "To show uncertainty, this dashboard adds tiny random noise (a fraction of each channel's std) "
            f"to the 3 starting days and makes {N_MEMBERS} separate live calls, one per member with a fixed seed. "
            "Where members disagree, the forecast is less certain.")
    c = ctx.controls
    col1, col2 = st.columns([2, 1])
    pct = col1.slider("Input noise (% of channel std)", 0.1, 2.0, 0.5, 0.1, key="ens_noise_pct")
    has_shock = c.perturbation is not None
    use_shock = col2.checkbox(f"Include current shock ({c.shock_label})", value=False, disabled=not has_shock,
                              key="ens_use_shock")
    if st.button(f"Run {N_MEMBERS}-member ensemble", type="primary", key="ens_run_button"):
        if ctx.health is None:
            st.error("The twin server is not reachable, so the ensemble cannot run.")
        else:
            try:
                _run_ensemble(ctx, pct / 100.0, use_shock and has_shock)
            except (client.ServerDown, ValueError) as e:
                st.error(f"Ensemble failed: {e}")

    ens = st.session_state.get("ensemble")
    if not ens:
        st.caption("Press the button to run the ensemble for the window and shock chosen in the sidebar.")
        return

    mem = ens["members"]
    std = np.asarray(ctx.archive.std, np.float64)
    st.success(f"{N_MEMBERS} live calls in {ens['seconds']:.2f} s · window ending {ens['last_day']} · noise "
               f"{100 * ens['frac']:.1f}% of std · shock: {ens['shock_label']}")
    if ens["last_day"] != c.last_day:
        st.warning("The sidebar window changed since this ensemble ran; press the button again to refresh it.")

    ch, day = c.channel, int(c.day)
    k = CH[ch]
    mean = mem[:, day - 1, k].mean(axis=0)
    spread = mem[:, day - 1, k].std(axis=0)
    lats, lons = ctx.archive.lats, ctx.archive.lons
    box = _box(ens["perturbation"])
    units = plots.display_units(ch)
    left, right = st.columns(2)
    left.plotly_chart(plots.field_map(plots.to_display(ch, mean), lats, lons, f"Ensemble mean · {ch} · day {day}",
                                      units, cmap=plots.CHANNEL_CMAPS.get(ch, "Viridis"), box=box),
                      key="ens_mean_map")
    right.plotly_chart(plots.field_map(plots.to_display(ch, spread, is_diff=True), lats, lons,
                                       f"Ensemble spread (std across members) · {ch} · day {day}", units,
                                       cmap="Oranges", zmin=0.0, box=box),
                       key="ens_spread_map")

    sp = M.spread_by_day(mem, std)                    # [steps, 6]
    fig = go.Figure()
    days = np.arange(1, sp.shape[0] + 1)
    for i, name in enumerate(CHANNELS):
        fig.add_trace(go.Scatter(x=days, y=sp[:, i], name=name, mode="lines+markers",
                                 line=dict(color=COLORS[i], width=2), marker=dict(size=8)))
    fig.update_layout(title="Spread vs lead day (mean ensemble std ÷ channel std)", height=360,
                      margin=dict(l=10, r=10, t=50, b=10), xaxis_title="lead day",
                      yaxis_title="spread in channel-std units", hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    st.plotly_chart(fig, key="ens_spread_line")
    st.caption("A spread that stays near the input noise level means the model damps small errors; "
               "a spread that grows with lead day means small starting differences are amplified, as in real weather.")
