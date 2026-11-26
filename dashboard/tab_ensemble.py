"""Ensemble tab: several live server calls from slightly noisy starting states, to show forecast uncertainty.

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
from tab_shocklab import CHANNEL_COLORS, MAP_H, NO_BAR, land_sea, map_fig, style_chart
from windows import get_window

N_MEMBERS = 8          # default member count
MAX_MEMBERS = 16


def _box(pert):
    return None if pert is None else M.box_from_perturbation(pert)


def _run_ensemble(ctx, frac: float, use_shock: bool, n_members: int = N_MEMBERS) -> None:
    c = ctx.controls
    win = get_window(ctx.archive, c.last_day)
    pert = c.perturbation if use_shock else None
    std = np.asarray(ctx.archive.std, np.float64)
    members = []
    bar = st.progress(0.0, text="Starting the ensemble…")
    t0 = time.perf_counter()
    for m in range(n_members):
        init = M.add_input_noise(win["init_states"], std, frac, seed=1000 + m)
        roll, _, _ = client.run_twin(c.server_url, init, win["doy_sincos"], perturbation=pert)
        members.append(roll)
        bar.progress((m + 1) / n_members,
                     text=f"Member {m + 1} of {n_members} done · {time.perf_counter() - t0:.1f} s (live calls)")
    bar.empty()
    st.session_state["ensemble"] = {
        "members": np.stack(members), "last_day": win["last_day"], "frac": frac,
        "shock_label": c.shock_label if pert is not None else "none", "perturbation": pert,
        "seconds": time.perf_counter() - t0,
    }


def render(ctx) -> None:
    st.subheader("Ensemble · how sure is the forecast?")
    st.markdown("The server is deterministic, so we nudge the 3 starting days with tiny random noise and make one "
                "live call per member: **where the members disagree, the forecast is less certain.**")
    c = ctx.controls
    has_shock = c.perturbation is not None
    col1, col2, col3 = st.columns([1, 1, 1], gap="medium", vertical_alignment="bottom")
    n_members = col1.slider("Members (live calls)", 2, MAX_MEMBERS, N_MEMBERS, 1, key="ens_members")
    pct = col2.slider("Input noise (% of channel std)", 0.1, 2.0, 0.5, 0.1, key="ens_noise_pct")
    use_shock = col3.checkbox(f"Include current shock ({c.shock_label})", value=False, disabled=not has_shock,
                              key="ens_use_shock")
    if st.button(f"▶ Run {n_members}-member ensemble", type="primary", key="ens_run_button"):
        if ctx.health is None:
            st.error("The twin server is not reachable, so the ensemble cannot run.")
        else:
            try:
                _run_ensemble(ctx, pct / 100.0, use_shock and has_shock, n_members)
            except (client.ServerDown, ValueError) as e:
                st.error(f"Ensemble failed: {e}")

    ens = st.session_state.get("ensemble")
    if not ens:
        st.caption("Press the button to run the ensemble for the window and shock chosen in the sidebar.")
        return

    mem = ens["members"]
    n = mem.shape[0]
    std = np.asarray(ctx.archive.std, np.float64)
    st.success(f"**{n} live calls in {ens['seconds']:.2f} s total** ({ens['seconds'] / n:.2f} s each) · "
               f"window ending {ens['last_day']} · noise {100 * ens['frac']:.1f}% of std · "
               f"shock: {ens['shock_label']}")
    if ens["last_day"] != c.last_day:
        st.warning("The sidebar window changed since this ensemble ran; press the button again to refresh it.")

    ch, day = c.channel, int(c.day)
    k = CH[ch]
    mean = mem[:, day - 1, k].mean(axis=0)
    spread = mem[:, day - 1, k].std(axis=0)
    lats, lons = ctx.archive.lats, ctx.archive.lons
    box = _box(ens["perturbation"])
    units = plots.display_units(ch)
    coast = land_sea(ctx)
    left, right = st.columns(2, gap="medium")
    left.plotly_chart(map_fig(plots.field_map, plots.to_display(ch, mean), lats, lons,
                              f"Ensemble mean · {ch} · day {day}", units,
                              cmap=plots.CHANNEL_CMAPS.get(ch, "Viridis"), box=box, height=MAP_H, coast=coast),
                      key="ens_mean_map", config=NO_BAR)
    right.plotly_chart(map_fig(plots.field_map, plots.to_display(ch, spread, is_diff=True), lats, lons,
                               f"Spread (std across members) · {ch} · day {day}", units.split(" (")[0],
                               cmap="Oranges", zmin=0.0, box=box, height=MAP_H, coast=coast),
                       key="ens_spread_map", config=NO_BAR)
    st.caption("Channel and lead day come from the sidebar. Darker orange = members disagree more.")

    sp = M.spread_by_day(mem, std)                    # [steps, 6]
    fig = go.Figure()
    days = np.arange(1, sp.shape[0] + 1)
    for i, name in enumerate(CHANNELS):
        fig.add_trace(go.Scatter(x=days, y=sp[:, i], name=name, mode="lines+markers",
                                 line=dict(color=CHANNEL_COLORS[name], width=2.5), marker=dict(size=8)))
    fig = style_chart(fig, "Spread vs lead day (ensemble std ÷ channel std)", "lead day", "spread (channel std)")
    fig.add_hline(y=ens["frac"], line_dash="dot", line_color="#666666", annotation_text="input noise level",
                  annotation_position="top left", annotation_font=dict(size=13))
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(range=[0, 1.3 * max(float(np.nanmax(sp)), ens["frac"])])
    st.plotly_chart(fig, key="ens_spread_line", config=NO_BAR)
    st.caption("Flat near the input noise level: the model damps small errors. Growing with lead day: small "
               "starting differences are amplified, as in real weather.")
