"""Shock Lab tab: where the shock's response goes, day by day, and the kit-style causality checks.

Everything here is computed from the last Run (two live server calls: baseline and shocked) plus, on demand,
two more live calls for the zero-shock check. Server access goes through client.run_twin only.

This module also holds the small chart-style helpers shared by the Ensemble and Model card tabs
(fixed colours, projector-friendly fonts, legend below the plot, optional coastline on maps).
"""
from __future__ import annotations

import inspect
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import client
import metrics as M
import plots
from common import BOXES, CHANNELS
from windows import get_window

COLORS = list(plots.CATEGORICAL)                                             # fixed categorical order, no yellow
CHANNEL_COLORS = {ch: COLORS[i] for i, ch in enumerate(CHANNELS)}            # same colour per channel everywhere
REGION_COLORS = {"inside": "#D55E00", "near": "#6B5B95", "outside": "#0072B2"}
DAY_COLORS = {1: "#9CC3E0", 4: "#4F8DBF", 8: "#1B3F66"}                       # one hue, light -> dark with time
DAY_LINES = {1: dict(width=7), 4: dict(width=4, dash="dash"), 8: dict(width=2)}  # identical curves stay visible
GOOD, BAD = plots.PADDY, plots.SIGNAL                                        # pass / fail only
CHART_H = 380
MAP_H = 340
DAYS = np.arange(1, 11)
NO_BAR = {"displayModeBar": False}

try:  # the organisers' own checks, used when the kit is installed
    from monsoon_r1 import scoring as KIT
except Exception:  # pragma: no cover - fallback path
    KIT = None


# ---------------------------------------------------------------- shared chart style
def style_chart(fig: go.Figure, title: str, xt: str = "", yt: str = "", height: int = CHART_H,
                legend: bool = True) -> go.Figure:
    """Projector style: 14px text, title at the top, legend in one row BELOW the plot (never over the title)."""
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", y=0.97, yanchor="top", font=dict(size=16)),
        height=height, font=dict(size=14), margin=dict(l=10, r=16, t=48, b=10),
        xaxis_title=xt, yaxis_title=yt, hovermode="x unified", showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0.0, xanchor="left", font=dict(size=14),
                    title_text=""),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor=plots.HAIRLINE, title_font=dict(size=14),
                     tickfont=dict(size=13))
    fig.update_yaxes(gridcolor="rgba(27,42,65,0.10)", title_font=dict(size=14), tickfont=dict(size=13))
    return fig


def _supports_coast(fn) -> bool:
    try:
        return "coast" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def map_fig(fn, *args, coast=None, **kwargs) -> go.Figure:
    """Call plots.field_map / diff_map, passing the land-sea mask as `coast=` when plots supports it."""
    if coast is not None and _supports_coast(fn):
        kwargs["coast"] = coast
    return fn(*args, **kwargs)


def land_sea(ctx):
    try:
        return np.asarray(ctx.archive.static[0])
    except Exception:
        return None


# ---------------------------------------------------------------- analysis helpers
def shock_box(label: str, pert) -> tuple | None:
    """Grader box for named shocks; otherwise the bounding box of the perturbation's nonzero cells."""
    if label in BOXES:
        return BOXES[label]
    return None if pert is None else M.box_from_perturbation(pert)


def _mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def verdict(chk: dict, is_persistence: bool) -> str:
    """One line for judges built from the kit's checks."""
    loc = (f"Shock stays local on day 1: {100 * chk['near_frac_day1']:.0f}% of energy within 3 cells "
           f"{_mark(chk['localized'])}")
    spread = f"Spreads outward by day 4–8: {_mark(chk['propagates'])}"
    if not chk["propagates"] and is_persistence:
        spread += " (persistence stub does not move)"
    return f"{loc}. {spread}"


def checks_rows(chk: dict) -> list[dict]:
    """Rows for the checks table: icon, check, measured number, plain-English meaning."""
    return [
        {"": _mark(chk["nonzero"]), "check": "Responds at all",
         "measured": "max |change| ≥ 0.001 std",
         "in plain English": "The shock changes the forecast somewhere."},
        {"": _mark(chk["localized"]), "check": "Local on day 1",
         "measured": f"{100 * chk['near_frac_day1']:.1f}% near the box (need ≥ 60%)",
         "in plain English": "Day 1 response sits on the shocked area (box + 3 cells)."},
        {"": _mark(chk["propagates"]), "check": "Spreads by day 4–8",
         "measured": "outside energy > 1.5× day 1 and ≥ 2% of peak",
         "in plain English": "Like real weather, the signal travels outward over the next days."},
        {"": _mark(chk["no_teleport"]), "check": "No teleporting",
         "measured": f"{100 * chk['far_frac_max']:.2f}% beyond reach (need < 5%)",
         "in plain English": "Nothing jumps more than 8 cells (~900 km) per day."},
        {"": _mark(chk["stable"]), "check": "Stable",
         "measured": "max |change| ≤ 5 std",
         "in plain English": "The response does not blow up numerically."},
    ]


def _styled(rows: list[dict]):
    """Green ✓ / red ✗ in the icon column (plain rows if pandas styling is unavailable)."""
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        colour = lambda v: f"color: {GOOD if v == '✓' else BAD}; font-weight: 700; font-size: 18px"  # noqa: E731
        return df.style.map(colour, subset=[""])
    except Exception:
        return rows


def _zero_shock_check(ctx, win) -> None:
    url = ctx.controls.server_url
    t0 = time.perf_counter()
    a, _, _ = client.run_twin(url, win["init_states"], win["doy_sincos"], perturbation=None)
    b, _, _ = client.run_twin(url, win["init_states"], win["doy_sincos"],
                              perturbation=np.zeros(win["init_states"].shape[1:]))
    st.session_state["zero_check"] = {"window": win["last_day"], "max_abs": float(np.abs(a - b).max()),
                                      "seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------- render
def render(ctx) -> None:
    st.subheader("Shock Lab: where does the shock's effect go?")
    run = ctx.run
    has_shock = run is not None and run.perturbation is not None and np.any(run.perturbation != 0)
    if run is None:
        st.info("Press **Run the twin** in the sidebar with a shock selected (for example *warm_ocean*). "
                "The Shock Lab then shows how the difference between the shocked and the baseline forecast "
                "moves across the map over 10 days.")
    elif not has_shock:
        st.info("The last Run had no shock. Choose a shock in the sidebar (warm_ocean, monsoon_stall or a custom "
                "box) and press **Run the twin** again.")
    else:
        _render_analysis(ctx, run)

    st.divider()
    _render_zero_check(ctx, run)


def _render_zero_check(ctx, run) -> None:
    left, right = st.columns([1, 2], vertical_alignment="center")
    with left:
        st.markdown("#### Zero-shock check")
        clicked = st.button("Run zero-shock check", key="zero_shock_button", type="primary", width="stretch")
    right.caption("Two fresh live calls on the same window: no perturbation vs. a perturbation of all zeros. "
                  "A well-behaved twin returns identical forecasts (max |difference| exactly 0).")
    if clicked:
        if ctx.health is None:
            st.error("The twin server is not reachable, so the check cannot run. Start it with "
                     "`python serve.py --port 8000` and press the button again.")
        else:
            try:
                win = run.window if run is not None else get_window(ctx.archive, ctx.controls.last_day)
                with st.spinner("Two live calls…"):
                    _zero_shock_check(ctx, win)
            except client.ServerDown as e:
                st.error(f"Check failed: {e}")
    zc = st.session_state.get("zero_check")
    if zc:
        ok = zc["max_abs"] == 0.0
        msg = (f"window ending {zc['window']}, max |difference| = {zc['max_abs']:.3g} "
               f"({zc['seconds']:.2f} s for both calls)")
        if ok:
            st.success(f"**✓ Pass: identical forecasts.** {msg[0].upper() + msg[1:]}")
        else:
            st.error(f"**✗ Fail: the forecasts differ.** {msg[0].upper() + msg[1:]}")


def _render_analysis(ctx, run) -> None:
    std = np.asarray(ctx.archive.std, np.float64)
    box = shock_box(run.shock_label, run.perturbation)
    mask = M.mask_from_box(box)
    d = run.shocked - run.base
    e = M.energy(d, std)
    reg = M.region_sums(e, mask)
    prof = M.radial_profile(e, mask)
    rch = M.reach(e, mask)
    if KIT is not None:
        chk = KIT.causality_checks(run.base, run.shocked, box, std)
        src = "organisers' code (monsoon_r1.scoring.causality_checks)"
    else:
        chk = M.causality_checks_local(run.base, run.shocked, mask, std)
        src = "dashboard/metrics.py (same thresholds; organisers' kit not importable here)"
    backend = str((ctx.health or {}).get("backend", ""))
    is_persist = "persist" in backend.lower()

    n_pass = sum(bool(chk[k]) for k in ("nonzero", "localized", "propagates", "no_teleport", "stable"))
    line = verdict(chk, is_persist)
    (st.success if chk["localized"] and chk["propagates"] else st.warning)(
        f"**{line}**  \n{n_pass} of 5 kit checks pass for shock **{run.shock_label}**, window ending {run.last_day}.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Day-1 energy near the box", f"{100 * M.near_frac_day1(e, mask):.0f}%")
    c2.metric("Reach on day 1", f"{rch[0]} cells" if rch[0] >= 0 else "none",
              help="Furthest ring (cells from the box) holding ≥ 5% of that day's energy.")
    c3.metric("Reach on day 8", f"{rch[7]} cells" if len(rch) > 7 and rch[7] >= 0 else "none")
    c4.metric("Largest change", f"{float(np.abs(d / std[None, :, None, None]).max()):.2f} std")

    left, right = st.columns(2, gap="medium")
    fig = go.Figure()
    for name, key in (("inside box", "inside"), ("near (1–3 cells)", "near"), ("outside", "outside")):
        fig.add_trace(go.Scatter(x=DAYS[:len(e)], y=reg[key], name=name, mode="lines+markers",
                                 line=dict(color=REGION_COLORS[key], width=6 if key == "near" else 2.5),
                                 marker=dict(size=11 if key == "near" else 8)))
    fig = style_chart(fig, "Response energy by region", "lead day", "energy (std² summed)")
    fig.update_xaxes(dtick=1)
    left.plotly_chart(fig, key="shocklab_energy", config=NO_BAR, theme=None)

    fig = go.Figure()
    for day in (1, 4, 8):
        if day <= len(prof):
            fig.add_trace(go.Scatter(x=np.arange(prof.shape[1]), y=100 * prof[day - 1], name=f"day {day}",
                                     mode="lines+markers", line=dict(color=DAY_COLORS[day], **DAY_LINES[day]),
                                     marker=dict(size=12 - day)))
    fig = style_chart(fig, "Share of energy by distance from the box", "cells from the box (0 = inside)",
                      "% of that day's energy")
    fig.update_xaxes(dtick=1)
    fig.update_layout(hovermode="x")
    right.plotly_chart(fig, key="shocklab_radial", config=NO_BAR, theme=None)

    _render_energy_maps(ctx, e, box)

    st.markdown("#### Causality checks")
    st.dataframe(_styled(checks_rows(chk)), hide_index=True, width="stretch",
                 column_config={"": st.column_config.TextColumn("", width=40)})
    st.caption(f"Computed by {src}. Energy = sum over the 6 channels of (shocked − baseline)² in units of each "
               "channel's std, the same measure the organisers' scorer uses. Box: lat "
               f"{box[0]:.0f}–{box[1]:.0f}°N, lon {box[2]:.0f}–{box[3]:.0f}°E ({int(mask.sum())} cells"
               f"{'' if run.shock_label in BOXES else ', bounding box of the perturbed cells'}).")


def _render_energy_maps(ctx, e: np.ndarray, box) -> None:
    lats, lons = ctx.archive.lats, ctx.archive.lons
    coast = land_sea(ctx)
    days = [k for k in (1, 8) if k <= e.shape[0]]
    zmax = max(float(e[[k - 1 for k in days]].max()), 1e-12)
    cols = st.columns(2, gap="medium")
    for col, k in zip(cols, days):
        fig = map_fig(plots.field_map, e[k - 1], lats, lons, f"Response energy on day {k}", "std²",
                      cmap="Reds", zmin=0.0, zmax=zmax, box=box, height=MAP_H + 40, coast=coast)
        col.plotly_chart(fig, key=f"shocklab_emap_{k}", config=NO_BAR, theme=None)
    st.caption("Where the response is. The gold box is the shocked area; both days share one colour scale. "
               "A model that moves weather would show the colour drifting away from the box by day 8.")

