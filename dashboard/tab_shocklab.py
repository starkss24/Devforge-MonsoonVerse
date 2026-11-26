"""Shock Lab tab: where the shock's response goes, day by day, and the kit-style causality checks.

Everything here is computed from the last Run (two live server calls: baseline and shocked) plus, on demand,
two more live calls for the zero-shock check. Server access goes through client.run_twin only.
"""
from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import client
import metrics as M
from common import BOXES
from windows import get_window

COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17"]   # fixed categorical order
DAYS = np.arange(1, 11)

try:  # the organisers' own checks, used when the kit is installed
    from monsoon_r1 import scoring as KIT
except Exception:  # pragma: no cover - fallback path
    KIT = None


def shock_box(label: str, pert) -> tuple | None:
    """Grader box for named shocks; otherwise the bounding box of the perturbation's nonzero cells."""
    if label in BOXES:
        return BOXES[label]
    return None if pert is None else M.box_from_perturbation(pert)


def _line_layout(fig: go.Figure, title: str, xt: str, yt: str, height: int = 340) -> go.Figure:
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title=xt, yaxis_title=yt, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    fig.update_xaxes(showgrid=False)
    return fig


def _checks_table(chk: dict, source: str) -> list[dict]:
    yes = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    return [
        {"check": "Non-zero", "result": yes(chk["nonzero"]),
         "number": "max |change| ≥ 0.001 std",
         "what it means": "The shock actually changes the forecast somewhere."},
        {"check": "Localised on day 1", "result": yes(chk["localized"]),
         "number": f"{100 * chk['near_frac_day1']:.1f}% of day-1 energy near the box (needs ≥ 60%)",
         "what it means": "On day 1 the response sits on the shocked area (box + 3 cells), "
                          "not spread over the whole map."},
        {"check": "Spreads outward (days 4–8)", "result": yes(chk["propagates"]),
         "number": "energy outside the near zone at days 4–8 > 1.5× day 1 and ≥ 2% of the peak",
         "what it means": "Like real weather, the signal travels away from the shock over the following days."},
        {"check": "No jump > 8 cells/day", "result": yes(chk["no_teleport"]),
         "number": f"worst share beyond reach: {100 * chk['far_frac_max']:.2f}% (needs < 5%)",
         "what it means": "Nothing appears far away faster than about 900 km per day (no teleporting)."},
        {"check": "Stable (< 5 std)", "result": yes(chk["stable"]),
         "number": "max |change| ≤ 5 channel std",
         "what it means": "The response does not blow up numerically."},
    ] + [{"check": "Computed by", "result": "", "number": source, "what it means": ""}]


def _zero_shock_check(ctx, win) -> None:
    url = ctx.controls.server_url
    t0 = time.perf_counter()
    a, _, _ = client.run_twin(url, win["init_states"], win["doy_sincos"], perturbation=None)
    b, _, _ = client.run_twin(url, win["init_states"], win["doy_sincos"],
                              perturbation=np.zeros(win["init_states"].shape[1:]))
    st.session_state["zero_check"] = {"window": win["last_day"], "max_abs": float(np.abs(a - b).max()),
                                      "seconds": time.perf_counter() - t0}


def render(ctx) -> None:
    st.subheader("Shock Lab · where does the shock's effect go?")
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
    st.markdown("#### Zero-shock check")
    st.caption("Two fresh live calls on the same window: one with no perturbation, one with a perturbation of "
               "all zeros. A well-behaved twin must return identical forecasts (max |difference| exactly 0).")
    if st.button("Zero-shock check", key="zero_shock_button"):
        if ctx.health is None:
            st.error("The twin server is not reachable, so the check cannot run.")
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
        msg = (f"Window ending {zc['window']}: max |difference| = {zc['max_abs']:.3g} "
               f"({zc['seconds']:.2f} s for both calls)")
        (st.success if ok else st.error)(("PASS · " if ok else "FAIL · ") + msg)


def _render_analysis(ctx, run) -> None:
    std = np.asarray(ctx.archive.std, np.float64)
    box = shock_box(run.shock_label, run.perturbation)
    mask = M.mask_from_box(box)
    d = run.shocked - run.base
    e = M.energy(d, std)
    reg = M.region_sums(e, mask)
    prof = M.radial_profile(e, mask)
    rch = M.reach(e, mask)

    st.caption(f"Shock **{run.shock_label}** on the window ending {run.last_day}. Box: lat {box[0]:.1f}–{box[1]:.1f}°N, "
               f"lon {box[2]:.1f}–{box[3]:.1f}°E ({int(mask.sum())} cells"
               f"{'' if run.shock_label in BOXES else ', bounding box of the perturbed cells'}). "
               "Energy = sum over the 6 channels of (shocked − baseline)² in units of each channel's std, "
               "the same measure the organisers' scorer uses.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Day-1 energy near the box", f"{100 * M.near_frac_day1(e, mask):.1f}%")
    c2.metric("Reach on day 1", f"ring {rch[0]}" if rch[0] >= 0 else "none")
    c3.metric("Reach on day 8", f"ring {rch[7]}" if len(rch) > 7 and rch[7] >= 0 else "none")
    c4.metric("Largest change", f"{float(np.abs(d / std[None, :, None, None]).max()):.2f} std")

    left, right = st.columns(2)
    fig = go.Figure()
    for name, key, col in (("inside box", "inside", COLORS[0]), ("near (1–3 cells)", "near", COLORS[1]),
                           ("outside near zone", "outside", COLORS[2])):
        fig.add_trace(go.Scatter(x=DAYS[:len(e)], y=reg[key], name=name, mode="lines+markers",
                                 line=dict(color=col, width=2), marker=dict(size=8)))
    left.plotly_chart(_line_layout(fig, "Response energy by region", "lead day", "energy (std² summed)"),
                      key="shocklab_energy")

    fig = go.Figure()
    for i, day in enumerate((1, 4, 8)):
        if day <= len(prof):
            fig.add_trace(go.Scatter(x=np.arange(prof.shape[1]), y=100 * prof[day - 1], name=f"day {day}",
                                     mode="lines+markers", line=dict(color=COLORS[i], width=2), marker=dict(size=8)))
    fig = _line_layout(fig, "Radial profile: share of energy by distance from the box",
                       "ring (0 = inside box, k = k cells away)", "% of that day's energy")
    fig.update_layout(hovermode="x")
    right.plotly_chart(fig, key="shocklab_radial")

    st.caption("Reach by day (furthest ring holding ≥ 5% of the energy): "
               + ", ".join(f"d{k + 1}: {r if r >= 0 else '–'}" for k, r in enumerate(rch)))

    st.markdown("#### Causality checks (the judge's four dynamics checks)")
    if KIT is not None:
        chk = KIT.causality_checks(run.base, run.shocked, box, std)
        src = "organisers' code: monsoon_r1.scoring.causality_checks"
    else:
        chk = M.causality_checks_local(run.base, run.shocked, mask, std)
        src = "dashboard/metrics.py (same thresholds; organisers' kit not importable here)"
    st.dataframe(_checks_table(chk, src), hide_index=True, width="stretch")
