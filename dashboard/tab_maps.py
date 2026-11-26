"""Maps tab: Baseline | Shocked | Difference for the chosen channel and lead day, plus a day 1-10 animation.

Channel and day changes re-slice the last run (no server call). Window or shock changes need a new Run.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

import plots
from common import CH


def is_stale(controls, run) -> bool:
    """True when the window or shock in the sidebar differ from what the last run used."""
    if controls.last_day != run.last_day or controls.shock_label != run.shock_label:
        return True
    a, b = controls.perturbation, run.perturbation
    if (a is None) != (b is None):
        return True
    return a is not None and not np.array_equal(a, b)


def _fmt_ms(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "–"
    return "–" if not np.isfinite(v) else f"{v:.0f} ms"


def render(ctx) -> None:
    run, c = ctx.run, ctx.controls
    if run is None:
        st.info("Pick a window and a shock in the sidebar, then press **Run the twin**. "
                "Each Run makes two live calls to the twin server: a baseline and a shocked rollout of 10 days.")
        return

    if is_stale(c, run):
        st.warning(f"The sidebar window or shock changed (showing the last run: window ending {run.last_day}, "
                   f"shock {run.shock_label}). Press Run to update.")

    arc = ctx.archive
    lats, lons = np.asarray(arc.lats), np.asarray(arc.lons)
    ch, d = c.channel, int(c.day) - 1
    ci = CH[ch]
    units = plots.display_units(ch)
    base = plots.to_display(ch, run.base[d, ci])
    shocked = plots.to_display(ch, run.shocked[d, ci])
    diff = plots.to_display(ch, run.shocked[d, ci] - run.base[d, ci], is_diff=True)
    box = plots.box_from_perturbation(run.perturbation, lats, lons)
    zmin = float(min(np.nanmin(base), np.nanmin(shocked)))
    zmax = float(max(np.nanmax(base), np.nanmax(shocked)))
    cmap = plots.CHANNEL_CMAPS.get(ch, "Viridis")
    zmid = 0.0 if ch in ("u10", "v10") else None
    if zmid is not None:
        zmax = max(abs(zmin), abs(zmax))
        zmin = -zmax

    def wind(arr):
        if not c.show_wind:
            return None
        return (arr[d, CH["u10"]], arr[d, CH["v10"]])

    st.markdown(f"**{ch}** · lead day **{c.day}** after {run.last_day} · shock: `{run.shock_label}`")
    h = 340
    cols = st.columns(3, gap="small")
    with cols[0]:
        st.plotly_chart(plots.field_map(base, lats, lons, "Baseline", units, cmap=cmap, zmid=zmid, zmin=zmin,
                                        zmax=zmax, box=box, wind=wind(run.base), height=h),
                        key="maps_base", config={"displayModeBar": False})
    with cols[1]:
        st.plotly_chart(plots.field_map(shocked, lats, lons, "Shocked", units, cmap=cmap, zmid=zmid, zmin=zmin,
                                        zmax=zmax, box=box, wind=wind(run.shocked), height=h),
                        key="maps_shocked", config={"displayModeBar": False})
    with cols[2]:
        st.plotly_chart(plots.diff_map(diff, lats, lons, "Shocked − Baseline", "Δ " + units.split(" (")[0],
                                       box=box, height=h),
                        key="maps_diff", config={"displayModeBar": False})

    t = run.timings_ms or {}
    hl = ctx.health or {}
    m = st.columns(4)
    m[0].metric("Baseline call (round trip)", _fmt_ms(t.get("base_rt")), help=f"server time {_fmt_ms(t.get('base_server'))}")
    m[1].metric("Shocked call (round trip)", _fmt_ms(t.get("shock_rt")) if run.perturbation is not None else "not needed",
                help=f"server time {_fmt_ms(t.get('shock_server'))}")
    m[2].metric("Server model", str(hl.get("model_name", "offline")), help=f"backend: {hl.get('backend', '?')}")
    m[3].metric(f"Max |Δ {ch}| on day {c.day}", f"{float(np.nanmax(np.abs(diff))):.3g} {units.split(' (')[0]}")

    with st.expander("Watch the shock spread: difference map, days 1–10", expanded=False):
        if run.perturbation is None:
            st.caption("No shock in the last run, so the difference is zero every day.")
        else:
            diffs = plots.to_display(ch, run.shocked[:, ci] - run.base[:, ci], is_diff=True)
            st.plotly_chart(plots.diff_animation(diffs, lats, lons, f"Δ {ch}", "Δ " + units.split(" (")[0],
                                                 box=box, height=460),
                            key="maps_anim", config={"displayModeBar": False})
            st.caption("Press Play. One colour range for all days, so growth and spread are comparable.")
