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


MAP_HEIGHT = 350       # px; with colourbar="bottom" the map fills a ~300 px column at 1280x720
WIND_STRIDE = 4        # one arrow every 4th cell
WIND_REF = 10.0        # m/s shown by the reference arrow; same scale on baseline and shocked maps
HOW_TO_READ = ("How to read this: left is the twin's 10-day forecast without a shock, middle is the same forecast "
               "with the shock painted in the dashed box, right is their difference (red = the shock raised the "
               "value, blue = lowered it); arrows are 10 m wind, black lines the coastline.")


def shock_cells(pert) -> int:
    """Number of grid cells where the perturbation is non-zero in any channel."""
    if pert is None:
        return 0
    nz = np.abs(np.asarray(pert)) > 0
    return int((nz.any(axis=0) if nz.ndim == 3 else nz).sum())


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

    du = units.split(" (")[0]
    coast = np.asarray(arc.static[0]) if getattr(arc, "static", None) is not None else None
    st.markdown(f"<div style='font-size:0.95rem;margin:-0.3rem 0 -0.4rem 0'><b>{ch}</b> · lead day <b>{c.day}</b> "
                f"after {run.last_day} · shock <code>{run.shock_label}</code></div>", unsafe_allow_html=True)
    h, common = MAP_HEIGHT, dict(coast=coast, colorbar="bottom")
    wind_kw = dict(wind_stride=WIND_STRIDE, wind_ref=WIND_REF)
    cols = st.columns(3, gap="small")
    with cols[0]:
        st.plotly_chart(plots.field_map(base, lats, lons, "Baseline", units, cmap=cmap, zmid=zmid, zmin=zmin,
                                        zmax=zmax, box=box, wind=wind(run.base), height=h, places=True,
                                        **wind_kw, **common),
                        key="maps_base", config={"displayModeBar": False})
    with cols[1]:
        st.plotly_chart(plots.field_map(shocked, lats, lons, "Shocked", units, cmap=cmap, zmid=zmid, zmin=zmin,
                                        zmax=zmax, box=box, wind=wind(run.shocked), height=h, **wind_kw, **common),
                        key="maps_shocked", config={"displayModeBar": False})
    with cols[2]:
        empty = f"No change in {ch} on day {c.day}"
        if run.perturbation is None:
            empty = "No shock in this run: nothing to compare"
        st.plotly_chart(plots.diff_map(diff, lats, lons, "Shocked − Baseline", "Δ " + du, box=box, height=h,
                                       empty_text=empty, **common),
                        key="maps_diff", config={"displayModeBar": False})
    st.caption(HOW_TO_READ)

    t = run.timings_ms or {}
    n_cells = shock_cells(run.perturbation)
    m = st.columns(4)
    m[0].metric("Baseline call (round trip)", _fmt_ms(t.get("base_rt")), help=f"server time {_fmt_ms(t.get('base_server'))}")
    m[1].metric("Shocked call (round trip)", _fmt_ms(t.get("shock_rt")) if run.perturbation is not None else "not needed",
                help=f"server time {_fmt_ms(t.get('shock_server'))}")
    m[2].metric("Shock changed", f"{n_cells} cells", help="grid cells (1° x 1°) touched by the perturbation on the "
                                                           "input day, any channel")
    m[3].metric(f"Max |Δ {ch}| on day {c.day}", f"{float(np.nanmax(np.abs(diff))):.3g} {du}")

    with st.expander("Watch the shock spread: difference map, days 1–10", expanded=False):
        if run.perturbation is None:
            st.caption("No shock in the last run, so the difference is zero every day.")
        else:
            diffs = plots.to_display(ch, run.shocked[:, ci] - run.base[:, ci], is_diff=True)
            st.plotly_chart(plots.diff_animation(diffs, lats, lons, f"Δ {ch}", "Δ " + du,
                                                 box=box, height=460, coast=coast),
                            key="maps_anim", config={"displayModeBar": False})
            st.caption("Press Play. One colour range for all days, so growth and spread are comparable.")
