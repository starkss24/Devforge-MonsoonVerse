"""Maps tab: a 10-day strip of small difference maps (how the shock spreads), then Baseline | Shocked | Difference
for the chosen channel and lead day, and one readout line.

Clicking a day in the strip, or the day buttons under it, sets the lead day for the three large maps.
Channel and day changes re-slice the last run (no server call). Window or shock changes need a new Run.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

import plots
import sidebar
from common import CH, CHANNELS


def is_stale(controls, run) -> bool:
    """True when the window or shock in the sidebar differ from what the last run used."""
    if controls.last_day != run.last_day or controls.shock_label != run.shock_label:
        return True
    a, b = controls.perturbation, run.perturbation
    if (a is None) != (b is None):
        return True
    return a is not None and not np.array_equal(a, b)


MAP_HEIGHT = 304       # px; with colourbar="bottom" each map fills a ~300 px column at 1280x720
STRIP_HEIGHT = 108     # px; ten small maps in one row
WIND_STRIDE = 4        # one arrow every 4th cell
WIND_REF = 10.0        # m/s shown by the reference arrow; same scale on baseline and shocked maps
DAYS = list(range(1, 11))
NO_BAR = {"displayModeBar": False}
HOW_TO_READ = ("How to read this: the strip shows the difference the shock makes on each of the 10 days, all on "
               "one colour scale, so you can see it start inside the gold box and then spread. Below, left is the "
               "twin's forecast without the shock, middle with it, right their difference (red = the shock raised "
               "the value, blue = lowered it). Arrows are 10 m wind, dark lines the coast.")


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


def _fmt_num(v: float) -> str:
    return f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"


def day_from_selection(event) -> int | None:
    """Lead day (1..10) from a plotly selection event on the strip, or None if nothing usable was clicked."""
    try:
        points = event["selection"]["points"]
    except (KeyError, TypeError):
        return None
    for p in points or []:
        cd = p.get("customdata") if isinstance(p, dict) else None
        if isinstance(cd, (list, tuple)):
            cd = cd[0] if cd else None
        try:
            day = int(cd)
        except (TypeError, ValueError):
            continue
        if 1 <= day <= 10:
            return day
    return None


def _on_strip_click() -> None:
    day = day_from_selection(st.session_state.get("maps_strip"))
    if day is not None:
        st.session_state["lead_day"] = day


def render_display_controls() -> None:
    """Lead day buttons, display channel and wind toggle in one row (keys lead_day, channel, show_wind)."""
    st.session_state.setdefault("lead_day", 1)
    st.session_state.setdefault("channel", sidebar.DEFAULT_CHANNEL)
    c0, c1, c2, c3 = st.columns([0.75, 4.1, 2.8, 1.35], gap="small", vertical_alignment="center")
    c0.markdown("<p class='mm-meta'>Lead day</p>", unsafe_allow_html=True)
    c1.segmented_control("Lead day", DAYS, key="lead_day", required=True, label_visibility="collapsed")
    c2.selectbox("Display channel", CHANNELS, format_func=sidebar.channel_label, key="channel",
                 label_visibility="collapsed")
    c3.toggle("Wind arrows", value=True, key="show_wind")



def render(ctx) -> None:
    run, c = ctx.run, ctx.controls
    if run is None:
        render_display_controls()
        st.info("Pick a window and a shock in the sidebar, then press **Run the twin**. "
                "Each Run makes two live calls to the twin server: a baseline and a shocked rollout of 10 days.")
        return

    if is_stale(c, run):
        st.warning(f"The sidebar window or shock changed. These maps still show the last run (window ending "
                   f"{run.last_day}, shock {run.shock_label}). Press Run to update them.")

    arc = ctx.archive
    lats, lons = np.asarray(arc.lats), np.asarray(arc.lons)
    ch, d = c.channel, int(c.day) - 1
    ci = CH[ch]
    units = plots.display_units(ch)
    du = units.split(" (")[0]
    coast = np.asarray(arc.static[0]) if getattr(arc, "static", None) is not None else None
    box = plots.box_from_perturbation(run.perturbation, lats, lons)

    # ---- the strip: how the shock spreads, day 1..10
    if run.perturbation is None:
        st.markdown("<p class='mm-meta'>No shock in the last run, so there is nothing to spread. Pick one in the "
                    "sidebar and press Run.</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p class='mm-meta'>How the {run.shock_label} shock spreads: change in {ch}, day by day "
                    f"after {run.last_day}. <span class='mm-quiet'>Click a day to enlarge it.</span></p>",
                    unsafe_allow_html=True)

        diffs = plots.to_display(ch, run.shocked[:, ci] - run.base[:, ci], is_diff=True)
        strip = plots.diff_strip(diffs, lats, lons, "Δ " + du, box=box, selected=c.day, height=STRIP_HEIGHT,
                                 coast=coast)
        st.plotly_chart(strip, key="maps_strip", on_select=_on_strip_click, selection_mode="points",
                        config=NO_BAR, theme=None)

    render_display_controls()

    # ---- the three large maps for the selected day
    base = plots.to_display(ch, run.base[d, ci])
    shocked = plots.to_display(ch, run.shocked[d, ci])
    diff = plots.to_display(ch, run.shocked[d, ci] - run.base[d, ci], is_diff=True)
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

    h, common = MAP_HEIGHT, dict(coast=coast, colorbar="bottom")
    wind_kw = dict(wind_stride=WIND_STRIDE, wind_ref=WIND_REF)
    cols = st.columns(3, gap="small")
    with cols[0]:
        st.plotly_chart(plots.field_map(base, lats, lons, "Baseline", units, cmap=cmap, zmid=zmid, zmin=zmin,
                                        zmax=zmax, box=box, wind=wind(run.base), height=h, places=True,
                                        **wind_kw, **common),
                        key="maps_base", config=NO_BAR, theme=None)
    with cols[1]:
        st.plotly_chart(plots.field_map(shocked, lats, lons, "Shocked", units, cmap=cmap, zmid=zmid, zmin=zmin,
                                        zmax=zmax, box=box, wind=wind(run.shocked), height=h, **wind_kw, **common),
                        key="maps_shocked", config=NO_BAR, theme=None)
    with cols[2]:
        empty = f"No change in {ch} on day {c.day}"
        if run.perturbation is None:
            empty = "No shock in this run: nothing to compare"
        st.plotly_chart(plots.diff_map(diff, lats, lons, "Shocked − baseline", "Δ " + du, box=box, height=h,
                                       empty_text=empty, **common),
                        key="maps_diff", config=NO_BAR, theme=None)

    # ---- one readout line
    t = run.timings_ms or {}
    n_cells = shock_cells(run.perturbation)
    max_d = float(np.nanmax(np.abs(diff)))
    calls = (f"round trip <b>{_fmt_ms(t.get('base_rt'))}</b> baseline, <b>{_fmt_ms(t.get('shock_rt'))}</b> shocked"
             if run.perturbation is not None else f"round trip <b>{_fmt_ms(t.get('base_rt'))}</b>, one call")
    st.markdown(
        f"<div class='mm-readout'><span class='mm-day'>Day {c.day}</span>"
        f"<span>shock changed <b>{n_cells}</b> cells</span>"
        f"<span>largest change <b>{_fmt_num(max_d)} {du}</b> in {ch}</span>"
        f"<span class='mm-quiet' title='server time {_fmt_ms(t.get('base_server'))} and "
        f"{_fmt_ms(t.get('shock_server'))}'>{calls}</span></div>", unsafe_allow_html=True)
    st.caption(HOW_TO_READ)
