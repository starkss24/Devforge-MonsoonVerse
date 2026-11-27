"""Maps tab: the twin as an app. One large interactive map (pan, scroll-zoom, click) with a control panel:
a timeline from the 3 real input days into the forecast, Play, a layer switch and a view switch (with shock,
no shock, shock effect, what really happened), plus a check against the observed day (peak, kept detail, error).
Below: the day-by-day strip of the shock's effect, a side-by-side view and the forecast at one place.

Everything comes from the last live run (two server calls). Only a new window, shock or forecast length calls the
server again, and in Live mode that happens by itself (wants_run). Keys: B / N step a day, P plays the timeline;
they are letters, not arrows, because arrows would also move whichever radio button has focus.
The map has no tiles on purpose: a tiled web map shows a blank "Map error" without internet (tested), and the demo
projector may be offline.
"""
from __future__ import annotations

import datetime as dt
import html
import math
import time

import numpy as np
import streamlit as st

import plots
import sidebar
from common import CH, CHANNELS


def is_stale(controls, run) -> bool:
    """True when the window, shock or forecast length in the sidebar differ from what the last run used."""
    if controls.last_day != run.last_day or controls.shock_label != run.shock_label:
        return True
    base = getattr(run, "base", None)
    if base is not None and int(getattr(controls, "steps", base.shape[0])) != int(base.shape[0]):
        return True
    a, b = controls.perturbation, run.perturbation
    if (a is None) != (b is None):
        return True
    return a is not None and not np.array_equal(a, b)


def wants_run(controls, run, health, live: bool) -> bool:
    """Run on a click; in Live mode also whenever the server is up and the sidebar differs from the last run
    (no run yet, other window, other shock, other length). A down server never triggers a live run."""
    if controls.run_clicked:
        return True
    return bool(live) and health is not None and (run is None or is_stale(controls, run))


MAP_HEIGHT = 486       # px; the big map, sized to fill the left column at 1280x720
PANEL_HEIGHT = 304     # px; each of the side-by-side maps
STRIP_HEIGHT = 108     # px; ten small maps in one row
WIND_STRIDE = 3
WIND_REF = 10.0        # m/s shown by the reference arrow; same scale on every map
PLAY_SECONDS = 0.4     # pause per frame when Play runs the timeline
DAYS = list(range(1, 11))
N_INPUT = 3            # real days the twin starts from (kit contract)
NO_BAR = {"displayModeBar": False}
LAYER_TEXT = {"tp": "Rain", "t2m": "Temperature", "u10": "Wind east–west", "v10": "Wind north–south",
              "msl": "Pressure", "tcwv": "Moisture"}
VIEWS = ("shocked", "base", "effect", "observed")
VIEW_TEXT = {"shocked": "With shock", "base": "No shock", "effect": "Effect", "observed": "Observed"}
CLICK_MOVE, CLICK_PROBE = "move", "probe"
CLICK_TEXT = {CLICK_MOVE: "Click moves the shock", CLICK_PROBE: "Click charts a place"}
BOX_CENTRE, CLICKED = "Shock box centre", "Clicked point"
CITIES = plots.CITY_LABELS
HOW_TO_READ = ("How to read this: the twin starts from the 3 real days before the forecast (days −2 to 0) and runs "
               "ahead from there. The timeline walks through them; Play runs it like a weather broadcast. 'With "
               "shock' is the forecast after your painted change (gold box), 'No shock' the same forecast without "
               "it, 'Effect' their difference (red = raised, blue = lowered), 'Observed' the real weather from "
               "the archive. Drag to pan the map, scroll to zoom, double-click to reset. Keys (click "
               "the page first): B and N step a day, P plays.")


# ---------------------------------------------------------------- timeline helpers (tested)
def timeline_days(steps: int) -> list[int]:
    """Days on the timeline: -2..0 are the real input days (0 = last one), 1..steps the forecast."""
    return list(range(1 - N_INPUT, int(steps) + 1))


def day_date(last_day: str, k: int) -> dt.date:
    return dt.date.fromisoformat(str(last_day)) + dt.timedelta(days=int(k))


def day_tag(k: int) -> str:
    return "real data" if k <= 0 else ("forecast" if k <= 10 else "forecast, experimental (beyond the judged 10 days)")


def timeline_label(k: int) -> str:
    return f"Day {k}" if k >= 1 else f"Day {k}, real"


def view_field(run, ch: str, k: int, view: str):
    """Physical [nlat, nlon] field shown for timeline day k and view, or None when there is nothing to show
    (observed day beyond the archive). Days <= 0 are the real input days; on day 0 the painted shock is added
    for the 'shocked' view and is itself the 'effect'."""
    c = CH[ch]
    inputs = run.window["init_states"]
    pert = run.perturbation
    if view == "effect":
        if k >= 1:
            return run.shocked[k - 1, c] - run.base[k - 1, c]
        if k == 0 and pert is not None:
            return np.asarray(pert)[c]
        return np.zeros_like(inputs[0, c])
    if view == "observed":
        if k <= 0:
            return inputs[k + N_INPUT - 1, c]
        truth = run.window.get("truth")
        return truth[k - 1, c] if truth is not None and k - 1 < len(truth) else None
    if k >= 1:
        return (run.shocked if view == "shocked" else run.base)[k - 1, c]
    x = inputs[k + N_INPUT - 1, c]
    if view == "shocked" and k == 0 and pert is not None:
        x = x + np.asarray(pert)[c]
    return x


def compare_observed(pred, obs) -> dict:
    """Peak, kept spatial detail (variance ratio, the judges' anti-blur measure) and RMSE of a forecast map
    against the observed map, both in display units."""
    p, o = np.asarray(pred, float), np.asarray(obs, float)
    vo = float(np.nanvar(o))
    return {"peak_pred": float(np.nanmax(p)), "peak_obs": float(np.nanmax(o)),
            "detail_pct": 100.0 * float(np.nanvar(p)) / vo if vo > 0 else float("nan"),
            "rmse": float(np.sqrt(np.nanmean((p - o) ** 2)))}


# ---------------------------------------------------------------- pure helpers (tested)
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


def _points(event):
    try:
        return event["selection"]["points"] or []
    except (KeyError, TypeError):
        return []


def day_from_selection(event) -> int | None:
    """Lead day (1..10) from a plotly selection event on the strip, or None if nothing usable was clicked."""
    for p in _points(event):
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


def latlon_from_selection(event) -> tuple[float, float] | None:
    """(lat, lon) of a click on a large map, or None.

    The click grid's customdata is [lat, lon, value]; Streamlit delivers it as a list or as a dict keyed
    "0", "1", "2". Without customdata the point's own y (lat) / x (lon) are used."""
    for p in _points(event):
        if not isinstance(p, dict):
            continue
        cd = p.get("customdata")
        if isinstance(cd, dict):
            cd = [cd.get("0"), cd.get("1")]
        pair = (cd[0], cd[1]) if isinstance(cd, (list, tuple)) and len(cd) >= 2 else (p.get("y"), p.get("x"))
        try:
            return float(pair[0]), float(pair[1])
        except (TypeError, ValueError):
            continue
    return None


def nearest_cell(lat: float, lon: float, lats, lons) -> tuple[int, int]:
    return (int(np.argmin(np.abs(np.asarray(lats, float) - lat))),
            int(np.argmin(np.abs(np.asarray(lons, float) - lon))))


def _compass(dlat: float, dlon_km: float) -> str:
    names = ["east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east"]
    ang = math.degrees(math.atan2(dlat, dlon_km)) % 360
    return names[int((ang + 22.5) // 45) % 8]


def describe_spread(diffs, box, lats, lons, day: int) -> str:
    """One plain sentence about where the change is on `day`: area (vs day 1) and how far its centre has moved
    from the shock box. Threshold: 10% of the largest change over all days (the strip's shared scale)."""
    d = np.abs(np.asarray(diffs, dtype=float))
    peak = float(np.nanmax(d)) if d.size else 0.0
    if peak < plots.ZERO_DIFF:
        return f"Day {day}: no change anywhere."
    thr = 0.1 * peak
    a = d[day - 1]
    cells, cells1 = int((a > thr).sum()), int((d[0] > thr).sum())
    if cells == 0:
        return f"Day {day}: the change has faded below a tenth of its peak everywhere."
    gy, gx = np.meshgrid(np.asarray(lats, float), np.asarray(lons, float), indexing="ij")
    w = np.where(a > thr, a, 0.0)
    clat, clon = float((w * gy).sum() / w.sum()), float((w * gx).sum() / w.sum())
    area = f"the change covers {cells} cells" + ("" if day == 1 else f" ({cells1} on day 1)")
    if box is None:
        return f"Day {day}: {area}, centred near {clat:.0f}°N, {clon:.0f}°E."
    blat, blon = (box[0] + box[1]) / 2, (box[2] + box[3]) / 2
    dy_km = (clat - blat) * 111.0
    dx_km = (clon - blon) * 111.0 * math.cos(math.radians((clat + blat) / 2))
    dist = math.hypot(dx_km, dy_km)
    where = ("its centre is still over the shock box" if dist < 60 else
             f"its centre is {dist:.0f} km {_compass(dy_km, dx_km)} of the shock box")
    return f"Day {day}: {area}; {where}."


def fade_summary(effect, box, units: str, lats, lons) -> str:
    """Live sentence about the change inside the shock box: mean change on days 1, 4, 8 (as available) and the first
    day it drops below a tenth of day 1, from the fetched rollouts (no hard-coded model claims)."""
    if box is None:
        return ""
    from common import box_mask
    m = box_mask(*box)
    e = np.asarray(effect, dtype=float)
    if not m.any() or e.ndim != 3:
        return ""
    series = np.array([float(np.nanmean(e[d][m])) for d in range(e.shape[0])])
    d1 = series[0]
    if abs(d1) < plots.ZERO_DIFF:
        return "Inside the box the change is too small to measure on day 1."
    parts = [f"day {d} {series[d - 1]:+.2f}" for d in (1, 4, 8) if d <= len(series)]
    below = np.where(np.abs(series) < 0.1 * abs(d1))[0]
    fade = (f"it falls below a tenth of day 1 on day {int(below[0]) + 1}" if below.size else
            f"it does not fade within {len(series)} days")
    return f"Inside the box: {', '.join(parts)} {units}; {fade}."


def probe_options() -> list[str]:
    return [BOX_CENTRE, *CITIES, CLICKED]


def probe_location(place: str, box, click) -> tuple[float, float, str]:
    """(lat, lon, label) for the chosen place; box centre falls back to Delhi without a shock."""
    if place in CITIES:
        la, lo = CITIES[place]
        return la, lo, place
    if place == CLICKED and click is not None:
        return float(click[0]), float(click[1]), f"{click[0]:.1f}°N, {click[1]:.1f}°E"
    if box is not None:
        return (box[0] + box[1]) / 2, (box[2] + box[3]) / 2, "the shock box centre"
    la, lo = CITIES["Delhi"]
    return la, lo, "Delhi"


# ---------------------------------------------------------------- callbacks (run before the script reruns)
def _timeline_bounds() -> tuple[int, int]:
    return 1 - N_INPUT, int(st.session_state.get("timeline_max", 10))


def _on_strip_click() -> None:
    day = day_from_selection(st.session_state.get("maps_strip"))
    if day is not None:
        st.session_state["lead_day"] = day


def _shift_day(delta: int) -> None:
    lo, hi = _timeline_bounds()
    d = int(st.session_state.get("lead_day") or 1)
    st.session_state["lead_day"] = min(hi, max(lo, d + delta))


def _on_play() -> None:
    st.session_state["play_request"] = True                  # the timeline jumps to the last day after the run


def _on_map_click(key: str) -> None:
    pt = latlon_from_selection(st.session_state.get(key))
    if pt is None:
        return
    if st.session_state.get("click_mode", CLICK_MOVE) == CLICK_MOVE:
        st.session_state.update(sidebar.shock_state_for_click(st.session_state.get("shock_preset", "none"), *pt))
    else:
        st.session_state["probe_click"] = pt
        st.session_state["probe_place"] = CLICKED


def _map_click_cb(key: str):
    return lambda: _on_map_click(key)


# ---------------------------------------------------------------- render
def _layer_row() -> None:
    st.session_state.setdefault("channel", sidebar.DEFAULT_CHANNEL)
    c1, c2 = st.columns([6.6, 1.4], gap="small", vertical_alignment="center")
    c1.segmented_control("Layer", CHANNELS, key="channel", required=True, label_visibility="collapsed",
                         format_func=lambda c: LAYER_TEXT[c])
    c2.toggle("Wind arrows", value=True, key="show_wind")


def _day_buttons(where, has_run: bool, steps: int) -> None:
    b1, b2, b3 = where.columns(3, gap="small")
    b1.button("Earlier", icon=":material/chevron_left:", key="day_prev", on_click=_shift_day, args=(-1,),
              shortcut="B", width="stretch", help="Previous day (key B)")
    b2.button("Later", icon=":material/chevron_right:", key="day_next", on_click=_shift_day, args=(1,),
              shortcut="N", width="stretch", help="Next day (key N)")
    b3.button("Play", icon=":material/play_arrow:", key="play_button", on_click=_on_play, shortcut="P",
              width="stretch", disabled=not has_run,
              help=f"Run the timeline from the real days to day {steps} (key P)")


def _dayhead_html(k: int, last_day: str) -> str:
    d = day_date(last_day, k)
    return (f"<div class='mm-dayhead'><span class='mm-day'>Day {k}</span>"
            f"<span class='mm-date'>{d:%A} {d.day} {d:%B}</span>"
            f"<span class='mm-tag mm-tag-{'real' if k <= 0 else 'fc'}'>{html.escape(day_tag(k))}</span></div>")


def render(ctx) -> None:
    run, c = ctx.run, ctx.controls
    steps = int(run.base.shape[0]) if run is not None else int(getattr(c, "steps", 10))
    days = timeline_days(steps)
    st.session_state["timeline_max"] = steps
    st.session_state.setdefault("lead_day", 1)
    if st.session_state.get("lead_day") not in days:
        st.session_state["lead_day"] = min(max(int(st.session_state.get("lead_day") or 1), days[0]), days[-1])
    st.session_state.setdefault("map_view", "shocked")
    st.session_state.setdefault("click_mode", CLICK_MOVE)
    st.session_state.setdefault("probe_place", BOX_CENTRE)
    _layer_row()

    left, right = st.columns([1.18, 0.82], gap="medium")
    map_ph = left.empty()
    playing = bool(st.session_state.pop("play_request", False)) and run is not None
    with right:
        head_ph = st.empty()
        slider_ph = st.empty()     # the timeline; while Play runs, a progress bar in the same place
        _day_buttons(st, run is not None, steps)
        st.segmented_control("Show", VIEWS, key="map_view", required=True, label_visibility="collapsed",
                             format_func=lambda v: VIEW_TEXT[v])
        st.segmented_control("Map clicks", [CLICK_MOVE, CLICK_PROBE], key="click_mode", required=True,
                             label_visibility="collapsed", format_func=lambda m: CLICK_TEXT[m])
        info_ph = st.empty()

    def timeline_widget() -> None:
        slider_ph.select_slider("Timeline", days, key="lead_day", format_func=timeline_label,
                                label_visibility="collapsed")

    k = int(st.session_state.get("lead_day", 1))
    if not playing:
        timeline_widget()
    if run is None:
        head_ph.markdown(_dayhead_html(k, c.last_day), unsafe_allow_html=True)
        live = st.session_state.get("live_mode", True)
        map_ph.info("Pick a window and a shock in the sidebar, then press **Run the twin**"
                    + (" (Live mode runs it by itself as soon as the server answers)." if live else ".")
                    + " Each run makes two live calls to the twin server: a baseline and a shocked rollout.")
        return
    if is_stale(c, run):
        st.warning(f"The sidebar changed. The map still shows the last run (window ending {run.last_day}, shock "
                   f"{run.shock_label}, {steps} days). Press Run to update it, or turn on Live.")

    arc = ctx.archive
    lats, lons = np.asarray(arc.lats), np.asarray(arc.lons)
    ch, view = c.channel, st.session_state.get("map_view", "shocked")
    ci = CH[ch]
    units = plots.display_units(ch)
    du = units.split(" (")[0]
    coast = np.asarray(arc.static[0]) if getattr(arc, "static", None) is not None else None
    box = plots.box_from_perturbation(run.perturbation, lats, lons)
    shocked_run = run.perturbation is not None
    cmap = plots.CHANNEL_CMAPS.get(ch, "Viridis")

    # one colour scale for the whole timeline (values) and one for the shock effect, so Play compares honestly
    vals = [run.window["init_states"][:, ci], run.base[:, ci], run.shocked[:, ci]]
    if run.window.get("truth") is not None and len(run.window["truth"]):
        vals.append(run.window["truth"][:, ci])
    allv = plots.to_display(ch, np.concatenate([v.reshape(-1) for v in vals]))
    zmin, zmax = float(np.nanmin(allv)), float(np.nanmax(allv))
    zmid = 0.0 if ch in ("u10", "v10") else None
    if zmid is not None:
        zmax = max(abs(zmin), abs(zmax))
        zmin = -zmax
    eff = plots.to_display(ch, run.shocked[:, ci] - run.base[:, ci], is_diff=True)
    pert_d = plots.to_display(ch, np.asarray(run.perturbation)[ci], is_diff=True) if shocked_run else 0.0
    elim = plots.symmetric_limit(eff, pert_d)
    elim = elim if elim >= plots.ZERO_DIFF else 1.0
    plat, plon, plabel = probe_location(st.session_state.get("probe_place", BOX_CENTRE), box,
                                        st.session_state.get("probe_click"))
    pi, pj = nearest_cell(plat, plon, lats, lons)
    action = "move the shock here" if st.session_state.get("click_mode") == CLICK_MOVE else "chart this place"
    shock_txt = f"with the {run.shock_label} shock" if shocked_run else "no shock painted"

    def wind_for(day: int):
        if not c.show_wind:
            return None
        if day >= 1:
            src = run.base if view == "base" else run.shocked if view != "observed" else None
            if src is None:
                tr = run.window.get("truth")
                if tr is None or day - 1 >= len(tr):
                    return None
                return tr[day - 1, CH["u10"]], tr[day - 1, CH["v10"]]
            return src[day - 1, CH["u10"]], src[day - 1, CH["v10"]]
        x = run.window["init_states"][day + N_INPUT - 1]
        return x[CH["u10"]], x[CH["v10"]]

    def draw(day: int, interactive: bool) -> None:
        raw = view_field(run, ch, day, view)
        is_eff = view == "effect"
        title = f"{LAYER_TEXT[ch]}: {VIEW_TEXT[view].lower()}, {timeline_label(day).lower()}"
        if raw is None:
            z = np.full((len(lats), len(lons)), np.nan)
            note = "The archive has no observed weather for this day"
        else:
            z = plots.to_display(ch, raw, is_diff=is_eff)
            note = ("No change on this day" if is_eff and float(np.nanmax(np.abs(z))) < plots.ZERO_DIFF else None)
        fig = plots.twin_map(z, lats, lons, title, ("Δ " + du) if is_eff else units,
                             plots.DIVERGING if is_eff else cmap,
                             zmin=-elim if is_eff else zmin, zmax=elim if is_eff else zmax,
                             zmid=0.0 if is_eff else zmid, box=box,
                             wind=None if is_eff else wind_for(day), coast=coast, height=MAP_HEIGHT,
                             wind_stride=WIND_STRIDE, wind_ref=WIND_REF, note=note)
        fig.add_trace(plots.probe_marker_trace(lats[pi], lons[pj]))
        if interactive:
            plots.add_click_grid(fig, np.nan_to_num(z), lats, lons, ("Δ " + du) if is_eff else du, action)
            fig.update_layout(dragmode="pan")
            map_ph.plotly_chart(fig, key="maps_main", on_select=_map_click_cb("maps_main"), selection_mode="points",
                                config=plots.MAP_CONFIG, theme=None)
        else:
            map_ph.plotly_chart(fig, key=f"maps_main_play_{day}", config=plots.MAP_CONFIG, theme=None)
        head_ph.markdown(_dayhead_html(day, run.last_day), unsafe_allow_html=True)

        # side panel facts: place value, spread story, check against the real day
        lines = []
        pv = view_field(run, ch, day, "shocked")
        bv = view_field(run, ch, day, "base")
        if pv is not None:
            v_s = float(plots.to_display(ch, pv)[pi, pj])
            v_b = float(plots.to_display(ch, bv)[pi, pj])
            extra = f" (no shock {_fmt_num(v_b)})" if shocked_run and abs(v_s - v_b) > 1e-9 else ""
            lines.append(f"<b>{html.escape(plabel[:1].upper() + plabel[1:])}</b>: {_fmt_num(v_s)} {du}{extra}")
        if shocked_run and day >= 1:
            lines.append(html.escape(describe_spread(eff[:min(steps, len(eff))], box, lats, lons, day)))
        if shocked_run:
            lines.append(html.escape(fade_summary(eff, box, du, lats, lons)))
        obs = view_field(run, ch, day, "observed")
        if day >= 1 and obs is not None:
            cmp = compare_observed(plots.to_display(ch, bv), plots.to_display(ch, obs))
            lines.append(f"Against the real day (no shock): peak <b>{_fmt_num(cmp['peak_pred'])}</b> vs "
                         f"<b>{_fmt_num(cmp['peak_obs'])} {du}</b> observed; detail kept "
                         f"<b>{cmp['detail_pct']:.0f}%</b> of the real variation (100% = no blur); typical error "
                         f"<b>{_fmt_num(cmp['rmse'])} {du}</b>.")
        elif day >= 1:
            lines.append("<span class='mm-quiet'>No observed weather in the archive for this day.</span>")
        t = run.timings_ms or {}
        calls = (f"Two live calls: {_fmt_ms(t.get('base_rt'))} and {_fmt_ms(t.get('shock_rt'))}; "
                 f"{shock_cells(run.perturbation)} cells shocked, {shock_txt}." if shocked_run else
                 f"One live call: {_fmt_ms(t.get('base_rt'))}; no shock painted.")
        lines.append(f"<span class='mm-quiet'>{calls}</span>")
        info_ph.markdown("<div class='mm-facts'>" + "".join(f"<p>{x}</p>" for x in lines) + "</div>",
                         unsafe_allow_html=True)

    if playing:
        for i, day in enumerate(days):
            slider_ph.progress((i + 1) / len(days), text=f"Playing: {timeline_label(day)} (to day {steps})")
            draw(day, interactive=False)
            time.sleep(PLAY_SECONDS)
        k = days[-1]
        st.session_state["lead_day"] = k          # set before the timeline widget exists in this run
        timeline_widget()
    draw(k, interactive=True)

    # ---- below the fold: the shock's effect day by day, side by side, and the forecast at one place
    if shocked_run:
        st.markdown(f"#### How the shock spreads, day by day ({ch})")
        n = min(steps, 10)
        sel = min(max(k, 1), n)
        strip = plots.diff_strip(eff[:n], lats, lons, "Δ " + du, box=box, selected=sel, height=STRIP_HEIGHT,
                                 coast=coast)
        st.plotly_chart(strip, key="maps_strip", on_select=_on_strip_click, selection_mode="points", config=NO_BAR,
                        theme=None)
    with st.expander("Side by side for this day: no shock, with shock, difference", expanded=False):
        if k < 1:
            st.caption("Pick a forecast day (1 or later) on the timeline.")
        else:
            d = k - 1
            cols = st.columns(3, gap="small")
            common = dict(coast=coast, colorbar="bottom", height=PANEL_HEIGHT)
            cols[0].plotly_chart(plots.field_map(plots.to_display(ch, run.base[d, ci]), lats, lons, "No shock",
                                                 units, cmap=cmap, zmid=zmid, zmin=zmin, zmax=zmax, box=box,
                                                 places=True, **common),
                                 key="maps_base", config=NO_BAR, theme=None)
            cols[1].plotly_chart(plots.field_map(plots.to_display(ch, run.shocked[d, ci]), lats, lons, "With shock",
                                                 units, cmap=cmap, zmid=zmid, zmin=zmin, zmax=zmax, box=box,
                                                 **common),
                                 key="maps_shocked", config=NO_BAR, theme=None)
            cols[2].plotly_chart(plots.diff_map(eff[d], lats, lons, "Difference", "Δ " + du, vmax=elim, box=box,
                                                empty_text=f"No change in {ch} on day {k}", **common),
                                 key="maps_diff", config=NO_BAR, theme=None)

    st.markdown("#### Forecast at one place")
    st.selectbox("Place", probe_options(), key="probe_place",
                 format_func=lambda p: "Clicked point (switch clicks to 'charts a place', then click the map)"
                 if p == CLICKED and st.session_state.get("probe_click") is None else p)
    truth = run.window.get("truth")
    tser = plots.to_display(ch, truth[:, ci, pi, pj]) if truth is not None and len(truth) else None
    start = float(plots.to_display(ch, run.window["init_states"][-1, ci, pi, pj]))
    fig = plots.point_series(plots.to_display(ch, run.base[:, ci, pi, pj]),
                             plots.to_display(ch, run.shocked[:, ci, pi, pj]), du,
                             f"{sidebar.CHANNEL_NAMES[ch]} at {plabel} ({lats[pi]:.1f}°N, {lons[pj]:.1f}°E)",
                             start=start, truth=tser, selected=k if k >= 1 else None)
    st.plotly_chart(fig, key="maps_point", config=NO_BAR, theme=None)
    st.caption(HOW_TO_READ)
