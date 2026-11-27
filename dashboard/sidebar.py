"""Sidebar controls: shock painter + Run, window, settings. Display options live on the Maps tab.

Returns a state.Controls each rerun. No server calls here (only dashboard/client.py talks to the server).
Pure helpers (build_custom_shock, custom_label, preset_shock, snap_day, status_pill, default_server_url)
are importable for tests.

Layout (top to bottom): status line (filled by app.py into `status_slot`), shock painter, Run button,
window date, collapsed Settings (server URL). The display channel, lead day and wind toggle are widgets on the
Maps tab (keys channel, lead_day, show_wind, drawn by tab_maps.render_display_controls); this module only reads
their values from session state, so the Controls handed to every tab are known before any tab draws.
"""
from __future__ import annotations

import datetime as dt
import html
import os

import numpy as np
import streamlit as st

import client
import plots
from common import BOXES, CH, CHANNELS, LATS, LONS, UNITS, STATE_SHAPE, box_mask, make_shock
from state import Controls
from windows import get_window, valid_last_days

DEFAULT_DAY = "2019-07-15"
SERVER_ENV = "DASHBOARD_SERVER_URL"
PRESETS = ("none", "warm_ocean", "monsoon_stall", "heatwave", "westerly_burst", "bay_low", "custom")
PRESET_TEXT = {
    "none": "No shock (baseline only)",
    "warm_ocean": "Warm ocean: Bay of Bengal +3 K, +15% moisture",
    "monsoon_stall": "Monsoon stall: Arabian Sea winds −40%",
    "heatwave": "Heatwave: north India +4 K",
    "westerly_burst": "Westerly burst: Arabian Sea wind +5 m/s",
    "bay_low": "Bay of Bengal low: pressure −1500 Pa",
    "custom": "Custom box (or click a map)",
}
# Demo shocks the model lane measured on the final model (docs/HANDOFF_VISHWAK.md, "Live-shock behaviour").
# Single-channel boxes on cell centres; built with build_custom_shock like a custom box.
DEMO_SHOCKS = {
    "heatwave": dict(channel="t2m", mode="absolute", mag=4.0, box=(24.5, 30.5, 73.5, 81.5)),
    "westerly_burst": dict(channel="u10", mode="absolute", mag=5.0, box=(10.5, 18.5, 65.5, 73.5)),
    "bay_low": dict(channel="msl", mode="absolute", mag=-1500.0, box=(15.5, 20.5, 85.5, 90.5)),
}
# What is painted, in plain words. Deliberately no claims about what the model will do: that is measured live on
# the Maps tab from the fetched rollouts (tab_maps.fade_summary / describe_spread), so it stays true when the
# weights change.
PRESET_EXPECT = {
    "warm_ocean": "One of the two shocks the judges score.",
    "monsoon_stall": "One of the two shocks the judges score.",
    "heatwave": "A hot spell over north India.",
    "westerly_burst": "Stronger west-to-east winds over the Arabian Sea.",
    "bay_low": "A low-pressure system; the map switches to rain, where its effect shows.",
}
DEFAULT_PRESET = "heatwave"     # first load: the clearest "starts local, then spreads" demo (model lane, 20/20 windows)
# Display channel picked automatically when a preset shock is chosen (the user can change it afterwards).
SHOCK_CHANNEL = {"warm_ocean": "t2m", "monsoon_stall": "u10", "heatwave": "t2m", "westerly_burst": "u10",
                 "bay_low": "tp"}
# A click on a map turns the current shock into a custom box of the same kind, centred on the click.
CLICK_AS_CUSTOM = {"none": ("t2m", "absolute", 3.0, 11), "warm_ocean": ("t2m", "absolute", 3.0, 11),
                   "monsoon_stall": ("u10", "percent", -40.0, 11),
                   **{k: (v["channel"], v["mode"], v["mag"], 7) for k, v in DEMO_SHOCKS.items()}}
BOX_SIZES = (3, 5, 7, 9, 11, 13, 15)     # cells across (1 cell = 1° ~ 110 km)
HORIZONS = (10, 15, 20)                  # forecast days; the judges score 10, beyond that is experimental
CUSTOM_DEFAULTS = {"t2m": 3.0, "tp": 10.0, "u10": -3.0, "v10": -3.0, "msl": -300.0, "tcwv": 5.0}
CHANNEL_NAMES = {"t2m": "2 m temperature", "tp": "Precipitation", "u10": "10 m wind (east-west)",
                 "v10": "10 m wind (north-south)", "msl": "Sea-level pressure", "tcwv": "Column water vapour"}
DEFAULT_CHANNEL = SHOCK_CHANNEL[DEFAULT_PRESET]


def channel_label(ch: str) -> str:
    return f"{CHANNEL_NAMES[ch]} ({ch}, {UNITS[ch]})"


def default_server_url() -> str:
    return os.environ.get(SERVER_ENV, "").strip() or client.DEFAULT_URL


def status_pill(health, error: str | None = None, rt_ms: float | None = None) -> str:
    """Small HTML status line from /health: green dot 'Server ready, running <model>' with backend, whether
    post-processing is loaded and the last round trip; or red dot 'Server down'."""
    if health is not None:
        name = html.escape(str(health.get("model_name", "?")))
        backend = html.escape(str(health.get("backend", "?")))
        post = str(health.get("post", ""))
        bits = [backend] + (["post-processing on"] if post == "loaded" else [])
        if rt_ms is not None and np.isfinite(rt_ms) and rt_ms > 0:
            bits.append(f"{rt_ms:.0f} ms")
        detail = html.escape(", ".join(bits))
        return (f'<span class="mm-pill mm-ok" title="backend: {backend}, post-processing: {html.escape(post)}">'
                f'<span class="mm-dot"></span>Server ready, running {name}'
                f'<span class="mm-pill-detail">({detail})</span></span>')
    tip = html.escape(error or "not reachable")
    return f'<span class="mm-pill mm-down" title="{tip}"><span class="mm-dot"></span>Server down</span>'


def snap_day(valid_days: np.ndarray, wanted) -> str:
    """Nearest valid last-input day to `wanted` (YYYY-MM-DD string or date)."""
    v = np.asarray(valid_days).astype("datetime64[D]")
    d = np.datetime64(str(wanted), "D")
    i = int(np.argmin(np.abs((v - d).astype(np.int64))))
    return str(v[i])


def build_custom_shock(last_state: np.ndarray, lat_range, lon_range, channel: str, magnitude: float,
                       mode: str = "absolute") -> np.ndarray:
    """[6,32,36] physical delta: `magnitude` on `channel` inside the lat/lon box (cell centres, inclusive).

    mode "absolute": magnitude is a physical delta in UNITS[channel].
    mode "percent":  magnitude is % of the current value (last_state = init_states[-1]).
    """
    last_state = np.asarray(last_state, dtype=np.float64)
    if last_state.shape != STATE_SHAPE:
        raise ValueError(f"last_state must be {STATE_SHAPE}, got {last_state.shape}")
    if channel not in CH:
        raise ValueError(f"unknown channel {channel!r}")
    delta = np.zeros(STATE_SHAPE, dtype=np.float64)
    m = box_mask(float(lat_range[0]), float(lat_range[1]), float(lon_range[0]), float(lon_range[1]))
    c = CH[channel]
    if mode == "absolute":
        delta[c][m] = float(magnitude)
    elif mode == "percent":
        delta[c][m] = float(magnitude) / 100.0 * last_state[c][m]
    else:
        raise ValueError(f"mode must be 'absolute' or 'percent', got {mode!r}")
    return delta


def custom_label(lat_range, lon_range, channel: str, magnitude: float, mode: str) -> str:
    amt = f"{magnitude:+g}%" if mode == "percent" else f"{magnitude:+g} {UNITS[channel]}"
    return (f"custom: {channel} {amt} in lat {lat_range[0]:g}–{lat_range[1]:g}°N, "
            f"lon {lon_range[0]:g}–{lon_range[1]:g}°E")


def preset_shock(name: str, last_state: np.ndarray):
    """Perturbation for a preset name ('none' -> None)."""
    return None if name == "none" else make_shock(name, last_state)


# ---------------------------------------------------------------- callbacks (run before the script reruns)
def _on_preset_change() -> None:
    p = st.session_state.get("shock_preset")
    if p in SHOCK_CHANNEL:
        st.session_state["channel"] = SHOCK_CHANNEL[p]
    elif p == "custom":
        st.session_state["channel"] = st.session_state.get("custom_channel", CHANNELS[0])


def _on_custom_channel_change() -> None:
    st.session_state["channel"] = st.session_state.get("custom_channel", CHANNELS[0])


def snap(v: float, grid) -> float:
    """Nearest grid value (cell centre) to v."""
    g = np.asarray(grid, dtype=float)
    return float(g[int(np.argmin(np.abs(g - float(v))))])


def box_from_center(lat: float, lon: float, size: int) -> tuple[float, float, float, float]:
    """(lat_min, lat_max, lon_min, lon_max) on cell centres: `size` cells across, centred on the nearest cell,
    clipped to the grid."""
    half = max(int(size), 1) // 2
    la, lo = snap(lat, LATS), snap(lon, LONS)
    return (max(la - half, float(LATS.min())), min(la + half, float(LATS.max())),
            max(lo - half, float(LONS.min())), min(lo + half, float(LONS.max())))


def shock_state_for_click(preset: str, lat: float, lon: float, ss=None) -> dict:
    """Session-state updates that move the shock to a clicked map point.

    A custom box keeps its channel, magnitude and size and only moves. Any other preset becomes a custom box of
    the same kind (channel, magnitude, size) centred on the click, so "paint it over there" is one click."""
    ss = ss if ss is not None else {}
    upd = {"shock_lat": snap(lat, LATS), "shock_lon": snap(lon, LONS), "shock_preset": "custom"}
    if preset != "custom":
        ch, mode, mag, size = CLICK_AS_CUSTOM.get(preset, CLICK_AS_CUSTOM["none"])
        upd.update({"custom_channel": ch, "custom_mode": mode, f"custom_mag_{mode}_{ch}": float(mag),
                    "shock_size": int(size)})          # the displayed layer stays what the user is looking at
    return upd


def display_state() -> tuple[str, int, bool]:
    """Current display channel, lead day (1..10) and wind toggle, as last set on the Maps tab."""
    ss = st.session_state
    ss.setdefault("channel", DEFAULT_CHANNEL)
    ch = ss.get("channel") if ss.get("channel") in CHANNELS else DEFAULT_CHANNEL
    try:
        day = int(ss.get("lead_day") or 1)
    except (TypeError, ValueError):
        day = 1
    return ch, min(max(day, 1), 10), bool(ss.get("show_wind", True))   # Maps timeline also has days -2..0, 11..20


def render_sidebar(arc, status_slot=None, display=None) -> Controls:
    """status_slot: an st.empty() already in the sidebar (app.py fills it). `display` is accepted for
    backwards compatibility and ignored: the display widgets are drawn by the Maps tab."""
    sb = st.sidebar
    painter = sb.container()
    run_box = sb.container()
    preview = sb.container()   # custom-box controls + preview sit under Run, so Run stays above the fold
    window = sb.container()
    settings = sb.expander("Settings", expanded=False)

    server_url = settings.text_input("Twin server URL", value=default_server_url(), key="server_url")
    settings.toggle("Reduce motion", key="reduce_motion",
                    help="No easing between days when the map plays (frames simply switch). For viewers who prefer "
                         "less motion.")

    valid = valid_last_days(arc)
    vmin, vmax = valid.min().astype(dt.date), valid.max().astype(dt.date)
    default = snap_day(valid, DEFAULT_DAY)
    st.session_state.setdefault("horizon", 10)
    steps = int(window.segmented_control("Forecast length (days)", HORIZONS, key="horizon", required=True,
                                         help="The twin runs this many days ahead in one live call. The judges "
                                              "score 10 days; beyond about a week the forecast leans on climatology."))
    picked = window.date_input("Window: last input day", value=dt.date.fromisoformat(default),
                               min_value=vmin, max_value=vmax, key="last_day",
                               help="The twin starts from the 3 real days ending here and runs 10 days ahead.")
    last_day = snap_day(valid, picked)
    if last_day != str(picked):
        window.caption(f"{picked} is not in the archive, so the window ends on {last_day} instead.")

    st.session_state.setdefault("shock_preset", DEFAULT_PRESET)
    preset = painter.radio("Shock painter", PRESETS, format_func=lambda p: PRESET_TEXT[p], key="shock_preset",
                           on_change=_on_preset_change)
    win = get_window(arc, last_day)
    last_state = win["init_states"][-1]
    lats, lons = np.asarray(arc.lats), np.asarray(arc.lons)

    perturbation, label, box = None, "none", None
    if preset in ("warm_ocean", "monsoon_stall"):
        perturbation, label, box = preset_shock(preset, last_state), preset, BOXES[preset]
    elif preset in DEMO_SHOCKS:
        p = DEMO_SHOCKS[preset]
        box = p["box"]
        perturbation = build_custom_shock(last_state, box[:2], box[2:], p["channel"], p["mag"], p["mode"])
        label = preset
    elif preset == "custom":
        ss = st.session_state
        ss.setdefault("shock_lat", 14.5)
        ss.setdefault("shock_lon", 88.5)
        ss.setdefault("shock_size", 11)
        preview.caption("Click a map on the Maps tab to move the box there, or type the centre.")
        pc1, pc2 = preview.columns(2)
        clat = pc1.number_input("Centre °N", float(LATS.min()), float(LATS.max()), step=1.0, format="%.1f",
                                key="shock_lat")
        clon = pc2.number_input("Centre °E", float(LONS.min()), float(LONS.max()), step=1.0, format="%.1f",
                                key="shock_lon")
        size = preview.select_slider("Box size (cells across, 1 cell ≈ 110 km)", BOX_SIZES, key="shock_size")
        cch = preview.selectbox("Shock channel", CHANNELS, format_func=channel_label, key="custom_channel",
                                on_change=_on_custom_channel_change)
        mc1, mc2 = preview.columns([1, 1.15], vertical_alignment="bottom")
        mode = mc1.radio("Magnitude as", ("absolute", "percent"), key="custom_mode",
                         format_func=lambda m: f"Δ {UNITS[cch]}" if m == "absolute" else "% of now")
        dflt = CUSTOM_DEFAULTS[cch] if mode == "absolute" else 20.0
        mkey = f"custom_mag_{mode}_{cch}"
        mkw = {} if mkey in ss else {"value": float(dflt)}   # no default when a map click already set it
        mag = mc2.number_input(f"Strength ({UNITS[cch]})" if mode == "absolute" else "Strength (%)",
                               step=float(max(abs(dflt) / 10.0, 0.5)), key=mkey, **mkw)
        box = box_from_center(clat, clon, size)
        perturbation = build_custom_shock(last_state, box[:2], box[2:], cch, mag, mode)
        label = custom_label(box[:2], box[2:], cch, mag, mode)
        if not box_mask(*box).any():
            preview.warning("The box holds no grid cells. Move it onto the map or make it bigger.")
            perturbation = None
        else:
            c = CH[cch]
            fig = plots.diff_map(plots.to_display(cch, perturbation[c], is_diff=True), lats, lons,
                                 "Shock preview (Δ)", plots.display_units(cch), box=box, height=240,
                                 coast=arc.static[0], colorbar="bottom")
            preview.plotly_chart(fig, key="shock_preview", config={"displayModeBar": False}, theme=None)

    if box is not None and preset != "custom":
        painter.caption(f"Box: lat {box[0]:g}–{box[1]:g}°N, lon {box[2]:g}–{box[3]:g}°E. "
                        f"{PRESET_EXPECT.get(preset, '')}")

    run_clicked = run_box.button("Run the twin", type="primary", width="stretch", key="run_button")
    st.session_state.setdefault("live_mode", True)
    run_box.toggle("Live: update on every change", key="live_mode",
                   help="Re-runs the twin (two live calls, about a quarter of a second) whenever the shock, "
                        "a map click or the date changes. Turn it off to run only when you press the button.")


    channel, day, show_wind = display_state()
    return Controls(server_url=server_url, last_day=last_day, channel=channel, day=int(day), shock_label=label,
                    perturbation=perturbation, run_clicked=bool(run_clicked), show_wind=bool(show_wind),
                    steps=steps)
