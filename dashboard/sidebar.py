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
PRESETS = ("none", "warm_ocean", "monsoon_stall", "custom")
PRESET_TEXT = {
    "none": "No shock (baseline only)",
    "warm_ocean": "Warm ocean: Bay of Bengal +3 K, +15% moisture",
    "monsoon_stall": "Monsoon stall: Arabian Sea winds −40%",
    "custom": "Custom box",
}
# Display channel picked automatically when a preset shock is chosen (the user can change it afterwards).
SHOCK_CHANNEL = {"warm_ocean": "t2m", "monsoon_stall": "u10"}
CUSTOM_DEFAULTS = {"t2m": 3.0, "tp": 10.0, "u10": -3.0, "v10": -3.0, "msl": -300.0, "tcwv": 5.0}
CHANNEL_NAMES = {"t2m": "2 m temperature", "tp": "Precipitation", "u10": "10 m wind (east-west)",
                 "v10": "10 m wind (north-south)", "msl": "Sea-level pressure", "tcwv": "Column water vapour"}
DEFAULT_CHANNEL = "tp"


def channel_label(ch: str) -> str:
    return f"{CHANNEL_NAMES[ch]} ({ch}, {UNITS[ch]})"


def default_server_url() -> str:
    return os.environ.get(SERVER_ENV, "").strip() or client.DEFAULT_URL


def status_pill(health, error: str | None = None) -> str:
    """Small HTML status line: green dot 'Server ready, running <model>' or red dot 'Server down'."""
    if health is not None:
        name = html.escape(str(health.get("model_name", "?")))
        backend = html.escape(str(health.get("backend", "?")))
        return (f'<span class="mm-pill mm-ok" title="backend: {backend}"><span class="mm-dot"></span>'
                f'Server ready, running {name}</span>')
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


def display_state() -> tuple[str, int, bool]:
    """Current display channel, lead day (1..10) and wind toggle, as last set on the Maps tab."""
    ss = st.session_state
    ss.setdefault("channel", DEFAULT_CHANNEL)
    ch = ss.get("channel") if ss.get("channel") in CHANNELS else DEFAULT_CHANNEL
    try:
        day = int(ss.get("lead_day") or 1)
    except (TypeError, ValueError):
        day = 1
    return ch, min(max(day, 1), 10), bool(ss.get("show_wind", True))


def render_sidebar(arc, status_slot=None, display=None) -> Controls:
    """status_slot: an st.empty() already in the sidebar (app.py fills it). `display` is accepted for
    backwards compatibility and ignored: the display widgets are drawn by the Maps tab."""
    sb = st.sidebar
    painter = sb.container()
    run_box = sb.container()
    preview = sb.container()   # custom-box preview sits under Run, so Run stays above the fold
    window = sb.container()
    settings = sb.expander("Settings", expanded=False)

    server_url = settings.text_input("Twin server URL", value=default_server_url(), key="server_url")

    valid = valid_last_days(arc)
    vmin, vmax = valid.min().astype(dt.date), valid.max().astype(dt.date)
    default = snap_day(valid, DEFAULT_DAY)
    picked = window.date_input("Window: last input day", value=dt.date.fromisoformat(default),
                               min_value=vmin, max_value=vmax, key="last_day",
                               help="The twin starts from the 3 real days ending here and runs 10 days ahead.")
    last_day = snap_day(valid, picked)
    if last_day != str(picked):
        window.caption(f"{picked} is not in the archive, so the window ends on {last_day} instead.")

    preset = painter.radio("Shock painter", PRESETS, format_func=lambda p: PRESET_TEXT[p], key="shock_preset",
                           on_change=_on_preset_change)
    win = get_window(arc, last_day)
    last_state = win["init_states"][-1]
    lats, lons = np.asarray(arc.lats), np.asarray(arc.lons)

    perturbation, label, box = None, "none", None
    if preset in ("warm_ocean", "monsoon_stall"):
        perturbation, label, box = preset_shock(preset, last_state), preset, BOXES[preset]
    elif preset == "custom":
        lat_range = painter.slider("Latitude box °N", float(LATS.min()), float(LATS.max()), (8.5, 19.5), step=1.0,
                                   key="custom_lat")
        lon_range = painter.slider("Longitude box °E", float(LONS.min()), float(LONS.max()), (82.5, 93.5),
                                   step=1.0, key="custom_lon")
        cch = painter.selectbox("Shock channel", CHANNELS, format_func=channel_label, key="custom_channel",
                                on_change=_on_custom_channel_change)
        mc1, mc2 = painter.columns([1, 1.15], vertical_alignment="bottom")
        mode = mc1.radio("Magnitude as", ("absolute", "percent"), key="custom_mode",
                         format_func=lambda m: f"Δ {UNITS[cch]}" if m == "absolute" else "% of now")
        dflt = CUSTOM_DEFAULTS[cch] if mode == "absolute" else 20.0
        mag = mc2.number_input(f"Size ({UNITS[cch]})" if mode == "absolute" else "Size (%)",
                               value=float(dflt), step=float(max(abs(dflt) / 10.0, 0.5)),
                               key=f"custom_mag_{mode}_{cch}")
        perturbation = build_custom_shock(last_state, lat_range, lon_range, cch, mag, mode)
        label = custom_label(lat_range, lon_range, cch, mag, mode)
        box = (lat_range[0], lat_range[1], lon_range[0], lon_range[1])
        if not box_mask(*box).any():
            painter.warning("The box holds no grid cells. Widen it with the two sliders above.")
            perturbation = None
        else:
            c = CH[cch]
            fig = plots.diff_map(plots.to_display(cch, perturbation[c], is_diff=True), lats, lons,
                                 "Shock preview (Δ)", plots.display_units(cch), box=box, height=270,
                                 coast=arc.static[0], colorbar="bottom")
            preview.plotly_chart(fig, key="shock_preview", config={"displayModeBar": False}, theme=None)

    if box is not None and preset != "custom":
        painter.caption(f"Box: lat {box[0]:g}–{box[1]:g}°N, lon {box[2]:g}–{box[3]:g}°E")

    run_clicked = run_box.button("Run the twin", type="primary", width="stretch", key="run_button")

    channel, day, show_wind = display_state()
    return Controls(server_url=server_url, last_day=last_day, channel=channel, day=int(day), shock_label=label,
                    perturbation=perturbation, run_clicked=bool(run_clicked), show_wind=bool(show_wind))
