"""Sidebar controls: window, display options and the shock painter. Returns a state.Controls each rerun.

No server calls here (only dashboard/client.py talks to the server). Pure helpers
(build_custom_shock, custom_label, preset_shock, snap_day) are importable for tests.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import streamlit as st

import client
import plots
from common import BOXES, CH, CHANNELS, LATS, LONS, UNITS, STATE_SHAPE, box_mask, make_shock
from state import Controls
from windows import get_window, valid_last_days

DEFAULT_DAY = "2019-07-15"
PRESETS = ("none", "warm_ocean", "monsoon_stall", "custom")
PRESET_TEXT = {
    "none": "No shock (baseline only)",
    "warm_ocean": "Warm ocean · Bay of Bengal +3 K, +15% moisture",
    "monsoon_stall": "Monsoon stall · Arabian Sea winds -40%",
    "custom": "Custom box",
}
CUSTOM_DEFAULTS = {"t2m": 3.0, "tp": 10.0, "u10": -3.0, "v10": -3.0, "msl": -300.0, "tcwv": 5.0}
CHANNEL_NAMES = {"t2m": "2 m temperature", "tp": "Precipitation", "u10": "10 m wind (east-west)",
                 "v10": "10 m wind (north-south)", "msl": "Sea-level pressure", "tcwv": "Column water vapour"}

_CSS = """
<style>
html, body, [class*="css"] { font-size: 17px; }
section[data-testid="stSidebar"] label p { font-size: 15px; }
div[data-testid="stMetricValue"] { font-size: 1.6rem; }
button[kind="primary"] { font-size: 1.25rem; font-weight: 700; padding: 0.6rem 0; }
.block-container { padding-top: 2.2rem; padding-left: 1.2rem; padding-right: 1.2rem; max-width: 100%; }
</style>
"""


def channel_label(ch: str) -> str:
    return f"{ch} · {CHANNEL_NAMES[ch]} ({UNITS[ch]})"


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


def render_sidebar(arc) -> Controls:
    st.markdown(_CSS, unsafe_allow_html=True)
    sb = st.sidebar
    sb.header("Controls")
    server_url = sb.text_input("Twin server URL", value=client.DEFAULT_URL, key="server_url")

    valid = valid_last_days(arc)
    vmin, vmax = valid.min().astype(dt.date), valid.max().astype(dt.date)
    default = snap_day(valid, DEFAULT_DAY)
    picked = sb.date_input("Last input day (3-day window ends here)", value=dt.date.fromisoformat(default),
                           min_value=vmin, max_value=vmax, key="last_day")
    last_day = snap_day(valid, picked)
    if last_day != str(picked):
        sb.caption(f"{picked} is not in the archive; using {last_day}.")

    channel = sb.selectbox("Display channel", CHANNELS, index=CHANNELS.index("tp"), format_func=channel_label,
                           key="channel")
    day = sb.slider("Lead day", 1, 10, 1, key="lead_day")
    show_wind = sb.toggle("Show wind arrows", value=True, key="show_wind")

    sb.subheader("Shock painter")
    preset = sb.radio("Shock", PRESETS, format_func=lambda p: PRESET_TEXT[p], key="shock_preset")
    win = get_window(arc, last_day)
    last_state = win["init_states"][-1]
    lats, lons = np.asarray(arc.lats), np.asarray(arc.lons)

    perturbation, label, box = None, "none", None
    if preset in ("warm_ocean", "monsoon_stall"):
        perturbation, label, box = preset_shock(preset, last_state), preset, BOXES[preset]
    elif preset == "custom":
        lat_range = sb.slider("Latitude box °N", float(LATS.min()), float(LATS.max()), (8.5, 19.5), step=1.0,
                              key="custom_lat")
        lon_range = sb.slider("Longitude box °E", float(LONS.min()), float(LONS.max()), (82.5, 93.5), step=1.0,
                              key="custom_lon")
        cch = sb.selectbox("Shock channel", CHANNELS, format_func=channel_label, key="custom_channel")
        mode = sb.radio("Magnitude as", ("absolute", "percent"), horizontal=True, key="custom_mode",
                        format_func=lambda m: f"Δ {UNITS[cch]}" if m == "absolute" else "% of current value")
        dflt = CUSTOM_DEFAULTS[cch] if mode == "absolute" else 20.0
        mag = sb.number_input(f"Magnitude ({UNITS[cch]})" if mode == "absolute" else "Magnitude (%)",
                              value=float(dflt), step=float(max(abs(dflt) / 10.0, 0.5)),
                              key=f"custom_mag_{mode}_{cch}")
        perturbation = build_custom_shock(last_state, lat_range, lon_range, cch, mag, mode)
        label = custom_label(lat_range, lon_range, cch, mag, mode)
        box = (lat_range[0], lat_range[1], lon_range[0], lon_range[1])
        if not box_mask(*box).any():
            sb.warning("The box contains no grid cells; widen it.")
            perturbation = None
        else:
            c = CH[cch]
            fig = plots.diff_map(plots.to_display(cch, perturbation[c], is_diff=True), lats, lons,
                                 "Shock preview (Δ)", plots.display_units(cch), box=box, height=230)
            fig.update_layout(margin=dict(l=4, r=4, t=30, b=4), font=dict(size=11))
            fig.update_xaxes(title_text=None)
            fig.update_yaxes(title_text=None)
            sb.plotly_chart(fig, key="shock_preview", config={"displayModeBar": False})

    if box is not None and preset != "custom":
        sb.caption(f"Box: lat {box[0]:g}–{box[1]:g}°N, lon {box[2]:g}–{box[3]:g}°E")

    run_clicked = sb.button("▶ Run the twin", type="primary", width="stretch", key="run_button")
    return Controls(server_url=server_url, last_day=last_day, channel=channel, day=int(day), shock_label=label,
                    perturbation=perturbation, run_clicked=bool(run_clicked), show_wind=bool(show_wind))
