"""Shared plotly map helpers for the dashboard. No streamlit imports, no server calls.

Data stays in physical units everywhere; conversion happens only for display, and the colourbar
label says which unit is shown (t2m in degC = K - 273.15, msl in hPa = Pa / 100).

Stable public API (other tabs import these; keep signatures backwards compatible):

    CHANNEL_CMAPS: dict[channel -> plotly colourscale name]
    display_units(channel) -> str
        Unit string shown on colourbars (e.g. "°C (from K)").
    to_display(channel, arr, is_diff=False) -> np.ndarray
        Physical -> display units. is_diff=True applies only the scale (a K difference equals a °C difference).
    box_from_perturbation(pert, lats, lons) -> (lat_min, lat_max, lon_min, lon_max) | None
        Bounding box (cell centres) of the non-zero cells of a [6,32,36] (or [32,36]) perturbation.
    field_map(field2d, lats, lons, title, units, cmap="Viridis", zmid=None, zmin=None, zmax=None,
              box=None, wind=None, height=360, wind_stride=3) -> go.Figure
        Heatmap on lon (x) / lat (y, ascending so north is up). field2d is [nlat, nlon] already in
        the units you want to show. box = (lat_min, lat_max, lon_min, lon_max) on cell centres, drawn as an
        outline around those cells. wind = (u2d, v2d) in m/s, drawn as subsampled arrows.
    diff_map(diff2d, lats, lons, title, units, vmax=None, box=None, wind=None, height=360) -> go.Figure
        Diverging blue-white-red map centred at 0 with symmetric range [-vmax, vmax]
        (vmax defaults to max |diff|).
    diff_animation(diffs, lats, lons, title, units, box=None, height=420, day_labels=None) -> go.Figure
        diffs [ndays, nlat, nlon]; one frame per day with Play/Pause buttons and a day slider; one
        symmetric colour range across all days so growth and spread are visible.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go

CHANNEL_CMAPS = {
    "t2m": "RdYlBu_r",
    "tp": "Blues",
    "u10": "RdBu_r",
    "v10": "RdBu_r",
    "msl": "Viridis",
    "tcwv": "YlGnBu",
}
DIVERGING = "RdBu_r"

_DISPLAY = {  # channel -> (scale, offset, display unit)
    "t2m": (1.0, -273.15, "°C (from K)"),
    "tp": (1.0, 0.0, "mm/day"),
    "u10": (1.0, 0.0, "m/s"),
    "v10": (1.0, 0.0, "m/s"),
    "msl": (0.01, 0.0, "hPa (from Pa)"),
    "tcwv": (1.0, 0.0, "kg/m²"),
}

FONT = dict(size=15)


def display_units(channel: str) -> str:
    return _DISPLAY.get(channel, (1.0, 0.0, ""))[2]


def to_display(channel: str, arr, is_diff: bool = False) -> np.ndarray:
    scale, offset, _ = _DISPLAY.get(channel, (1.0, 0.0, ""))
    a = np.asarray(arr, dtype=np.float64) * scale
    return a if is_diff else a + offset


def box_from_perturbation(pert, lats, lons) -> Optional[tuple]:
    if pert is None:
        return None
    p = np.asarray(pert)
    nz = np.abs(p) > 0
    if nz.ndim == 3:
        nz = nz.any(axis=0)
    if not nz.any():
        return None
    li = np.where(nz.any(axis=1))[0]
    lj = np.where(nz.any(axis=0))[0]
    lats, lons = np.asarray(lats), np.asarray(lons)
    return (float(lats[li.min()]), float(lats[li.max()]), float(lons[lj.min()]), float(lons[lj.max()]))


def _half_step(v) -> float:
    v = np.asarray(v, dtype=float)
    return float(abs(v[1] - v[0]) / 2.0) if v.size > 1 else 0.5


def _layout(fig: go.Figure, title: str, lats, lons, height: int) -> go.Figure:
    hy, hx = _half_step(lats), _half_step(lons)
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=17)),
        height=height, margin=dict(l=48, r=8, t=44, b=40), font=FONT,
        plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
    )
    fig.update_xaxes(title_text="Longitude °E", range=[float(np.min(lons)) - hx, float(np.max(lons)) + hx],
                     showgrid=False, zeroline=False, constrain="domain")
    fig.update_yaxes(title_text="Latitude °N", range=[float(np.min(lats)) - hy, float(np.max(lats)) + hy],
                     showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1, constrain="domain")
    return fig


def _box_shape(box, lats, lons) -> dict:
    lat_min, lat_max, lon_min, lon_max = box
    hy, hx = _half_step(lats), _half_step(lons)
    return dict(type="rect", xref="x", yref="y", x0=lon_min - hx, x1=lon_max + hx, y0=lat_min - hy, y1=lat_max + hy,
                line=dict(color="black", width=2.5, dash="dash"), fillcolor="rgba(0,0,0,0)")


def _wind_trace(u2d, v2d, lats, lons, stride: int = 3) -> Optional[go.Scatter]:
    """Arrows as one line trace (shaft + two head strokes per arrow, separated by None)."""
    u = np.asarray(u2d, dtype=float)[::stride, ::stride]
    v = np.asarray(v2d, dtype=float)[::stride, ::stride]
    la = np.asarray(lats, dtype=float)[::stride]
    lo = np.asarray(lons, dtype=float)[::stride]
    speed = np.hypot(u, v)
    smax = float(np.nanmax(speed)) if np.isfinite(speed).any() else 0.0
    if smax <= 0:
        return None
    k = 0.9 * stride / smax  # longest arrow ~0.9 of the arrow spacing (degrees)
    xs, ys = [], []
    for i, y0 in enumerate(la):
        for j, x0 in enumerate(lo):
            dx, dy = u[i, j] * k, v[i, j] * k
            if not (np.isfinite(dx) and np.isfinite(dy)) or np.hypot(dx, dy) < 0.08:
                continue
            xa, ya = x0 - dx / 2, y0 - dy / 2
            xb, yb = x0 + dx / 2, y0 + dy / 2
            xs += [xa, xb, None]
            ys += [ya, yb, None]
            ang = np.arctan2(dy, dx)
            hl = 0.35 * np.hypot(dx, dy)
            for s in (+1, -1):
                a = ang + np.pi - s * 0.45
                xs += [xb, xb + hl * np.cos(a), None]
                ys += [yb, yb + hl * np.sin(a), None]
    return go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(20,20,20,0.85)", width=1.6),
                      hoverinfo="skip", name="wind")


def field_map(field2d, lats, lons, title: str, units: str, cmap: str = "Viridis", zmid=None, zmin=None, zmax=None,
              box=None, wind=None, height: int = 360, wind_stride: int = 3) -> go.Figure:
    z = np.asarray(field2d, dtype=float)
    hm = go.Heatmap(z=z, x=np.asarray(lons, float), y=np.asarray(lats, float), colorscale=cmap,
                    zmin=zmin, zmax=zmax, zmid=zmid,
                    colorbar=dict(title=dict(text=units, side="right"), thickness=14, len=0.9),
                    hovertemplate="lat %{y:.1f}°N, lon %{x:.1f}°E<br>%{z:.2f} " + units + "<extra></extra>")
    fig = go.Figure(hm)
    if wind is not None:
        tr = _wind_trace(wind[0], wind[1], lats, lons, stride=wind_stride)
        if tr is not None:
            fig.add_trace(tr)
    if box is not None:
        fig.add_shape(_box_shape(box, lats, lons))
    return _layout(fig, title, lats, lons, height)


def symmetric_limit(*arrays, floor: float = 1e-9) -> float:
    m = 0.0
    for a in arrays:
        a = np.asarray(a, dtype=float)
        if a.size and np.isfinite(a).any():
            m = max(m, float(np.nanmax(np.abs(a))))
    return max(m, floor)


def diff_map(diff2d, lats, lons, title: str, units: str, vmax=None, box=None, wind=None, height: int = 360) -> go.Figure:
    v = symmetric_limit(diff2d) if vmax is None else max(float(vmax), 1e-9)
    return field_map(diff2d, lats, lons, title, units, cmap=DIVERGING, zmid=0.0, zmin=-v, zmax=v,
                     box=box, wind=wind, height=height)


def diff_animation(diffs, lats, lons, title: str, units: str, box=None, height: int = 420,
                   day_labels: Optional[Sequence[str]] = None) -> go.Figure:
    d = np.asarray(diffs, dtype=float)
    n = d.shape[0]
    labels = list(day_labels) if day_labels is not None else [f"Day {i + 1}" for i in range(n)]
    v = symmetric_limit(d)
    fig = diff_map(d[0], lats, lons, f"{title} · {labels[0]}", units, vmax=v, box=box, height=height)
    fig.frames = [go.Frame(data=[go.Heatmap(z=d[i])], name=labels[i], traces=[0],
                           layout=dict(title=dict(text=f"{title} · {labels[i]}")))
                  for i in range(n)]
    anim = dict(frame=dict(duration=700, redraw=True), transition=dict(duration=0), fromcurrent=True, mode="immediate")
    fig.update_layout(
        margin=dict(l=48, r=8, t=44, b=90),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.14, xanchor="left", yanchor="top",
                          pad=dict(t=4, r=8), showactive=False, buttons=[
                              dict(label="▶ Play", method="animate", args=[None, anim]),
                              dict(label="❚❚ Pause", method="animate",
                                   args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, x=0.22, len=0.78, y=-0.12, yanchor="top", currentvalue=dict(visible=False),
                      pad=dict(t=4), steps=[dict(label=lab, method="animate",
                                                 args=[[lab], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                                            for lab in labels])],
    )
    return fig
