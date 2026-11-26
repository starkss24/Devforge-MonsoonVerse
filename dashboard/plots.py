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
              box=None, wind=None, height=360, wind_stride=3, *, coast=None, places=False,
              colorbar="right", wind_ref=None, note=None) -> go.Figure
        Heatmap on lon (x) / lat (y, ascending so north is up). field2d is [nlat, nlon] already in
        the units you want to show. box = (lat_min, lat_max, lon_min, lon_max) on cell centres, drawn as an
        outline around those cells. wind = (u2d, v2d) in m/s, drawn as subsampled arrows.
        Keyword-only extras: coast = land-sea mask [nlat, nlon] (1 = land) drawn as a coastline (0.5 contour);
        places = True writes "Arabian Sea" / "Bay of Bengal" in small grey italics; colorbar = "right" |
        "bottom" ("bottom" = thin horizontal bar under the map plus compact margins, so the map gets the full
        column width); wind_ref = m/s of a fixed arrow scale shared by every map (weaker wind -> shorter
        arrows), with a reference arrow in the top-left corner, instead of auto-scaling to the field max;
        note = text box in the middle of the map.
    diff_map(diff2d, lats, lons, title, units, vmax=None, box=None, wind=None, height=360, *,
             empty_text=None, **field_map_kwargs) -> go.Figure
        Diverging blue-white-red map centred at 0 with symmetric range [-vmax, vmax]
        (vmax defaults to max |diff|). If max |diff| < ZERO_DIFF the range is fixed to [-1, 1] and
        `empty_text` (if given) is shown on the map. Extra kwargs go to field_map (coast, places, ...).
    diff_animation(diffs, lats, lons, title, units, box=None, height=420, day_labels=None, *, coast=None)
        diffs [ndays, nlat, nlon]; one frame per day with Play/Pause buttons and a day slider; one
        symmetric colour range across all days so growth and spread are visible.
    coastline_trace(mask, lats, lons) -> go.Contour ;  add_coast(fig, mask, lats, lons) -> fig
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
ZERO_DIFF = 1e-4          # display units; a difference map below this counts as "no change"
COAST_COLOR = "rgba(35,35,35,0.85)"
WIND_COLOR = "rgba(25,25,25,0.55)"
PLACES = (("Arabian Sea", 22.3, 65.4), ("Bay of Bengal", 16.5, 85.0))   # (text, lat, lon of the left end)
LAT_SCALE = float(1.0 / np.cos(np.deg2rad(22.0)))  # 1° latitude is this many times longer than 1° longitude


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


def _compact_layout(fig: go.Figure, title: str, lats, lons, height: int) -> go.Figure:
    """Map fills its column: no axis titles (ticks carry °E/°N), thin margins, room below for the colourbar."""
    hy, hx = _half_step(lats), _half_step(lons)
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", y=0.99, yanchor="top", font=dict(size=15)),
        height=height, margin=dict(l=40, r=6, t=26, b=74), font=dict(size=12),
        plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
    )
    fig.update_xaxes(title_text=None, range=[float(np.min(lons)) - hx, float(np.max(lons)) + hx],
                     showgrid=False, zeroline=False, constrain="domain", tickvals=[70, 80, 90], ticktext=["70°E", "80°E", "90°E"], ticks="outside", ticklen=3, tickfont=dict(size=11))
    fig.update_yaxes(title_text=None, range=[float(np.min(lats)) - hy, float(np.max(lats)) + hy],
                     showgrid=False, zeroline=False, scaleanchor="x", scaleratio=LAT_SCALE, constrain="domain",
                     tickvals=[10, 20, 30], ticktext=["10°N", "20°N", "30°N"], ticks="outside", ticklen=3, tickfont=dict(size=11))
    return fig


def _box_shape(box, lats, lons) -> dict:
    lat_min, lat_max, lon_min, lon_max = box
    hy, hx = _half_step(lats), _half_step(lons)
    return dict(type="rect", xref="x", yref="y", x0=lon_min - hx, x1=lon_max + hx, y0=lat_min - hy, y1=lat_max + hy,
                line=dict(color="black", width=2.5, dash="dash"), fillcolor="rgba(0,0,0,0)")


def coastline_trace(mask, lats, lons) -> go.Contour:
    """Coastline: the 0.5 contour of a land-sea mask [nlat, nlon] (1 = land), as a thin dark line."""
    return go.Contour(z=np.asarray(mask, dtype=float), x=np.asarray(lons, float), y=np.asarray(lats, float),
                      contours=dict(start=0.5, end=0.5, size=1.0, coloring="none"), autocontour=False,
                      line=dict(color=COAST_COLOR, width=1.5, smoothing=1.0), showscale=False,
                      hoverinfo="skip", name="coast")


def add_coast(fig: go.Figure, mask, lats, lons) -> go.Figure:
    if mask is not None:
        fig.add_trace(coastline_trace(mask, lats, lons))
    return fig


def _arrow_xy(x0, y0, dx, dy, xs, ys) -> None:
    """Append one arrow (shaft + two head strokes, centred on x0, y0) to the xs/ys line lists."""
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


def _wind_trace(u2d, v2d, lats, lons, stride: int = 3, ref: Optional[float] = None) -> Optional[go.Scatter]:
    """Arrows as one line trace. ref=None: longest arrow ~0.9 x spacing; ref=m/s: `ref` m/s = 0.6 x spacing."""
    u = np.asarray(u2d, dtype=float)[::stride, ::stride]
    v = np.asarray(v2d, dtype=float)[::stride, ::stride]
    la = np.asarray(lats, dtype=float)[::stride]
    lo = np.asarray(lons, dtype=float)[::stride]
    speed = np.hypot(u, v)
    smax = float(np.nanmax(speed)) if np.isfinite(speed).any() else 0.0
    if smax <= 0:
        return None
    k = 0.6 * stride / float(ref) if ref else 0.9 * stride / smax  # degrees per m/s
    xs, ys = [], []
    for i, y0 in enumerate(la):
        for j, x0 in enumerate(lo):
            dx, dy = u[i, j] * k, v[i, j] * k
            if not (np.isfinite(dx) and np.isfinite(dy)) or np.hypot(dx, dy) < 0.08:
                continue
            _arrow_xy(x0, y0, dx, dy, xs, ys)
    line = dict(color=WIND_COLOR, width=1.1) if ref else dict(color="rgba(20,20,20,0.85)", width=1.6)
    return go.Scatter(x=xs, y=ys, mode="lines", line=line, hoverinfo="skip", name="wind")


def _wind_key(fig: go.Figure, lats, lons, stride: int, ref: float) -> None:
    """Reference arrow + 'N m/s' label in the top-left corner, on the same scale as the map arrows."""
    L = 0.6 * stride
    hy, hx = _half_step(lats), _half_step(lons)
    left, top = float(np.min(lons)) - hx, float(np.max(lats)) + hy
    x0, y0 = left + 0.7 + L / 2, top - 1.3
    fig.add_shape(type="rect", xref="x", yref="y", x0=left, x1=x0 + L / 2 + 6.2, y0=top - 2.6, y1=top,
                  line=dict(width=0), fillcolor="rgba(255,255,255,0.8)", layer="above")
    xs, ys = [], []
    _arrow_xy(x0, y0, L, 0.0, xs, ys)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(25,25,25,0.95)", width=1.4),
                             hoverinfo="skip", name="wind_key"))
    fig.add_annotation(x=x0 + L / 2 + 0.4, y=y0, xref="x", yref="y", text=f"{ref:g} m/s", showarrow=False,
                       xanchor="left", font=dict(size=11, color="#222"))


def field_map(field2d, lats, lons, title: str, units: str, cmap: str = "Viridis", zmid=None, zmin=None, zmax=None,
              box=None, wind=None, height: int = 360, wind_stride: int = 3, *, coast=None, places: bool = False,
              colorbar: str = "right", wind_ref: Optional[float] = None, note: Optional[str] = None) -> go.Figure:
    z = np.asarray(field2d, dtype=float)
    if colorbar == "bottom":
        cb = dict(title=dict(text=units, side="top", font=dict(size=12)), orientation="h", thickness=9, len=1.0,
                  x=0.5, xanchor="center", y=-0.17, yanchor="top", ypad=0, xpad=0, tickfont=dict(size=11),
                  nticks=6, outlinewidth=0)
    else:
        cb = dict(title=dict(text=units, side="right"), thickness=14, len=0.9)
    hm = go.Heatmap(z=z, x=np.asarray(lons, float), y=np.asarray(lats, float), colorscale=cmap,
                    zmin=zmin, zmax=zmax, zmid=zmid, colorbar=cb,
                    hovertemplate="lat %{y:.1f}°N, lon %{x:.1f}°E<br>%{z:.2f} " + units + "<extra></extra>")
    fig = go.Figure(hm)
    has_wind = False
    if wind is not None:
        tr = _wind_trace(wind[0], wind[1], lats, lons, stride=wind_stride, ref=wind_ref)
        if tr is not None:
            fig.add_trace(tr)
            has_wind = True
    if coast is not None:
        add_coast(fig, coast, lats, lons)
    if box is not None:
        fig.add_shape(_box_shape(box, lats, lons))
    if places:
        for text, la, lo in PLACES:
            fig.add_annotation(x=lo, y=la, xref="x", yref="y", text=f"<i>{text}</i>", showarrow=False,
                               xanchor="left", font=dict(size=11, color="#333"), bgcolor="rgba(255,255,255,0.6)",
                               borderpad=1)
    if has_wind and wind_ref:
        _wind_key(fig, lats, lons, wind_stride, float(wind_ref))
    if note:
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text=note, showarrow=False,
                           font=dict(size=13, color="#333"), bgcolor="rgba(255,255,255,0.9)",
                           bordercolor="rgba(0,0,0,0.25)", borderwidth=1, borderpad=6)
    if colorbar == "bottom":
        return _compact_layout(fig, title, lats, lons, height)
    return _layout(fig, title, lats, lons, height)


def symmetric_limit(*arrays, floor: float = 1e-9) -> float:
    m = 0.0
    for a in arrays:
        a = np.asarray(a, dtype=float)
        if a.size and np.isfinite(a).any():
            m = max(m, float(np.nanmax(np.abs(a))))
    return max(m, floor)


def diff_map(diff2d, lats, lons, title: str, units: str, vmax=None, box=None, wind=None, height: int = 360, *,
             empty_text: Optional[str] = None, **kwargs) -> go.Figure:
    """Diverging map centred at 0. A ~zero difference gets a fixed [-1, 1] range instead of a '1n' colourbar."""
    empty = symmetric_limit(diff2d, floor=0.0) < ZERO_DIFF
    if vmax is None:
        v = 1.0 if empty else symmetric_limit(diff2d)
    else:
        v = max(float(vmax), 1e-9)
    if empty and empty_text and not kwargs.get("note"):
        kwargs["note"] = empty_text
    return field_map(diff2d, lats, lons, title, units, cmap=DIVERGING, zmid=0.0, zmin=-v, zmax=v,
                     box=box, wind=wind, height=height, **kwargs)


def diff_animation(diffs, lats, lons, title: str, units: str, box=None, height: int = 420,
                   day_labels: Optional[Sequence[str]] = None, *, coast=None) -> go.Figure:
    d = np.asarray(diffs, dtype=float)
    n = d.shape[0]
    labels = list(day_labels) if day_labels is not None else [f"Day {i + 1}" for i in range(n)]
    v = symmetric_limit(d)
    if symmetric_limit(d, floor=0.0) < ZERO_DIFF:
        v = 1.0
    fig = diff_map(d[0], lats, lons, f"{title} · {labels[0]}", units, vmax=v, box=box, height=height, coast=coast)
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
