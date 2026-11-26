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
        the units you want to show. box = (lat_min, lat_max, lon_min, lon_max) on cell centres, drawn as a
        turmeric outline around those cells. wind = (u2d, v2d) in m/s, drawn as subsampled arrows.
        Keyword-only extras: coast = land-sea mask [nlat, nlon] (1 = land) drawn as a coastline (0.5 contour);
        places = True writes "Arabian Sea" / "Bay of Bengal" in small italics; colorbar = "right" |
        "bottom" ("bottom" = thin horizontal bar under the map plus compact margins, so the map gets the full
        column width); wind_ref = m/s of a fixed arrow scale shared by every map (weaker wind -> shorter
        arrows), with a reference arrow in the top-left corner, instead of auto-scaling to the field max;
        note = text box in the middle of the map.
    diff_map(diff2d, lats, lons, title, units, vmax=None, box=None, wind=None, height=360, *,
             empty_text=None, **field_map_kwargs) -> go.Figure
        Diverging blue-white-red map centred at 0 with symmetric range [-vmax, vmax]
        (vmax defaults to max |diff|). If max |diff| < ZERO_DIFF the range is fixed to [-1, 1] and
        `empty_text` (if given) is shown on the map. Extra kwargs go to field_map (coast, places, ...).
    diff_strip(diffs, lats, lons, units, box=None, selected=1, height=150, *, coast=None) -> go.Figure
        Small multiples: one small difference map per lead day in one row, one symmetric colour range for all
        days (so growth and spread compare), the shock box on every panel and the selected day outlined.
        Each panel carries an invisible grid of points whose customdata is the day number, so a click on a
        panel can be read back as a selection.
    coastline_trace(mask, lats, lons) -> go.Contour ;  add_coast(fig, mask, lats, lons) -> fig
        When dashboard/assets/india_outline.geojson exists, add_coast draws its lines (sharper than the
        1° grid) instead of the grid-mask contour; without the file it falls back to the contour.
    apply_theme() registers the dashboard's plotly template ("monsoon") and makes it the default.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

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

# Design tokens (app.py's CSS uses the same values). Turmeric means "look here": shock box and selected day only.
INK = "#1B2A41"           # cloud ink: headings, sidebar
MIST = "#EEF3F6"          # rain mist: page background
BLUE = "#2F6F9F"          # monsoon blue: actions, links
TURMERIC = "#D9A21B"      # the one highlight
PADDY = "#3E7D4F"         # pass
SIGNAL = "#B3261E"        # fail / server down
HAIRLINE = "#C3CED6"      # map frames
MUTED = "#4F5D6B"         # secondary text (over 6:1 on white)
FONT_FAMILY = "Archivo, 'Segoe UI', system-ui, -apple-system, sans-serif"
CATEGORICAL = ["#0072B2", "#D55E00", "#009E73", "#56B4E9", "#6B5B95", "#CC79A7"]   # no yellow: turmeric is taken
FONT = dict(size=14)
ZERO_DIFF = 1e-4          # display units; a difference map below this counts as "no change"
COAST_COLOR = "rgba(27,42,65,0.85)"
WIND_COLOR = "rgba(25,25,25,0.55)"
PLACES = (("Arabian Sea", 22.3, 65.4), ("Bay of Bengal", 16.5, 85.0))   # (text, lat, lon of the left end)
LAT_SCALE = float(1.0 / np.cos(np.deg2rad(22.0)))  # 1° latitude is this many times longer than 1° longitude
OUTLINE_PATH = Path(__file__).resolve().parent / "assets" / "india_outline.geojson"


def apply_theme() -> None:
    """Register the "monsoon" plotly template (one type family, ink text, white plots) and make it the default."""
    axis = dict(linecolor=HAIRLINE, gridcolor="rgba(27,42,65,0.10)", zerolinecolor=HAIRLINE, automargin=True,
                tickfont=dict(size=13, color=MUTED), title=dict(font=dict(size=14, color=INK)))
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT_FAMILY, size=14, color=INK),
        title=dict(font=dict(family=FONT_FAMILY, size=16, color=INK, weight=600)),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", colorway=CATEGORICAL,
        hoverlabel=dict(font=dict(family=FONT_FAMILY, size=14, color=INK), bgcolor="#ffffff",
                        bordercolor=HAIRLINE),
        xaxis=axis, yaxis=axis, legend=dict(font=dict(size=14)),
    )
    pio.templates["monsoon"] = t
    pio.templates.default = "monsoon"


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


def _frame(fig: go.Figure) -> None:
    """Hairline frame round a map: axis lines mirrored on all four sides (no extra shapes)."""
    fig.update_xaxes(showline=True, mirror=True, linecolor=HAIRLINE, linewidth=1)
    fig.update_yaxes(showline=True, mirror=True, linecolor=HAIRLINE, linewidth=1)


def _layout(fig: go.Figure, title: str, lats, lons, height: int) -> go.Figure:
    hy, hx = _half_step(lats), _half_step(lons)
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16)),
        height=height, margin=dict(l=48, r=8, t=44, b=40), font=FONT,
        plot_bgcolor="#ffffff", showlegend=False,
    )
    fig.update_xaxes(title_text="Longitude °E", range=[float(np.min(lons)) - hx, float(np.max(lons)) + hx],
                     showgrid=False, zeroline=False, constrain="domain")
    fig.update_yaxes(title_text="Latitude °N", range=[float(np.min(lats)) - hy, float(np.max(lats)) + hy],
                     showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1, constrain="domain")
    _frame(fig)
    return fig


def _compact_layout(fig: go.Figure, title: str, lats, lons, height: int) -> go.Figure:
    """Map fills its column: no axis titles (ticks carry °E/°N), thin margins, room below for the colourbar."""
    hy, hx = _half_step(lats), _half_step(lons)
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", y=0.99, yanchor="top", font=dict(size=16)),
        height=height, margin=dict(l=44, r=6, t=26, b=76), font=dict(size=13),

        plot_bgcolor="#ffffff", showlegend=False,
    )
    fig.update_xaxes(title_text=None, range=[float(np.min(lons)) - hx, float(np.max(lons)) + hx],
                     showgrid=False, zeroline=False, constrain="domain", tickvals=[70, 80, 90],
                     ticktext=["70°E", "80°E", "90°E"], ticks="outside", ticklen=3, tickfont=dict(size=13))
    fig.update_yaxes(title_text=None, range=[float(np.min(lats)) - hy, float(np.max(lats)) + hy],
                     showgrid=False, zeroline=False, scaleanchor="x", scaleratio=LAT_SCALE, constrain="domain",
                     tickvals=[10, 20, 30], ticktext=["10°N", "20°N", "30°N"], ticks="outside", ticklen=3,
                     tickfont=dict(size=13))
    _frame(fig)
    return fig


def _box_shape(box, lats, lons) -> dict:
    lat_min, lat_max, lon_min, lon_max = box
    hy, hx = _half_step(lats), _half_step(lons)
    return dict(type="rect", xref="x", yref="y", x0=lon_min - hx, x1=lon_max + hx, y0=lat_min - hy, y1=lat_max + hy,
                line=dict(color=TURMERIC, width=2.5), fillcolor="rgba(0,0,0,0)")


def coastline_trace(mask, lats, lons) -> go.Contour:
    """Coastline: the 0.5 contour of a land-sea mask [nlat, nlon] (1 = land), as a thin dark line."""
    return go.Contour(z=np.asarray(mask, dtype=float), x=np.asarray(lons, float), y=np.asarray(lats, float),
                      contours=dict(start=0.5, end=0.5, size=1.0, coloring="none"), autocontour=False,
                      line=dict(color=COAST_COLOR, width=1.5, smoothing=1.0), showscale=False,
                      hoverinfo="skip", name="coast")


@lru_cache(maxsize=1)
def _outline_xy(path: str) -> Optional[tuple]:
    """Lines of the bundled Natural Earth outline as (xs, ys) with None breaks, or None if unavailable."""
    try:
        gj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    xs, ys = [], []

    def ring(coords):
        for x, y in coords:
            xs.append(float(x))
            ys.append(float(y))
        xs.append(None)
        ys.append(None)

    for feat in gj.get("features", []):
        g = feat.get("geometry") or {}
        t, c = g.get("type"), g.get("coordinates", [])
        if t == "LineString":
            ring(c)
        elif t in ("MultiLineString", "Polygon"):
            for r in c:
                ring(r)
        elif t == "MultiPolygon":
            for poly in c:
                for r in poly:
                    ring(r)
    return (tuple(xs), tuple(ys)) if xs else None


def outline_trace(width: float = 1.2) -> Optional[go.Scatter]:
    """Natural Earth coast/border lines from dashboard/assets/india_outline.geojson, or None if the file is missing."""
    xy = _outline_xy(str(OUTLINE_PATH)) if OUTLINE_PATH.is_file() else None
    if xy is None:
        return None
    return go.Scatter(x=list(xy[0]), y=list(xy[1]), mode="lines", line=dict(color=COAST_COLOR, width=width),
                      hoverinfo="skip", name="outline", connectgaps=False)


def add_coast(fig: go.Figure, mask, lats, lons, width: float = 1.2, **subplot) -> go.Figure:
    """Outline file if present (sharp lines), else the grid-mask contour; mask=None draws nothing."""
    if mask is None:
        return fig
    tr = outline_trace(width)
    if tr is None:
        tr = coastline_trace(mask, lats, lons)
        tr.line.width = width + 0.3
    fig.add_trace(tr, **subplot)
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
    x0, y0 = left + 0.7 + L / 2, top - 1.4
    fig.add_shape(type="rect", xref="x", yref="y", x0=left, x1=x0 + L / 2 + 8.0, y0=top - 2.8, y1=top,
                  line=dict(width=0), fillcolor="rgba(255,255,255,0.85)", layer="above")
    xs, ys = [], []
    _arrow_xy(x0, y0, L, 0.0, xs, ys)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(25,25,25,0.95)", width=1.4),
                             hoverinfo="skip", name="wind_key"))
    fig.add_annotation(x=x0 + L / 2 + 0.4, y=y0, xref="x", yref="y", text=f"{ref:g} m/s", showarrow=False,
                       xanchor="left", font=dict(size=13, color=INK))


def field_map(field2d, lats, lons, title: str, units: str, cmap: str = "Viridis", zmid=None, zmin=None, zmax=None,
              box=None, wind=None, height: int = 360, wind_stride: int = 3, *, coast=None, places: bool = False,
              colorbar: str = "right", wind_ref: Optional[float] = None, note: Optional[str] = None) -> go.Figure:
    z = np.asarray(field2d, dtype=float)
    if colorbar == "bottom":
        cb = dict(title=dict(text=units, side="bottom", font=dict(size=13)), orientation="h", thickness=9,
                  len=1.0, x=0.5, xanchor="center", y=-0.14, yanchor="top", ypad=0, xpad=0,
                  tickfont=dict(size=13), nticks=6, outlinewidth=0)
    else:
        cb = dict(title=dict(text=units, side="right", font=dict(size=13)), thickness=12, len=0.9,
                  tickfont=dict(size=13), outlinewidth=0)
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
                               xanchor="left", font=dict(size=13, color=INK), bgcolor="rgba(255,255,255,0.7)",
                               borderpad=1)
    if has_wind and wind_ref:
        _wind_key(fig, lats, lons, wind_stride, float(wind_ref))
    if note:
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text=note, showarrow=False,
                           font=dict(size=14, color=INK), bgcolor="rgba(255,255,255,0.92)",
                           bordercolor=HAIRLINE, borderwidth=1, borderpad=6)
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


def diff_strip(diffs, lats, lons, units: str, box=None, selected: int = 1, height: int = 150, *,
               coast=None, hit_stride: int = 3) -> go.Figure:
    """One row of small difference maps, day 1..n, on one symmetric colour range; the selected day is outlined.

    Every panel also holds an invisible grid of points (customdata = day number), so clicking anywhere on a
    panel selects that day when the figure is shown with point selections enabled.
    """
    d = np.asarray(diffs, dtype=float)
    n = d.shape[0]
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    v = 1.0 if symmetric_limit(d, floor=0.0) < ZERO_DIFF else symmetric_limit(d)
    fig = make_subplots(rows=1, cols=n, horizontal_spacing=0.01,
                        subplot_titles=[f"Day {k + 1}" for k in range(n)])
    hy, hx = _half_step(lats), _half_step(lons)
    gx, gy = np.meshgrid(lons[1::hit_stride], lats[1::hit_stride])
    x_range = [float(lons.min()) - hx, float(lons.max()) + hx]
    y_range = [float(lats.min()) - hy, float(lats.max()) + hy]
    for k in range(n):
        col = k + 1
        cb = dict(title=dict(text=units, side="right", font=dict(size=13)), thickness=9, len=1.0, x=1.0,
                  xpad=6, tickfont=dict(size=13), nticks=3, outlinewidth=0)
        fig.add_trace(go.Heatmap(z=d[k], x=lons, y=lats, colorscale=DIVERGING, zmin=-v, zmax=v, zmid=0.0,
                                 showscale=(k == 0), hoverinfo="skip", colorbar=cb, name=f"diff day {k + 1}"),
                      row=1, col=col)
        if coast is not None:
            add_coast(fig, coast, lats, lons, width=0.7, row=1, col=col)
        if box is not None:
            sh = _box_shape(box, lats, lons)
            sh.update(xref="x" if col == 1 else f"x{col}", yref="y" if col == 1 else f"y{col}",
                      line=dict(color=TURMERIC, width=1.5))
            fig.add_shape(sh)
        fig.add_trace(go.Scatter(x=gx.ravel(), y=gy.ravel(), mode="markers", name=f"day {k + 1}",
                                 customdata=np.full(gx.size, k + 1),
                                 marker=dict(size=14, opacity=0, color=INK),
                                 hovertemplate=f"Day {k + 1}: show it in the large maps<extra></extra>"),
                      row=1, col=col)
        fig.update_xaxes(range=x_range, showticklabels=False, ticks="", showgrid=False, zeroline=False,
                         showline=True, mirror=True, linecolor=HAIRLINE, linewidth=1, constrain="domain",
                         fixedrange=True, row=1, col=col)
        fig.update_yaxes(range=y_range, showticklabels=False, ticks="", showgrid=False, zeroline=False,
                         showline=True, mirror=True, linecolor=HAIRLINE, linewidth=1,
                         scaleanchor="x" if col == 1 else f"x{col}", scaleratio=LAT_SCALE, constrain="domain",
                         fixedrange=True, row=1, col=col)
    sel = int(selected)
    for k, a in enumerate(fig.layout.annotations):   # subplot titles
        a.font = dict(size=13, color=INK, weight=700 if k + 1 == sel else 400)
    if 1 <= sel <= n:
        dom = fig.layout["xaxis" if sel == 1 else f"xaxis{sel}"].domain
        fig.add_shape(type="rect", xref="paper", yref="paper", x0=dom[0] - 0.004, x1=dom[1] + 0.004,
                      y0=-0.04, y1=1.2, line=dict(color=TURMERIC, width=3), fillcolor="rgba(0,0,0,0)",
                      name="selected_day")
    fig.update_layout(height=height, margin=dict(l=4, r=48, t=24, b=6), showlegend=False,
                      plot_bgcolor="#ffffff", dragmode=False, hovermode="closest")
    return fig
