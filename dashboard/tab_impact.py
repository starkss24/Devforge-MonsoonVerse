"""Impact tab: crop-stress index per approximate farming region, baseline vs shocked, from the last live Run.

Inputs are the two rollouts already returned by the server for the last Run (no extra server calls) and a
day-of-year climatology computed once from the archive's TRAIN years 2005-2018 (impact.climatology). The maths
lives in impact/crop_stress.py and docs/IMPACT.md.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import plots
from impact.climatology import TRAIN_YEARS, clim_for_dates, forecast_dates, train_climatology
from impact.crop_stress import DEFAULT_WEIGHTS, crop_stress
from impact.regions import REGIONS, region_boxes
from tab_shocklab import shock_box

UP, DOWN, FLAT = "#dc3c3d", "#257de0", "#9aa3ab"       # more stress / less stress / no change
REGION_LINE = "#3b2a7a"
SHOCK_LINE = "#f47230"
NO_BAR = {"displayModeBar": False}
MAP_H = 410
REGION_HELP = "Approximate boxes, not district boundaries: " + "; ".join(
    f"{r['short']} = {name}" for name, r in REGIONS.items())


@st.cache_resource(show_spinner="Computing the 2005–2018 climatology (once)…")
def _climatology(_arc) -> np.ndarray:
    """[366, 2, nlat, nlon] smoothed day-of-year climatology of (t2m, tp) from the train years only."""
    return train_climatology(_arc.data, _arc.dates, channels=(0, 1))


def compute(run, arc, hot_anom_k: float) -> tuple[dict, np.ndarray]:
    """crop_stress for the run's baseline vs shocked rollouts; also returns the climatological tp [steps,...]."""
    steps = int(run.base.shape[0])
    c = clim_for_dates(_climatology(arc), forecast_dates(run.last_day, steps))
    res = crop_stress(run.base, run.shocked, c[:, 1], c[:, 0], region_boxes(), land_mask=land_sea_mask(arc),
                      hot_anom_k=hot_anom_k, lats=arc.lats, lons=arc.lons)
    return res, c[:, 1]


def land_sea_mask(arc):
    try:
        return np.asarray(arc.static[0])
    except Exception:
        return None


def _pct(v: float, dry: bool) -> str:
    return "dry season" if dry else f"{v:+.0f}%"


def table_rows(res: dict) -> list[dict]:
    """One row per region: baseline -> shocked for deficit, hot days and index, plus the index change."""
    rows = []
    for name, r in res.items():
        b, s, ch = r["baseline"], r["shocked"], r["change"]
        rows.append({
            "Region": REGIONS[name]["short"] if name in REGIONS else name,
            "Rain deficit": f"{_pct(b['deficit_pct'], b['clim_dry'])} → {_pct(s['deficit_pct'], s['clim_dry'])}",
            "Hot days": f"{b['hot_days']} → {s['hot_days']}",
            "Δ t2m (K)": round(float(ch["t2m_anom_k"]), 2),
            "Index": f"{b['index']:.0f} → {s['index']:.0f}",
            "Δ index": round(float(ch["index"]), 1),
        })
    return rows


def headline(res: dict, shocked_run: bool) -> tuple[str, bool]:
    """Sentence for the banner and whether anything changed."""
    if not shocked_run:
        return ("No shock in the last Run, so baseline = shocked. The table still shows how the baseline "
                "forecast compares with the normal rain and temperature for these dates.", False)
    name, r = max(res.items(), key=lambda kv: abs(kv[1]["change"]["index"]))
    ch = r["change"]
    if abs(ch["index"]) < 0.05:
        tname, t = max(res.items(), key=lambda kv: abs(kv[1]["change"]["t2m_anom_k"]))
        pname, q = max(res.items(), key=lambda kv: abs(kv[1]["change"]["rain_mm"]))
        dt, dp = t["change"]["t2m_anom_k"], q["change"]["rain_mm"]
        if abs(dt) < 0.01 and abs(dp) < 0.05:
            return ("The shock leaves rain and temperature unchanged in every region, so the crop-stress index "
                    "does not move (Δ = 0 everywhere).", False)
        parts = []
        if abs(dt) >= 0.01:
            parts.append(f"{REGIONS.get(tname, {}).get('short', tname)} mean t2m anomaly {dt:+.2f} K")
        if abs(dp) >= 0.05:
            parts.append(f"{REGIONS.get(pname, {}).get('short', pname)} rain {dp:+.1f} mm")
        return (f"The shock moves the regional weather ({'; '.join(parts)}) but no region crosses a hot-day or "
                "rain threshold it had not already crossed, so the index does not move. Try the hot-day slider.",
                False)
    b, s = r["baseline"], r["shocked"]
    return (f"Largest change: **{name}** index {b['index']:.0f} → {s['index']:.0f} (**{ch['index']:+.1f}**) · "
            f"rain deficit {_pct(b['deficit_pct'], b['clim_dry'])} → {_pct(s['deficit_pct'], s['clim_dry'])} · "
            f"hot days {b['hot_days']} → {s['hot_days']}", True)


def _rect(box, color, dash, width) -> dict:
    lat0, lat1, lon0, lon1 = box
    return dict(type="rect", xref="x", yref="y", x0=lon0 - 0.5, x1=lon1 + 0.5, y0=lat0 - 0.5, y1=lat1 + 0.5,
                line=dict(color=color, width=width, dash=dash), fillcolor="rgba(0,0,0,0)")


def region_map(arc, field2d, title: str, units: str, sbox=None) -> go.Figure:
    v = max(float(np.nanmax(np.abs(field2d))), 1.0)
    fig = plots.field_map(field2d, arc.lats, arc.lons, title, units, cmap="BrBG", zmid=0.0, zmin=-v, zmax=v,
                          height=MAP_H, coast=land_sea_mask(arc), colorbar="bottom")
    if sbox is not None:
        fig.add_shape(_rect(sbox, SHOCK_LINE, "solid", 2.5))
        fig.add_annotation(x=sbox[2] - 0.5, y=sbox[0] - 0.5, xref="x", yref="y", text="shock", showarrow=False,
                           xanchor="left", yanchor="top", font=dict(size=11, color=SHOCK_LINE))
    for name, r in REGIONS.items():
        box = r["box"]
        fig.add_shape(_rect(box, REGION_LINE, "dash", 2.0))
        fig.add_annotation(x=(box[2] + box[3]) / 2, y=(box[0] + box[1]) / 2, xref="x", yref="y",
                           text=r["short"].replace(" ", "<br>"), showarrow=False,
                           font=dict(size=10, color=REGION_LINE),
                           bgcolor="rgba(255,255,255,0.75)", borderpad=1)
    return fig


def delta_bars(res: dict) -> go.Figure:
    names = [REGIONS[n]["short"] if n in REGIONS else n for n in res]
    d = np.array([r["change"]["index"] for r in res.values()])
    colors = [UP if x > 0.05 else DOWN if x < -0.05 else FLAT for x in d]
    fig = go.Figure(go.Bar(x=d, y=names, orientation="h", marker_color=colors, text=[f"{x:+.1f}" for x in d],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: Δ index %{x:+.1f}<extra></extra>"))
    lim = max(float(np.abs(d).max()) * 1.35, 5.0)
    fig.update_layout(title=dict(text="Δ crop-stress index (shocked − baseline)", x=0.0, xanchor="left",
                                 font=dict(size=15)),
                      height=230, margin=dict(l=8, r=16, t=30, b=24), font=dict(size=13),
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    fig.update_xaxes(range=[-lim, lim], zeroline=True, zerolinecolor="#555", zerolinewidth=1.5,
                     gridcolor="rgba(128,128,128,0.25)", title_text="red = more stress, blue = less")
    fig.update_yaxes(autorange="reversed")
    if not np.any(np.abs(d) > 0.05):
        fig.add_annotation(x=1.0, y=1.0, xref="paper", yref="paper", xanchor="right", yanchor="top", showarrow=False,
                           text="no change in any region",
                           font=dict(size=13, color="#333"), bgcolor="rgba(255,255,255,0.9)",
                           bordercolor="rgba(0,0,0,0.25)", borderwidth=1, borderpad=5)
    return fig


def render(ctx) -> None:
    st.subheader("Impact · crop stress by farming region")
    run = ctx.run
    if run is None:
        st.info("Press **Run the twin** in the sidebar (for example with *monsoon_stall*). This tab then turns the "
                "baseline and shocked forecasts into a rain deficit, a hot-day count and a 0–100 crop-stress "
                "index for six approximate farming regions.")
        return

    top_l, top_r = st.columns([3, 1], vertical_alignment="center")
    hot_k = top_r.slider("Hot day: t2m anomaly above (K)", 0.5, 5.0, 2.0, 0.5, key="impact_hot_k",
                         help="A day counts as hot when the region's mean DAILY-MEAN 2 m temperature is more than "
                              "this many kelvin above the 2005–2018 normal for that date.")
    try:
        res, clim_tp = compute(run, ctx.archive, float(hot_k))
    except ValueError as e:
        st.error(f"Impact could not be computed: {e}")
        return
    shocked_run = run.perturbation is not None and bool(np.any(run.perturbation != 0))
    text, moved = headline(res, shocked_run)
    with top_l:
        (st.success if moved else st.info)(text)

    left, right = st.columns([5, 6], gap="medium")
    rain_anom = run.shocked[:, 1].sum(axis=0) - clim_tp.sum(axis=0)
    sbox = shock_box(run.shock_label, run.perturbation) if shocked_run else None
    fig = region_map(ctx.archive, rain_anom, f"{'Shocked' if shocked_run else 'Baseline'} {run.base.shape[0]}-day "
                     "rain vs normal", "mm over the horizon (brown = drier)", sbox)
    left.plotly_chart(fig, key="impact_map", config=NO_BAR)

    with right:
        st.dataframe(table_rows(res), hide_index=True, width="stretch",
                     column_config={
                         "Region": st.column_config.TextColumn(help=REGION_HELP),
                         "Rain deficit": st.column_config.TextColumn(
                             help="Baseline → shocked. 100 × (normal − forecast) / normal rain total; + = drier"),
                         "Hot days": st.column_config.TextColumn(
                             help=f"Baseline → shocked, out of {run.base.shape[0]} forecast days"),
                         "Δ t2m (K)": st.column_config.NumberColumn(format="%+.2f", width="small",
                                                                    help="Change in the 10-day mean t2m anomaly"),
                         "Δ index": st.column_config.NumberColumn(format="%+.1f", width="small"),
                     })
        st.plotly_chart(delta_bars(res), key="impact_bars", config=NO_BAR)

    backend = str((ctx.health or {}).get("backend", ""))
    stub = "persist" in backend.lower()
    w_rain, w_heat = DEFAULT_WEIGHTS
    st.caption(
        f"**How it is computed.** Per region, cos-latitude weighted mean over land cells (mask > 0.5) for the "
        f"{run.base.shape[0]} forecast days after {run.last_day}. Rain deficit = 100 × (normal − forecast) / normal "
        f"rain total (+ = drier); \"dry season\" when normal < 1 mm. Hot day = region-mean t2m anomaly > {hot_k:g} K. "
        f"Index = 100 × ({w_rain:g} × clip(deficit, 0–100%) + {w_heat:g} × hot days / days). Normal = smoothed "
        f"day-of-year climatology from {TRAIN_YEARS[0]}–{TRAIN_YEARS[1]} only.  \n"
        "**Honest limits.** t2m is a daily mean, so these are warm-anomaly days, not heatwave maxima. Region boxes "
        "are approximate (dashed), not district boundaries. Values come from the live twin run"
        + (" — currently the **persistence stub** (the last input day repeated), so only a shock that touches "
           "t2m or rain directly can move the index; wind-only shocks such as monsoon_stall give Δ = 0 until the "
           "model is loaded." if stub else f" (backend {backend or 'unknown'}).")
    )
