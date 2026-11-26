"""Model card tab: offline forecast skill from our scorer (eval/scores.csv) plus honest known limits.

eval/scores.csv is the documented exception to "every number comes from a live call": it is the output of
eval/score.py, which uses the organisers' own metric code. The tab shows its source and file time.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from common import CHANNELS

ROOT = Path(__file__).resolve().parent.parent
SCORES = ROOT / "eval" / "scores.csv"
SUMMARY = ROOT / "eval" / "scores_summary.json"
GEN_CMD = "python -m eval.score --data data/india_climate_train.nc --stats data/stats.json --twin"
COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17"]   # fixed categorical order
BAND = (0.40, 2.5)


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_scores(path: Path = SCORES) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _layout(fig, title, yt, height=340):
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=50, b=10), yaxis_title=yt,
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0), barmode="group")
    return fig


def render(ctx) -> None:
    st.subheader("Model card")
    backend = (ctx.health or {}).get("backend", "unknown (server down)")
    model_name = (ctx.health or {}).get("model_name", "?")
    if "persist" in str(backend).lower():
        st.warning(f"The live server is running the **persistence stub** (backend `{backend}`): every forecast day "
                 "repeats the last input day. It is a placeholder until the trained model is loaded, so skill "
                 "versus persistence is exactly 0 by construction.")
    else:
        st.info(f"Live server backend: `{backend}` · model `{model_name}`")

    if not SCORES.is_file():
        st.warning("eval/scores.csv was not found, so no offline skill numbers are shown. Generate it with:")
        st.code(GEN_CMD, language="bash")
    else:
        _render_scores()
    _render_limits(backend)


def _render_scores() -> None:
    rows = load_scores()
    mtime = dt.datetime.fromtimestamp(SCORES.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    summ = {}
    if SUMMARY.is_file():
        try:
            summ = json.loads(SUMMARY.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            summ = {}
    n_win = sorted({r.get("n_windows", "?") for r in rows})
    st.caption(f"Source: `eval/scores.csv` written by our scorer (eval/score.py, organisers' metric code) · "
               f"file time {mtime} · produced by backend `{summ.get('backend', 'not recorded')}` · "
               f"windows pooled: {', '.join(n_win)} (validation years 2019–2021; climatology from 2005–2018). "
               f"Regenerate with `{GEN_CMD}`.")
    leads = sorted({int(r["lead"]) for r in rows})

    def series(col, ch):
        by = {int(r["lead"]): _f(r.get(col)) for r in rows if r["channel"] == ch}
        return [by.get(lead, float("nan")) for lead in leads]

    left, right = st.columns(2)
    for col, title, where, key in (("ss_persist", "Skill vs persistence (1 − RMSE/RMSE persistence)", left, "mc_ssp"),
                                   ("ss_clim", "Skill vs climatology (1 − RMSE/RMSE climatology)", right, "mc_ssc")):
        fig = go.Figure()
        for i, ch in enumerate(CHANNELS):
            fig.add_trace(go.Bar(x=[f"day {lead}" for lead in leads], y=series(col, ch), name=ch,
                                 marker_color=COLORS[i]))
        fig.add_hline(y=0, line_color="#666666", line_width=1)
        where.plotly_chart(_layout(fig, title, "skill score (> 0 is better than the reference)"), key=key)

    fig = go.Figure()
    for i, ch in enumerate(CHANNELS):
        fig.add_trace(go.Scatter(x=leads, y=series("ss", ch), name=ch, mode="lines+markers",
                                 line=dict(color=COLORS[i], width=2), marker=dict(size=8)))
    fig.add_hline(y=0.12, line_dash="dot", line_color="#666666", annotation_text="full points at +0.12")
    fig.update_xaxes(tickvals=leads, title="lead day")
    st.plotly_chart(_layout(fig, "Skill vs the better reference (what the judge scores)", "skill score"),
                    key="mc_ss")

    left, right = st.columns(2)
    tp = {int(r["lead"]): r for r in rows if r["channel"] == "tp"}
    csi_rows = [{"lead day": lead, "CSI model": _f(tp[lead]["csi_model"]), "CSI persistence": _f(tp[lead]["csi_persist"]),
                 "CSI climatology": _f(tp[lead]["csi_clim"]), "CSI skill": _f(tp[lead]["csi_skill"])}
                for lead in leads if lead in tp]
    left.markdown("**Heavy rain (tp > 10 mm/day) · CSI skill**")
    left.dataframe(csi_rows, hide_index=True, width="stretch")
    left.caption("CSI = hits / (hits + misses + false alarms). CSI skill > 0 beats the better of persistence "
                 "and climatology; +0.10 earns full points.")

    vr_lead = 8 if 8 in leads else leads[-1]
    vr = [(ch, _f(next((r["var_ratio"] for r in rows if r["channel"] == ch and int(r["lead"]) == vr_lead), "nan")))
          for ch in CHANNELS]
    fig = go.Figure(go.Bar(x=[c for c, _ in vr], y=[100 * v for _, v in vr],
                           marker_color=[COLORS[0] if BAND[0] <= v <= BAND[1] else "#dc3c3d" for _, v in vr],
                           hovertemplate="%{x}: %{y:.0f}%<extra></extra>"))
    fig.add_hrect(y0=100 * BAND[0], y1=100 * BAND[1], fillcolor="#00b894", opacity=0.12, line_width=0,
                  annotation_text="full-marks band 40–250%", annotation_position="top left")
    right.plotly_chart(_layout(fig, f"Day-{vr_lead} spatial variance ratio (forecast ÷ truth)", "% of real variance"),
                       key="mc_var")
    right.caption("Below 40% means a blurry forecast; above 250% means inflated noise. Red bars are outside the band.")


def _render_limits(backend: str) -> None:
    st.markdown("#### Known limits")
    items = [
        "**t2m is a daily mean**, not a daily maximum, so heatwave flags use temperature anomalies against "
        "climatology rather than absolute thresholds such as 45 °C.",
        "**Region boxes are approximate**: the grid is 1° (about 110 km), so state or basin outlines are "
        "represented by whole grid cells.",
        "**Skill falls with lead time**: day-1 forecasts are much closer to the truth than day-8 ones; beyond "
        "about a week, climatology is a hard reference to beat.",
        "**Ensemble spread is dashboard-made**: the server is deterministic, so spread only reflects the input "
        "noise the Ensemble tab adds, not a calibrated probability.",
        f"**Current backend**: `{backend}`"
        + (" — the persistence stub, not a trained model. Shock responses stay inside the box and do not travel."
           if "persist" in str(backend).lower() else "."),
    ]
    st.markdown("\n".join(f"- {t}" for t in items))
