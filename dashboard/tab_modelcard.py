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
from tab_shocklab import BAD, CHANNEL_COLORS, CHART_H, GOOD, NO_BAR, style_chart

ROOT = Path(__file__).resolve().parent.parent
SCORES = ROOT / "eval" / "scores.csv"
SUMMARY = ROOT / "eval" / "scores_summary.json"
GEN_CMD = "python -m eval.score --data data/india_climate_train.nc --stats data/stats.json --twin"
BAND = (0.40, 2.5)


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_scores(path: Path = SCORES) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _layout(fig, title, yt, height=CHART_H):
    fig = style_chart(fig, title, "", yt, height)
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    return fig


def scorecard(rows: list[dict], leads=(1, 4, 8)) -> list[dict]:
    """Per lead day: mean over channels of skill vs the better reference (`ss`, what the judge scores)."""
    out = []
    for lead in leads:
        vals = [_f(r.get("ss")) for r in rows if int(r["lead"]) == lead]
        vals = [v for v in vals if v == v]
        if vals:
            out.append({"lead": lead, "skill": sum(vals) / len(vals), "n": len(vals)})
    return out


def _tile(label: str, value: str, sub: str, color: str) -> str:
    return (f'<div style="border:1px solid rgba(128,128,128,0.3);border-left:6px solid {color};border-radius:8px;'
            f'padding:10px 14px;height:100%">'
            f'<div style="font-size:15px;opacity:0.8">{label}</div>'
            f'<div style="font-size:34px;font-weight:700;color:{color};line-height:1.2">{value}</div>'
            f'<div style="font-size:13px;opacity:0.75">{sub}</div></div>')


def _skill_color(v: float) -> str:
    return GOOD if v > 0 else BAD if v < 0 else "#6b7280"


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
    leads = sorted({int(r["lead"]) for r in rows})

    def series(col, ch):
        by = {int(r["lead"]): _f(r.get(col)) for r in rows if r["channel"] == ch}
        return [by.get(lead, float("nan")) for lead in leads]

    vr_lead = 8 if 8 in leads else leads[-1]
    vr = [(ch, _f(next((r["var_ratio"] for r in rows if r["channel"] == ch and int(r["lead"]) == vr_lead), "nan")))
          for ch in CHANNELS]
    in_band = sum(BAND[0] <= v <= BAND[1] for _, v in vr)

    # ---- scorecard
    st.markdown("#### Scorecard · skill vs the better baseline (persistence or climatology)")
    cards = scorecard(rows, [lead for lead in (1, 4, 8) if lead in leads])
    cols = st.columns(len(cards) + 1, gap="small")
    for col, sc in zip(cols, cards):
        v = sc["skill"]
        col.markdown(_tile(f"Day {sc['lead']}", f"{v:+.2f}",
                           "beats the baseline" if v > 0 else "worse than the baseline" if v < 0 else "ties the baseline",
                           _skill_color(v)), unsafe_allow_html=True)
    cols[-1].markdown(_tile(f"Day-{vr_lead} sharpness", f"{in_band}/{len(vr)}", "channels in the 40–250% variance band",
                            GOOD if in_band == len(vr) else BAD), unsafe_allow_html=True)
    st.caption("Mean over the 6 channels of 1 − RMSE / RMSE of the better reference. Above 0 beats it; "
               "+0.12 earns full points.")

    # ---- charts
    left, right = st.columns(2, gap="medium")
    for col, title, where, key in (("ss_persist", "Skill vs persistence", left, "mc_ssp"),
                                   ("ss_clim", "Skill vs climatology", right, "mc_ssc")):
        fig = go.Figure()
        for ch in CHANNELS:
            fig.add_trace(go.Bar(x=[f"day {lead}" for lead in leads], y=series(col, ch), name=ch,
                                 marker_color=CHANNEL_COLORS[ch],
                                 hovertemplate=ch + ": %{y:+.3f}<extra></extra>"))
        fig.add_hline(y=0, line_color="#666666", line_width=1)
        vals = [v for ch in CHANNELS for v in series(col, ch) if v == v]
        if vals and all(v == 0 for v in vals):
            fig.add_annotation(x=0.5, y=0.78, xref="paper", yref="paper", showarrow=False, font=dict(size=15),
                               text="0 at every lead: the forecast is persistence itself")
        where.plotly_chart(_layout(fig, title, "skill (> 0 beats it)"), key=key, config=NO_BAR)

    left, right = st.columns(2, gap="medium")
    fig = go.Figure()
    for ch in CHANNELS:
        fig.add_trace(go.Scatter(x=leads, y=series("ss", ch), name=ch, mode="lines+markers",
                                 line=dict(color=CHANNEL_COLORS[ch], width=2.5), marker=dict(size=8)))
    fig.add_hline(y=0.12, line_dash="dot", line_color=GOOD, annotation_text="full points +0.12",
                  annotation_position="top left", annotation_font=dict(size=13))
    fig.add_hline(y=0, line_color="#666666", line_width=1)
    fig = _layout(fig, "Skill vs the better reference (judged)", "skill score")
    fig.update_xaxes(tickvals=leads, ticktext=[f"day {lead}" for lead in leads])
    left.plotly_chart(fig, key="mc_ss", config=NO_BAR)

    ymax = max(100 * BAND[1] * 1.12, max((100 * v for _, v in vr if v == v), default=0) * 1.15)
    fig = go.Figure(go.Bar(x=[c for c, _ in vr], y=[100 * v for _, v in vr],
                           marker_color=[CHANNEL_COLORS[c] for c, _ in vr],
                           marker_line=dict(color=[BAD if not (BAND[0] <= v <= BAND[1]) else "rgba(0,0,0,0)"
                                                   for _, v in vr], width=3),
                           text=[f"{100 * v:.0f}%" + ("" if BAND[0] <= v <= BAND[1] else " ✗") for _, v in vr],
                           textposition="outside", textfont=dict(size=13),
                           hovertemplate="%{x}: %{y:.0f}% of real variance<extra></extra>"))
    fig.add_hrect(y0=100 * BAND[0], y1=100 * BAND[1], fillcolor=GOOD, opacity=0.12, line_width=0, layer="below")
    fig.add_annotation(x=1.0, xref="paper", xanchor="right", y=100 * BAND[1], yanchor="bottom", showarrow=False,
                       text="full-marks band 40–250%", font=dict(size=13, color=GOOD))
    fig = _layout(fig, f"Day-{vr_lead} variance ratio (forecast ÷ truth)", "% of real variance")
    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[0, ymax])
    right.plotly_chart(fig, key="mc_var", config=NO_BAR)

    st.markdown("**Heavy rain (tp > 10 mm/day) · CSI skill**")
    tp = {int(r["lead"]): r for r in rows if r["channel"] == "tp"}
    csi_rows = [{"lead day": lead, "CSI model": _f(tp[lead]["csi_model"]), "CSI persistence": _f(tp[lead]["csi_persist"]),
                 "CSI climatology": _f(tp[lead]["csi_clim"]), "CSI skill": _f(tp[lead]["csi_skill"])}
                for lead in leads if lead in tp]
    st.dataframe(csi_rows, hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="%.3f")
                                for k in ("CSI model", "CSI persistence", "CSI climatology", "CSI skill")})
    st.caption("CSI = hits / (hits + misses + false alarms). CSI skill > 0 beats the better of persistence and "
               "climatology; +0.10 earns full points. Variance ratio below 40% means a blurry forecast, above 250% "
               "inflated noise.")
    st.caption(f"Source: `eval/scores.csv` written by our scorer (eval/score.py, organisers' metric code) · "
               f"file time {mtime} · backend `{summ.get('backend', 'not recorded')}` · "
               f"windows pooled: {', '.join(n_win)} (validation 2019–2021; climatology 2005–2018). "
               f"Regenerate with `{GEN_CMD}`.")


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
