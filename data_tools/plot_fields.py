"""README figures -> docs/img/*.png (dev only; needs matplotlib).

  fields_2019-07-15.png       one monsoon day, all 6 channels, coastline from land_sea_mask
  tp_monthly_climatology.png  mean daily rain per month over the train years 2005-2018
  baseline_rmse_vs_lead.png   persistence vs climatology lat-weighted RMSE, leads 1..10 (from baselines.csv)

Run:  .venv/Scripts/python -m data_tools.plot_fields [--date 2019-07-15]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import CHANNELS, LATS, LONS, UNITS  # noqa: E402
from data_tools.loader import DEFAULT_NC, load_archive, split_years  # noqa: E402

IMG = ROOT / "docs" / "img"
BASELINES_CSV = ROOT / "data_tools" / "baselines.csv"
PERSIST_COLOR, CLIM_COLOR = "#257de0", "#f47230"   # categorical slots 1 and 2
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"
LABELS = {"t2m": "2 m temperature", "tp": "precipitation", "u10": "10 m zonal wind (u)",
          "v10": "10 m meridional wind (v)", "msl": "mean sea-level pressure", "tcwv": "total column water vapour"}
EXTENT = (LONS[0] - 0.5, LONS[-1] + 0.5, LATS[0] - 0.5, LATS[-1] + 0.5)
DPI = 110

plt.rcParams.update({"font.size": 9, "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2,
                     "ytick.color": TEXT2, "text.color": TEXT, "axes.titlesize": 10, "figure.facecolor": "white"})


def _coast(ax, lsm):
    ax.contour(LONS, LATS, lsm, levels=[0.5], colors="k", linewidths=0.7)


def _map(ax, field, lsm, cmap, vmin=None, vmax=None, title="", unit=""):
    im = ax.imshow(field, origin="lower", extent=EXTENT, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
                   interpolation="nearest")
    _coast(ax, lsm)
    ax.set_title(title, loc="left")
    ax.set_xticks([70, 80, 90, 100])
    ax.set_yticks([10, 20, 30])
    ax.set_xticklabels(["70E", "80E", "90E", "100E"])
    ax.set_yticklabels(["10N", "20N", "30N"])
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(unit)
    cb.outline.set_visible(False)
    return im


def plot_day(arc, date: str, out: Path) -> Path:
    i = int(np.nonzero(arc.dates == np.datetime64(date, "D"))[0][0])
    day = arc.data[i]
    lsm = arc.static[list(arc.static_names).index("land_sea_mask")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), constrained_layout=True)
    specs = {
        "t2m": dict(cmap="inferno", field=day[0] - 273.15, unit="deg C (file is K)"),
        "tp": dict(cmap="Blues", field=day[1], vmin=0, vmax=max(20.0, float(np.percentile(day[1], 99))),
                   unit="mm/day"),
        "u10": dict(cmap="RdBu_r", field=day[2], sym=True, unit="m/s (+ = eastward)"),
        "v10": dict(cmap="RdBu_r", field=day[3], sym=True, unit="m/s (+ = northward)"),
        "msl": dict(cmap="viridis", field=day[4] / 100.0, unit="hPa (file is Pa)"),
        "tcwv": dict(cmap="YlGnBu", field=day[5], vmin=0, unit="kg/m2"),
    }
    for ax, name in zip(axes.ravel(), CHANNELS):
        s = specs[name]
        vmin, vmax = s.get("vmin"), s.get("vmax")
        if s.get("sym"):
            m = float(np.abs(s["field"]).max())
            vmin, vmax = -m, m
        _map(ax, s["field"], lsm, s["cmap"], vmin, vmax, f"{name}: {LABELS[name]}", s["unit"])
    fig.suptitle(f"One monsoon day, {date} (1 deg grid, 32 x 36; black line = coast from land_sea_mask)",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def plot_tp_monthly(arc, out: Path) -> Path:
    tr, _ = split_years(arc.dates)
    months = arc.dates.astype("datetime64[M]").astype(np.int64) % 12 + 1
    tp = arc.data[:, CHANNELS.index("tp")]
    lsm = arc.static[list(arc.static_names).index("land_sea_mask")]
    clim = np.stack([tp[tr & (months == m)].mean(axis=0) for m in range(1, 13)])
    vmax = float(np.ceil(np.percentile(clim, 99.5)))
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
    for m, ax in enumerate(axes.ravel()):
        im = ax.imshow(clim[m], origin="lower", extent=EXTENT, cmap="Blues", vmin=0, vmax=vmax,
                       interpolation="nearest")
        _coast(ax, lsm)
        ax.set_title(f"{names[m]}  (domain mean {clim[m].mean():.1f})", loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes, shrink=0.6, location="right", pad=0.01)
    cb.set_label("mean precipitation, mm/day")
    cb.outline.set_visible(False)
    fig.suptitle("Monthly mean rain, train years 2005-2018: the June-September monsoon (west coast, "
                 "Bay of Bengal, north-east)", x=0.01, ha="left", fontsize=11)
    fig.savefig(out, dpi=95)
    plt.close(fig)
    return out


def read_baselines(path=BASELINES_CSV) -> dict:
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d = out.setdefault(r["channel"], {"lead": [], "p": [], "c": [], "units": r["units"]})
            d["lead"].append(int(r["lead"]))
            d["p"].append(float(r["rmse_lat_persist"]))
            d["c"].append(float(r["rmse_lat_clim"]))
    return out


def plot_baselines(out: Path, path=BASELINES_CSV) -> Path:
    b = read_baselines(path)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), constrained_layout=True)
    for ax, name in zip(axes.ravel(), CHANNELS):
        d = b[name]
        ax.plot(d["lead"], d["p"], color=PERSIST_COLOR, lw=2, marker="o", ms=4, label="persistence")
        ax.plot(d["lead"], d["c"], color=CLIM_COLOR, lw=2, marker="o", ms=4, label="climatology (train yrs)")
        for lead in (1, 4, 8):                      # the judge's scored leads
            ax.axvline(lead, color=GRID, lw=1, zorder=0)
        ax.set_title(f"{name}: {LABELS[name]}", loc="left")
        ax.set_ylabel(f"lat-weighted RMSE, {d['units']}")
        ax.set_xticks(range(1, 11))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        # direct labels at the right end
        ax.annotate("persist", (d["lead"][-1], d["p"][-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=TEXT2)
        ax.annotate("clim", (d["lead"][-1], d["c"][-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=TEXT2)
        ax.set_xlim(0.6, 11.2)
    for ax in axes[1]:
        ax.set_xlabel("lead day")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Reference forecasts on all 1084 validation windows (2019-2021). Grey lines = judge leads 1, 4, 8. "
                 "Persistence wins day 1, climatology from day 3.", x=0.01, ha="left", fontsize=11)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="README figures")
    p.add_argument("--data", default=str(DEFAULT_NC))
    p.add_argument("--date", default="2019-07-15")
    a = p.parse_args(argv)
    IMG.mkdir(parents=True, exist_ok=True)
    arc = load_archive(a.data)
    outs = [plot_day(arc, a.date, IMG / f"fields_{a.date}.png"),
            plot_tp_monthly(arc, IMG / "tp_monthly_climatology.png"),
            plot_baselines(IMG / "baseline_rmse_vs_lead.png")]
    for o in outs:
        print(f"wrote {o.relative_to(ROOT)}  ({o.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
