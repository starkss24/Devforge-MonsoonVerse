"""Reference baselines: persistence and smoothed day-of-year climatology, leads 1..10, on every validation window.

- Climatology: kit `monsoon_r1.stats.climatology` (per calendar day mean, 15-day circular smoothing) built from
  TRAIN years 2005-2018 only (via eval.score.train_climatology), looked up for the TARGET day.
- Persistence: the last input day repeated.
- Windows: every validation window (2019-2021) whose 3 input days and 10 target days all fall in 2019-2021
  (1084 windows), the same set as `eval.score --windows 0`.
- Metrics, pooled over all windows per lead: kit `scoring.rmse_lat` (judge metric, cos-lat weighted), plain RMSE,
  and kit `scoring.csi` for tp at 10 mm/day.

Writes data_tools/baselines.csv. Run:  .venv/Scripts/python -m data_tools.baselines
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monsoon_r1 import scoring as S  # noqa: E402
from monsoon_r1.stats import clim_for  # noqa: E402

from common import CH, CHANNELS, UNITS  # noqa: E402
from data_tools.loader import DEFAULT_NC, TRAIN_YEARS, VAL_YEARS, load_archive, split_years, window_starts  # noqa: E402
from eval.score import to_date, train_climatology  # noqa: E402

OUT_CSV = ROOT / "data_tools" / "baselines.csv"
COLUMNS = ["lead", "channel", "units", "rmse_persist", "rmse_clim", "rmse_lat_persist", "rmse_lat_clim",
           "best_ref", "csi_persist", "csi_clim"]
MAX_LEAD = 10


def rmse_plain(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Unweighted RMSE per channel for [..., 6, 32, 36], pooled over all leading dims and grid cells."""
    se = (np.asarray(pred, np.float64) - true) ** 2
    return np.sqrt(se.mean(axis=(-2, -1)).reshape(-1, len(CHANNELS)).mean(axis=0))


def compute_baselines(data: np.ndarray, dates: np.ndarray, max_lead: int = MAX_LEAD,
                      train=TRAIN_YEARS, val=VAL_YEARS) -> tuple[list[dict], int]:
    """Rows (one per lead x channel) of reference scores over every val window. Returns (rows, n_windows)."""
    tr, va = split_years(dates, train, val)
    clim = train_climatology(data, dates, train)                 # [366,6,32,36], train years only
    win = window_starts(dates, va, k=max_lead)                    # last input day indices
    if win.size == 0:
        raise ValueError("no complete validation windows")
    clim_by_day = np.stack([clim_for(clim, to_date(d)) for d in dates])  # [T,6,32,36] clim of each calendar day
    persist = data[win].astype(np.float64)                        # same for every lead
    rows = []
    for lead in range(1, max_lead + 1):
        truth = data[win + lead].astype(np.float64)
        climf = clim_by_day[win + lead].astype(np.float64)
        rp, rc = rmse_plain(persist, truth), rmse_plain(climf, truth)
        rlp, rlc = S.rmse_lat(persist, truth), S.rmse_lat(climf, truth)
        tp = CH["tp"]
        csi_p = S.csi(persist[:, tp], truth[:, tp])
        csi_c = S.csi(climf[:, tp], truth[:, tp])
        for c, name in enumerate(CHANNELS):
            rows.append({"lead": lead, "channel": name, "units": UNITS[name],
                         "rmse_persist": rp[c], "rmse_clim": rc[c],
                         "rmse_lat_persist": rlp[c], "rmse_lat_clim": rlc[c],
                         "best_ref": "persist" if rlp[c] <= rlc[c] else "clim",
                         "csi_persist": csi_p if name == "tp" else None,
                         "csi_clim": csi_c if name == "tp" else None})
    return rows, int(win.size)


def write_csv(rows: list[dict], path=OUT_CSV) -> None:
    def cell(v):
        if v is None:
            return ""
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.6g}"
        return v

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: cell(r[k]) for k in COLUMNS})


def print_summary(rows: list[dict], n_windows: int) -> None:
    print(f"lat-weighted RMSE (judge metric), persistence / climatology, {n_windows} val windows")
    print(f"{'lead':>4s}  " + "  ".join(f"{c + ' ' + UNITS[c]:>17s}" for c in CHANNELS) + "  best ref        CSI tp>10 p/c")
    for lead in sorted({r["lead"] for r in rows}):
        rr = [r for r in rows if r["lead"] == lead]
        cells = "  ".join(f"{r['rmse_lat_persist']:8.3g}/{r['rmse_lat_clim']:<8.3g}" for r in rr)
        refs = {r["best_ref"] for r in rr}
        best = refs.pop() + " (all)" if len(refs) == 1 else ",".join(f"{r['channel']}:{r['best_ref']}" for r in rr)
        tp = next(r for r in rr if r["channel"] == "tp")
        print(f"{lead:4d}  {cells}  {best:14s}  {tp['csi_persist']:.3f}/{tp['csi_clim']:.3f}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="persistence / climatology baselines, leads 1..10")
    p.add_argument("--data", default=str(DEFAULT_NC))
    p.add_argument("--out", default=str(OUT_CSV))
    a = p.parse_args(argv)
    t0 = time.perf_counter()
    arc = load_archive(a.data)
    rows, n = compute_baselines(arc.data, arc.dates)
    write_csv(rows, a.out)
    print_summary(rows, n)
    print(f"wrote {a.out} ({len(rows)} rows) in {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
