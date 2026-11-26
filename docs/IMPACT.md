# Impact extension: crop-stress index

The Impact tab turns the twin's two live rollouts (baseline and shocked, `[10, 6, 32, 36]`) into three numbers per
farming region: a rain deficit, a count of hot days and a 0–100 crop-stress index. It also shows how each number
changes under the shock. Everything is numpy, deterministic, and recomputed from the last Run. Nothing is cached
across runs except the climatology, which comes from the archive and not from any run.

| File | Role |
|---|---|
| `impact/regions.py` | 6 approximate farming regions as lat/lon boxes |
| `impact/climatology.py` | smoothed day-of-year climatology from the train years 2005–2018 |
| `impact/crop_stress.py` | `crop_stress(base, shocked, clim_tp, clim_t2m, regions, land_mask=None, hot_anom_k=2.0, weights=(0.6, 0.4))` |
| `dashboard/tab_impact.py` | the Impact tab (map, table, Δ bars, hot-day slider, honest-limits caption) |
| `tests/test_impact.py`, `tests/test_dashboard_impact.py` | hand-computed unit tests and a live headless app test |

## The maths

Notation: `t = 1..N` forecast days (N = 10), cells `(i, j)`, region `r`.

1. **Climatology** (same definition as the organisers' `monsoon_r1.stats.climatology`; a test checks equality).
   Each date maps to a slot 0..365 of the leap year 2000, so Feb 29 has its own slot (59). The mean per slot is
   taken over 2005–2018 only. Then a circular 15-day running mean is applied (offsets −7..+7, wrapping Dec 31 → Jan 1).
   For a run whose last input day is `d0`, the forecast dates are `d0 + 1 .. d0 + N`. `clim_tp(t)` and
   `clim_t2m(t)` are the slots for those dates.
2. **Weights.** `w_ij = cos(lat_i)` on the region's cells, and 0 elsewhere. Only land cells count
   (`land_sea_mask > 0.5`). Weights are normalised to sum to 1. A region with no cell raises an error, so it never
   produces NaN.
3. **Region means.** `X_r(t) = Σ_ij w_ij X_ij(t)`, for forecast and climatology alike.
4. **Rain.** `P = Σ_t tp_r(t)` is the forecast rain total in mm and `C = Σ_t clim_tp_r(t)` is the normal total.
   **Rain deficit** `D = 100 · (C − P) / C` in %. Positive means drier than normal.
   Guard: if `C < 1 mm`, the region is in its dry season. Then `D = 0` and the table shows "dry season".
5. **Hot days.** `H = #{t : t2m_r(t) − clim_t2m_r(t) > k}`, with `k` set by the slider (default 2 K). The
   comparison is strict. **t2m is a daily mean**, so this counts warm daily-mean anomalies. It is not a heatwave
   on daily maxima. A 40 °C-maximum rule would almost never trigger on daily means.
6. **Index.** `S = 100 · [0.6 · clip(D, 0, 100)/100 + 0.4 · H/N]`, which always lies in [0, 100]. Rain that is
   wetter than normal adds no stress. The weights are a transparent choice (rain matters more for kharif crops)
   and are not fitted to yield data.
7. **Change.** shocked − baseline for `P`, `D`, `H`, `S` and the mean t2m anomaly.

## Regions (approximate boxes, not administrative boundaries)

Boxes are inclusive on cell centres (x.5°). They do not overlap and are trimmed to keep the Himalaya out.

| Region | lat °N | lon °E | cells / land cells | note |
|---|---|---|---|---|
| NW plains (Punjab–Haryana) | 28.5–30.5 | 73.5–76.5 | 12 / 12 | irrigated wheat–rice |
| Gangetic plain (UP–Bihar) | 24.5–27.5 | 77.5–85.5 | 36 / 36 | rice, wheat; NE corner touches the foothills |
| Central India (Vidarbha–Marathwada) | 19.5–22.5 | 74.5–80.5 | 28 / 28 | rain-fed cotton, soybean, pulses |
| Deccan interior (Telangana–Rayalaseema) | 13.5–17.5 | 77.5–80.5 | 20 / 18 | semi-arid groundnut, millets |
| East coast (Odisha–Andhra) | 16.5–21.5 | 81.5–86.5 | 36 / 22 | rice deltas; southern part inside the warm_ocean box |
| West coast (Konkan–Malabar) | 8.5–18.5 | 72.5–76.5 | 55 / 25 | rice, spices; overlaps the monsoon_stall box |

## What it shows today (persistence stub, window ending 2019-07-15)

The stub repeats the last input day, so the forecast cannot move weather. It only carries a shock that changes
t2m or rain directly.
- **monsoon_stall** (winds only): Δ = 0 in every region. The banner says so. Baseline indices are NW plains 0,
  Gangetic plain 43, Central India 98, Deccan 1, East coast 87, West coast 24.
- **warm_ocean** (+3 K): the East coast mean t2m anomaly rises by 0.82 K (only part of the box is shocked).
  At 2 K the region is already hot on 10 of 10 days, so the index does not move. At **2.5 K**, East coast hot
  days go from 0 to 10 and the index from 47 to 87 (**+40**).

With trained weights, the wind shock should change rain downstream. That will be the real test of the panel.

## Limits

- t2m is a daily mean: anomalies only, no heatwave maxima.
- Boxes are approximate. At 1° (about 110 km) a district is a few cells at most.
- The index is a planning indicator, not a yield model. It has no soil moisture, irrigation or crop calendar.
- The climatology uses 2005–2018. The test seasons (2019+) are judged against a normal that has not seen them.

## Run the tests

```
.venv/Scripts/python -m pytest tests/test_impact.py tests/test_dashboard_impact.py -q
```

`test_impact.py` needs only numpy. The kit-equality test skips if `monsoon_r1` is not importable.
`test_dashboard_impact.py` starts its own `serve.py` (port 8082, or a free port if that one is busy) and
runs monsoon_stall and warm_ocean headless. It skips without streamlit or `data/india_climate_train.nc`.
