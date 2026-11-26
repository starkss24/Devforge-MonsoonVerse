# Baseline: persistence Twin, judge-projected points

What the hour-0 persistence Twin scores under the judge's own functions (`monsoon_r1.scoring`). Aether's model has to beat these numbers.
Climatology comes from 2005–2018 only (kit `climatology`, 15-day smoothing). Windows come from 2019–2021. Data is `data/india_climate_train.nc` (sha256 verified).

```
.venv/Scripts/python -m eval.score --data data/india_climate_train.nc --stats data/stats.json --twin --windows 0   # all 1084 val windows
.venv/Scripts/python -m eval.score --data data/india_climate_train.nc --stats data/stats.json --twin              # 40 windows (default), writes eval/scores.csv
.venv/Scripts/python -m eval.shock_test --data data/india_climate_train.nc --stats data/stats.json --date 2019-07-15 --date 2020-01-15 --date 2021-04-20 --date 2019-10-10
```

| component | max | all 1084 windows | 40 windows (seed 0) | judge-like 5 windows (mean, p10–p90) |
|---|---|---|---|---|
| RMSE skill | 30 | 0.00 (mean SS −0.173) | 0.00 (SS −0.162) | 0.00 |
| heavy-rain CSI skill | 15 | 0.00 (mean −0.039) | 0.00 (−0.033) | 0.00 |
| anti-blur | 10 | 8.33 (tp day-8 ratio 2.80 > 2.5) | 10.00 | 9.44 (8.33–10) |
| causality | 15 | 11.25 on all 4 dates tried (nonzero, localized and stable; does not propagate) | | |
| **total of 70 auto points** | 70 | **19.6** | **21.25** | ~20.7 |

Day-4 gate (beat persistence): FAIL by construction (SS_persist = 0).

Lat-weighted RMSE over all windows (model = persist), with climatology for comparison and SS vs the best reference:

| lead | t2m K | tp mm/d | u10 m/s | v10 m/s | msl Pa | tcwv | best ref |
|---|---|---|---|---|---|---|---|
| 1 persist / clim | 1.03 / 1.76 | 6.28 / 6.75 | 1.09 / 1.65 | 1.11 / 1.47 | 197 / 281 | 3.69 / 6.42 | persistence, all channels |
| 4 persist / clim | 1.93 / 1.76 | 9.16 / 6.74 | 1.99 / 1.65 | 1.93 / 1.47 | 340 / 280 | 7.36 / 6.42 | climatology, all channels |
| 8 persist / clim | 2.25 / 1.76 | 9.40 / 6.73 | 2.21 / 1.65 | 2.01 / 1.47 | 375 / 280 | 8.22 / 6.42 | climatology, all channels |

CSI (tp > 10 mm/day), persistence vs climatology: lead 1 0.433 vs 0.278, lead 4 0.222 vs 0.279, lead 8 0.201 vs 0.279.

What this means for the model:
- Full RMSE points need a mean window SS of +0.12 against the better reference. That reference is persistence at day 1 and climatology at days 4 and 8. So the model has to beat climatology by about 12 % at long leads, not just persistence.
- Anti-blur averages the variance ratios over windows. Rain that persists into a dry day 8 can push the tp ratio above 2.5, which scores 0 for that channel. Over-sharp rain is punished as well as blur.
- Propagation: our tests show that one 3x3 blur per day does not count. Energy outside the 3-cell neighbourhood at days 4–8 must reach 2 % of the peak response energy (see `tests/test_shocks.py`).
