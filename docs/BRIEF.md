# Multiverse of Monsoon, Track 2: a learned digital twin of Indian daily weather

*Draft. `[TO CONFIRM: …]` = owner confirms; `[RESULT]` = from `eval/score.py` / `eval/shock_test.py`.*

## Page 1: model design, drift and blur, failure modes

**Problem.** From 3 real days of 6 fields (t2m, tp, u10, v10, msl, tcwv) on a 32 × 36, 1° grid over India, roll the weather 10 days forward. Answer "what if" shocks causally: a change starts where it is painted and then spreads.

**Model (planned design).** A small residual U-Net `[TO CONFIRM: Aether]`.
- Input: 3 days × 6 channels, the 2 static maps (land-sea mask, orography) and day-of-year as sin/cos, giving 22 input planes.
- Output: the normalised change to the next day, added to the last input day. Predicting the change, not the full state, makes "no change" the default.
- Rain is trained in log1p space. A softplus output keeps it non-negative `[TO CONFIRM: Aether]`.
- Channels and day-to-day changes are normalised, so each variable counts equally.
- Padding is replicate or reflect, not circular, because the grid is a regional box, not a globe `[TO CONFIRM: Aether]`.
- The forecast is autoregressive: each predicted day is fed back in to make the next.

**How drift and blur are prevented.**
- *Multi-step fine-tune.* After 1-step training, we fine-tune on 2-, then 4-, then 8-step rollouts. Gradients flow through the whole chain, so the model learns from its own errors `[TO CONFIRM: Aether]`.
- *Input noise* during 1-step training makes the model robust to imperfect inputs `[TO CONFIRM: Aether]`.
- *Gradient or spectral loss* penalises lost fine detail, which plain MSE rewards `[TO CONFIRM: Aether]`.
- *Checkpoint selection by rollout score.* We pick the checkpoint by day-4 and day-8 skill plus sharpness, not by training loss.
- *Clamps between steps.* Rain ≥ 0 and t2m within −40 to +60 °C are enforced before each day is fed back.
- The judge's anti-blur test wants day-8 spatial variance between 40% and 250% of the truth. Over-sharp rain fails it too.

**Determinism and the null-shock guarantee.** The server runs the model in eval and inference mode, on CPU, with 4 threads and no randomness. A zero perturbation adds exactly 0, so it gives a bit-identical rollout. Our own test demands max |difference| = 0, which is stricter than the judge's 1e-3 std. If the weights fail to load or a step returns NaN/Inf, the twin falls back to persistence, logged and counted in `/health`. The server never crashes or sends invalid numbers.

**Validation.** We train on 2005–2018 and validate on 2019–2021. The split is by year, because neighbouring days are nearly identical and would leak answers. Climatology and the heavy-rain threshold come from training years only. Every metric is the organisers' own code (`monsoon_r1.scoring`), and the shocks use the kit's `make_shock`. The baselines are persistence and day-of-year climatology.

| Judge component (validation 2019–2021) | Persistence baseline | Our model |
|---|---|---|
| RMSE skill vs better reference (30) | 0.00 (mean SS −0.173) | `[RESULT]` |
| Heavy-rain CSI skill, > 10 mm/day (15) | 0.00 (mean −0.039) | `[RESULT]` |
| Anti-blur, day-8 variance (10) | 8.33 (tp ratio 2.80 > 2.5) | `[RESULT]` |
| Shock causality (15) | 11.25 (localised, does not propagate) | `[RESULT]` |
| Beats persistence at day 4 | fails by construction | `[RESULT]` |
| Total of 70 automatic points | 19.6 | `[RESULT]` |

*Persistence: all 1,084 validation windows (`eval/BASELINE.md`). The better reference is persistence at day 1 and climatology at days 4 and 8, so full RMSE points need about 12% better than climatology at long leads.*

**Failure modes and honest limits.**
- **t2m is a daily mean.** Heatwave thresholds written for daily maxima rarely trigger, so we use anomalies against climatology.
- **Skill falls with lead time.** Beyond about a week, climatology is hard to beat. Day 10 is the weakest day. Blending towards climatology after day 7 is a stretch goal, not built.
- **Box edges.** The domain is a regional box. Nothing outside it can flow in, so the edge cells are the least reliable.
- **Region boxes are approximate.** At 1° (about 110 km), state and basin outlines are whole grid cells.
- **Learned, not physical.** The dynamics are learned from data. Only the clamps, non-negative rain and the residual form are imposed. There is no conservation law: a moisture-budget loss was judged too risky for 20 hours.
- **Hidden test dates** are after 2023-01-10, outside our training and validation years, so the real test is on unseen seasons.

## Page 2: the visualisation and the impact use case

**The dashboard** (`streamlit run dashboard/app.py`) calls the live `/simulate` on every Run, with no pre-rendered results, and says clearly if the server is down.
- **Shock painter.** Presets for *warm_ocean* (Bay of Bengal +3 K, +15% moisture) and *monsoon_stall* (Arabian Sea winds −40%), plus a custom box with a channel and a magnitude, either absolute or a percentage.
- **Maps.** Three panels side by side: baseline, shocked and difference. Any of the 6 channels, lead days 1–10, wind arrows and coastline, plus a day 1–10 animation of the difference.
- **Shock Lab: proof, not claims.** It plots response energy inside, near (1–3 cells) and outside the box by day, and the share of energy by distance from the box at days 1, 4 and 8. It runs the kit's own causality checks (non-zero, localised: ≥ 60% of day-1 energy within 3 cells; propagates; no teleport faster than 8 cells/day; stable below 5 std). A one-click zero-shock check makes two fresh live calls and shows PASS only if the forecasts are identical.
- **Ensemble.** 8 live calls (2–16 adjustable), each from the 3 start days nudged by small noise (default 0.5% of channel std), shown as mean and spread maps. The server stays deterministic. The spread shows how sensitive the forecast is to its start, not a calibrated probability.
- **Model card.** Skill against both baselines by lead day, CSI skill and the day-8 variance band from the scorer's CSV, plus known limits and the live backend.

**Impact use case: crop stress.** For six approximate farming-region boxes (land only), the Impact tab turns both rollouts into the 10-day rain deficit (%) against the 2005–2018 climatology and the count of hot days (daily-mean t2m anomaly above a slider, default 2 K). A 0–100 index combines them (0.6 × deficit + 0.4 × hot-day share), baseline → shocked. On the persistence stub, warm_ocean at 2.5 K lifts the East coast index from 47 to 87; monsoon_stall gives Δ = 0. Tests use hand-computed numbers.

**Engineering.**
- **Fast.** A 10-step rollout takes ~100 ms (persistence backend) and ~60 ms for a 2.1M-parameter test conv net on 4 CPU threads. The judge's limit is 5 s. Final model: `[TO CONFIRM: Vishwak]`.
- **Offline-safe.** `serve.py` imports only libraries on the sandbox list and never the kit or dashboard libraries. The weights are inside the repo, and the server writes nothing to disk. The model loads and warms up before `/health` reports ready.
- **Tested.** 93 automated tests cover the server, the twin, the scorer, the shocks and the dashboard. We also run the organisers' schema validator, a repo scan and a clean-clone script.
