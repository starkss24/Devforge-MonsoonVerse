# Multiverse of Monsoon, Track 2: a learned digital twin of Indian daily weather

*Model `aether-unet-v21-ens4`. All scores use the organisers' scoring code; the judge simulation ran through our own `serve.py` on 2026-11-27.*

## Page 1: model design, drift and blur, failure modes

**Problem.** From 3 real days of 6 fields (t2m, tp, u10, v10, msl, tcwv) on a 32 × 36, 1° grid over India, roll the weather 10 days forward, and answer "what if" shocks causally: a change starts where it is painted, then spreads.

**Model.** Four residual U-Nets (2.0 M parameters each, 3 levels), trained identically with different seeds; their predicted daily changes are averaged at every step (1 → 2 → 4 networks: 55 → 59 → 64 of 70 on real 2023 windows).
- Input: 3 days × 6 channels, land-sea mask, orography and day-of-year (sin/cos): 22 planes. Output: the normalised change to the next day, added to the last input day. The output layer starts at zero, so the untrained model *is* persistence.
- Rain is modelled in log1p space and floored at exactly 0 with a straight-through ReLU, so dry cells stay dry over 10 days.
- Replicate padding (the grid is a regional box, not a globe). Each predicted day is fed back in to make the next.

**How drift and blur are prevented.**
- *Multi-step fine-tune* on 2-, 4- then 8-day rollouts, back-propagating through the chain, so the model learns from its own fed-back errors. This alone took the score from 28 to 60 of 70.
- *Anti-blur loss*: a gradient-*magnitude* term, so the forecast's spatial gradients stay as strong as the truth's without punishing a slightly misplaced sharp feature.
- *Post-processing fitted on 2005–2018*, applied to the finished rollout: a climatology blend growing with lead time, capped at 0.5 so shocks are never erased, and a per-lead rain quantile map restoring heavy-rain intensity. Both are fixed functions of forecast and date, so the null shock stays exact.
- *Checkpoints chosen by rollout score*, never training loss; rain ≥ 0 and t2m within −40…+60 °C enforced before each day is fed back.
- Tried and rejected (in `experiments.csv`): input noise (52 vs 61 of 70), a wider U-Net (58), training on all years (worse on real 2023–2025 data).

**Determinism.** CPU, 4 threads, eval mode, no randomness: a zero perturbation gives a bit-identical rollout (our test demands exactly 0, stricter than the judge's 1e-3 std). If weights fail to load or a step returns NaN, the twin falls back to persistence and reports it in `/health`; the server never crashes.

**Validation.** Train 2005–2018, validate 2019–2021, split by year so neighbouring days cannot leak. Climatology and thresholds come from training years only; every metric is the organisers' own code.

| Judge component (2019–2021) | Persistence | Our model |
|---|---|---|
| RMSE skill vs better reference (30) | 0.0 | 30.0 |
| Heavy-rain CSI skill, > 10 mm/day (15) | 0.0 | 14.5 |
| Anti-blur, day-8 variance (10) | 8.3 | 10.0 |
| Shock causality (15) | 11.3 | 13.1 |
| Beats persistence at day 4 | no | yes, every channel |
| **Total of 70 automatic points** | **19.6** | **67.7** |

*Persistence: all 1,084 validation windows; model: 100 windows. The hidden test is after 2023, so we also scored 12 real 2023–2025 windows fetched with the organisers' pipeline: 62.9 of 70 (68.1 in a full judge simulation through our server). Expect the low-to-mid 60s on unseen dates.*

**Honest limits.**
- **Skill falls with lead.** By day 8 we only match climatology; the blend keeps late days honest but damps long-range shock effects by up to half.
- **monsoon_stall rarely spreads** (about 5 of 20 windows): a weakened wind field decays rather than travels. It costs about 2 causality points where it fails.
- **Single rain peaks are smoothed** at longer leads (day-8 rain variance ratio 0.57, inside the judge's band but below temperature's).
- **t2m is a daily mean**, so "hot days" are anomalies, not heatwave maxima. Region boxes are approximate (1° ≈ 110 km); edge cells are least reliable.
- **Learned, not physical**: only the clamps, non-negative rain and the residual form are imposed; there is no conservation law.

## Page 2: the visualisation and the impact use case

**The dashboard** (`streamlit run dashboard/app.py`) calls the live server on every change: nothing is pre-rendered, and it says clearly when the server is down.
- **One interactive map** of India with cities, coastline and wind arrows; layers for rain, temperature, winds, pressure and moisture. Click the map to move the shock box; Live mode re-runs the twin in under half a second.
- **Shock painter**: the judges' warm_ocean and monsoon_stall, demo presets (north-India heatwave +4 K, Arabian Sea westerly burst, Bay of Bengal low −1,500 Pa) and a custom box, channel and magnitude.
- **Timeline and Play** from the 3 real input days through the forecast, with views for *with shock*, *no shock*, *effect* and *observed*, plus a day-by-day strip that shows the shock start in its box. A side panel compares each day against what really happened (peak, detail kept, error).
- **Shock Lab: proof, not claims.** Response energy inside, near and outside the box by day; spread by distance; the organisers' causality checks in plain English; a one-click zero-shock check that passes only if two live forecasts are identical.
- **Ensemble** spread from small input nudges (the server itself stays deterministic), and a **model card** with skill by lead day and known limits.

**Impact use case: crop stress.** For six approximate farming regions (land cells only), the Impact tab turns the baseline and shocked forecasts into a 10-day rain deficit against the 2005–2018 climatology and a count of hot days (daily-mean anomaly above a slider, default 2 K). A 0–100 index combines them (0.6 × deficit + 0.4 × hot-day share). On the 2019-07-15 window, warm_ocean wets the East coast (rain deficit −10 points, +0.38 K, index 14.9 → 12.9) because the extra moisture rains out; monsoon_stall dries Central India (deficit +2.9 points, index 28.3 → 30.1). Formulas are tested against hand-computed numbers.

**Engineering.** About 100 ms per 10-day rollout for the four networks (limit 5 s); `/health` ready 4–7 s after start. `serve.py` imports only sandbox libraries, loads ONNX weights committed in the repo (TorchScript fallback), and writes nothing to disk. 244 automated tests, the organisers' schema validator, repo scan (15/15) and a clean-clone test all pass.
