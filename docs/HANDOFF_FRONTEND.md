# Model-lane hand-off for the dashboard (feat/dashboard-v2)

Paste everything below the line into the dashboard agent's session, **in addition to** the dashboard-v2 brief it
already has. It adds what the model lane knows, so the dashboard shows the model honestly and stays correct when
the weights change.

---

You are working on the dashboard (`dashboard/`, branch `feat/dashboard-v2`). Keep following your dashboard-v2 brief
and its hard rules. This adds the **model lane's facts and requests**, from Aether (model lead). Read
`docs/MODEL_V2.md` on branch `feat/aether-v2` (`git show origin/feat/aether-v2:docs/MODEL_V2.md`) before changing
any text that describes the model.

## 1. Which model the dashboard talks to
- **On main now:** `aether-unet-e5b` (single residual U-Net, onnx + `post.npz`).
- **Candidate on `feat/aether-v2`:** `aether-unet-v2-ens2` (2-seed ensemble of the same U-Net, post-processing with
  blend capped at 0.5 and rain-map slope capped at 1.5). It meets the team's acceptance bar (kit score 67.06 vs 65.78
  on 100 held-out windows, 59.11 vs 55.31 on real 2023 windows). Vishwak re-verifies and merges it; it may land on
  main while you work. **The full test suite, including all `tests/test_dashboard_*.py` live AppTests, already passes
  with the v2 weights** (218 passed on Aether's machine).
- Same API, same request format, same `/health` fields: `model_name`, `backend` (`onnx`), `post` (`"loaded"`),
  `post_failures`, `load_error`, `nonfinite_fallbacks`. Show `model_name` in the connection status so judges see which
  model is live. Don't special-case a model name anywhere.
- Latency budget for Live mode: model compute is ~25 ms (e5b) / ~50 ms (v2) per 10-day rollout on 4 threads; a full
  call with JSON is ~0.08-0.12 s. On a laptop, a server started as a background process can be throttled (Aether saw
  ~0.4 s per call for v2); design Live mode to feel right at 0.5 s too (keep the last good result visible, mark the
  map "updating", drop stale requests).

## 2. Replace hard-coded model claims with live measurements (most important)
`dashboard/sidebar.py` `PRESET_EXPECT` contains claims measured on e5b that become wrong with v2, e.g.
"Heatwave: ... fades by day 8" (v2: it does not fade) and "bay_low: up to +55 mm/day" (v2: up to ~44). Hard-coded
behaviour text will go stale whenever the weights change. Please:
- Compute the "what to expect / what happened" line **from the live baseline + shocked rollouts already fetched**:
  local on day 1 (near-box energy fraction >= 60%), spreads by day 4-8, stays stable (< 5 std), the day the in-box
  change falls below 10% of day 1 (or "does not fade within 10 days"), the in-box change at days 1/4/8, and the largest
  rain change. The Shock Lab already computes the kit checks live; reuse that code path, don't duplicate it.
- Keep at most a neutral one-line description per preset (what is painted), not what the model will do.
- If you need reference numbers for tests or copy, use these, measured on 20 windows (8 held-out 2019-2021 + 12 real
  2023-2025) with the dashboard's exact preset boxes:

| Preset (dashboard box) | Model | Local d1 | Spreads | Stable | Peak std | Fades by | In-box change d1 / d4 / d8 / d10 | Max rain up |
|---|---|---|---|---|---|---|---|---|
| warm_ocean (kit) | v2 | 20/20 | 13/20 | 20/20 | 3.3 | d9 on 7/20, else lasts | t2m +2.66 / +1.52 / +1.44 / +1.42 K | 25 mm/d |
| | e5b | 20/20 | 16/20 | 17/20 | 10.6 | d8 on 9/20 | +2.65 / +1.47 / +0.64 / +0.48 K | 80 mm/d |
| monsoon_stall (kit) | v2 | 20/20 | 5/20 | 20/20 | 3.5 | d5 | u10 -0.92 / -0.39 / -0.17 / -0.10 m/s | 27 mm/d |
| | e5b | 20/20 | 5/20 | 17/20 | 8.0 | d4 | -0.94 / -0.46 / -0.17 / -0.10 m/s | 61 mm/d |
| heatwave t2m +4 K, 24.5-30.5 N 73.5-81.5 E | v2 | 20/20 | **20/20** | 20/20 | 1.1 | **does not fade** | +3.57 / +2.00 / +1.88 / +1.85 K | 8 mm/d |
| | e5b | 20/20 | 16/20 | 20/20 | 1.8 | d10 on 6/20 | +3.53 / +1.89 / +0.78 / +0.58 K | 14 mm/d |
| westerly_burst u10 +5 m/s, 10.5-18.5 N 65.5-73.5 E | v2 | 20/20 | 7/20 | 20/20 | 4.1 | d5 | +3.88 / +1.55 / +0.43 / +0.20 m/s | 31 mm/d |
| | e5b | 20/20 | 7/20 | 18/20 | 7.8 | d5 | +3.95 / +1.61 / +0.45 / +0.21 m/s | 38 mm/d |
| bay_low msl -1500 Pa, 15.5-20.5 N 85.5-90.5 E | v2 | 20/20 | 4/20 | 18/20 | 5.9 | d8 | -1479 / -893 / -414 / -290 Pa | 44 mm/d |
| | e5b | 20/20 | 6/20 | 13/20 | 10.7 | d7 | -1530 / -751 / -325 / -226 Pa | 81 mm/d |

Reproduce on any weights: `python -m training.dashboard_presets <checkpoint> ...` (model lane tool; read-only for you).

## 3. Demo guidance from the model lane
- **Best live demo:** the heatwave (local on day 1, spreads on every window, persists through day 10 with v2), then
  warm_ocean. Default the first-load shock to one of these so the judges immediately see "starts local, then spreads".
- **Bay of Bengal low:** the visible effect is in **rain** (switch the display to precipitation automatically for
  this preset, as you already do for winds); pressure itself stays near the box.
- **monsoon_stall is the model's honest weak spot:** local and stable, but it spreads on only ~5/20 windows, in the
  raw model too (not a display or post-processing artefact). Say so in the Shock Lab when its "spreads" check fails;
  don't hide it.
- The forecast beyond day 10 (your 15/20-day option) is outside what was trained and scored; keep the
  "experimental" label.

## 4. Model card: real numbers
- `eval/scores.csv` is git-ignored local output. With real weights in `weights/`, regenerate it:
  `python -m eval.score --data data/india_climate_train.nc --stats data/stats.json --twin`. Never ship the old
  persistence-stub file. If it is missing, say "run this command" (as now), don't show stub numbers.
- Skill vs the better of persistence and climatology (what the judge scores), 100 held-out windows, for the copy:

| Lead | t2m | tp | u10 | v10 | msl | tcwv |
|---|---|---|---|---|---|---|
| v2 day 1 | +0.148 | +0.199 | +0.290 | +0.319 | +0.325 | +0.297 |
| v2 day 4 | +0.056 | +0.027 | +0.043 | +0.081 | +0.020 | +0.120 |
| v2 day 8 | -0.021 | +0.001 | -0.022 | -0.013 | +0.006 | +0.008 |
| e5b day 1 | +0.127 | +0.166 | +0.259 | +0.292 | +0.292 | +0.266 |
| e5b day 4 | +0.046 | -0.078 | +0.032 | +0.082 | +0.014 | +0.098 |
| e5b day 8 | -0.004 | -0.115 | -0.027 | -0.014 | -0.007 | -0.008 |

- Optional "How the model was built" panel: the experiment ladder is real, committed data in the repo root
  (`experiments.csv`, `kit_scores.csv`): E1 one-step U-Net 28.1/70 -> E4 multi-step fine-tune 60.3 -> E5b
  gradient-magnitude anti-blur loss 61.1 -> + post-processing 65.8 (100 windows) -> v2 ensemble + shock-safe post
  67.1. Rejected runs are in the log with reasons (input noise, wider model, all-years retrain): good for "what did you
  try" questions. Read these files, don't copy numbers into code.
- Known limits for the card (keep them visible): monsoon_stall propagation (above); temperature beyond ~day 5 leans on
  climatology (blend capped at 0.5, so long-range shock effects are damped by up to half); rain peaks are smoothed
  (e.g. 2019-07-15 day-3 forecast peak 37.5 vs 93.6 mm/day observed with e5b; check the number again on the live
  model rather than hard-coding it); scores vary +-1-2 points between windows.

## 5. Boundaries (unchanged)
- Don't edit `twin_model/`, `training/`, `weights/`, `serve.py`, `twin.py`, `postproc.py`, `eval/`.
- Don't import model-lane code into `dashboard/` (the dashboard must run against the server only). Computing kit-style
  checks from fetched rollouts inside `dashboard/` is fine (the Shock Lab already does).
- If you find a model behaviour that looks wrong on the map (e.g. a shock that teleports, a NaN, a jump between days),
  note window date, preset and view in docs/STATUS.md for Aether instead of working around it in the UI.
