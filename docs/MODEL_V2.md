# aether-unet-v2-ens2: candidate replacing aether-unet-e5b (branch `feat/aether-v2`)

Status: **candidate, verified on Aether's machine; Vishwak re-verifies through `serve.py` before main.**
Cut-off agreed: 06:30. All numbers use the kit's own scoring functions (`monsoon_r1.scoring`).

## What changed (and what did not)
- **Model:** 2-member ensemble of the existing U-Nets, e5b (seed 0) + e9b (seed 1), same recipe; at every step the two
  members' predicted daily change is averaged (`twin_model.model.EnsembleUNet`). Trained on 2005-2018 only.
- **Post-processing (same `post.npz` format, `postproc.save_npz`):** refit for the ensemble on 2005-2018, then
  - climatology blend weights **capped at 0.5** (was up to 0.83 for t2m at day 10): shocks no longer fade to nothing;
  - rain quantile-map **segment slope capped at 1.5** (was up to ~2): stops rain *differences* being stretched, which
    caused 6-11 std responses to painted shocks.
  Both are still deterministic functions of the rollout and doy only: zero shock = no shock stays bit-identical.
- **Unchanged:** API, hand-over format, export wrapper (doy shifted by -1 day), allowed packages, plain-git weights.
  `serve.py`, `twin.py`, `postproc.py`, `eval/`, `dashboard/` untouched.

## Acceptance bar
| Check | Bar | v2 candidate | e5b (main) |
|---|---|---|---|
| kit_eval, 100 held-out windows 2019-2021 | >= 65.8 | **67.06** | 65.78 |
| real 2023 windows (4) | >= 55.3 | **59.11** | 55.31 |
| real 2023-2025 windows (12, extra check) | - | **60.26** | 58.40 |
| judge simulation through serve.py (4 x 2023 + 1 x 2021 window + 3 shocks) | - | **64.7** | 61.0 |
| hard gates (beats persistence d4, validity, localised shock) | pass | pass | pass |
| latency, 10-step rollout, onnx, 4 threads | < 1 s | **47 ms** in-process (see note) | 25 ms |

Verified through the unchanged `serve.py` + `twin.py` + `postproc.py` with the four files in `weights/`: `/health`
backend onnx, `model_name: aether-unet-v2-ens2`, `post: loaded`; `tools/validate.py` PASS; kit `schema_validator`
all checks passed; `scripts/repo_scan_lite.py` PASS; determinism and zero shock exact. Export equivalence: ONNX vs
training rollout 1.5e-7 relative over 10 days, TorchScript identical, ONNX loop + `postproc` vs reference 0.016 Pa.

Latency note: `twin.rollout` takes 47 ms in-process (onnxruntime with `twin.py`'s exact session options). When the
server ran as a background process on Aether's laptop it measured ~355 ms inference / ~400 ms per call; the same
happened to the reference PyTorch model, so it is the laptop scheduling background processes on efficiency cores,
not the model. Still far inside the 1 s target and the 5 s limit. Please re-time on your machine.

## Skill per lead and channel (v2, 100 held-out windows; skill vs the better of persistence and climatology)
| Lead | t2m | tp | u10 | v10 | msl | tcwv |
|---|---|---|---|---|---|---|
| day 1 | +0.148 | +0.199 | +0.290 | +0.319 | +0.325 | +0.297 |
| day 4 | +0.056 | +0.027 | +0.043 | +0.081 | +0.020 | +0.120 |
| day 8 | -0.021 | +0.001 | -0.022 | -0.013 | +0.006 | +0.008 |

e5b for comparison: tp day 4 -0.078, day 8 -0.115 (rain at days 4-8 was below climatology; now level or above).
Day-8 variance ratio (anti-blur, needs >= 0.40): t2m 0.99, tp 0.57, u10 0.84, v10 0.79, msl 0.90, tcwv 0.91.
Model card: `eval/scores.csv` is git-ignored local output; regenerate it with the real weights in place:
`python -m eval.score --data data/india_climate_train.nc --stats data/stats.json --twin`.

## Shock gallery (v2, 20 windows: 8 held-out 2019-2021 + 12 real 2023-2025)
Box = lat0..lat1 N, lon0..lon1 E (grid cell centres on .5 degrees). "Spreads" = kit propagation check (energy outside
the box neighbourhood grows from day 1 to days 4-8, no teleport). Peak = largest response in channel std, any day.

| Shock (paint this) | Channel, box, size | Local day 1 | Spreads | Stable (< 5 std) | Peak std | Fades by |
|---|---|---|---|---|---|---|
| **Heatwave NW India** (cleanest demo) | t2m +4 K, 25-32 N 72-82 E | 20/20 | **19/20** | 20/20 | 1.2 | does not fade (day 8 in-box +1.8 K) |
| **Warm Bay of Bengal** (kit `warm_ocean`) | t2m +3 K, tcwv +15%, 8-20 N 82-94 E | 20/20 | 13/20 | 20/20 | 3.3 | day 9 (7/20), else lasts |
| Westerly burst | u10 +5 m/s, 10-16 N 66-74 E | 20/20 | 6/20 | 20/20 | 2.7 | day 5 |
| Bay depression | msl -1000 Pa, 15-21 N 86-92 E | 20/20 | 4/20 | 20/20 | 4.7 (rain) | day 8 |
| Deep Bay depression | msl -1500 Pa, same box | 20/20 | 3/20 | 17/20 | 5.6 (rain) | day 8 |
| Dry spell central India | tcwv -30%, 18-26 N 76-84 E | 20/20 | 7/20 | 20/20 | 3.4 | day 7 |
| Moist surge west coast | tcwv +20%, 10-20 N 70-76 E | 18/20 | 5/20 | 20/20 | 2.1 | day 6 |
| Monsoon stall (kit `monsoon_stall`) | u10, v10 -40%, 8-20 N 65-75 E | 20/20 | 5/20 | 20/20 | 3.5 | day 5 |

Compared with e5b on the same 20 windows: peaks were up to 11 std (rain), the deep depression was stable on only
12/20, and the warm-ocean signal faded on 9/20 windows. Best live demos: the heatwave (local, spreads, persists) and
the warm Bay of Bengal. The depressions show a strong, physically sensible rain response but stay near the box.

## Known limits (for the brief and Q&A)
- **monsoon_stall rarely propagates (~5/20 windows).** Not a post-processing effect: the raw model alone spreads it on
  ~5/12 windows (outside-energy growth 1.14, the kit needs > 1.5). A weakened wind field decays rather than spreading.
  Fixing it needs retraining (e.g. a propagation-aware loss or a shallower net), beyond the 06:30 cut-off. Expect to lose
  ~1.9 causality points when the judge's window is one where it fails.
- Moisture shocks (tcwv) can leak a little energy far from the box on day 1 (teleport check fails on 6/20 for the west
  coast surge): the 3-level U-Net sees the whole domain in one step, and GroupNorm normalises over the whole map.
- t2m beyond ~day 5 still leans on climatology (blend now capped at 0.5), so long-range shock effects are damped by up
  to half: honest behaviour, worth saying.
- Scores vary ~+-1-2 points between windows and seeds; the judge uses 5 windows.

## Rejected this round (logged in `experiments.csv`)
- Plain ensemble without the caps: 68.06 held-out but 57.05 on 12 real windows (worse than e5b's 58.40).
- e5b single model with the same caps: 65.73 held-out (just under the bar), 60.95 real.
- All-years retrain (2005-2021, fixed schedule): 46.38 on 2023, 52.94 on 12 real windows.
- Blend cap alone (no slope cap): does not fix stability.

## Reproduce
```
python -m training.make_ensemble runs/e5b_gradmag/best.pt runs/e9b_seed1_multi/best.pt --out runs/ens2_e5b_e9b/best.pt
python -m training.fit_post runs/ens2_e5b_e9b/best.pt --out runs/ens2_e5b_e9b/best_post.pt
python -m training.post_variants runs/ens2_e5b_e9b/best_post.pt --out runs/ens2_e5b_e9b/pv     # -> cap05_slope15.pt
python -m training.export runs/ens2_e5b_e9b/pv/cap05_slope15.pt --out export --name aether-unet-v2-ens2
python -m training.kit_eval runs/ens2_e5b_e9b/pv/cap05_slope15.pt --windows 100
python -m training.eval_real runs/ens2_e5b_e9b/pv/cap05_slope15.pt [--only 2023]
python -m training.shock_lab runs/ens2_e5b_e9b/pv/cap05_slope15.pt --windows 8 --gallery
```
