# aether-unet-v21-ens4: 4-seed ensemble (candidate replacing aether-unet-v2-ens2)

Status: **candidate on `feat/aether-v3`, verified on Aether's machine; Vishwak re-verifies through `serve.py` before main.**
Everything else is identical to v2 (`docs/MODEL_V2.md`): same network, same training recipe, same post-processing
(climatology blend capped at 0.5, rain-map slope capped at 1.5), same export format and wrapper. The only change: **4**
identically trained U-Nets (seeds 0, 1, 2, 3) averaged per step instead of 2.

## Acceptance bar (kit scoring code)
| Check | Bar (v2 on main) | v2.1 | v2 |
|---|---|---|---|
| kit_eval, 100 held-out windows 2019-2021 | >= 67.06 | **67.66** | 67.06 |
| real 2023 windows (4) | >= 59.11 | **63.64** | 59.11 |
| real 2023-2025 windows (12) | >= 60.26 | **62.94** | 60.26 |
| judge simulation through serve.py (4 x 2023 + 1 x 2021 + 3 shocks) | - | **68.1** (RMSE 30, CSI 15, blur 10, causality 13.1) | 64.7 |
| hard gates | pass | pass | pass |
| latency, 10 steps, onnx, server session options | < 1 s | **100 ms** in-process | 50 ms |

Verified through the unchanged `serve.py` + `twin.py` + `postproc.py` with the four files in `weights/`: `/health` backend
onnx, `model_name: aether-unet-v21-ens4`, `post: loaded`, ready in 1.6 s; `tools/validate.py` PASS; kit `schema_validator`
all checks passed; `scripts/repo_scan_lite.py` PASS; null shock exact; deterministic. Export equivalence: ONNX vs training
rollout 1.5e-7 relative over 10 days, TorchScript identical. Files: model.onnx 33.1 MB, model.pt 32.1 MB, post.npz 10.1 MB
(plain git, each under the 95 MB limit). Full test suite with these weights: see the commit message.

Latency note: as a background process on Aether's laptop (with the dashboard running) the server measured ~0.78 s per call;
in-process the rollout is 100 ms. Please re-time on your machine.

## Shock presets (dashboard boxes, 20 windows: 8 held-out + 12 real 2023-2025)
| Preset | Local d1 | Spreads | Stable | Peak std | Fades by | In-box change d1 / d4 / d8 / d10 |
|---|---|---|---|---|---|---|
| warm_ocean (kit) | 20/20 | **16/20** (v2 13) | 20/20 | 3.9 | lasts (d10 on 5/20) | t2m +2.64 / +1.60 / +1.41 / +1.38 K |
| monsoon_stall (kit) | 20/20 | 4/20 (v2 5) | 20/20 | 3.5 | d5 | u10 -0.86 / -0.38 / -0.20 / -0.12 m/s |
| heatwave +4 K | 20/20 | 20/20 | 20/20 | 0.9 | does not fade | +3.59 / +2.14 / +1.87 / +1.84 K |
| westerly_burst +5 m/s | 20/20 | 6/20 | 20/20 | 3.6 | d5 | +3.83 / +1.43 / +0.41 / +0.20 m/s |
| bay_low -1500 Pa | 20/20 | 6/20 | **20/20** (v2 18) | 5.0 | d10 | -1484 / -940 / -555 / -385 Pa |

Limits are unchanged from v2: monsoon_stall rarely spreads, day-8 skill about level with climatology, smoothed rain peaks,
ensemble-tab spread shrinks with lead day.

## Reproduce
```
# seeds 0-3: one-step (E1 recipe) then multistep k=2,4,8, grad-mode mag 0.3 (README "Model and how to train it")
python -m training.make_ensemble runs/e5b_gradmag/best.pt runs/e9b_seed1_multi/best.pt runs/s2_multi/best.pt runs/s3_multi/best.pt --out runs/ens4/best.pt
python -m training.fit_post runs/ens4/best.pt --out runs/ens4/best_post.pt
python -m training.post_variants runs/ens4/best_post.pt --out runs/ens4/pv
python -m training.export runs/ens4/pv/cap05_slope15.pt --out export --name aether-unet-v21-ens4
```
