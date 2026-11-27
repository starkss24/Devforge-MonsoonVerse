# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Hackathon "Multiverse of Monsoon" (Track 2): a learned digital twin of Indian weather, graded offline. Full plan: `Team Playbook - Multiverse of Monsoon (Track 2).pdf` (not part of the code; do not commit it). Grading: forecast skill 45, dynamics (no blur, shocks local-then-spreading) 25, repo check 15, brief 15.

The user is **Aether — model lead** (Claude pair): residual U-Net, losses, multi-step fine-tune, experiment log, weights. Teammates own other lanes; do not edit their areas unless asked:

| Area | Owner | Reviewer |
|---|---|---|
| Model and training code (`twin_model/`, `training/`) | Aether | Junaid |
| Scorer, shock tests, `serve.py`, `weights/` | Vishwak | Akash |
| Data scripts, impact module, README, brief | Junaid | Aether |
| Dashboard | Akash | Vishwak |

## Contract (from the kit's R1_CONTRACT.md — never change without telling the team)

- `POST /health` → `{"status": "ready", ...}`; `POST /simulate` → `{"rollout": ..., "inference_ms": ...}`.
- `init_states` `[3,6,32,36]`, oldest day first, physical units. `steps` is 10 in scored calls.
- `doy_sincos` = sin, cos of 2π·doy/365 for the **last input day**.
- `perturbation`: null or `[6,32,36]` physical deltas added to `init_states[-1]`.
- Output `rollout` `[steps,6,32,36]`; `rollout[k]` is day k+1 after the last input day.
- Channels in order: t2m K, tp mm/day, u10 m/s, v10 m/s, msl Pa, tcwv kg/m².
- Grid 32×36, 1°, lat 6.5–37.5, lon 65.5–100.5 (regional box → replicate/reflect padding, never circular).
- Validity: finite only, rain ≥ 0, t2m within −40..60 °C; a zero perturbation must equal no perturbation (within 1e-3 of each channel's std).
- Limits: `/health` within 120 s; each `/simulate` < 5 s on 4 CPU cores, 6 GB RAM, no GPU, no internet. Target < 1 s.
- Repo: `serve.py` at root, listens on `0.0.0.0`, weights inside repo (plain git files, no LFS: see `.gitattributes`), writes only to `/tmp`.

## Hard rules (reject these in review)

- No weather APIs, downloads, or internet at inference; no hardcoded dates or lookup tables of past weather.
- Never commit data (`data/`, `*.nc`) or pre-rendered dashboard images. Weights (`weights/model.onnx`, `model.pt`, `post.npz`, `model_meta.json`) are committed as plain git files (team decision, so a clone without `git lfs pull` still works); weight changes reach `main` only after Vishwak re-verifies them through `serve.py`.
- Anything `serve.py` imports (including `twin_model/`) may use only: torch (CPU), numpy, scipy, pandas, xarray, scikit-learn, lightgbm, onnxruntime, fastapi, flask, uvicorn. Never Streamlit/Plotly.
- Never "fix" a failing check by loosening it; never wrap errors in a silent try/except that returns zeros.
- Split data by year (train 2005–2018, validate 2019–2021), never by random day.

## Architecture

Serving path (team, Vishwak's lane): `serve.py` → root **`twin.py`** (`Twin.rollout`, frozen signature in `docs/CONTRACT.md`) → exported weights in `weights/` (`model.onnx` preferred, TorchScript `model.pt` fallback, persistence if neither loads). The weights hand-over contract is `docs/WEIGHTS.md`: a one-day `step(states [3,6,32,36] physical, doy_sincos [2] of the day being predicted) -> [6,32,36] physical`; `twin.py` runs the autoregressive loop and clamps. `common.py` holds shared shapes/clamps.

Model lane (Aether): research/training code, not imported by the server. Module interfaces are pinned in `docs/interfaces.md` — read it before touching `twin_model/` or `training/`.

- `twin_model/` — the model package used for training and evaluation: normalization (`norm.py`), U-Net + single step/rollout (`model.py`), output post-processing (`post.py`: climatology blend + rain quantile map), and a reference `Twin` (`twin.py`) that runs a full checkpoint (`.pt` dict, weights + norm + static + post). **Named `twin_model`, not `twin`, so it never shadows the root `twin.py`.**
- `training/` — data windows, losses, training CLI, `fit_post.py` (fits post-processing), `kit_eval.py` (scores a checkpoint with the kit's own scoring → projected points /70, appends to `kit_scores.csv`).
- Other lanes: `eval/` (scorer, shock tests), `data_tools/`, `impact/`, `dashboard/`, `scripts/`, `tools/validate.py`.

Post-processing (climatology blend per lead + rain quantile map per lead) applies to the finished rollout, so it lives in `weights/post.npz` and root `twin.py` applies it with `postproc.py` after its loop (never inside the one-day `step`). Write `post.npz` with `postproc.save_npz` (`training/export.py` does).

Models: the served model is `aether-unet-v21-ens4` (4-seed `EnsembleUNet`, same recipe and post as v2; 67.66 on 100 held-out, 63.64 on real 2023, 62.94 on 12 real windows, judge simulation 68.1; `docs/MODEL_V21.md`), merged after Vishwak's re-verification through `serve.py`. Before it: `aether-unet-v2-ens2` (2-seed `EnsembleUNet` + post with blend cap 0.5 and rain-map slope cap 1.5; merged 2026-11-27 after re-verification through `serve.py`: 67.06/70 on 100 held-out windows); the previous model was `aether-unet-e5b` (65.78), see `docs/MODEL_V2.md`. Judge new weights by `training/kit_eval.py` on 100 held-out windows AND `training/eval_real.py` on the real 2023-2025 windows in `data/eval2023/` (git-ignored); shock behaviour with `training/shock_lab.py`.

Key ML invariants (PyTorch traps):
- Rain lives in log1p space inside the model; denormalize in order: un-standardize, then `expm1`, then clamp ≥ 0.
- Inference must use `model.eval()` + `torch.no_grad()`; otherwise the zero-shock test breaks.
- Normalization stats and static maps are saved inside the checkpoint; the server never reads data files.
- Select checkpoints by rollout score (day-4/day-8 ratios + sharpness), not by training loss.
- Test inference on CPU with `torch.set_num_threads(4)`.

## Commands

Environment: project venv at `.venv` (CUDA 12.8+ PyTorch build, needed for the RTX 50-series).

```
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv\Scripts\python -m pytest tests -q                                  # all tests
.venv\Scripts\python -m pytest tests/test_twin_model.py::test_zero_shock -q    # single test
.venv\Scripts\python -m training.kit_eval runs/<run>/best_post.pt --windows 100  # official-style score /70
.venv\Scripts\python -m training.train --synthetic --stage onestep --epochs 1   # smoke run
.venv\Scripts\python -m training.train --data data/india_climate_train.nc --stage onestep
.venv\Scripts\python -m training.train --data data/india_climate_train.nc --stage multistep --k 2,4,8 --resume runs/<run>/best.pt
```
