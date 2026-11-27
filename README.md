# Devforge-MonsoonVerse

Digital twin of Indian daily weather for **Multiverse of Monsoon (Track 2)**: a model that rolls the
monsoon forward 10 days from 3 input days and responds causally to "what-if" shocks.
The API contract is the organisers' `R1_CONTRACT.md` (kit: https://github.com/OfficialAbhinavSingh/yug-monsoon-kit);
our summary is in [docs/CONTRACT.md](docs/CONTRACT.md). Team progress: [docs/PROGRESS.md](docs/PROGRESS.md).

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt   # Linux: add --extra-index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/OfficialAbhinavSingh/yug-monsoon-kit   # organisers' kit (scoring, validator)

python serve.py --port 8000                               # start the server (blocks: run the rest in a 2nd terminal, venv activated)
python tools/validate.py --url http://127.0.0.1:8000      # our pre-check
python -m monsoon_r1.schema_validator http://127.0.0.1:8000   # organisers' validator
python -m pytest -q                                       # all tests
pip install -r dashboard/requirements.txt && streamlit run dashboard/app.py   # dashboard (needs data/ + running server)
```

Use `127.0.0.1`, not `localhost`, on Windows (localhost adds ~2 s per request).

## Training data (never commit it)

Download `india_climate_train.nc` and `stats.json` from the kit's release
https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/releases/tag/v1.0.0 into `data/` (gitignored).
Check them against `SHA256SUMS.txt` from the same release.

## Data and baselines

Daily gridded weather fields over India, 2005–2021, on a 1° 32×36 grid with 6 channels (t2m, tp, u10, v10, msl, tcwv).
We train on 2005–2018 and validate on 2019–2021. Persistence is the better reference at day 1; train-year climatology
is better from day 3 on, for every channel. Details, the baseline table and the figures are in [docs/DATA.md](docs/DATA.md);
the impact maths is in [docs/IMPACT.md](docs/IMPACT.md).

![Baseline RMSE vs lead](docs/img/baseline_rmse_vs_lead.png)

## Model and how to train it

The served model is `aether-unet-v21-ens4`: four residual U-Nets (2 M parameters each, trained identically with seeds
0-3) whose predicted daily change is averaged at every step, plus post-processing fitted on training years
(climatology blend per lead, capped at 0.5; rain quantile map per lead, stretch capped at 1.5×). Design, results,
shock behaviour and limits: [docs/MODEL_V21.md](docs/MODEL_V21.md) and [docs/MODEL_V2.md](docs/MODEL_V2.md); every experiment and why it was kept or rejected:
`experiments.csv`, `kit_scores.csv`.

Training needs a GPU for comfort (about 10 minutes per model on an RTX 5070 laptop GPU; CPU works, slowly) and the
data in `data/`. Train-only packages: `torch`, `xarray`, `netCDF4`; scoring also uses the organisers' kit
(`pip install --no-deps -e <kit clone>`, gives `monsoon_r1`).

```bash
# 1. one-step training (30 epochs; checkpoint picked by 8-day rollout score), per seed
python -m training.train --data data/india_climate_train.nc --stage onestep --epochs 30 --seed 0 --select old --run-name e1_onestep
# 2. multi-step fine-tune on 2-, 4-, 8-day rollouts with the gradient-magnitude anti-blur loss (picked by kit score)
python -m training.train --data data/india_climate_train.nc --stage multistep --k 2,4,8 --epochs 5 \
    --grad-mode mag --grad-weight 0.3 --resume runs/e1_onestep/best.pt --run-name e5b_gradmag
#    (repeat 1-2 with --seed 1, 2, 3 -> runs/e9b_seed1_multi, runs/s2_multi, runs/s3_multi)
# 3. ensemble the four seeds, fit post-processing on 2005-2018, apply the caps
python -m training.make_ensemble runs/e5b_gradmag/best.pt runs/e9b_seed1_multi/best.pt runs/s2_multi/best.pt runs/s3_multi/best.pt --out runs/ens4/best.pt
python -m training.fit_post runs/ens4/best.pt --out runs/ens4/best_post.pt
python -m training.post_variants runs/ens4/best_post.pt --out runs/ens4/pv     # -> cap05_slope15.pt
# 4. score (kit scoring code): 100 held-out windows, real post-2023 windows, shock behaviour
python -m training.kit_eval runs/ens4/pv/cap05_slope15.pt --windows 100
python -m training.eval_real runs/ens4/pv/cap05_slope15.pt --fetch 2023-07-20     # downloads real windows
python -m training.shock_lab runs/ens4/pv/cap05_slope15.pt --windows 8 --gallery
# 5. export for the server (docs/WEIGHTS.md): model.onnx, TorchScript model.pt, post.npz, model_meta.json
python -m training.export runs/ens4/pv/cap05_slope15.pt --out export --name aether-unet-v21-ens4
```

Tests for the model lane: `python -m pytest tests/test_norm.py tests/test_model.py tests/test_twin_model.py
tests/test_training.py tests/test_post.py tests/test_ensemble.py -q` (synthetic data, CPU, seconds).

## API

- `POST /health` → `{"status": "ready", "model_name": ..., "backend": "onnx" | "torchscript" | "persistence", "post": "loaded" | "absent" | ..., "load_error": ...}`
- `POST /simulate` with `{"init_states": [3,6,32,36], "doy_sincos": [sin, cos], "steps": 10, "perturbation": null | [6,32,36]}`
  → `{"rollout": [steps,6,32,36], "inference_ms": ...}`. Physical units; channels t2m, tp, u10, v10, msl, tcwv.

## Layout

| Path | What | Owner |
|---|---|---|
| `serve.py` | FastAPI server, contract checks, persistence fallback | Vishwak |
| `twin.py` | `Twin.rollout()`; loads `weights/model.onnx` (else TorchScript `model.pt`, else persistence), applies `weights/post.npz` via `postproc.py` | Vishwak |
| `common.py` | Shared constants: channels, grid, shock boxes, validity clamps | Vishwak |
| `eval/score.py` | Judge-identical forecast scoring (uses the kit's `monsoon_r1.scoring`) | Vishwak |
| `eval/shock_test.py` | warm_ocean / monsoon_stall / null shock causality checks | Vishwak |
| `eval/BASELINE.md` | Persistence baseline scores the model must beat | Vishwak |
| `weights/` | Model weights, plain git files, no Git LFS (only Vishwak commits here) | Vishwak |
| `docs/WEIGHTS.md` | How to export the model so the server loads it | Aether → Vishwak |
| `tools/validate.py`, `scripts/` | Pre-checks, repo scan, clean-clone test | Vishwak |
| `twin_model/` | Model package: residual U-Net, ensemble, normalisation (log1p rain), post-processing, reference `Twin` for checkpoints (not imported by the server) | Aether |
| `training/` | Data windows, losses, training CLI, ensemble, post-processing fit, export, kit-faithful scoring (`kit_eval`, `eval_real`, `shock_lab`) | Aether |
| `data_tools/` | Data loader (year split, 3-day windows), baselines, README figures | Junaid (built by Vishwak) |
| `impact/` | Crop-stress index per farming region (rain deficit, hot days); dashboard Impact tab | Junaid (built by Vishwak) |
| `dashboard/` | Streamlit app: live maps, shock painter, Shock Lab, ensemble, model card (`streamlit run dashboard/app.py`) | Akash (built by Vishwak) |

## Working rules

One branch per task (`feat/<name>-<task>`), pull request into `main`, reviewed by someone on the other tool.
Never commit data files. `serve.py` may import only torch, numpy, scipy, pandas, xarray, scikit-learn,
lightgbm, onnxruntime, fastapi, flask, uvicorn, gunicorn.
