# Multiverse of Monsoon: a live digital twin of India's weather

**Track 2, Multiverse of Monsoon.** A learned weather model for India that takes the **3 most recent days** of six
weather fields, **forecasts the next 10 days**, and answers **"what if?"**: paint a change on the map (a warmer Bay of
Bengal, weaker monsoon winds, a heatwave) and the twin shows how that change **starts locally and then spreads**.

It runs on a laptop CPU in well under a second, needs no internet, and serves the organisers' API from `serve.py`.
A dashboard (`dashboard/`) lets anyone paint shocks, play the forecast day by day and compare it with what really
happened.

New here? Read [docs/PROJECT_EXPLAINED.md](docs/PROJECT_EXPLAINED.md) (the whole project in plain words) and the
two-page brief [docs/brief.pdf](docs/brief.pdf).

---

## Results

All numbers use the **organisers' own scoring code** (`monsoon_r1.scoring`), 70 automatic points.

| Test | Score |
|---|---|
| Held-out years 2019–2021, 100 windows | **67.7 / 70** (RMSE skill 30.0/30, heavy rain 14.5/15, anti-blur 10/10, shocks 13.1/15) |
| Real 2023 weather, 4 windows (fetched with the organisers' pipeline) | **63.6 / 70** |
| Real 2023–2025 weather, 12 windows | **62.9 / 70** |
| Full judge simulation through our own server (5 windows + 3 shocks) | **68.1 / 70** |
| Hard gates for Round 2 (beats persistence at day 4, valid output, local shock response) | **all pass** |
| Organisers' repo scan on a fresh clone of this repository | **15 / 15**, no flags |

For comparison, "tomorrow = today" (persistence) scores 19.6. The model and how each score was produced:
[docs/MODEL_V21.md](docs/MODEL_V21.md). Every experiment, including the ones we threw away, is logged in
`experiments.csv` and `kit_scores.csv`.

**How we checked.** The model lane scored each checkpoint on its training machine; we then re-scored the exported
weights **through `serve.py`**, the way the judges call it, and got the same numbers (67.66 vs 67.66). Before every
merge: `tools/validate.py`, the organisers' schema validator, our shock tests, the full test suite (253 tests,
including tests that start the real server and drive the dashboard) and a **fresh clone from GitHub** into an empty
folder (install, cold start, validators, repo scan).

---

## How it works

```
 3 real days            one day ahead, 10 times                         finished 10-day rollout
 [3, 6, 32, 36]  --->  4 residual U-Nets, change averaged  --->  ...  --->  post-processing  --->  [10, 6, 32, 36]
 (+ optional shock      (each step feeds back as input)                   (climatology blend per day,
  added to the last day)                                                   rain quantile map per day)
```

- **Data:** daily fields over India on a 1° grid (32 × 36 cells, about 110 km each), 2005–2021: 2 m temperature,
  rain, east–west and north–south wind, sea-level pressure, column water vapour. Train 2005–2018, test 2019–2021,
  split by year so neighbouring days cannot leak.
- **Model:** a residual U-Net (about 2 M parameters) predicts the change from one day to the next. Four copies trained
  with different random seeds are averaged at every step (`aether-unet-v21-ens4`). Rain is learned in log space and
  can never go negative.
- **Training:** first one day ahead, then fine-tuned on 2-, 4- and 8-day rollouts with an extra loss that punishes
  blurry maps, so the forecast keeps its detail instead of smearing towards the average.
- **Post-processing:** a per-day blend towards the normal weather for that date (capped at half) and a per-day rain
  correction, both fitted on training years only, applied to the finished rollout.
- **Shocks:** a "perturbation" is added to the last real day; a zero perturbation gives a bit-identical forecast.
  The judges score two shocks (`warm_ocean`, `monsoon_stall`) for "local on day 1, spreading by days 4–8, stable".

Full explanation, with the experiment ladder from 28/70 to 67.7/70: [docs/PROJECT_EXPLAINED.md](docs/PROJECT_EXPLAINED.md).

---

## Run it

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate        Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt           # server only (Linux: add --extra-index-url https://download.pytorch.org/whl/cpu)

python serve.py --port 8000               # the twin server (keep it running; use a second terminal below)
python tools/validate.py --url http://127.0.0.1:8000
```

On Windows use `127.0.0.1`, not `localhost` (localhost adds about 2 s per request).

**Dashboard** (needs the training data in `data/` and the server running):

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py            # opens http://127.0.0.1:8501
```

**Development** (tests, organisers' validator and scoring):

```bash
pip install -r requirements-dev.txt
pip install git+https://github.com/OfficialAbhinavSingh/yug-monsoon-kit     # gives monsoon_r1
python -m monsoon_r1.schema_validator http://127.0.0.1:8000
python -m pytest tests -q
```

**Data (never committed):** download `india_climate_train.nc` and `stats.json` from the kit's release
https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/releases/tag/v1.0.0 into `data/` and check them against
`SHA256SUMS.txt`. The server never reads data files; the dashboard uses them for real starting days.

---

## The dashboard in one minute

1. **Pick a shock** in the sidebar (it opens on a north-India heatwave), or **click the map** to drop one anywhere.
   Live mode re-runs the twin by itself on every change: two live calls, about a quarter of a second each.
2. The **big map** shows the forecast with the shock painted (gold box). Switch the layer (rain, temperature, wind,
   pressure, moisture) and the view: *With shock*, *No shock*, *Effect* (the difference) or *Observed* (what really
   happened, from the archive).
3. Press **Play** under the map to animate all days, from the 3 real days into the forecast. In the *Effect* view a
   dotted ring marks the shock's spreading front. **Step all** (key P) walks the map, the side panel and the day strip
   through the days together; **Back / Next** (keys B / N) step one day.
4. The side panel explains the day in plain sentences, measured from the live run: how far the change has spread,
   how much is left inside the box, and how the forecast compares with the real day (peak, detail kept, error).
5. Other tabs: **Shock Lab** (the judges' own checks, live), **Ensemble** (how sure the forecast is), **Model card**
   (skill and known limits), **Impact** (crop stress by farming region).

Every number on screen comes from a live call to the server in this session; nothing is pre-rendered. The map works
offline (no map tiles), for a projector without internet.

---

## What to expect from shocks (measured on the final model)

Measured by the model lane on 20 windows (8 held-out 2019–2021, 12 real 2023–2025) with the dashboard's exact boxes
([docs/MODEL_V21.md](docs/MODEL_V21.md)):

| Shock | Painted | Local on day 1 | Spreads by days 4–8 | Fades |
|---|---|---|---|---|
| `warm_ocean` (judged) | Bay of Bengal +3 K, +15 % moisture | 20 / 20 | 16 / 20 | mostly lasts to day 10 |
| `monsoon_stall` (judged) | Arabian Sea winds −40 % | 20 / 20 | **4 / 20** (known weak spot) | by day 5 |
| Heatwave | north India +4 K | 20 / 20 | 20 / 20 | does not fade in 10 days |
| Westerly burst | Arabian Sea wind +5 m/s | 20 / 20 | 6 / 20 | by day 5 |
| Bay of Bengal low | pressure −1500 Pa (effect shows in rain) | 20 / 20 | 6 / 20 | by day 10 |

All five stay stable (under the judges' 5-standard-deviation limit) on every window. The dashboard measures this live
for whatever you paint, so the table is a guide, not a promise.

## Known limits

- Skill falls with lead time: by day 8 the model only matches the normal weather for the date, and the blend towards
  it damps long-range shock effects by up to half.
- `monsoon_stall` rarely spreads (4 of 20 windows): a weakened wind field decays instead of travelling (about 2
  shock points at risk when the judges' window is one of those).
- Single heavy-rain peaks are smoothed, more at longer leads (day-8 rain keeps about 68 % of the real spatial
  variation, inside the judges' band; temperature keeps about 99 %).
- Temperature is a daily mean, so "hot days" are anomalies, not heatwave maxima. The grid is 1°; region boxes are
  approximate.
- Scores vary by about ±1–2 points between windows; the judges use 5.

---

## API (organisers' contract, frozen)

- `POST /health` → `{"status": "ready", "model_name": ..., "backend": "onnx" | "torchscript" | "persistence",
  "post": "loaded" | "absent" | ..., "load_error": ...}`, ready within 120 s (about 5 s here).
- `POST /simulate` with `{"init_states": [3,6,32,36], "doy_sincos": [sin, cos], "steps": 10,
  "perturbation": null | [6,32,36]}` → `{"rollout": [steps,6,32,36], "inference_ms": ...}`. Physical units;
  channels t2m (K), tp (mm/day), u10, v10 (m/s), msl (Pa), tcwv (kg/m²). Each call well under 5 s (about 0.1–0.3 s).

Contract summary: [docs/CONTRACT.md](docs/CONTRACT.md). Weight format: [docs/WEIGHTS.md](docs/WEIGHTS.md).

---

## Model and how to train it

The served model is `aether-unet-v21-ens4`: four residual U-Nets (2 M parameters each, trained identically with seeds
0–3) whose predicted daily change is averaged at every step, plus post-processing fitted on training years
(climatology blend per lead, capped at 0.5; rain quantile map per lead, stretch capped at 1.5×). Design, results,
shock behaviour and limits: [docs/MODEL_V21.md](docs/MODEL_V21.md) and [docs/MODEL_V2.md](docs/MODEL_V2.md).

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

---

## Repository map

| Path | What it is |
|---|---|
| `serve.py` | The server the judges run: FastAPI, contract checks, never crashes (persistence fallback) |
| `twin.py`, `postproc.py`, `common.py` | Runs the exported model day by day (onnx, else TorchScript, else persistence), applies post-processing, shared shapes and validity clamps |
| `weights/` | The final model: `model.onnx`, `model.pt`, `post.npz`, `model_meta.json` (plain git files, no LFS) |
| `twin_model/`, `training/` | The model and its training, ensembling, post-processing fit, export and kit-faithful scoring (not imported by the server) |
| `eval/` | Our judge-identical scorer and shock tests (`eval/score.py`, `eval/shock_test.py`, `eval/BASELINE.md`) |
| `dashboard/` | Streamlit app (own requirements; talks to the server only through `dashboard/client.py`) |
| `impact/` | Crop-stress index per farming region (rain deficit, hot days), used by the Impact tab |
| `data_tools/` | Data loader, baselines and the figures in [docs/DATA.md](docs/DATA.md) |
| `tools/validate.py`, `scripts/` | Pre-submission checks: validator, repo scan, clean-clone test |
| `tests/` | 253 tests: contract, server, model, training, scoring, shocks, impact, dashboard (live) |
| `docs/` | Brief, explainers, model reports, contract, demo script, judge Q&A: see [docs/README.md](docs/README.md) |
| `experiments.csv`, `kit_scores.csv` | Every training run and score, kept or rejected, with the reason |

## Team

| Area | Owner |
|---|---|
| Model and training (`twin_model/`, `training/`) | Aether |
| Server, weights, scoring, shock tests, repo checks, clean-clone test | Vishwak |
| Data tools, impact module, README, brief | Junaid |
| Dashboard | Akash |

Built with AI pair-programmers (Claude). Rules we kept: never commit data; the server imports only torch, numpy,
scipy, pandas, xarray, scikit-learn, lightgbm, onnxruntime, fastapi, flask, uvicorn, gunicorn; never loosen a failing
check; every merged change re-verified through the server.
