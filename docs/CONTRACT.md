# Shared contract

**Authority:** the kit's `docs/R1_CONTRACT.md` (yug-monsoon-kit). It wins on any conflict; this file records how we meet it.

## Confirmed against the kit
- Latitude is south -> north (6.5..37.5), lon west -> east: confirmed from `india_climate_train.nc` coords and `contract.py`.
- Shocks are relative to current values of `init_states[-1]`: warm_ocean +3 K t2m and +15% of current tcwv (Bay of Bengal box
  lat 8..20, lon 82..94); monsoon_stall -40% of current u10/v10 (Arabian Sea box lat 8..20, lon 65..75); null = zeros.
- Kit `MAX_STEPS = 14`, scored rollouts are 10 steps. We accept 1..60.
- Judge std per channel comes from `stats.json` (kit `STD_DEFAULT` is only a fallback).

## Code interface
- `common.py`: channel order, grid (`LATS`, `LONS`), validity bounds, `BOXES`, `box_mask()`, `make_shock()`, `doy_sincos()`, `enforce_validity()`.
- `twin.py`: `Twin().rollout(init_states, doy_sincos, steps=10, perturbation=None) -> [steps,6,32,36]`. Signature frozen.
  Backend "model" if `weights/model.pt` (TorchScript, see `docs/WEIGHTS.md`) loads and warms up, otherwise "persistence".

## HTTP API (serve.py, repo root, 0.0.0.0, `python serve.py --port 8000`)
- `POST /health` (GET also works) -> `{"status": "ready", "model_name": str, "backend": "persistence"|"model", "load_error", "nonfinite_fallbacks", "torch", "fallbacks"}`.
  The model loads and warms up before the server listens, so the cold start falls inside the judge's /health polling (120 s).
- `POST /simulate` body: `{"init_states": [3][6][32][36], "doy_sincos": [s, c], "steps": 10, "perturbation": null | [6][32][36]}`
  -> `{"rollout": [steps][6][32][36], "inference_ms": float}`. Arrays are nested lists (canonical); `{"b64","shape"}` input is tolerated.
  Bad input -> 422. No redirects (`redirect_slashes=False`). `doy_sincos` is for the last input day; step k is doy+k.

## Scoring (kit)
Skill 45 = lat-weighted RMSE skill vs the better of persistence/climatology at leads 1, 4, 8 (30) + heavy-rain CSI at 10 mm/day (15).
Dynamics 25 = anti-blur, day-8 spatial variance 40..250% of truth (10) + shock causality (15: non-zero, >= 60% of day-1 response
energy within 3 cells of the box, spread <= 8 cells/day, never above 5 std, null shock within 1e-3 std).
Repo 15 (`monsoon_r1.repo_scan.scan`) + brief 15.

## Rules
- Channels: t2m K, tp mm/day, u10 m/s, v10 m/s, msl Pa, tcwv kg/m2. Grid 32x36, 1 deg.
- Valid output: finite only, tp >= 0, t2m within -40..60 C.
- Limits: /simulate < 5 s on 4 CPU cores, 6 GB RAM, no GPU, no internet. Our target: < 1 s.
- serve.py may import only: torch (CPU), numpy, scipy, pandas, xarray, scikit-learn, lightgbm, onnxruntime, fastapi, flask,
  uvicorn, gunicorn (+ our own modules). Never `monsoon_r1` (not in the judge sandbox).
- repo_scan treats every .py that mentions `/simulate` (and what it imports) as inference code: no weather-API names and fewer than
  3 date literals in those files.
- Never commit data (`data/`, `*.nc`). Weights go in `weights/` (needs a checkpoint > 1 KB in the repo or anti-cheat scores 0).
- Banned: weather APIs / internet, hardcoded dates or lookup tables, loosening a check to pass it, silent try/except returning zeros.
