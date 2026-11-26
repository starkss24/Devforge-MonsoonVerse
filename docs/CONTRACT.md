# Shared contract (mirror of the kit's R1_CONTRACT.md — the kit wins on any conflict)

## Code interface
- `common.py`: channel order, grid (`LATS`, `LONS`), validity bounds, `BOXES`, `box_mask()`, `make_shock()`, `doy_sincos()`, `enforce_validity()`. Import from here, never re-type constants.
- `twin.py`: `Twin().rollout(init_states, doy_sincos, steps=10, perturbation=None) -> [steps,6,32,36]`.
  Hour-0 backend is persistence. Signature is frozen.

## HTTP API (serve.py, repo root, 0.0.0.0, default port 8000)
- `POST /health` -> `{"status": "ready", "backend": "persistence"|"model", ...}`
- `POST /simulate` body: `{"init_states": [3][6][32][36], "doy_sincos": [s, c], "steps": 10, "perturbation": null | [6][32][36]}`
  -> `{"rollout": [steps][6][32][36], "inference_ms": float}`

## Rules
- Channels: t2m K, tp mm/day, u10 m/s, v10 m/s, msl Pa, tcwv kg/m2. Grid 32x36, 1°, lat 6.5–37.5, lon 65.5–100.5.
- Valid output: finite only, tp >= 0, t2m within -40..60 °C.
- Null/zero shock must match the unshocked run within 1e-3 × channel std (we aim for exactly 0).
- Limits: /health within 120 s; /simulate < 5 s on 4 CPU cores, 6 GB RAM, no GPU, no internet. Our target: < 1 s.
- serve.py may import only: torch (CPU), numpy, scipy, pandas, xarray, scikit-learn, lightgbm, onnxruntime, fastapi, flask, uvicorn (+ our own modules). Writes only to /tmp.
- Never commit data (`data/`, `*.nc`). Weights only via Git LFS in `weights/`.
- Banned: weather APIs / internet, hardcoded dates or lookup tables, loosening a check to pass it, silent try/except returning zeros.

## Kit files we do not have yet
`R1_CONTRACT.md`, `stats.json`, `india_climate_train.nc`, the schema validator, `repo_scan`. Code must take their paths as arguments and work on synthetic data until they arrive.
