# Hand-off prompt for Vishwak's agent

Paste everything below the line into a new Claude session, at the root of the team repo.

---

You are continuing work on a hackathon project already in progress. Read this whole brief before acting. I am **Vishwak**, owner of the **scorer, shock tests, `serve.py`, `weights/`, Git LFS, repo scan and the clean-clone test** (Claude pair, with Aether). Aether (model lead) has finished the model lane; this brief is their hand-off. Treat every number here as measured, not assumed. If something on disk disagrees with this brief, trust the disk and tell me.

## 1. The competition (Multiverse of Monsoon, Track 2)

We build a learned **digital twin of Indian weather**: given 3 days of 6 weather fields, roll the weather forward 10 days, and respond sensibly to "shocks" (perturbations) painted on the input.

- **Automated grader (Stage 1, 100 pts):** forecast skill 45, dynamics 25, repo check 15, brief 15. The top 10 teams per track go to desk review; 2 per track reach the finals, where judges paint new shocks live and ask about design and limits.
- **How the grader runs us:** clones our public repo into an **offline container (4 CPU cores, 6 GB RAM, no GPU, no internet)** and runs `python serve.py --port 8000`. It sends **8 blind requests: 5 real windows from dates after 2023-01-10, plus 3 shock requests**.
- **Kit:** https://github.com/OfficialAbhinavSingh/yug-monsoon-kit. Read `docs/R1_CONTRACT.md` and `monsoon_r1/scoring.py` there. The data files `india_climate_train.nc` and `stats.json` are on the kit's **Release page**, not in the repo.

### API contract (from R1_CONTRACT.md — never change it)
- `POST /health` → `{"status": "ready", ...}` within 120 s of start.
- `POST /simulate` body: `init_states` [3,6,32,36] oldest day first, physical units; `doy_sincos` = [sin, cos] of 2π·doy/365 for the **last input day**; `steps` (10 in scored calls); `perturbation` null or [6,32,36] physical deltas **added to init_states[-1]**. Returns `{"rollout": [steps,6,32,36], "inference_ms": ...}`; `rollout[k]` is day k+1 after the last input day. Nested JSON lists are canonical; base64 `{"b64","shape"}` is tolerated.
- Channels in order: **t2m K, tp mm/day, u10 m/s, v10 m/s, msl Pa, tcwv kg/m²**. Grid 32×36, 1°, lat 6.5–37.5, lon 65.5–100.5 (cell centres on .5°).
- Validity: finite only, rain ≥ 0, t2m within −40..60 °C. A zero perturbation must equal no perturbation (within 1e-3 of each channel's std). Each `/simulate` < 5 s.
- Repo: `serve.py` at the root, listens on `0.0.0.0`, weights inside the repo, writes only to `/tmp`.

### Exact scoring (kit `scoring.py`)
- **RMSE skill (30):** lat-weighted RMSE skill vs the **better of persistence and climatology**, lead days 1, 4, 8, all 6 channels. Full 30 at mean skill **0.12**.
- **Heavy-rain CSI (15):** threshold **10 mm/day**, skill vs the better reference. Full 15 at **0.10**.
- **Anti-blur (10):** day-8 spatial variance per channel ≥ **40%** of the real one (and ≤ 2.5×).
- **Causality (15):** two shocks (`warm_ocean`: Bay of Bengal box lat 8–20, lon 82–94; `monsoon_stall`: Arabian Sea box lat 8–20, lon 65–75). Response non-zero; ≥ 60% of day-1 response energy within 3 cells of the box; propagates outward by days 4–8 at ≤ 8 cells/day; never above 5 std (stable); null shock consistent.
- **Hard gates for Round 2:** beat persistence at day 4; validity on all 8 cases; non-zero localised response to at least one shock.

### Hard rules (reject these in any review)
- No weather APIs, downloads or internet at inference; no hardcoded dates or lookup tables of past weather (a **one-year climatology is explicitly allowed** by the kit's `repo_scan`).
- Never commit `data/`, `*.nc`, multi-year arrays or pre-rendered dashboard images. Weights only via Git LFS (`*.pt`, `*.onnx`). **Only Vishwak commits to `weights/`**, so there is only ever one `model.pt`.
- `serve.py` and everything it imports may use only: torch (CPU), numpy, scipy, pandas, xarray, scikit-learn, lightgbm, onnxruntime, fastapi, flask, uvicorn. Never Streamlit/Plotly (the dashboard has its own folder and requirements).
- Never "fix" a failing check by loosening it; never wrap errors in a silent try/except that returns zeros.
- Every PR is reviewed by someone on the other tool (Akash reviews Vishwak's work). Commit before and after every AI session; never merge AI code nobody has run.

## 2. Team and ownership

| Area | Owner | Reviewer |
|---|---|---|
| Model and training (`twin_model/`, `training/`) | Aether | Junaid |
| Scorer, shock tests, `serve.py`, `weights/`, LFS, repo scan, clean clone | **Vishwak** | Akash |
| Data scripts, impact module, README, brief | Junaid | Aether |
| Dashboard (Streamlit, calls live `/simulate`) | Akash | Vishwak |

Timeline gates: **hour 17 code freeze** (final weights committed via LFS, only bug fixes after), **hour 19 clean-clone test** (Vishwak reports to all) and demo rehearsal.

## 3. What Aether has already done (complete, tested)

### Code in the repo (branch `feat/aether-model`, based on `main` at 3021523)
- `CLAUDE.md` — contract, rules, ownership, commands, and the serving-path vs model-lane split. `docs/interfaces.md` — pinned model-lane interfaces.
- **Your serving path is unchanged**: `serve.py` → root `twin.py` → `weights/model.onnx` / `model.pt` (TorchScript) per `docs/WEIGHTS.md`. Nothing on this branch edits `serve.py`, `twin.py`, `common.py`, `eval/` or any other lane's files.
- `twin_model/` — Aether's model package (training + evaluation; **named `twin_model` so it never shadows your root `twin.py`**). Uses only torch, numpy, stdlib. Not imported by `serve.py` today.
  - `twin_model/norm.py`: channel constants, physical ↔ model space (rain in log1p), `Normalizer`.
  - `twin_model/model.py`: residual U-Net (~2M params, replicate padding, zero-initialised head so an untrained model = persistence), `step`, `rollout_std`, `advance_doy`. Rain floor is a straight-through ReLU (dry cells stay exactly dry).
  - `twin_model/post.py`: output post-processing stored in the checkpoint (climatology blend per lead/channel, then rain quantile mapping per lead; optional `qmap_strength`, unused by the final model).
  - `twin_model/twin.py`: **`Twin(ckpt_path="weights/model.pt")`** and **`Twin.rollout(init_states [3,6,32,36], doy_sincos [2], steps:int, perturbation [6,32,36] | None) -> np.float32 [steps,6,32,36]`** in physical units. Calls `torch.set_num_threads(4)`, eval mode, no grad, deterministic. Missing/unloadable checkpoint → logs a warning, `twin.fallback = True`, returns persistence (never crashes). Malformed `post` block → logs a warning and runs the raw model. Applies validity clamps (rain ≥ 0, t2m 233.15–333.15 K, non-finite → persistence value). Any `steps` works (tested 1, 10, 20). Perturbation `None` and zeros give bit-identical output.
- `training/` — `dataset.py` (loader understands the kit's .nc layout: one `data(time,channel,lat,lon)` var + `static(static_channel,lat,lon)` with `land_sea_mask`, `orography_m`), `losses.py`, `train.py` (one-step + multi-step fine-tune, kit-aligned checkpoint selection `--select kit`), `fit_post.py` (fits post-processing on 2005–2018), **`kit_eval.py` (scores any checkpoint with the kit's own scoring functions → projected points /70; appends to `kit_scores.csv`)**.
- `tests/` — Aether's tests added next to yours (`test_twin_model.py`, `test_norm.py`, `test_model.py`, `test_post.py`, `test_qmap_strength.py`, `test_training.py`, `test_selection.py`, `conftest.py`). On this branch **167 passed, 3 skipped** for the whole suite (yours + Aether's), excluding `tests/test_serve.py`, which wasn't run on Aether's machine because `fastapi.testclient` needs an extra package (`httpx2`) that wasn't installed there. Run it on yours.
- `experiments.csv` (every run and why it was kept or rejected), `kit_scores.csv` (official-style scores).

### Final model — frozen
- Checkpoint on Aether's machine: `runs/e5b_gradmag/best_post.pt`, **18.1 MB**, a `torch.save` dict (weights, normalisation stats, static maps, config, post-processing block: 366-day climatology [366,6,32,36], blend weights [10,6], rain quantile knots [10,Q]). **This is NOT the TorchScript/ONNX format `twin.py` loads** — don't drop it into `weights/` as is.
- **Exported per `docs/WEIGHTS.md` (done, verified):** `python -m training.export runs/e5b_gradmag/best_post.pt --out export --name aether-unet-e5b` writes `model.onnx` (8.3 MB), TorchScript `model.pt` (8.0 MB, `jit.trace`), `model_meta.json` (`model_name: aether-unet-e5b`) and `post.npz` (9.1 MB). Aether sends you these four files (not committed by Aether — `weights/` is yours).
  - The net was trained with `doy_sincos` of the **last input day**; `twin.py` passes the **day being predicted**, so the export wrapper rotates the angle back one day (2π/365) before the net.
  - Verified: export wrapper + your exact `twin.py` loop vs the training rollout: max rel diff 2.3e-7 over 10 days on 5 real windows; TorchScript vs wrapper: identical; ONNX vs wrapper: 7.7e-8; ONNX loop + `post_numpy.apply` vs the frozen reference model: max abs diff 0.016 (Pa on msl, i.e. float32 round-off). ONNX 10-step rollout: 25 ms on 4 threads.
  - **Through your unchanged `serve.py` + `twin.py`** (files copied into `weights/` locally, then removed): `/health` → `backend: onnx`, `model_name: aether-unet-e5b`; `tools/validate.py` → **PASS** (all 9 checks, 80 ms median wall); kit `schema_validator` → **all checks passed**; judge simulation → **44.0/70 without post-processing** vs **61.0/70 with it**. That gap is task 2.
- Recipe: one-step U-Net (30 epochs, keep epoch 8) → multi-step fine-tune on 2, 4, 8-day rollouts with a gradient-magnitude anti-blur loss (weight 0.3) → post-processing fitted on 2005–2018.

### Measured results (all with the kit's own scoring functions)
| Test | Score /70 (automated part) |
|---|---|
| 2019–2021 held-out, 40 windows | 68.1 |
| 2019–2021 held-out, 100 windows | 65.8 (RMSE 30, CSI 13.3, blur 10, causality 12.5) |
| Same recipe, different seed | 66.4 |
| **Real 2023 data, 4 windows (Feb/Apr/Jul/Oct)**, fetched with the kit's ARCO pipeline | **55.3** |
| Full judge simulation over HTTP (4 × 2023 + 1 × 2021 window + 3 shocks) | **61.0** (RMSE 24.8, CSI 11.2, blur 10, causality 15) |

**Expected automated score: about 55–66 of 70.** Hard gates pass in every test.

Experiment history (official scorer): E1 one-step 28.1 → E4 multi-step 60.3 → E5b gradient-magnitude loss 61.1 → **E5b + post-processing 68.1 (final)**. Rejected with evidence: higher weight on the original gradient loss (no effect), input noise (52.3), wider model (57.9), kit-aligned selection at the one-step stage (67.5), weaker rain calibration on 2023 data (all variants 52.1–54.9 vs 55.3).

### Server checks already passed (on Aether's machine, CPU only, using `twin_model.twin.Twin` behind the kit's starter server — not yet your `serve.py`)
- Kit's starter `serve.py` with `simulate()` replaced by `twin_model.twin.Twin.rollout()` + **`python -m monsoon_r1.schema_validator` → all checks passed** (health; JSON lists and base64; no shock, warm_ocean, zeros).
- Latency: model 35–150 ms per 10-step rollout on 4 threads; ~0.1–0.15 s per HTTP call (limit 5 s).
- Judge simulation: `warm_ocean` and `monsoon_stall` both non-zero, localised, propagating, stable; null shock identical. Robustness: +80 K shock, −500 mm/day rain shock, +20,000 Pa shock all return valid output; repeated requests identical.
- Live-shock behaviour (for the demo): north-India heatwave +4 K and westerly burst +5 m/s are clean (local, spread, fade by day 8). A Bay of Bengal low of −1,500 Pa triggers up to +55 mm/day rain (7.3 std, over the 5-std "stable" line — not an auto-scored shock). A central-India dry spell stays local and does not spread.

### Known limits (go in the brief and model card)
- t2m after ~day 5 leans on climatology (blend weight ~0.5 at day 4, ~0.8 at day 8), so shock effects fade by day 8.
- `monsoon_stall` propagation fails on many windows (~1.5–2 pts at risk on causality).
- Rain skill at days 4–8 is near or below climatology; CSI is the most variable score part.
- Scores vary ~±1 pt with the random seed.

## 4. Your tasks, in order

1. **Get the code and confirm the environment.** `git fetch && git checkout feat/aether-model` (review it; it only adds Aether's lane). Install Aether's extra deps for the model lane: `xarray netCDF4` (already in your env), and the kit with `pip install --no-deps -e <path-to-kit-clone>` (gives `monsoon_r1`, needed by `training/kit_eval.py`). Data: `india_climate_train.nc` must be 122,831,709 bytes with header `89 48 44 46 0D 0A 1A 0A` — a teammate's copy was corrupted by a text-mode transfer earlier; never open it in an editor. Run the full suite → expect 167 passed, 3 skipped, plus your `test_serve.py`.
2. **Decide the post-processing integration with Aether (biggest open item, ~10 pts).** The final model's post-processing (climatology blend per lead/channel, then rain quantile map per lead) is applied to the **finished rollout**, depends on the lead day, and must NOT be fed back into the loop — so it can't live inside the one-day exported `step`. Ready to use: `weights/post.npz` (blend [10,6], clim [366,6,32,36] = one-year climatology, allowed by the kit's repo_scan; qmap_fq/qmap_oq [10,160]; 9.1 MB) and **`twin_model/post_numpy.py`** — numpy + stdlib only, importable without torch (tested), identical output to the reference implementation (tested). **Ready-made, tested patch: `git apply docs/patches/vishwak_post_integration.patch`** (your files, your call — review it first). It changes two files:
   - `twin.py`: loads `weights/post.npz` (LFS-pointer check, loud error and raw-model fallback if it can't load) and applies `post_numpy.apply` after the loop, model path only.
   - `scripts/repo_scan_lite.py`: adds `"twin_model"` to `LOCAL` — **without this the lite scan FAILS** with `twin.py: disallowed import 'twin_model'`.
   Tested on Aether's machine with all four weight files in `weights/` and the patch applied: `/health` backend onnx; `tools/validate.py` PASS; kit `schema_validator` all checks passed; **judge simulation 61.0/70** (RMSE 24.8, CSI 11.2, blur 10, causality 15; both shocks non-zero, localised, propagating, stable; null shock identical; deterministic); `repo_scan_lite` PASS. Kit `repo_scan` lists `twin_model/__init__.py` and `twin_model/post_numpy.py` as inference files (fine: numpy/stdlib only) and gave 13/15, its only flag being `data/india_climate_train.nc` — it scans the working folder, so that flag should vanish in the clean clone (data is git-ignored); confirm at hour 19.
   For reference, the essence of the `twin.py` change:
   ```python
   from twin_model import post_numpy                       # numpy-only
   # in __init__, after weights load (wdir = weights folder):
   self.post = None
   pf = wdir / "post.npz"
   if pf.is_file():
       try:
           _check_not_lfs_pointer(pf); self.post = post_numpy.load(pf)
       except Exception as e:                              # log loudly, keep serving the raw model
           self.post = None; log.error("post.npz not loaded (%s)", e)
   # in rollout(), only on the model path, after the finite check, before enforce_validity:
   if self.post is not None and self.model is not None:
       out = post_numpy.apply(out, d, self.post)           # d = doy_sincos of the LAST INPUT day
   ```
   Rules: apply to the finished rollout only, never inside the loop; only when a real model ran (not on persistence); keep it deterministic (it is — zero shock stays exact). Clim slot for lead k = day of year of (last input day + k) on the kit's 366-slot calendar (Feb 29 = slot 59), handled inside `post_numpy`. If `post.npz` is absent, behaviour is exactly as today. Show it in `/health` (e.g. `"post": true`). Note `.gitattributes` routes `*.npz` through LFS — same plain-git-vs-LFS question as `model.onnx` in `docs/WEIGHTS.md`.
3. **Weights:** Aether hands over `model.onnx`, `model.pt` (TorchScript), `model_meta.json` and `post.npz` (from `export/`, or regenerate with `training/export.py` if you have the checkpoint). Commit them in `weights/` per `docs/WEIGHTS.md` (the team still has to settle plain git vs LFS for `model.onnx`). Run the WEIGHTS.md self-test: `/health` shows `backend: onnx` and the model name; `TWIN_BACKEND=torchscript` also works.
4. **Validate on the real weights:** `python serve.py --port 8000`, `python tools/validate.py --url http://127.0.0.1:8000`, `python -m monsoon_r1.schema_validator http://127.0.0.1:8000`. On Windows use `127.0.0.1`, not `localhost` (localhost adds ~2 s per call). Then score through your server: zero shock, latency, both shocks. `training/kit_eval.py` scores a checkpoint with the kit's functions; Aether's `judge_sim.py` (full judge simulation over HTTP: 5 windows + 3 shocks + painted shocks + robustness) can be added to `eval/` if you want it.
5. **Your scorer and shock-test lane:** `training/kit_eval.py` already implements the kit-faithful scorer and shock causality. Rather than rewriting, review it (you're the scorer owner) and move/wrap it into `eval/` if the team wants that layout. Keep the kit's functions as the source of truth.
6. **Repo scan:** `python -c "from monsoon_r1.repo_scan import scan; import json; print(json.dumps(scan('.'), indent=1))"`. Must be clean: no data files, no multi-year arrays, no disallowed imports in inference files. Make sure the scratch folders `runs/`, `data/`, `.venv/` are git-ignored and the playbook PDF is not committed (synthetic smoke rows are already removed from `experiments.csv`).
7. **Give Akash a live `/simulate`** for the dashboard (same server). Tell Akash: shocks fade by day 8; a deep Bay low gives a very strong rain response; heatwave and westerly-burst shocks are the cleanest live demos.
8. **Hour 19 clean-clone test:** fresh clone into an empty folder, no internet, `pip install` only the allowed packages, `python serve.py --port 8000`, validator passes, repo scan clean, `/health` < 120 s. Report the result to the whole team.

## 5. Gotchas already found (save yourself the time)
- Kit `build_dataset arco` hangs with zarr-python 3: the ARCO store is zarr **v2** on disk; open with `zarr.open_group(store=fsspec.get_mapper(url, token="anon"), mode="r", zarr_format=2)`. Each window of real data costs ~700 MB to fetch (whole-globe hourly chunks).
- `schema_validator` needs `requests` installed. The kit client assumes `steps=10` when decoding; other step counts work on the server but show as "failed" in the kit client.
- Under Git Bash on Windows, `CUDA_VISIBLE_DEVICES=""` crashes torch ("Invalid device id"); use `-1` to force CPU.
- `training/kit_eval.py` sets `CUDA_VISIBLE_DEVICES` on import, so don't import it inside training code.
- The day-of-year convention is 1-based (`date.timetuple().tm_yday`), matching the training data and the kit.

## 6. How to work
Small steps. Show me the diff and the command that tests it. The scores above were measured with Aether's `twin_model.twin.Twin`; once the exported ONNX/TorchScript weights (and post-processing) run through your `twin.py`, re-score and compare — they should match to float32 round-off. Don't touch `twin_model/` or `training/` (Aether's lane) — report problems there to Aether instead of editing. Run code before calling it done. Start by telling me what you found on disk versus this brief, then do task 1.
