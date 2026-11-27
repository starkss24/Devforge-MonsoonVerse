# Final review checklist for Vishwak's Claude agent (from Aether, 09:15)

Paste everything below the line into Vishwak's Claude session.

---

Aether has finished all model-lane work and the remaining finishing touches on branch `feat/aether-v3`. It already
contains `main` (6ced80c), so merging is a **fast-forward with no conflicts**. Please REVIEW, VERIFY, then MERGE.

## What the branch changes (vs main 6ced80c)
1. **Weights → `aether-unet-v21-ens4`**: same network, recipe and post-processing as v2 (blend cap 0.5, rain-map slope
   cap 1.5), but 4 identically trained seeds averaged per step instead of 2. `weights/model.onnx` 33.1 MB,
   `model.pt` 32.1 MB, `post.npz` 10.1 MB, `model_meta.json` (plain git, each < 95 MB). Details: `docs/MODEL_V21.md`.
2. **Dashboard polish** (dashboard/app.py, dashboard/tab_maps.py, tests/test_dashboard_polish.py):
   - "Drawing the map…" placeholder while the frames are built, plus a CSS backdrop behind the big map
     (`.st-key-maps_main`), so the projector never shows a blank white box during the first paint.
   - A short note under Earlier/Later: the panel shows the *selected* day; Play animates the map in the browser and
     the map's title + info box follow the playing day. (Streamlit receives no event per client-side animation frame,
     so a true live sync isn't possible; this makes the behaviour explicit instead of looking like a bug.)
3. **Docs**: `docs/brief.pdf` rebuilt (exactly 2 pages, v2.1 numbers); your tightened `docs/BRIEF.md` with only the
   model facts/numbers changed; `docs/DEMO_SCRIPT.md` setup (backend=onnx, model name, run `eval.score` once per
   machine for the Model card, first-load note); `docs/JUDGE_QA.md`, `docs/PROJECT_EXPLAINED.md` (team explainer),
   `README.md` (how to train, 4 seeds), `CLAUDE.md`, `docs/STATUS.md` line, `experiments.csv`, `kit_scores.csv`.
4. Training code + tests from the v3 attempt (`--anom-weight`, default 0 = unchanged behaviour). v3 itself was a
   no-go (logged).

## Verify (what Aether measured, please reproduce on your machine)
| Check | Aether's result | Bar |
|---|---|---|
| `python -m training.kit_eval runs/...` (100 held-out windows) | 67.66 | >= 67.06 |
| real 2023 windows / 12 real windows (`training/eval_real.py`) | 63.64 / 62.94 | >= 59.11 / >= 60.26 |
| judge simulation through serve.py (4 x 2023 + 1 x 2021 windows + 3 shocks) | 68.1 (v2: 64.7) | - |
| `/health` | onnx, `aether-unet-v21-ens4`, post loaded, ready 1.6 s | < 120 s |
| `tools/validate.py`, kit `schema_validator` | PASS, all checks passed | PASS |
| `scripts/repo_scan_lite.py`, kit `repo_scan` on a fresh clone | PASS, 15/15 no flags | 15/15 |
| rollout latency | 100 ms in-process (background server on a busy laptop ~0.7 s/call) | < 1 s |
| pytest | see Aether's final commit message | all green |

Your minimal path: `git fetch && git checkout feat/aether-v3`, `python serve.py --port 8000` → `/health` shows
`aether-unet-v21-ens4`; `python tools/validate.py --url http://127.0.0.1:8000`; `python -m monsoon_r1.schema_validator
http://127.0.0.1:8000`; `pytest tests/ -q` (incl. test_serve.py, which Aether could not run: httpx2 missing there);
open the dashboard at 1280x720 and check the Maps tab (placeholder, panel note, Play). If all good: merge to main,
then a fresh clone from GitHub + repo scan once more. If anything fails: keep main as is (v2 is verified and safe)
and tell Aether what failed.

## After merging (team)
- Repo public, all four as collaborators.
- On the presentation laptop: follow `docs/DEMO_SCRIPT.md` "Before the judges arrive" (incl. `eval.score` once).
- Record the backup demo video with `aether-unet-v21-ens4` loaded.
- Rehearse ~10:00: everyone answers the 5 questions in `docs/JUDGE_QA.md`; lead the demo with the heatwave, then the
  warm Bay of Bengal; present monsoon_stall as the honest limit (and the tested anti-damping fix as "what we tried").
