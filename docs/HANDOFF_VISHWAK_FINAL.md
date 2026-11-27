# Final review note for Vishwak's Claude agent (from Aether, 09:25)

Paste everything below the line into Vishwak's Claude session.

---

Thanks for merging v2.1 and adding Step all. `main` (5611b8b) serves `aether-unet-v21-ens4`; Aether verified main's
serving path from a fresh GitHub clone earlier (repo_scan 15/15, validators PASS, 228 tests). One small branch is left
for review: **`feat/aether-final`**, made from your 5611b8b (merging it is a fast-forward). It changes:

1. **Dashboard first paint** (dashboard/app.py, dashboard/tab_maps.py): a "Drawing the map…" placeholder while the
   frames are built, plus a CSS backdrop behind the big map (`.st-key-maps_main`), so the projector never shows a blank
   white box for the 10–20 s the first paint can take on a busy laptop. The figure's white paper covers the backdrop
   once painted.
2. **One-line note under Back / Next / Step all**: "Play under the map previews the animation (the map's title and info
   box follow it). Step all (key P) moves the map, this panel and the day strip together." (font 0.85 rem ≈ 13.6 px.)
3. **tests/test_dashboard_polish.py** for 1 and 2.
4. **docs/DEMO_SCRIPT.md** step 2: run `python -m eval.score --data data/india_climate_train.nc --stats data/stats.json
   --twin` once per machine so the Model card shows v2.1 skill (eval/scores.csv is git-ignored), and the first-load note.
5. **CLAUDE.md** (served model = v2.1, how to judge new weights), **docs/PROJECT_EXPLAINED.md** (pointer to MODEL_V21),
   this file, and a docs/STATUS.md line.

No weights, server, twin, postproc or eval changes. Please: check out the branch, run `pytest tests/ -q` (incl.
test_serve.py, which Aether can't run: httpx2 missing there), open the dashboard at 1280×720 and look at the Maps tab
(placeholder on first load, note under Step all, nothing overlapping), then merge. If you'd rather not take it, main is
complete and safe as it is.

After merging: repo public + all four collaborators; presentation laptop follows docs/DEMO_SCRIPT.md "Before the
judges arrive"; backup recording with aether-unet-v21-ens4; rehearsal (5 questions in docs/JUDGE_QA.md; demo order
heatwave → warm Bay of Bengal → monsoon_stall as the honest limit).
