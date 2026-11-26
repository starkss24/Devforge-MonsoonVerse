# Team progress (against the Team Playbook)

Last updated: 2026-11-26. Source of truth for what is on `main`; per-agent log is in [STATUS.md](STATUS.md).

## Gates

| Hour | Gate | Status |
|---|---|---|
| 0 | Kickoff: repo, Twin stub, validator passes on stub | ✅ Done |
| 3 | Baselines: loader, year split, baselines CSV; scorer runs on any model | ✅ Scorer, loader, year split and `data_tools/baselines.csv` done on real data |
| 6 | First U-Net beats persistence at day 4, live behind the server | ⏳ Waiting on Aether's weights (server loader ready) |
| 10 | Architecture freeze; shock tests pass on real boxes | ⏳ Shock tests ready; need the real model |
| 14 | Impact module + ensemble toggle in dashboard | ✅ Dashboard, ensemble and Impact tab done |
| 17 | Code freeze, final weights committed | ⏳ |
| 19 | Clean-clone test + rehearsal | 🟡 Clean clone from GitHub passed (2026-11-27); rerun after final weights |

## By person

### Vishwak — scorer, shock tests, server (on `main`)
- ✅ Shared contract: `common.py`, `twin.py` (persistence stub), `docs/CONTRACT.md`
- ✅ `serve.py`: `/health` with `model_name`, `/simulate`, input checks (422), persistence fallback, clamps, 4 threads, model warm-up before listening
- ✅ Organisers' `schema_validator` passes (all 6 cases, ~0.05 s each)
- ✅ `eval/score.py` and `eval/shock_test.py` use the kit's own scoring code; run on real data
- ✅ Weights loader: drop a TorchScript `weights/model.pt` in and the server uses it (`docs/WEIGHTS.md`)
- ✅ Repo hygiene: `.gitignore` (data never committed), weights as plain git files (no LFS), pinned requirements, repo-scan and clean-clone scripts
- ✅ 65 tests passing
- ⏳ Load and score Aether's checkpoints; re-time latency with the real model; commit weights
- ⏳ Review Junaid's baselines.csv and Akash's dashboard PR; hour-19 clean-clone test

### Aether — model and training
- ⏳ Nothing in the repo yet. Needs: training code (+3 repo-scan points), `weights/model.pt` exported per `docs/WEIGHTS.md`

### Junaid — data, baselines, impact, README/brief (built by Vishwak's agents, on `main`)
- ✅ `data_tools/`: loader with checks, year split, 3-day windows (reusable for training), `baselines.csv` (leads 1–10, all 1084 validation windows), README figures, `docs/DATA.md`
- ✅ `impact/`: crop-stress index per approximate farming region (rain deficit vs train-year climatology, daily-mean t2m anomaly hot days, 0–100 index), hand-checked tests, `docs/IMPACT.md`
- ✅ Dashboard Impact tab: region map, baseline → shocked table, Δ index bars, hot-day threshold slider, honest limits
- ✅ Drafts: `docs/BRIEF.md` (2-page brief), `docs/DEMO_SCRIPT.md`, `docs/JUDGE_QA.md`; model facts marked `[TO CONFIRM: Aether]`, results `[RESULT]`
- ⏳ Brief to PDF (2 pages), backup screen recording with the final model
- Note: organisers' `stats.json` covers 2005–2021 incl. validation years (<1% of std difference); say so in the brief

### Akash — dashboard (built by Vishwak's agents, on `main`)
- ✅ `dashboard/`: Streamlit app calling the live server on every Run; sidebar with real start windows, 6 channels, day slider, wind arrows
- ✅ Shock painter: warm_ocean / monsoon_stall presets + custom box; baseline / shocked / difference maps; day 1–10 animation
- ✅ Shock Lab: response energy inside/near/outside the box by day, radial spread, kit causality checks, one-click zero-shock check
- ✅ Ensemble tab (8 runs, small input noise, mean + spread); model card from `eval/scores.csv` with known limits
- ✅ Impact tab connected to the crop-stress module; English labels only
- ✅ Server-down message; route only in `dashboard/client.py` (repo scan: frontend +1, server still separate from UI)
- ⏳ Visual check at 1280×720 on a projector

## Current score estimate (persistence stub, judge's own scoring)

| Component | Points | What lifts it |
|---|---|---|
| Forecast skill (RMSE 30 + rain CSI 15) | 0 / 45 | A model ~12 % better than climatology at days 4 and 8 |
| Anti-blur | 8.3 / 10 | Real model keeping day-8 rain variance within 40–250 % |
| Shock causality | 11.25 / 15 | Model whose shock response spreads outward by days 4–8 |
| Repo scan | 3.5 / 15 | Checkpoint in repo (+8 anti-cheat), training code (+3) |
| Brief | – / 15 | Written by the team |

Round 2 hard gate (beat persistence at day 4) currently **fails** — expected until the real model lands.

## Open decisions
1. ~~Git LFS vs plain file for weights~~ **Decided: plain git files** (LFS rules removed from `.gitattributes`; `repo_scan_lite.py` fails on LFS pointers or files > 95 MB).
2. **Protect `main`** in GitHub settings (require pull requests).
3. **Collaborators:** add Aether, Junaid and Akash to the repo so they can push branches.
