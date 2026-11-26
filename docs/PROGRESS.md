# Team progress (against the Team Playbook)

Last updated: 2026-11-26. Source of truth for what is on `main`; per-agent log is in [STATUS.md](STATUS.md).

## Gates

| Hour | Gate | Status |
|---|---|---|
| 0 | Kickoff: repo, Twin stub, validator passes on stub | ✅ Done |
| 3 | Baselines: loader, year split, baselines CSV; scorer runs on any model | 🟡 Scorer done and run on real data; Junaid's loader/baselines.csv not in repo yet |
| 6 | First U-Net beats persistence at day 4, live behind the server | ⏳ Waiting on Aether's weights (server loader ready) |
| 10 | Architecture freeze; shock tests pass on real boxes | ⏳ Shock tests ready; need the real model |
| 14 | Impact module + ensemble toggle in dashboard | ⏳ Junaid / Akash, not in repo yet |
| 17 | Code freeze, final weights committed | ⏳ |
| 19 | Clean-clone test + rehearsal | ⏳ Script ready (`scripts/clean_clone_check.sh`) |

## By person

### Vishwak — scorer, shock tests, server (on `main`)
- ✅ Shared contract: `common.py`, `twin.py` (persistence stub), `docs/CONTRACT.md`
- ✅ `serve.py`: `/health` with `model_name`, `/simulate`, input checks (422), persistence fallback, clamps, 4 threads, model warm-up before listening
- ✅ Organisers' `schema_validator` passes (all 6 cases, ~0.05 s each)
- ✅ `eval/score.py` and `eval/shock_test.py` use the kit's own scoring code; run on real data
- ✅ Weights loader: drop a TorchScript `weights/model.pt` in and the server uses it (`docs/WEIGHTS.md`)
- ✅ Repo hygiene: `.gitignore` (data never committed), Git LFS rules, pinned requirements, repo-scan and clean-clone scripts
- ✅ 65 tests passing
- ⏳ Load and score Aether's checkpoints; re-time latency with the real model; commit weights
- ⏳ Review Junaid's baselines.csv and Akash's dashboard PR; hour-19 clean-clone test

### Aether — model and training
- ⏳ Nothing in the repo yet. Needs: training code (+3 repo-scan points), `weights/model.pt` exported per `docs/WEIGHTS.md`

### Junaid — data, baselines, impact, README/brief
- ⏳ Nothing in the repo yet. Needs: data loader, `baselines.csv`, `impact/crop_stress.py` with tests, brief page 2, demo script

### Akash — dashboard
- ⏳ Reported done, but nothing in the repo yet. Push to branch `feat/akash-dashboard` under `dashboard/` and open a PR (+1 repo-scan point for a frontend)

## Current score estimate (persistence stub, judge's own scoring)

| Component | Points | What lifts it |
|---|---|---|
| Forecast skill (RMSE 30 + rain CSI 15) | 0 / 45 | A model ~12 % better than climatology at days 4 and 8 |
| Anti-blur | 8.3 / 10 | Real model keeping day-8 rain variance within 40–250 % |
| Shock causality | 11.25 / 15 | Model whose shock response spreads outward by days 4–8 |
| Repo scan | 2.5 / 15 | Checkpoint in repo (+8 anti-cheat), training code (+3), frontend (+1) |
| Brief | – / 15 | Written by the team |

Round 2 hard gate (beat persistence at day 4) currently **fails** — expected until the real model lands.

## Open decisions
1. **Git LFS vs plain file for `model.pt`.** repo_scan reads files on disk; if the judge's clone skips LFS, the checkpoint is a pointer and 8 points are lost. A small U-Net (a few MB) fits in git directly. Recommendation: plain file.
2. **Protect `main`** in GitHub settings (require pull requests).
3. **Collaborators:** add Aether, Junaid and Akash to the repo so they can push branches.
