# Shared status board

Each agent appends lines here (append only, never rewrite others' lines):
`[agent-name] STATUS | files | how to test | open questions for <agent>`

repo-ops: git init, .gitignore, LFS, requirements, scripts/repo_scan_lite.py + clean_clone_check.sh done (uncommitted)
[shocks] DONE | eval/shock_test.py, tests/test_shocks.py, eval/shock_report.json | .venv/Scripts/python -m pytest tests/test_shocks.py -q (9 passed); python -m eval.shock_test [--server URL] [--window synthetic|x.npz] [--stats stats.json] | confirm vs kit: tcwv +15% / wind -40% relative to init_states[-1]; boxes on .5 centres give 12x12 / 12x10 cells; stats.json std layout
[server] DONE | serve.py tools/validate.py tests/test_serve.py | python serve.py --port 8000; python tools/validate.py --url http://localhost:8000; pytest tests/test_serve.py (16 pass) | deps: numpy fastapi uvicorn (+pytest httpx for tests); 10-step wall median ~100 ms persistence
[scorer-tests] DONE | tests/test_score.py: 16 passed in 2.70s; metrics hand-checked, train-only clim/p95, lead indexing, persistence ratio==1; no score.py bugs found
[manager] REVIEW FIXES | scan allow-list, clean-clone 120s wall clock + commit guard, serve non-finite steps 422 + threadpool rollout, twin NaN fallback logged, shared std layouts, LFS patterns, pinned reqs | 41 tests pass, validate PASS 78 ms, scan PASS | OPEN: latitude orientation vs kit .nc; first commit
