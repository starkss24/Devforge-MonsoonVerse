# Final 2.5 hours: message for Vishwak's Claude agent (from Aether)

Paste everything below the line into Vishwak's Claude session.

---

FROM AETHER (model lead). We two Claude agents share the whole project for the final stretch. You own the
PRESENTATION DASHBOARD + final integration; I own the model (v3 attempt on `feat/aether-v3`: delivered or "no-go" by
09:15) and the model-lane docs. Work on branch `feat/dashboard-v2`, small commits, Vishwak merges.

STATE: main = 95f964f serves aether-unet-v2-ens2 (you verified it: 67.06/70 on 100 held-out windows, judge simulation
64.7, repo_scan 15/15). I ran main's dashboard locally at 1280 wide with v2: it works end to end (status pill shows the
model, heatwave first load, live facts, Shock Lab 5/5 checks). Now make it look and feel like a polished weather
SIMULATOR for a projector. Goal: judges paint a shock and SEE it start locally and then spread over 10 days, smoothly.

1. ANIMATED MAP (highest priority)
   - Replace the server-side Play loop with ONE Plotly figure with a frame per timeline day (input days -2..0, forecast
     1..N): heatmap z, wind arrows, title ("Day 3 · Thu 18 Jul · forecast"). Plotly Play/Pause + day slider under the
     map; transition ~400 ms cubic-in-out, frame ~700 ms; speed 0.5x/1x/2x; fixed colour scale across frames.
   - "Effect" view = the money shot: the shock difference map spreading day by day, gold shock box fixed, plus a thin
     contour where the response is >= 10% of its day-1 peak, so the spreading front shows as a moving ring.
   - Side panel (day/date, facts) syncs to the slider / when the animation stops.
   - Smooth transitions on layer/view switches; hover tooltip (value + unit, nearest city, lat/lon, day); visible marker
     at the clicked place.
   - Day strip: fewer, larger frames (days 1, 2, 4, 6, 8, 10), readable labels, turmeric outline on the frame matching
     the big map's day, click to jump.
2. LIVE MODE THAT FEELS INSTANT
   - Debounce rapid changes, drop stale requests; keep the last good map visible with a small "updating…" chip; never a
     blank page or page-wide spinner. Warm the first call while the page builds (first load took ~20 s on a busy
     laptop; live calls 0.6-0.9 s there, ~0.1-0.16 s on an idle machine).
   - Status pill from /health: model name, backend, post loaded, last round-trip ms; friendly server-down card with the
     fix ("start it with: python serve.py --port 8000") and automatic recovery.
3. SHOCK LAB CLARITY
   - Fix the contradiction: "Reach on day 1: 0 cells / Reach on day 8: 0 cells" next to "Spreads outward by day 4-8 ✓"
     (heatwave, 2019-07-15). Show the kit's own measure instead: energy outside the 3-cell neighbourhood on day 1 vs the
     mean of days 4-8, their ratio, and the 1.5x threshold; plus the day-1 near-box share (>= 60%).
   - Moving day marker on the energy-by-region chart, in step with the map.
   - monsoon_stall: keep saying plainly when it does not spread (real weak spot, ~5/20 windows).
4. PRESENTATION POLISH (1280x720)
   - One app: map ~60% width, control panel right, Run + shock painter visible without scrolling.
   - Text >= 13 px (sidebar labels and strip are too small now), contrast >= 4.5:1, visible keyboard focus, keyboard
     usable (keep B/N/P), prefers-reduced-motion = no animation. Motion only in response to the user.
   - Optional "Presenter mode": hides sidebar and secondary panels, map full width, larger fonts.
   - Keep the design tokens (ink #1B2A41, mist #EEF3F6, monsoon blue #2F6F9F, turmeric #D9A21B only for shock box +
     selected day, paddy #3E7D4F pass, signal red #B3261E fail, Archivo).
   - Ensemble tab: spread SHRINKS with lead day on this model; the caption must not say it grows (docs/DEMO_SCRIPT.md).
   - Model card: regenerate eval/scores.csv with the real weights (`python -m eval.score --data
     data/india_climate_train.nc --stats data/stats.json --twin`) and show the limits from docs/MODEL_V2.md.
   SKIP if short on time: paint-by-dragging, Hindi labels, 15/20-day polish.

HARD RULES: only dashboard/client.py talks to the server and names the route; no UI library in that file; never write
the route word in quotes/after a slash elsewhere in dashboard/; every forecast shown comes from a live call this
session; works OFFLINE (no tiles/CDNs; Plotly's bundled JS is fine); real numbers incl. weak spots; don't edit
serve.py, twin.py, postproc.py, weights/, twin_model/, training/, eval/; don't special-case a model name.

TIMELINE: build until 09:30 (commit every ~30 min, tests green: update tests for changed widgets, add tests for new
behaviour, never loosen meaningful assertions). 09:15: Aether hands over v3 or says no-go; merge v3 only if it passes:
kit_eval 100 >= 67.06, real 2023 >= 59.11 (`python -m training.eval_real <ckpt> --only 2023`), real 12 >= 60.26, gates,
validators, repo_scan 15/15, < 1 s. 09:30 frontend freeze. 09:30-09:50 fresh GitHub clone: serve.py, tools/validate.py,
kit schema_validator, repo_scan 15/15, pytest tests/ -q. 09:50-10:15 screenshots at 1280x720 (first load Live/heatwave,
Bay of Bengal low on rain, map click moving the shock, Play mid-animation, Observed, Shock Lab, Ensemble, Model card,
Impact), brief PDF with Junaid, rehearsal. Note results and anything not done in docs/STATUS.md.

UPDATE 07:55: v3 is a NO-GO (anti-damping loss spread monsoon_stall 13/20 in the raw network, but only 7/20 after
post-processing and -3 points on real 2023 windows); weights on main stay aether-unet-v2-ens2. Nothing to verify for
weights. Please MERGE `feat/aether-v3` anyway (no weights change): README training section, docs/PROJECT_EXPLAINED.md
(team explainer), updated JUDGE_QA monsoon_stall answer, training code + tests, experiments log.

ALSO MERGE from `feat/aether-v3` even if no new weights land: README now has the "Model and how to train it" section
(the organisers' checklist needs "README says how to train"); it only touches README.md + docs.
