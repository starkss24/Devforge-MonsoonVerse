# Four-minute desk-review demo

Follows the playbook's 7 steps. The labels below are copied from the dashboard code (`dashboard/app.py`, `sidebar.py`,
`tab_*.py`) as of the final version (animated map, Live mode).

## Before the judges arrive (not timed)

1. Start the server: `python serve.py --port 8000`. The log must say `backend=onnx` and the model name
   (`aether-unet-v21-ens4`).
2. Start the dashboard: `streamlit run dashboard/app.py`. Open it at 1280 × 720 or larger. The first load can take
   10–20 s on a busy laptop (the map area says "Drawing the map…"). Once per machine, before this step, generate the
   Model card's scores with the final weights (local file, not in git; about a minute):
   `python -m eval.score --data data/india_climate_train.nc --stats data/stats.json --twin`.
3. Check the header reads **● Server ready, running aether-unet-v21-ens4**. The sidebar status line shows the details:
   `onnx, post-processing on` and the last round trip in ms. It must not say `persistence`.
4. Sidebar, **Window: last input day**: leave it on **2019-07-15** (the default, a monsoon day). **Live: update on
   every change** stays on, so every change re-runs the twin by itself (no need to press Run).
5. The dashboard opens on the **Heatwave: north India +4 K** shock and runs it once on load, so the first live call is
   not the one the judges see. Press **Play** under the map once to check it animates.
6. Keep the backup screen recording open in another window (record it from this script with the final model loaded).

**Fallback for the whole demo.** If the header shows **● Server down**, a card explains the fix: restart
`python serve.py --port 8000`; the page reconnects by itself within 3 s. If the URL is wrong, open sidebar
**Settings → Twin server URL** and use `http://127.0.0.1:8000` (not `localhost`, which adds about 2 s per call on
Windows). If it still fails within 30 s, play the backup recording and talk over it.

---

## Step 1: baseline 10-day rollout (0:00 to 0:30)

**Click.** Sidebar **Shock painter** → **No shock (baseline only)**. Tab **Maps**. In the layer row above the map press
**Rain**. Leave **Wind arrows** on. Press **Play** under the map (or **Fast**).

**Say.** "This is a live call to our twin: 3 real days in, 10 days out, on a 1° grid over India. The timeline starts on
the three real days, marked real data, then runs the forecast. Watch the rain bands and the winds evolve."

**If it fails.** If the run is slow, a small **Updating…** chip shows in the header and the last map stays visible; keep
talking. The side panel's last line shows the round-trip time of the live call. If the server is down, use the
whole-demo fallback above.

## Step 2: paint warm_ocean, watch it start in the box and spread (0:30 to 1:15)

**Click.** **Shock painter** → **Warm ocean: Bay of Bengal +3 K, +15% moisture** (the layer switches to Temperature by
itself, and Live re-runs). On the right, switch the view to **Effect**. Press **Play** under the map. For the details,
press **Step all** (key P): the map, the big day heading, the side panel and the day strip below walk through the days
together.

**Say.** "This is the difference the shock makes: red is warmer, blue cooler. On day 1 it sits inside the gold box. The
dotted ring marks where the change is at least a tenth of its day-1 strength, and you can see it move outward by days 4
to 8 while the far side of India stays untouched." Point at the side panel: it measures this live, e.g. how much of the
change is still inside the box and whether it fades.

**If it fails.** If the difference does not spread on this window, say so honestly: "on this window the shock stays
near the box; the Shock Lab measures exactly how much." Then move on.

## Step 3: Shock Lab, spread measure and zero-shock check (1:15 to 2:00)

**Click.** Tab **Shock Lab**. Point at the verdict line and the metrics **Day-1 energy near the box**, **Outside the
near zone** (day 1 → days 4–8) and **Growth outside (needs > 1.5×)**. Point at **Response energy by region** (the gold
line marks the day chosen on the Maps tab) and **Share of energy by distance from the box**. Scroll to **Causality
checks**. Then press **Run zero-shock check**.

**Say.** "These are the organisers' own causality checks run on our live output: localised on day 1, the energy outside
the box's neighbourhood growing by days 4 to 8 by more than one and a half times, never faster than 8 cells a day,
stable. The zero-shock check makes two fresh calls, no perturbation versus all zeros. It passes only if the forecasts
are identical, and they are: max difference exactly 0."

**If it fails.** If the zero-shock check shows **Fail**, say: "our server is deterministic by design; this means the
wrong server is running," and check the backend in the header. If the server is down, the button shows an error; quote
the result from `eval/shock_report.json` instead.

## Step 4: impact panel under monsoon_stall (2:00 to 2:45)

**Click.** **Shock painter** → **Monsoon stall: Arabian Sea winds −40%** (Live re-runs). Tab **Impact**.

**Say.** "This is the crop-stress index for six approximate farming regions, the dashed boxes on the map. The table shows,
baseline → shocked, the **Rain deficit** against the 2005–2018 normal, **Hot days** (days when the region's t2m anomaly
is above the slider, 2 K by default; t2m is a daily mean, so these are anomalies, not heatwave maxima) and the 0–100
**Index**. The bars show **Δ crop-stress index (shocked − baseline)**." Point at the banner above the map: it names the
region that changes most, or says that nothing moved.

**If it fails.** If nothing moves (the stalled winds may not cross a rain or heat threshold on this window), say so
honestly and switch to **Warm ocean**, then move **Hot day: t2m anomaly above (K)** down or up and show the hot-day
counts change. Do not show numbers that are not live.

## Step 5: ensemble spread (2:45 to 3:15)

**Click.** Tab **Ensemble**. Leave **Members (live calls)** at 8 and **Input noise (% of channel std)** at 0.5. Press
**Run 8-member ensemble**. Point at the spread map, the **Spread vs lead day** chart and the caption under it (it states
what the curve does, measured from this ensemble).

**Say.** "The server itself is deterministic. Here the dashboard nudges the 3 start days with tiny noise and makes 8 live
calls. Notice the spread does not grow: it *shrinks* with lead day. Our model damps small start errors instead of
amplifying them, and the climatology blend halves them further at long leads. That is stable, but it also means this
spread is not a forecast of uncertainty: real weather errors grow, which is why our skill against climatology is near
zero by day 8. A calibrated ensemble would need perturbed models, not just perturbed starts." (Measured by Aether on the
final models, 2019-07-15, 2020-01-20, 2021-04-25: spread falls from about 0.004 to 0.002 of channel std from day 1 to
day 10 on every channel.)

**If it fails.** If it is slow, cut to 4 members with the slider and say "each member is one live call." If the server is
down, skip the step and say what the tab shows.

## Step 6: model card, one honest limit (3:15 to 3:45)

**Click.** Tab **Model card**. Point at the **Scorecard** row (day 1, 4 and 8 skill, and day-8 sharpness). Scroll to
**Known limits**.

**Say.** "Skill against the better of persistence and climatology, scored with the organisers' code on 2019–2021. One
honest limit: skill falls with lead time, and beyond about a week climatology is hard to beat. Another: the monsoon
stall shock rarely spreads in our model; we show that rather than hide it."

**If it fails.** If the scores file is missing, the tab says which command regenerates it. Quote the numbers from the
brief's results table instead.

## Step 7: hand the mouse to a judge (3:45 onward)

**Click.** Tab **Maps**, then let the judge **click anywhere on the map**: the shock moves there as a custom box of the
same kind and Live re-runs. In the sidebar, **Custom box (or click a map)** shows **Centre °N**, **Centre °E**, **Box
size**, **Shock channel**, **Magnitude as** and **Strength** for fine control. Then **Play**, **Effect**, and the
**Shock Lab**.

**Say.** "Paint anything you like: anywhere, any channel, absolute or percent. Every result is computed live, and the
side panel tells you in plain words what happened."

**If it fails.** If the box lands off the grid, the sidebar says "The box holds no grid cells. Move it onto the map or
make it bigger." If a huge shock gives odd values, explain that outputs are clamped to valid physics (rain ≥ 0, t2m
within −40 to +60 °C).
