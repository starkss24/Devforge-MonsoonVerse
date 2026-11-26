# Four-minute desk-review demo

Follows the playbook's 7 steps. The labels below are copied from the dashboard code (`dashboard/app.py`, `sidebar.py`, `tab_*.py`).

## Before the judges arrive (not timed)

1. Start the server: `python serve.py --port 8000`. The log must say `backend=model`.
2. Start the dashboard: `streamlit run dashboard/app.py`. Open it at 1280 × 720 or larger.
3. Check the header pill reads **● Server ready · <model name>**. The sidebar status line must say `backend model`, not `persistence`.
4. Sidebar, **Window: last input day**: leave it on **2019-07-15** (the default, a monsoon day).
5. Do one warm-up **▶ Run the twin** so the first live call is not the one the judges see.
6. Keep the backup screen recording open in another window. Record it from this script with the final model loaded; it does not exist yet.

**Fallback for the whole demo.** If the pill shows **● Server down**, restart `python serve.py --port 8000`. If the URL is wrong, open sidebar **Settings → Twin server URL** and use `http://127.0.0.1:8000` (not `localhost`, which adds about 2 s per call on Windows). If it still fails within 30 s, play the backup recording and talk over it.

---

## Step 1: baseline 10-day rollout (0:00 to 0:30)

**Click.** Sidebar **Shock painter** → **No shock (baseline only)** → **▶ Run the twin**. Tab **Maps**. In the row above the tabs, set **Display channel** to `tp · Precipitation (mm/day)`. Drag **Lead day** from 1 to 10. Leave **Wind arrows** on.

**Say.** "This is a live call to our twin: 3 real days in, 10 days out, on a 1° grid over India. Watch the rain bands and the winds evolve as I move the lead day."

**If it fails.** If the run is slow, keep talking; when it lands, point at the **Baseline call (round trip)** metric under the maps to show it was live. If the server is down, use the whole-demo fallback above.

## Step 2: paint warm_ocean, difference at days 1, 4 and 8 (0:30 to 1:15)

**Click.** **Shock painter** → **Warm ocean · Bay of Bengal +3 K, +15% moisture**. Display channel switches to t2m by itself. **▶ Run the twin**. On **Maps**, set **Lead day** to 1, then 4, then 8. Optionally open **Watch the shock spread: difference map, days 1–10** and press Play.

**Say.** "Left is the baseline, middle is the shocked run, right is the difference. On day 1 the change sits inside the dashed box. By day 4 and 8 it has spread out from there, and the far side of India is still untouched."

**If it fails.** If the difference does not spread, say so honestly: "this backend damps the shock; the Shock Lab measures exactly how much." Then move on. If the call is slow, talk through the three panels while it loads.

## Step 3: Shock Lab, spread plot and zero-shock check (1:15 to 2:00)

**Click.** Tab **Shock Lab**. Point at the verdict line and the metrics **Day-1 energy near the box**, **Reach on day 1** and **Reach on day 8**. Point at the two charts **Response energy by region** and **Share of energy by distance from the box**. Scroll to **Causality checks**. Then press **▶ Run zero-shock check**.

**Say.** "These are the organisers' own causality checks run on our live output: localised on day 1, spreading by day 4 to 8, never faster than 8 cells a day, stable. The zero-shock check makes two fresh calls, no perturbation versus all zeros. It passes only if the forecasts are identical, and they are: max difference exactly 0."

**If it fails.** If the zero-shock check shows FAIL, say: "our server is deterministic by design; this means the wrong server is running," and check the backend in `/health`. If the server is down, the button shows an error; quote the result from `eval/shock_report.json` instead.

## Step 4: impact panel under monsoon_stall (2:00 to 2:45)

**Click.** **Shock painter** → **Monsoon stall · Arabian Sea winds -40%** → **▶ Run the twin**. Tab **Impact**.

**Say.** "This is the crop-stress index for six approximate farming regions, the dashed boxes on the map. The table shows, baseline → shocked, the **Rain deficit** against the 2005–2018 normal, **Hot days** (days when the region's t2m anomaly is above the slider, 2 K by default; t2m is a daily mean, so these are anomalies, not heatwave maxima) and the 0–100 **Index**. The bars show **Δ crop-stress index (shocked − baseline)**. The stalled monsoon hits the **West coast** box first, and with the model loaded the rain change spreads into Central India." Point at the green or blue banner above the map: it names the region that changes most, or says that nothing moved.

**If it fails.** On the persistence stub, monsoon_stall changes only winds, so the banner says Δ = 0 everywhere. Say so honestly and switch to **Warm ocean** → **▶ Run the twin**. Then move **Hot day: t2m anomaly above (K)** to 2.5: **East coast** hot days go from 0 to 10 and its index from 47 to 87 (+40). If the tab shows an error, go to **Maps** with **Display channel** `u10` and `tp` and say: "Here are the weaker winds and the rain change that the index is computed from." Do not show numbers that are not live.

## Step 5: ensemble spread (2:45 to 3:15)

**Click.** Tab **Ensemble**. Leave **Members (live calls)** at 8 and **Input noise (% of channel std)** at 0.5. Press **▶ Run 8-member ensemble**. Point at the spread map and the **Spread vs lead day** chart.

**Say.** "The server itself is deterministic. Here the dashboard nudges the 3 start days with tiny noise and makes 8 live calls. Where the members disagree, the forecast is less certain, and the chart shows how that grows with lead day." `[TO CONFIRM: Aether]`: which channel actually spreads most with the final model.

**If it fails.** If it is slow, cut to 4 members with the slider and say "each member is one live call." If the server is down, skip the step and say what the tab shows.

## Step 6: model card, one honest limit (3:15 to 3:45)

**Click.** Tab **Model card**. Point at the **Scorecard** tiles (day 1, 4 and 8 skill). Scroll to **Known limits**.

**Say.** "Skill against the better of persistence and climatology, scored with the organisers' code on 2019–2021. One honest limit: skill falls with lead time, and beyond about a week climatology is hard to beat."

**If it fails.** If the scores file is missing, the tab says so. Quote the numbers from the brief's results table instead.

## Step 7: hand the mouse to a judge (3:45 onward)

**Click.** **Shock painter** → **Custom box**. Show the judge **Latitude box °N**, **Longitude box °E**, **Shock channel**, **Magnitude as** and **Size**. Let them press **▶ Run the twin**, then open **Maps** and **Shock Lab**.

**Say.** "Paint anything you like: any box, any channel, absolute or percent. Every result is computed live."

**If it fails.** If the box shows "The box contains no grid cells; widen it.", widen the sliders. If a huge shock gives odd values, explain that outputs are clamped to valid physics (rain ≥ 0, t2m within −40 to +60 °C).
