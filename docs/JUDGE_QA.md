# Judge Q&A: answers anyone on the team can give

Short answers, grounded in our docs and the organisers' contract. `[TO CONFIRM: Aether]` marks model facts to check against the final training run. Rehearse at hour 19, each person answering about another person's lane.

## The five must-answer questions

**1. Why a U-Net?**
Weather on a grid is local. What happens in a cell depends mostly on its neighbours, and convolutions are built for exactly that. The U-Net's down-and-up path sees both local detail and the larger pattern, and the skip connections keep fine detail, which fights blur. It is small, fast on CPU and reliable to train in 20 hours for a team new to PyTorch. We predict the change to the next day (residual), so "no change" is the easy default `[TO CONFIRM: Aether]`.

**2. How are drift and blur prevented?**
Drift: we fine-tune on 2-, 4- and 8-step rollouts, so the model trains on its own fed-back errors. Small input noise during 1-step training also helps `[TO CONFIRM: Aether]`. Blur: a gradient or spectral loss penalises lost detail `[TO CONFIRM: Aether]`, and we choose the checkpoint by day-4 and day-8 skill plus sharpness, not by training loss. Clamps between steps keep rain ≥ 0 and t2m within −40 to +60 °C. The judge checks day-8 variance stays within 40–250% of the truth, and our scorer runs that same check.

**3. Why does a shock stay local at first?**
Each forecast step is a stack of local convolutions, with no global or fully connected layers `[TO CONFIRM: Aether]`, so on day 1 a change mostly reaches nearby cells. If it spread too far, the playbook fix is a shallower U-Net. The model predicts a change per cell, so far-away cells keep following their own weather. Over the following days the effect is passed on step by step, so it spreads outward. The Shock Lab shows this live with the organisers' own checks: at least 60% of day-1 energy within 3 cells of the box, then spreading no faster than 8 cells a day. Our measured day-1 share is `[RESULT]`.

**4. What fails at day 10?**
Skill falls with lead time. Even for the baselines, persistence wins at day 1 but climatology wins from day 4 on (`eval/BASELINE.md`), and beating climatology by a lot at long leads is hard. Small errors build up over 10 fed-back steps, rain gets either too smooth or too sharp, and the box edges get no information from outside the domain. We show this honestly in the model card instead of hiding it. Blending towards climatology after day 7 was a stretch idea and is not built.

**5. What in the twin is learned rather than physical?**
The dynamics are entirely learned: how the 6 fields change from one day to the next, learned from 2005–2018 data. What is imposed by us: the units, non-negative rain (softplus, plus a clamp) `[TO CONFIRM: Aether]`, the t2m bounds, the static land-sea and orography maps as inputs, the day-of-year signal and the residual form. There is no conservation law. A moisture-budget loss was judged too risky for 20 hours.

## Likely follow-ups

**Why not a transformer or a graph network?**
Our grid is only 32 × 36, and we had one GPU and 20 hours. A transformer's global attention would also let a shock reach the whole map on day 1, which the judge's localisation check penalises. A small CNN is boringly reliable, and reliable is what scores here.

**How do you know the shock response is causal, not noise?**
The server is deterministic, so any difference between the shocked and baseline runs comes from the shock alone. The zero shock gives exactly zero difference. The Shock Lab runs the organisers' own `causality_checks` on live output: non-zero, localised, propagating without teleporting, and stable below 5 std. You can paint your own box and watch it happen.

**How is the null test exact?**
A zero perturbation adds exactly 0 to the last input day, so the input is bit-identical. The model runs in eval and inference mode on CPU, with no dropout and no random sampling `[TO CONFIRM: Aether]`. So the output is bit-identical too. Our own test demands max |difference| = 0, stricter than the judge's 1e-3 of each channel's std.

**Why a persistence fallback? Isn't that cheating?**
It is a safety net, not the forecast. It triggers only if the weights fail to load or a step produces NaN/Inf. Every fallback is logged loudly and counted in `/health`, and the backend name is shown in the dashboard. Without it, one bad step would crash the server, and a failed window scores 0.

**How did you prevent data leakage?**
We split by year: train 2005–2018, validate 2019–2021. We never split by random day, because neighbouring days are nearly identical. Climatology and the heavy-rain threshold are computed from training years only. The judge's hidden test dates are after 2023-01-10, and no dates or weather tables are stored in the server.

**What does the ensemble mean?**
It shows how sensitive the forecast is to small errors in the starting state. The dashboard adds tiny noise (default 0.5% of channel std) to the 3 start days and makes 8 live calls. The server itself stays deterministic. It is not a calibrated probability, and the model card says so.

**How fast is it on CPU?**
A 10-step rollout on the persistence backend takes about 100 ms. A 2.1M-parameter test conv net took about 60 ms on 4 threads. The judge's limit is 5 s on 4 cores. The final model: `[TO CONFIRM: Vishwak]`. The model loads and warms up before `/health` says ready, so no scored call pays the start-up cost.

**Why predict the change rather than the full next day?**
Day-to-day changes are small next to the fields themselves, so predicting the change is an easier learning problem. It also makes the model start from persistence, which is the strong baseline at day 1. Each change is normalised per channel, so rain and pressure count equally `[TO CONFIRM: Aether]`.

**How do you score yourselves?**
`eval/score.py` imports the organisers' own metric code: lat-weighted RMSE skill, heavy-rain CSI at 10 mm/day, the validity gate and the anti-blur ratio. `eval/shock_test.py` uses the kit's own shocks and causality checks. We score many validation windows, not just 5, so the numbers are stable.

**What would you do with more time?**
From the playbook's stretch list: a second model with a 2-day step, or blending towards climatology after day 7, to fix late-lead error. Then a low-pressure-centre track for Bay of Bengal shocks and Hindi labels checked by a fluent speaker. A physics term such as a moisture budget only with proper time to test it.
