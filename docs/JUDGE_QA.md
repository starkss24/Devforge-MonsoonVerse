# Judge Q&A: answers anyone on the team can give

Short answers, grounded in our docs and the organisers' contract. Model facts confirmed by Aether against the final model (`aether-unet-v2-ens2`; numbers in `docs/MODEL_V2.md`). Rehearse at hour 19, each person answering about another person's lane.

## The five must-answer questions

**1. Why a U-Net?**
Weather on a grid is local. What happens in a cell depends mostly on its neighbours, and convolutions are built for exactly that. The U-Net's down-and-up path sees both local detail and the larger pattern, and the skip connections keep fine detail, which fights blur. It is small (2 M parameters), fast on CPU (about 50 ms for 10 days) and reliable to train in 20 hours for a team new to PyTorch: one training run takes about 10 minutes on our laptop GPU. We predict the change to the next day (residual), and the output layer starts at zero, so the untrained model is exactly persistence and training only has to learn the departures.

**2. How are drift and blur prevented?**
Drift: we fine-tune on 2-, 4- and 8-step rollouts, back-propagating through the whole chain, so the model trains on its own fed-back errors; that step alone took us from 28 to 60 of 70. (We also tried input noise during training: it scored worse, 52, so it is not used.) Blur: a gradient-*magnitude* loss makes the forecast's spatial gradients as strong as the truth's; our first version, a plain gradient-matching loss, rewarded hedging and did nothing, which we measured and replaced. A per-lead rain quantile map then restores heavy-rain intensity. We choose the checkpoint by day-4 and day-8 skill plus sharpness, not by training loss. Clamps between steps keep rain ≥ 0 and t2m within −40 to +60 °C. The judge checks day-8 variance stays within 40–250% of the truth, and our scorer runs that same check.

**3. Why does a shock stay local at first?**
Each forecast step is a stack of convolutions with no fully connected or attention layers, so on day 1 a change mostly reaches nearby cells. Two parts can see further: the deepest U-Net level works at 1/8 resolution, and GroupNorm normalises over the whole map, so a shock nudges far cells very slightly. That is why we measure locality instead of assuming it. If it spread too far, the playbook fix is a shallower U-Net. The model predicts a change per cell, so far-away cells keep following their own weather. Over the following days the effect is passed on step by step, so it spreads outward. The Shock Lab shows this live with the organisers' own checks: at least 60% of day-1 energy within 3 cells of the box, then spreading no faster than 8 cells a day. Our measured day-1 share: 93% for warm_ocean and 95% for monsoon_stall (20 windows, 2019–2025), and 95% for a painted north-India heatwave, which then spreads on all 20 windows.

**4. What fails at day 10?**
Skill falls with lead time. Even for the baselines, persistence wins at day 1 but climatology wins from day 4 on (`eval/BASELINE.md`), and beating climatology by a lot at long leads is hard. Small errors build up over 10 fed-back steps, rain gets either too smooth or too sharp, and the box edges get no information from outside the domain. By day 8 our skill against climatology is about zero on every channel (t2m −0.02, rain 0.00), against +0.15 to +0.33 on day 1. So we blend towards climatology with a weight that grows with lead time, fitted on training years and capped at 0.5 so a painted shock is damped, never erased. We show this honestly in the model card instead of hiding it.

**5. What in the twin is learned rather than physical?**
The dynamics are entirely learned: how the 6 fields change from one day to the next, learned from 2005–2018 data. What is imposed by us: the units, non-negative rain (an exact floor at 0 in log space, plus a clamp), the t2m bounds, the static land-sea and orography maps as inputs, the day-of-year signal, the residual form, and the post-processing (climatology blend and rain quantile map, fitted on training years). There is no conservation law. A moisture-budget loss was judged too risky for 20 hours.

## Likely follow-ups

**Why not a transformer or a graph network?**
Our grid is only 32 × 36, and we had one GPU and 20 hours. A transformer's global attention would also let a shock reach the whole map on day 1, which the judge's localisation check penalises. A small CNN is boringly reliable, and reliable is what scores here.

**How do you know the shock response is causal, not noise?**
The server is deterministic, so any difference between the shocked and baseline runs comes from the shock alone. The zero shock gives exactly zero difference. The Shock Lab runs the organisers' own `causality_checks` on live output: non-zero, localised, propagating without teleporting, and stable below 5 std. You can paint your own box and watch it happen.

**How is the null test exact?**
A zero perturbation adds exactly 0 to the last input day, so the input is bit-identical. The model runs in eval and inference mode on CPU, with no dropout and no random sampling, and both ensemble members and the post-processing are fixed functions of the input and the date. So the output is bit-identical too. Our own test demands max |difference| = 0, stricter than the judge's 1e-3 of each channel's std.

**Why a persistence fallback? Isn't that cheating?**
It is a safety net, not the forecast. It triggers only if the weights fail to load or a step produces NaN/Inf. Every fallback is logged loudly and counted in `/health`, and the backend name is shown in the dashboard. Without it, one bad step would crash the server, and a failed window scores 0.

**How did you prevent data leakage?**
We split by year: train 2005–2018, validate 2019–2021. We never split by random day, because neighbouring days are nearly identical. Climatology and the heavy-rain threshold are computed from training years only. The judge's hidden test dates are after 2023-01-10, and no dates or weather tables are stored in the server.

**What does the ensemble mean?**
It shows how sensitive the forecast is to small errors in the starting state. The dashboard adds tiny noise (default 0.5% of channel std) to the 3 start days and makes 8 live calls. The server itself stays deterministic. It is not a calibrated probability, and the model card says so.

**How fast is it on CPU?**
A 10-step rollout on the persistence backend takes about 100 ms. The final model (two 2 M-parameter U-Nets, onnx) takes about 50 ms on 4 threads, about 0.1 s per call including JSON. The judge's limit is 5 s on 4 cores. Measured through `serve.py` on our Windows laptop: about 80 ms per 10-day rollout on the server, 0.16 s per call including JSON and HTTP, `/health` ready in 4-7 s. The model loads and warms up before `/health` says ready, so no scored call pays the start-up cost.

**Why predict the change rather than the full next day?**
Day-to-day changes are small next to the fields themselves, so predicting the change is an easier learning problem. It also makes the model start from persistence, which is the strong baseline at day 1. Each change is normalised per channel by the spread of its own day-to-day changes, so rain and pressure count equally (rain then gets double weight in the loss, because heavy rain is scored separately).

**How do you score yourselves?**
`eval/score.py` imports the organisers' own metric code: lat-weighted RMSE skill, heavy-rain CSI at 10 mm/day, the validity gate and the anti-blur ratio. `eval/shock_test.py` uses the kit's own shocks and causality checks. We score many validation windows, not just 5, so the numbers are stable.

**Why two models averaged (an ensemble)?**
Two networks trained the same way but from different random starts make partly independent errors; averaging their predicted change each day cancels some of them. It raised our validation score from 65.8 to 67.1 and the real 2023 score from 55.3 to 59.1, and it costs about 25 ms. It is still deterministic: same input, same output.

**Isn't the post-processing a lookup table of past weather?**
No dates or past weather are stored. It is a one-year climatology (366 average days, which the organisers' repo scan explicitly allows) plus a curve per lead day that maps forecast rain amounts to observed ones, both fitted on 2005–2018 only. It is applied to our own forecast, always the same way, so a zero shock still gives an identical result.

**Why does monsoon_stall not spread?**
It is our honest weak spot. The shock (Arabian Sea winds −40%) is local and stable, but it spreads outward on only about 5 of 20 windows. We checked: the raw network behaves the same, so it is not caused by the post-processing. The network learned that a weakened wind field decays back to normal rather than pushing its effect inland. Fixing it needs retraining with a propagation-aware objective, which we could not test properly in the time. A temperature shock like the north-India heatwave spreads on all 20 windows.

**How did you test on unseen dates?**
The hidden test uses dates after 2023-01-10. We used the organisers' own data pipeline to download 12 real windows from 2023–2025 (all seasons) and scored every candidate on them with the kit's code. That is how we caught two ideas that looked good on validation but were worse on real new data (a plain ensemble without capped post-processing, and retraining on all years).

**What would you do with more time?**
Fix monsoon_stall propagation (a propagation-aware loss or a shallower network, tested on the 12 real windows). Then a second model with a 2-day step for late leads, a low-pressure-centre track for Bay of Bengal shocks, and Hindi labels checked by a fluent speaker. A physics term such as a moisture budget only with proper time to test it.
