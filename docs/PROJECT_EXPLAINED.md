# Multiverse of Monsoon: the whole project, explained for the team

This is the "explain it like I'm new" guide to what we built. Read it once before the presentation; you should be able
to answer any judge question after it. No prior machine-learning knowledge needed.

---

## 1. The problem in one paragraph

We built a **digital twin of India's weather**: a program that takes the last **3 days** of weather over India and
**predicts the next 10 days**, day by day. It also answers **"what if"** questions: if someone "paints" a change on the
map (for example "make the Bay of Bengal 3 °C warmer"), the twin shows how that change plays out over the following
days. The organisers grade it automatically and then judges watch a live demo.

**What it is not:** it is not a climate model for decades, not a physics simulator solving equations, and **not
reinforcement learning (RL)**. It is **supervised deep learning**: a neural network that learned weather patterns by
looking at 14 years of real weather and being corrected every time it guessed wrong.

---

## 2. The data

- **Source:** ERA5, the standard global weather record (a mix of observations and physics, by the European weather
  centre), already cut out for India by the organisers: file `india_climate_train.nc` (download it from the kit's
  release page; it is **never committed** to GitHub).
- **Time:** one map per day, **2005 to 2021** (6,209 days).
- **Space:** a grid of **32 × 36 cells**, each **1° × 1°** (about 110 km × 110 km), covering latitude 6.5–37.5° N and
  longitude 65.5–100.5° E (all of India plus the Arabian Sea and Bay of Bengal).
- **6 weather variables ("channels")** per cell per day:

| Channel | Meaning | Unit |
|---|---|---|
| `t2m` | air temperature 2 m above ground (daily mean) | kelvin (K); °C = K − 273.15 |
| `tp` | total precipitation (rain) in the day | mm/day |
| `u10` | wind blowing east (+) / west (−) at 10 m | m/s |
| `v10` | wind blowing north (+) / south (−) at 10 m | m/s |
| `msl` | air pressure at sea level | pascal (Pa); 1000 Pa = 10 hPa |
| `tcwv` | total water vapour in the air column ("how moist") | kg/m² |

- **2 fixed maps ("static"):** land-sea mask (land or sea) and orography (height of the ground, e.g. Himalayas).

So one day is a stack of 6 maps of 32 × 36 numbers; one input to the twin is 3 such days.

---

## 3. What the organisers test (how we get points)

The judge runs our code **offline** on their computer: `python serve.py --port 8000`, then sends **8 requests**:
5 real 3-day windows from dates **after January 2023** (we never saw those) and 3 "shock" requests.

| Part | Points | What it measures |
|---|---|---|
| Forecast skill (RMSE) | 30 | How close our 10-day forecasts are to what really happened, on days 1, 4 and 8 |
| Heavy-rain skill (CSI) | 15 | Do we predict heavy rain (≥ 10 mm/day) in the right places |
| Anti-blur | 10 | Day-8 maps must keep realistic detail (not a smooth smear) |
| Shock causality | 15 | A painted shock must start **local** on day 1, then **spread** by days 4–8, and stay **stable** |
| Repo check | 15 | Code is clean: no data files, no cheating tricks, training code + frontend present |
| Brief | 15 | Our 2-page write-up (`docs/BRIEF.md`) |

**"Skill" means: better than a simple guess.** The judge compares us with two dumb forecasts and takes whichever of
the two is better on each day:
- **Persistence:** "tomorrow = today". Excellent for 1 day, useless after a week.
- **Climatology:** "the usual weather for this date" (average of past years). Useless for tomorrow, hard to beat after
  a week.

Hard rules to qualify: beat persistence on day 4, never output impossible numbers (no NaN, rain never negative,
temperature within −40 to +60 °C), and respond locally to at least one shock.

---

## 4. How the model works (the neural network, in plain words)

### 4.1 The basic idea
A neural network is a big function with ~2 million adjustable numbers ("weights"). It takes the 3 input days and
outputs tomorrow. **Training** = showing it thousands of real examples ("here were 3 days, here is what happened
next") and nudging the weights a tiny bit after each one so its guesses get closer to reality. After enough examples,
it has learned typical weather behaviour: how rain bands move, how temperature follows the seasons, how winds and
pressure relate.

### 4.2 The architecture: a U-Net (a kind of CNN)
- **CNN (convolutional neural network):** looks at each cell together with its neighbours (like a 3 × 3 magnifying
  glass sliding over the map). Weather is local, so this fits well.
- **U-Net:** the map is shrunk step by step (32×36 → 16×18 → 8×9 → 4×5) so the network sees bigger patterns
  (monsoon-scale), then grown back to full size, with "skip connections" that carry the fine detail across. Shape of
  a "U", hence the name.
- **Input:** 22 maps = 3 days × 6 channels + 2 static maps + 2 maps encoding the day of the year (so it knows the
  season).
- **Output:** the **change** from the last day to tomorrow (not tomorrow itself). Predicting the change makes "no
  change" the default, which is a strong starting point.

### 4.3 Rolling forward 10 days ("autoregressive")
The network only predicts **one day**. To get 10 days, we feed its own prediction back in: days 1–3 → day 4; days 2–4
→ day 5; and so on. Small errors pile up over these 10 steps, which is why skill drops with lead time.

### 4.4 Tricks that make it good (each was tested; numbers out of 70 automatic points)
1. **Multi-step training (biggest win: 28 → 60).** First we trained on 1-day guesses only (28/70: day 1 good, later
   days drifted off). Then we fine-tuned on 2-, 4- and 8-day rollouts so the network learns from its own fed-back
   mistakes.
2. **Rain handled carefully.** Rain is skewed (mostly 0, sometimes huge), so the network works with log(1 + rain), and
   rain is floored at exactly 0 (no negative rain, no fake drizzle on dry days).
3. **Anti-blur loss (60 → 61).** Networks trained only to minimise error learn to "hedge" with blurry maps. We added a
   term that requires the forecast's sharpness (strength of spatial differences) to match reality's.
4. **Post-processing (61 → 66).** Two fixed corrections applied to the finished 10-day forecast, fitted on training
   years only:
   - **Climatology blend:** as days go on, mix the forecast a little toward "usual weather for this date" (weight
     grows with lead time, capped at 50%). Late days become more sensible.
   - **Rain quantile map:** the network under-predicts heavy rain; this stretches its rain amounts so their
     distribution matches real rain (stretch capped at 1.5× so painted shocks don't explode).
5. **Ensemble (66 → 67).** Two networks trained the same way with different random starts; we average their
   predictions every day. Their errors partly cancel.

### 4.5 What we tried and threw away (good for "what did you try?")
Adding noise to inputs during training (worse: 52), a wider network (worse: 58), retraining on all years 2005–2021
(worse on real 2023–2025 data), a plain ensemble without the post-processing caps (good on old data, worse on new
data), an "anti-damping" loss to make wind shocks spread (the raw network spread monsoon_stall on 13/20 windows instead
of 8/20, but after post-processing only 7/20 vs 5/20, at a cost of about 3 points of skill on real 2023 data; see
section 8). All in `experiments.csv` with the reason.

### 4.6 The final model: `aether-unet-v2-ens2`
2 U-Nets (2 M weights each) averaged + post-processing. Exported as `weights/model.onnx` (ONNX = a portable model file
that runs fast on CPU without the training code) with a backup `weights/model.pt` (TorchScript) and the
post-processing numbers in `weights/post.npz`.

---

## 5. Results (all computed with the organisers' own scoring code)

| Test | Score (automatic part, out of 70) |
|---|---|
| 100 held-out days from 2019–2021 (never used for training) | **67.1** |
| 12 real windows from 2023–2025 (same period as the judge's test) | **60.3** |
| 4 real 2023 windows | 59.1 |
| Full judge simulation through our server | 64.7 |

**Expect the low 60s from the judge**, because the hidden test is on new dates. Plus repo (we pass the organisers'
own repo scan with 15/15) and brief.

Skill by forecast day (100 held-out windows, "+" = better than the best simple guess):

| Day | Temperature | Rain | Wind E-W | Wind N-S | Pressure | Moisture |
|---|---|---|---|---|---|---|
| 1 | +15% | +20% | +29% | +32% | +33% | +30% |
| 4 | +6% | +3% | +4% | +8% | +2% | +12% |
| 8 | −2% | 0% | −2% | −1% | +1% | +1% |

Read it as: very useful for the first few days, about as good as "usual weather" by day 8. That is normal: weather is
chaotic, and after about a week nobody can predict daily weather much better than the seasonal average.

---

## 6. Shocks ("what if") and why they matter

A **shock** is a change painted on the last input day, e.g. +4 °C over north India. The twin runs twice (with and
without the shock); the **difference** shows the effect. The judge checks the organisers' two shocks:
- **warm_ocean:** Bay of Bengal +3 °C and +15% moisture.
- **monsoon_stall:** Arabian Sea winds −40%.

Each must be **local on day 1** (≥ 60% of the effect within 3 cells of the box), **spread outward by days 4–8**
(without jumping faster than 8 cells/day), and **stay stable** (never more than 5 standard deviations).

How our model behaves (measured on 20 windows, 2019–2025):

| Shock | Local day 1 | Spreads | Notes |
|---|---|---|---|
| Heatwave north India +4 °C | ✅ | ✅ 20/20 | Best demo; still +1.9 °C in the box on day 8 |
| warm_ocean (judge) | ✅ | ✅ 13/20 | Good |
| Bay of Bengal low −1500 Pa | ✅ | rarely | Visible as a strong rain response |
| monsoon_stall (judge) | ✅ | ❌ ~5/20 | Our honest weak spot: the weaker winds recover in place instead of spreading inland |

**Why the null test is exact:** a zero shock adds exactly 0, and the model has no randomness, so "no shock" and "zero
shock" give identical numbers (the judge requires this).

---

## 7. How the pieces fit together (the software)

```
data/india_climate_train.nc  (not in git)
        │  training (GPU, ~10 min per network)       training/  +  twin_model/
        ▼
runs/…/best.pt  (checkpoint, not in git)
        │  export (training/export.py)
        ▼
weights/model.onnx + model.pt + post.npz + model_meta.json   (in git, plain files)
        │  loaded by
        ▼
twin.py  ── Twin.rollout(): runs the 10-day loop + post-processing (postproc.py)
        │  called by
        ▼
serve.py ── web server with 2 routes: /health and the forecast route   ← the judge talks to this
        │  called by
        ▼
dashboard/ (Streamlit app on http://localhost:8501) ← humans and live judges use this
```

| Folder / file | What it is | Lane |
|---|---|---|
| `serve.py`, `twin.py`, `postproc.py`, `common.py` | The server the judge runs | Vishwak |
| `weights/` | The trained model files | Aether makes them, Vishwak commits/verifies |
| `twin_model/`, `training/` | The network, training, evaluation tools | Aether |
| `eval/` | Our scorer and shock tests (organisers' code inside) | Vishwak |
| `data_tools/`, `impact/` | Data loader, baselines, crop-stress impact index | Junaid |
| `dashboard/` | The visual app | Akash (built on Vishwak's machine) |
| `docs/` | Brief, demo script, judge Q&A, model docs, this file | everyone |
| `tests/` | ~250 automatic tests | everyone |

### Run it yourself
```bash
python serve.py --port 8000                 # terminal 1: the twin (needs only the repo)
streamlit run dashboard/app.py              # terminal 2: the app (needs data/ for real start days)
# open http://localhost:8501
```

---

## 8. Limits (say them before the judges find them)

1. **Day 8 ≈ climatology.** Normal for any weather model at this resolution; we blend toward climatology so late days
   stay sensible, which also damps long-range shock effects by up to half.
2. **monsoon_stall rarely spreads.** The network learned that a local wind change usually just dies out. We tried an
   "anti-damping" training loss in the final hours (it penalises forecasts whose departure from normal weather is
   weaker than reality's): the raw network then spread the shock on 13/20 windows instead of 8/20. But the
   post-processing (rain stretching and the climatology blend) took most of that back (7/20 vs 5/20 for v2), and
   forecast skill on real 2023 data dropped by about 3 points, so by our acceptance rule we kept v2. A full fix needs
   post-processing that preserves spreading plus more training data.
3. **Heavy rain peaks are smoothed** (e.g. one test: forecast 37 mm/day where 94 fell).
4. **The ensemble tab's spread shrinks with time** instead of growing like real uncertainty: the model damps small
   differences. It shows stability, not a true probability.
5. **Coarse grid:** 110 km cells; city-level detail is not possible.

---

## 9. Glossary

| Word | Meaning |
|---|---|
| Digital twin | A computer copy of a real system you can run forward and experiment on |
| Supervised learning | Learning from examples with known answers (here: real past weather) |
| RL (reinforcement learning) | Learning by trial and reward in an environment — **not used here** |
| Neural network / weights | A big adjustable function / its ~2 million adjustable numbers |
| CNN / U-Net | Network type that scans neighbourhoods on a grid / one that shrinks then regrows the map |
| Training / epoch | Adjusting weights from examples / one pass over all training examples |
| Loss | The number training tries to make small (how wrong the forecast is, plus our extra terms) |
| Rollout / autoregressive | Running day after day by feeding predictions back in |
| Lead time / day k | How far ahead a forecast is (day 1 = tomorrow) |
| Persistence / climatology | "Tomorrow = today" / "usual weather for this date" |
| RMSE / skill | Typical error size / how much better than the best simple guess (0 = equal, + = better) |
| CSI | Heavy-rain hit score: hits ÷ (hits + misses + false alarms) |
| Blur / variance ratio | Smoothed-out maps / how much spatial detail survives (100% = as detailed as reality) |
| Shock / perturbation | A change painted on the start state to ask "what if" |
| Held-out / validation | Data never used for training, kept to test honestly (2019–2021) |
| Ensemble | Several models averaged |
| Post-processing | Fixed corrections applied after the network's forecast |
| ONNX / TorchScript | Portable model file formats that run without the training code |
| Checkpoint | A saved copy of the network's weights during training |
| Normalisation | Rescaling each variable so they are comparable (done inside the model) |

---

## 10. The five questions everyone must be able to answer

1. **Why a U-Net?** Weather is local and grid-shaped; a U-Net sees local detail and large patterns, keeps fine detail,
   and is small and fast (≈ 50 ms for 10 days on a CPU).
2. **How do you stop drift and blur?** Multi-step training on its own fed-back forecasts (28 → 60 points); an
   anti-blur loss; post-processing that blends toward climatology late and restores heavy rain.
3. **Why does a shock stay local at first?** Convolutions only look at neighbours, so on day 1 a change reaches nearby
   cells; we measured 93–97% of the day-1 effect within 3 cells. It then spreads step by step.
4. **What fails at day 10?** Skill falls to about the seasonal average by day 8 because small errors grow; heavy rain is
   smoothed; monsoon_stall doesn't spread.
5. **What is learned vs physical?** All the weather behaviour is learned from 2005–2018 data. We only impose units,
   non-negative rain, temperature limits, the static maps, the day of year, and the post-processing fitted on training
   years. No physics equations.

More answers: `docs/JUDGE_QA.md`. Full model details: `docs/MODEL_V2.md`. Demo steps: `docs/DEMO_SCRIPT.md`.
