# Documentation index

**Start here**

| File | What it is |
|---|---|
| [brief.pdf](brief.pdf) / [BRIEF.md](BRIEF.md) | The two-page brief for the judges (PDF built from the Markdown with `python docs/build_brief_pdf.py`) |
| [PROJECT_EXPLAINED.md](PROJECT_EXPLAINED.md) | The whole project in plain words: problem, data, model, results, shocks, limits, glossary |
| [PITCH_GUIDE.md](PITCH_GUIDE.md) | Pitch prep: 60-second pitch, 2/5/10-minute scripts, 47 likely judge questions with answers, numbers to memorise |
| [DEMO_SCRIPT.md](DEMO_SCRIPT.md) | What to click and say in the live demo |
| [JUDGE_QA.md](JUDGE_QA.md) | Likely judge questions with short, honest answers |

**Model and results**

| File | What it is |
|---|---|
| [MODEL_V21.md](MODEL_V21.md) | The served model `aether-unet-v21-ens4`: design, scores, shock behaviour, limits |
| [MODEL_V2.md](MODEL_V2.md) | The previous model and the experiments that led to the current one |
| [DATA.md](DATA.md) | The data, the year split, persistence and climatology baselines, figures in `img/` |
| [IMPACT.md](IMPACT.md) | The crop-stress index behind the dashboard's Impact tab |

**Engineering**

| File | What it is |
|---|---|
| [CONTRACT.md](CONTRACT.md) | The organisers' API contract, summarised |
| [WEIGHTS.md](WEIGHTS.md) | How the model is exported so the server loads it (`model.onnx`, `model.pt`, `post.npz`) |
| [interfaces.md](interfaces.md) | Model-lane module interfaces (`twin_model/`, `training/`) |

**Team working notes** (how the work was split and handed over; kept for transparency)

| File | What it is |
|---|---|
| [STATUS.md](STATUS.md) | Running log of finished tasks, with the commands and results that verified them |
| [PROGRESS.md](PROGRESS.md) | Progress tracker and decisions |
| [HANDOFF_VISHWAK.md](HANDOFF_VISHWAK.md), [HANDOFF_VISHWAK_FINAL.md](HANDOFF_VISHWAK_FINAL.md), [HANDOFF_FRONTEND.md](HANDOFF_FRONTEND.md) | Hand-off briefs between the model lane, the server lane and the dashboard |
