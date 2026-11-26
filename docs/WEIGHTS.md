# Weights hand-over contract (Aether -> serve)

The server doesn't need any of your training code. Export the same wrapper twice and drop both files in `weights/`:

| file | backend in `/health` | runtime |
|---|---|---|
| `weights/model.onnx` | `"onnx"` (**preferred**) | onnxruntime, CPU, 4 intra-op threads, 1 inter-op, all graph optimisations, deterministic |
| `weights/model.pt` | `"torchscript"` (fallback) | `torch.jit.load` |
| neither / both fail | `"persistence"` | numpy |

plus an optional `weights/model_meta.json`. `twin.py` picks them up at start-up. Nothing else changes.

**Why both, and why ONNX first:** on Python 3.14+ torch warns `torch.jit.load is not supported in Python 3.14+
and may break`. The judge's Python version is unknown, so TorchScript alone is a risk; onnxruntime 1.32 ships
wheels for 3.14 (and older) and is on the judge's allowed list. If `model.onnx` is present it wins (logged); if
it is missing, an LFS pointer, corrupt, or fails warm-up, Twin tries `model.pt`, then persistence, and
`/health.load_error` lists every failure. For testing, env `TWIN_BACKEND=onnx|torchscript|persistence` forces one.

## The interface (frozen)

```
step(states: float32[3, 6, 32, 36], doy_sincos: float32[2]) -> float32[6, 32, 36]
```
- `states`: 3 consecutive days, oldest first, **physical units**, float32, channel order
  `t2m K, tp mm/day, u10 m/s, v10 m/s, msl Pa, tcwv kg/m2`, lat south -> north (6.5..37.5), lon west -> east.
- `doy_sincos`: `[sin(2*pi*doy/365), cos(2*pi*doy/365)]` of the **day being predicted** (Twin passes doy+k at step k).
- Return: the next day, **physical units**, float32, shape `[6, 32, 36]`. No batch dimension in or out.
- ONNX names: inputs `states`, `doy_sincos`, output `next_state` (Twin uses the first output; if the two inputs
  have other names it uses them by position). ONNX only: a leading batch dim of 1 is tolerated
  (`states [1,3,6,32,36]`, `doy_sincos [1,2]` -> `[1,6,32,36]`); Twin detects it from the `states` input rank.
- Everything model-specific goes **inside** the exported model: normalisation (data/stats.json), static maps
  (land-sea mask, orography), log1p/expm1 for rain, the residual add, de-normalisation.

What Twin does (you don't): the autoregressive loop (slides the 3-day window, feeding your output back),
`model.eval()` + `torch.inference_mode()` / onnxruntime session, 4 CPU threads, float32. Between steps it clamps
`tp >= 0` and `t2m` to [-40, 60] C. If your model returns the wrong shape or any NaN/Inf, the whole
rollout falls back to persistence (logged + counted). At load, Twin runs a 2-step warm-up on a constant
plausible state (so the first scored call is fast); if that fails the server stays on persistence and
`/health` shows `backend` and `load_error`.

## Name

`weights/model_meta.json`: `{"model_name": "aether-unet-v3", "trained_on": "...", "val_rmse": ...}`.
Only `model_name` is read (shown in `/health`), for both backends. Fallbacks: TorchScript `self.model_name`
str attribute; ONNX metadata prop `model_name`. Default: `onnx:model.onnx` / `torchscript:model.pt`.

## Wrapper (shared by both exports)

1. Stats: `json.load(open("data/stats.json"))` -> `mean`, `std` lists of 6 (same order as above).
   If you trained rain in log1p space, use your own log-space stats for channel 1 instead.
2. Static maps: `xr.open_dataset("data/india_climate_train.nc")["static"].values` -> `[2, 32, 36]`
   (`land_sea_mask`, `orography_m`); normalise orography yourself (e.g. / 1000).
3. Wrap your trained net (adapt the input layout to what your net expects). Keep it static-shape and free of
   data-dependent Python `if` on tensor values (use `torch.where`), so both exporters accept it.

```python
class Exported(nn.Module):
    def __init__(self, net, mean, std, static):            # mean/std: lists of 6; static: [2,32,36]
        super().__init__()
        self.net = net.eval()
        self.register_buffer("mean", torch.tensor(mean).view(1, 6, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 6, 1, 1))
        self.register_buffer("static", torch.as_tensor(static, dtype=torch.float32)[None])  # [1,2,32,36]
        self.model_name = "aether-unet-v1"

    def forward(self, states: torch.Tensor, doy_sincos: torch.Tensor) -> torch.Tensor:
        x = states.clone()
        x[:, 1] = torch.log1p(x[:, 1].clamp(min=0.0))      # rain to log space (if trained that way)
        z = (x - self.mean) / self.std                     # normalise   [3,6,32,36]
        doy = doy_sincos.view(1, 2, 1, 1).expand(1, 2, 32, 36)
        inp = torch.cat([z.reshape(1, 18, 32, 36), self.static, doy], dim=1)  # [1,22,32,36]
        nxt = z[-1:] + self.net(inp)                       # net + residual on the last day
        y = (nxt * self.std + self.mean)[0]                # denormalise [6,32,36]
        y[1] = torch.expm1(y[1]).clamp(min=0.0)            # back to mm/day
        return y
```

## Export 1: ONNX (preferred, `weights/model.onnx`)

Needs `pip install -r requirements-dev.txt` (onnx + onnxscript; the server itself only needs onnxruntime).

```python
import onnx, torch
w = Exported(net, mean, std, static).eval()
ex = (torch.full((3, 6, 32, 36), 1.0) * torch.tensor([295, 4, 2, 1, 100500, 35.]).view(1, 6, 1, 1),
      torch.tensor([0.0, 1.0]))
torch.onnx.export(w, ex, "weights/model.onnx",
                  input_names=["states", "doy_sincos"], output_names=["next_state"],
                  opset_version=18,          # >= 17; the torch 2.15 (dynamo) exporter emits 18+
                  do_constant_folding=True,  # folds mean/std/static maps into constants
                  external_data=False,       # ONE self-contained file, no model.onnx.data next to it
                  verbose=False)             # verbose prints emoji that crash a cp1252 Windows console
m = onnx.load("weights/model.onnx"); onnx.checker.check_model(m)
onnx.helper.set_model_props(m, {"model_name": "aether-unet-v1"}); onnx.save(m, "weights/model.onnx")
```

Verify it on onnxruntime against the PyTorch wrapper:

```python
import numpy as np, onnxruntime as ort
s = ort.InferenceSession("weights/model.onnx", providers=["CPUExecutionProvider"])
y = s.run(["next_state"], {"states": ex[0].numpy(), "doy_sincos": ex[1].numpy()})[0]
print(y.shape, np.abs(y - w(*ex).detach().numpy()).max())   # (6, 32, 36), float32 round-off only
```

`tests/test_twin.py` exports dummy nets the same way: ONNX and TorchScript rollouts of the same net agree to
float32 round-off (day 1 within 1e-6 relative, 10 days within 1e-5 relative).

## Export 2: TorchScript (fallback, `weights/model.pt`)

`torch.jit.script(w).save("weights/model.pt")`. If scripting fails,
`torch.jit.trace(w, ex)` works too (no data-dependent control flow).

## File size and git

A ~2M-param net is ~8.5 MB as `.onnx` and about the same as `.pt`. **Team recommendation (pending decision):
commit `model.onnx` as a plain git file, not LFS**, so the judge's clone has the real bytes even without
`git lfs pull`. Today `.gitattributes` still routes `*.onnx` and `*.pt` through LFS: drop the `*.onnx` line
before committing it (team decision; `scripts/repo_scan_lite.py` currently expects `*.pt` in LFS). If a file
does arrive as an LFS pointer, Twin detects it (both formats), skips it and says so in `/health.load_error`.
Check `git lfs ls-files` after you commit.

## Self-test (before handing over)

```
python -c "import onnxruntime as o, numpy as np; s=o.InferenceSession('weights/model.onnx'); print(s.run(None, {'states': np.ones((3,6,32,36),'f4')*np.array([295,4,2,1,100500,35],'f4').reshape(1,6,1,1), 'doy_sincos': np.array([0,1],'f4')})[0].shape)"
python serve.py --port 8000                    # log must say backend=onnx
python tools/validate.py --url http://127.0.0.1:8000
python -m monsoon_r1.schema_validator http://127.0.0.1:8000
TWIN_BACKEND=torchscript python serve.py --port 8000   # check the fallback file too: backend=torchscript
python -m pytest tests/test_twin.py -q         # the contract tests use dummy models
```
`/health` must show `"backend": "onnx"` and your `model_name`. Measured budget: a 2.1M-param 3-level
conv net does a 10-step rollout in ~50 ms with onnxruntime and ~85 ms with TorchScript on 4 threads
(limit 5 s), so there's lots of headroom.

Note: our dev venv runs Python 3.14 + torch 2.15, where `torch.jit` prints a FutureWarning but still works
today; that warning is why ONNX is the primary format. Export with the same torch version as
`requirements.txt` (torch==2.15.0); the server pins onnxruntime==1.32.0.
