# Weights hand-over contract (Aether -> serve)

The server doesn't need any of your training code. Drop **one TorchScript file** at `weights/model.pt`
(plus an optional `weights/model_meta.json`) and `twin.py` picks it up at start-up. Nothing else changes.

## The interface (frozen)

```
forward(states: Tensor[3, 6, 32, 36], doy_sincos: Tensor[2]) -> Tensor[6, 32, 36]
```
- `states`: 3 consecutive days, oldest first, **physical units**, float32, channel order
  `t2m K, tp mm/day, u10 m/s, v10 m/s, msl Pa, tcwv kg/m2`, lat south -> north (6.5..37.5), lon west -> east.
- `doy_sincos`: `[sin(2*pi*doy/365), cos(2*pi*doy/365)]` of the **day being predicted** (Twin passes doy+k at step k).
- Return: the next day, **physical units**, float32, shape `[6, 32, 36]`. No batch dimension in or out.
- Everything model-specific goes **inside** the module: normalisation (data/stats.json), static maps
  (land-sea mask, orography), log1p/expm1 for rain, the residual add, de-normalisation.

What Twin does (you don't): the autoregressive loop (slides the 3-day window, feeding your output back),
`model.eval()`, `torch.inference_mode()`, `torch.set_num_threads(4)`, CPU, float32. Between steps it clamps
`tp >= 0` and `t2m` to [-40, 60] C. If your module returns the wrong shape or any NaN/Inf, the whole
rollout falls back to persistence (logged + counted). At load, Twin runs a 2-step warm-up on a constant
plausible state; if that fails the server stays on persistence and `/health` shows `backend` and `load_error`.

## Name

`weights/model_meta.json`: `{"model_name": "aether-unet-v3", "trained_on": "...", "val_rmse": ...}`.
Only `model_name` is read (shown in `/health`). Or set `self.model_name = "..."` in `__init__` (str attribute).
Default: `torchscript:model.pt`.

## Export recipe

1. Stats: `json.load(open("data/stats.json"))` -> `mean`, `std` lists of 6 (same order as above).
   If you trained rain in log1p space, use your own log-space stats for channel 1 instead.
2. Static maps: `xr.open_dataset("data/india_climate_train.nc")["static"].values` -> `[2, 32, 36]`
   (`land_sea_mask`, `orography_m`); normalise orography yourself (e.g. / 1000).
3. Wrap your trained net (example below; adapt the input layout to what your net expects), then
   `torch.jit.script(wrapper.eval()).save("weights/model.pt")`.
   If scripting fails, `torch.jit.trace(wrapper.eval(), (states_example, doy_example))` works too,
   but only if there's no data-dependent Python control flow.

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
        z = (x - self.mean) / self.std                     # [3,6,32,36]
        doy = doy_sincos.view(1, 2, 1, 1).expand(1, 2, 32, 36)
        inp = torch.cat([z.reshape(1, 18, 32, 36), self.static, doy], dim=1)  # [1,22,32,36]
        nxt = z[-1:] + self.net(inp)                       # residual on the last day
        y = (nxt * self.std + self.mean)[0]                # [6,32,36]
        y[1] = torch.expm1(y[1]).clamp(min=0.0)            # back to mm/day
        return y
```

Keep `weights/model.pt` small (a ~2M-param net is ~8 MB). `.gitattributes` routes `*.pt` through Git LFS:
if the judge's checkout doesn't pull LFS, the file is a pointer, Twin detects it and stays on persistence
(and repo_scan sees no checkpoint). Check `git lfs ls-files` after you commit.

## Self-test (before handing over)

```
python -c "import torch; m=torch.jit.load('weights/model.pt'); print(m(torch.full((3,6,32,36),1.0)*torch.tensor([295,4,2,1,100500,35.]).view(1,6,1,1), torch.tensor([0.,1.])).shape)"
python serve.py --port 8000                    # log must say backend=model
python tools/validate.py --url http://127.0.0.1:8000
python -m monsoon_r1.schema_validator http://127.0.0.1:8000
python -m pytest tests/test_twin.py -q         # the contract tests use dummy modules
```
`/health` must show `"backend": "model"` and your `model_name`. Measured budget: a 2.1M-param 3-level
conv net does a 10-step rollout in ~60 ms on 4 threads (limit 5 s), so there's lots of headroom.

Note: our dev venv runs Python 3.14 + torch 2.15, where `torch.jit` prints a FutureWarning but works.
Export with the same torch major version as `requirements.txt` (torch==2.15.0).
