# Module interfaces (model lane)

Pinned so the model, training and test work can proceed in parallel. Change a signature here first, then tell the other owners.

Constants (`twin_model/norm.py`): `CHANNELS = ["t2m","tp","u10","v10","msl","tcwv"]`, `RAIN = 1`, `H, W = 32, 36`, `N_IN = 3`.

## twin_model/norm.py
- `to_model(x_phys)` / `from_model(x_mod)`: physical ↔ "model space" (identical except rain: `log1p(max(tp,0))` / `expm1`, clamp ≥ 0). Works on numpy and torch, any leading dims, channel axis = -3.
- `class Normalizer` holding float32 arrays of shape `[6]`: `state_mean`, `state_std` (of model-space states), `delta_std` (of day-to-day change in model space, after dividing by `state_std`, i.e. std of the standardized delta); plus `static_mean`, `static_std` of shape `[2]`.
  - `Normalizer.fit(states_phys [T,6,H,W], static [2,H,W])` — called on training years only.
  - `standardize(x_mod)` / `unstandardize(x_std)`; `to_dict()` / `from_dict(d)` (plain lists, for the checkpoint).

## twin_model/model.py
- `class UNet(nn.Module)`: `__init__(in_ch=22, out_ch=6, base=32, depth=3)`; `forward(x [B,22,H,W]) -> [B,6,H,W]` (raw normalized delta). Replicate/reflect padding; handles 32×36 (pad internally if needed, crop back).
  - Input channels: 3 days × 6 standardized states (oldest first, 18) + 2 standardized static maps + doy sin, cos broadcast (2) = 22.
- `step(model, window_std [B,3,6,H,W], static_std [B,2,H,W], doy_sincos [B,2], norm: Normalizer) -> next_std [B,6,H,W]`
  - `next = last + raw * delta_std` (per channel, in standardized model space). Rain channel is floored so the un-standardized log1p rain is ≥ 0. Default `softplus_beta=0`: straight-through ReLU (forward exact max(x,0), backward identity) — no dry-cell drizzle drift. `softplus_beta>0` uses softplus instead (adds ≈ln2/beta per step to dry cells; experiment only). Output head is zero-initialised, so an untrained model is exact persistence.
- `advance_doy(doy_sincos [B,2], days=1) -> [B,2]` (rotate angle by 2π/365 per day).
- `rollout_std(model, window_std, static_std, doy_sincos, steps, norm) -> [B,steps,6,H,W]` — feeds each prediction back (window shifts by one day, doy advances). Differentiable; used by training (with grad) and Twin (no_grad).

## twin_model/twin.py
- `class Twin`: `__init__(ckpt_path: str | None = "weights/model.pt")`. Load failure or missing file → persistence fallback (logs a warning, `self.fallback = True`). Sets `torch.set_num_threads(4)`, `model.eval()`.
- `rollout(init_states [3,6,H,W], doy_sincos [2], steps: int, perturbation [6,H,W] | None) -> np.ndarray float32 [steps,6,H,W]` physical units.
  - Adds perturbation to `init_states[-1]` (None ≡ zeros). Deterministic. Output passes validity clamps (rain ≥ 0, t2m in 233.15..333.15 K, finite; non-finite → persistence value for that cell, never zeros).
- Checkpoint (`torch.save` dict): `{"model_state", "config": {"base","depth","in_ch","out_ch","softplus_beta"}, "norm": Normalizer.to_dict(), "static": float32 [2,H,W] (physical), "static_names": [..], "meta": {...}}`. Helper `save_checkpoint(path, model, config, norm, static, static_names, meta)` and `load_checkpoint(path)` live in `twin_model/twin.py`.
  - `save_checkpoint`: `config` may be None/partial (missing keys come from `model.config()`); `static` is stored as a float32 torch tensor so the file loads with `weights_only=True`.
  - `load_checkpoint(path, device="cpu") -> dict` with keys `model` (UNet, eval), `norm` (Normalizer), `static` (np.float32 [2,H,W] physical), `static_names`, `config`, `meta`, `raw` (stored dict, for `--resume`).
  - Additive helpers (no signature changes): `UNet(..., softplus_beta=0.0)` + `UNet.config()`; `step(..., softplus_beta=None)` (defaults to `model.softplus_beta`); `Normalizer.standardize_static/unstandardize_static`; `doy_sincos` given to `step` is that of the last day in the window.

  - Optional key `"post"` (output post-processing, `twin_model/post.py`; absent → Twin output unchanged). `load_checkpoint` also returns it as `post` (stored dict or None). All float32 torch tensors, L = 10 fitted leads (leads beyond L reuse the last row):
    - `"blend"` `[L,6]` in [0,1] and `"clim"` `[366,6,H,W]` (physical; `monsoon_r1.stats.climatology` of 2005–2018, kit slots with Feb 29 = slot 59) — given together. `y[k,c] = (1-a[k,c])·y[k,c] + a[k,c]·clim[slot]`, slot = kit slot of (doy from `atan2(doy_sincos)`) + k.
    - `"qmap_fq"`, `"qmap_oq"` `[L,Q]` — rain quantile map per lead (forecast → observed tp knots; `fq` strictly increasing, `oq` non-decreasing, both start at 0 so 0 → 0; slope-1 continuation past the last knot) — given together.
    - `"qmap_strength"` `[L]` (optional, values in [0,1], absent = 1): `tp ← tp + s[k]·(qmap(tp) − tp)`. Tested 2026-11-27 (constant 0.25/0.5/0.75 and linear 1→0.5, 1→0.25): all scored below full strength on real 2023 windows, so the frozen model does not use it.
    - `"meta"` dict (plain types). Order inside `Twin.rollout`: model physical output → non-finite cells to persistence → blend → rain qmap → validity clamps. Exact deterministic functions of the model output (zero perturbation stays bit-identical). A malformed `"post"` logs a warning and Twin runs without it.
  - `twin_model/post.py`: `class Post(blend, clim, qmap_fq, qmap_oq, meta)` with `from_dict/to_dict/apply(y [steps,6,H,W], doy_sincos)`; helpers `doy_from_sincos`, `clim_slot`, `target_slots`, `qmap`.

## training/
- `dataset.py`: `load_nc(path) -> (states [T,6,H,W] float32, dates datetime64[D] [T], static [2,H,W], static_names)`; `make_synthetic(T=…, seed=0)` with the same return signature (smooth advected fields, realistic magnitudes, skewed rain with zeros); `year_split(dates, train=(2005,2018), val=(2019,2021))`; `class WindowDataset(states, dates, static, norm, k)` → items `(window_std [3,6,H,W], static_std [2,H,W], doy_sincos [2], target_std [k,6,H,W])`, only windows whose 3+k days are consecutive calendar days.
- `losses.py`: `rollout_loss(pred_std, target_std, rain_weight=…, grad_weight=…)` — MSE averaged over steps, optional heavier weight on rain, optional spatial-gradient (anti-blur) term.
- `train.py`: CLI (`python -m training.train`) with `--data/--synthetic`, `--stage onestep|multistep`, `--k 2,4,8` (curriculum), `--epochs`, `--lr`, `--batch`, `--input-noise`, `--rain-weight`, `--grad-weight`, `--resume`, `--run-name`, `--seed`. AdamW + cosine; saves `runs/<run>/best.pt` selected on validation rollout score, appends to `experiments.csv`: `run,change,day1_ratio,day4_ratio,day8_ratio,sharpness,heavy_rain_hit,zero_shock_ok,notes`.
- `fit_post.py`: CLI `python -m training.fit_post runs/<run>/best.pt --out runs/<run>/best_post.pt [--no-blend] [--no-qmap] [--windows 600] [--q 160] [--blend-cap 1.0] [--blend-channels t2m,...] [--blend-min-lead 1]`. CPU only. Fits the `"post"` key on 2005–2018 windows only (lat-weighted closed-form blend per lead/channel, then rain qmap on the blended rain), prints a held-out 2019–2021 before/after summary, writes a copy of the checkpoint with `"post"` added.
