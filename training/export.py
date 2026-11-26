"""Export a trained checkpoint to the team's serving format (docs/WEIGHTS.md) and verify it.

    python -m training.export runs/e5b_gradmag/best_post.pt --out export/ --name aether-unet-e5b

Writes into --out:
    model.onnx        one-day step, inputs (states [3,6,32,36], doy_sincos [2]) -> next_state [6,32,36]
    model.pt          the same step as TorchScript (fallback backend of the root twin.py)
    model_meta.json   {"model_name": ...} plus provenance
    post.npz          post-processing for the FINISHED rollout (twin_model/post_numpy.py), if the checkpoint has it

Conventions bridged here (verified below, not assumed):
- WEIGHTS.md passes doy_sincos of the DAY BEING PREDICTED; the net was trained with the doy of the
  LAST INPUT day, so the wrapper rotates the angle back by one day (2*pi/365) before the net.
- Physical units in and out; log1p rain, standardisation, static maps, residual add and the rain
  floor (forward of the straight-through ReLU = exact max(x, 0) in log1p space) live in the wrapper.

Weights are NOT committed by this script: hand the files to the weights owner (only Vishwak commits
to weights/).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from twin_model.norm import RAIN, from_model, to_model
from twin_model.model import rollout_std
from twin_model.twin import load_checkpoint
from twin_model.post import Post
import postproc  # the serving path's numpy post-processing (root postproc.py, docs/WEIGHTS.md format)

DELTA = 2.0 * math.pi / 365.0
H, W = 32, 36


class StepExport(nn.Module):
    """One-day step in physical units: (states [3,6,H,W], doy_sincos of the predicted day [2]) -> [6,H,W]."""

    def __init__(self, net, norm, static_phys, softplus_beta: float = 0.0, model_name: str = "aether-unet"):
        super().__init__()
        self.net = net.eval()
        f = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))
        self.register_buffer("mean", f(norm.state_mean).view(1, 6, 1, 1))
        self.register_buffer("std", f(norm.state_std).view(1, 6, 1, 1))
        self.register_buffer("dstd", f(norm.delta_std).view(1, 6, 1, 1))
        self.register_buffer("static", f(norm.standardize_static(static_phys))[None])  # [1,2,H,W]
        self.beta = float(softplus_beta)
        self.model_name = model_name
        self.cd, self.sd = math.cos(DELTA), math.sin(DELTA)

    def forward(self, states: torch.Tensor, doy_sincos: torch.Tensor) -> torch.Tensor:
        rain = torch.log1p(torch.clamp(states[:, 1:2], min=0.0))
        xm = torch.cat([states[:, :1], rain, states[:, 2:]], dim=1)                     # model space
        z = (xm - self.mean) / self.std                                                  # [3,6,H,W]
        s, c = doy_sincos[0], doy_sincos[1]
        dsc = torch.stack([s * self.cd - c * self.sd, c * self.cd + s * self.sd])        # angle - 1 day
        doy = dsc.view(1, 2, 1, 1).expand(1, 2, H, W)
        inp = torch.cat([z.reshape(1, 18, H, W), self.static, doy], dim=1)               # [1,22,H,W]
        nxt = z[-1:] + self.net(inp) * self.dstd                                         # residual
        y = nxt * self.std + self.mean                                                   # model space
        r = y[:, 1:2]
        r = F.softplus(r, beta=self.beta) if self.beta > 0 else torch.relu(r)            # rain floor
        r = torch.clamp(torch.expm1(r), min=0.0)                                         # mm/day
        return torch.cat([y[:, :1], r, y[:, 2:]], dim=1)[0]                              # [6,H,W]


def team_loop(step_fn, init, last_dsc, steps):
    """Exactly the root twin.py loop: doy of the predicted day, clamps between steps, physical feedback."""
    theta = math.atan2(float(last_dsc[0]), float(last_dsc[1]))
    window = np.ascontiguousarray(init, dtype=np.float32); out = []
    for k in range(1, steps + 1):
        a = theta + 2.0 * math.pi * k / 365.0
        y = np.array(step_fn(window, np.array([math.sin(a), math.cos(a)], np.float32)), np.float32)
        y[1] = np.maximum(y[1], 0.0); y[0] = np.clip(y[0], 233.15, 333.15)
        out.append(y); window = np.concatenate([window[1:], y[None]], 0)
    return np.stack(out)


def reference_raw(ck, init, last_dsc, steps):
    """The training-time rollout (twin_model), without post-processing or clamps."""
    norm, model = ck["norm"], ck["model"]
    stt = torch.from_numpy(np.ascontiguousarray(norm.standardize_static(ck["static"]), np.float32))[None]
    win = torch.from_numpy(np.ascontiguousarray(norm.standardize(to_model(init)), np.float32))[None]
    with torch.no_grad():
        o = rollout_std(model, win, stt, torch.tensor(np.asarray(last_dsc, np.float32))[None], steps, norm)
    return from_model(norm.unstandardize(o[0].numpy().astype(np.float32)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--out", default="export")
    ap.add_argument("--name", default="aether-unet-e5b"); ap.add_argument("--data", default="data/india_climate_train.nc")
    ap.add_argument("--scores", default="", help='JSON of measured scores for model_meta.json, e.g. \'{"val100": 67.06}\'')
    a = ap.parse_args()
    torch.set_num_threads(4)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ck = load_checkpoint(a.ckpt, device="cpu")
    cfg = ck["config"]
    w = StepExport(ck["model"], ck["norm"], ck["static"], cfg.get("softplus_beta", 0.0), a.name).eval()

    # real test windows (validation years) for verification
    from training.dataset import load_nc
    states, dates, _, _ = load_nc(a.data)
    days = ["2019-01-15", "2019-07-20", "2020-04-10", "2021-08-10", "2021-10-25"]
    wins = []
    for d in days:
        i = int(np.where(dates == np.datetime64(d))[0][0]); dd = dates[i].astype(object)
        doy = dd.timetuple().tm_yday
        wins.append((d, states[i - 2:i + 1].astype(np.float32),
                     np.array([math.sin(2 * math.pi * doy / 365), math.cos(2 * math.pi * doy / 365)], np.float32)))

    # 1) wrapper loop vs training-time rollout
    tstep = lambda s, d: w(torch.from_numpy(s), torch.from_numpy(d)).detach().numpy()
    with torch.no_grad():
        e = max(float(np.abs(team_loop(tstep, x, dsc, 10) - reference_raw(ck, x, dsc, 10)).max() /
                      (np.abs(reference_raw(ck, x, dsc, 10)).max() + 1e-6)) for _, x, dsc in wins)
    print(f"[1] wrapper + team loop vs training rollout, 10 days: max rel diff {e:.2e}")

    # 2) TorchScript
    ex = (torch.from_numpy(wins[0][1]), torch.tensor([0.0, 1.0]))
    try:
        ts = torch.jit.script(w); how = "script"
    except Exception as err:  # noqa: BLE001 - trace is the documented alternative
        print(f"    jit.script failed ({type(err).__name__}); using jit.trace"); ts = torch.jit.trace(w, ex); how = "trace"
    ts.save(str(out / "model.pt"))
    ts2 = torch.jit.load(str(out / "model.pt"), map_location="cpu").eval()
    with torch.no_grad():
        e_ts = max(float(np.abs(team_loop(lambda s, d: ts2(torch.from_numpy(s), torch.from_numpy(d)).numpy(), x, dsc, 10)
                                - team_loop(tstep, x, dsc, 10)).max()) for _, x, dsc in wins)
    print(f"[2] TorchScript ({how}) model.pt: 10-day max abs diff vs wrapper {e_ts:.2e}")

    # 3) ONNX
    import onnx, onnxruntime as ort
    torch.onnx.export(w, ex, str(out / "model.onnx"), input_names=["states", "doy_sincos"],
                      output_names=["next_state"], opset_version=18, do_constant_folding=True,
                      external_data=False, verbose=False)
    m = onnx.load(str(out / "model.onnx")); onnx.checker.check_model(m)
    onnx.helper.set_model_props(m, {"model_name": a.name}); onnx.save(m, str(out / "model.onnx"))
    so = ort.SessionOptions(); so.intra_op_num_threads = 4; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(out / "model.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
    ostep = lambda s, d: sess.run(["next_state"], {"states": s, "doy_sincos": d})[0]
    with torch.no_grad():
        refs = [team_loop(tstep, x, dsc, 10) for _, x, dsc in wins]
    e_ox = max(float(np.abs(team_loop(ostep, x, dsc, 10) - r).max() / (np.abs(r).max() + 1e-6))
               for (_, x, dsc), r in zip(wins, refs))
    print(f"[3] ONNX model.onnx: 10-day max rel diff vs wrapper {e_ox:.2e}")
    t = []
    for _ in range(5):
        t0 = time.perf_counter(); team_loop(ostep, wins[0][1], wins[0][2], 10); t.append(time.perf_counter() - t0)
    print(f"    ONNX 10-step rollout on 4 threads: median {1000 * np.median(t):.0f} ms")

    # 4) post-processing file + equivalence with the full reference Twin
    post = ck["raw"].get("post")
    if post is not None:
        pp = Post.from_dict(post)
        postproc.save_npz(out / "post.npz", pp.blend, pp.clim, pp.fq, pp.oq, pp.strength,
                          meta={"model_name": a.name, "source_checkpoint": str(a.ckpt).replace("\\", "/")})
        p = postproc.PostProc.load(out / "post.npz")
        from twin_model.twin import Twin
        ref_twin = Twin(a.ckpt)
        def served(x, dsc):  # what root twin.py returns: loop -> postproc -> validity clamps
            y = p.apply(team_loop(ostep, x, dsc, 10), dsc)
            y[:, 1] = np.maximum(y[:, 1], 0.0); y[:, 0] = np.clip(y[:, 0], 233.15, 333.15)
            return y
        e_post = max(float(np.abs(served(x, dsc) - ref_twin.rollout(x, dsc, 10, None)).max()) for _, x, dsc in wins)
        print(f"[4] post.npz ({p.describe()}): ONNX loop + postproc vs reference Twin(best_post), "
              f"max abs diff {e_post:.3g}")
    (out / "model_meta.json").write_text(json.dumps({
        "model_name": a.name, "source_checkpoint": str(a.ckpt).replace("\\", "/"),
        "trained_on": "india_climate_train.nc 2005-2018 (train), 2019-2021 held out",
        "recipe": "residual U-Net, 1-step then multi-step (2,4,8) fine-tune, gradient-magnitude anti-blur loss 0.3",
        "doy_convention": "export takes doy of the predicted day (WEIGHTS.md); net trained on last-input-day doy, wrapper shifts by -1 day",
        "post_processing": "post.npz: climatology blend per lead + rain quantile map per lead, applied by root twin.py via postproc.py to the finished rollout",
        "kit_eval": json.loads(a.scores) if a.scores else None,
    }, indent=1))
    for f in sorted(out.iterdir()):
        print(f"    {f.name:16s} {f.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
