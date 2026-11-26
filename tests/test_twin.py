"""Tests for twin.py's TorchScript weights contract (docs/WEIGHTS.md), using dummy modules in a temp dir."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import CH, INIT_SHAPE, STATE_SHAPE, doy_sincos  # noqa: E402
from twin import DEFAULT_MODEL_NAME, Twin  # noqa: E402

MEAN = [292.36, 3.62, 0.89, 0.19, 101122.0, 30.03]
STD = [13.46, 7.55, 2.90, 2.43, 619.0, 20.49]


def _init(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(INIT_SHAPE)
    x[:, CH["t2m"]] = 300 + rng.normal(0, 2, x[:, 0].shape)
    x[:, CH["tp"]] = rng.gamma(0.8, 8.0, x[:, 0].shape)
    x[:, CH["u10"]] = rng.normal(5, 2, x[:, 0].shape)
    x[:, CH["v10"]] = rng.normal(3, 2, x[:, 0].shape)
    x[:, CH["msl"]] = rng.normal(100500, 100, x[:, 0].shape)
    x[:, CH["tcwv"]] = rng.uniform(20, 60, x[:, 0].shape)
    return x


class Diffuse(nn.Module):
    """next = last + 0.1 * local 3x3 diffusion (replicate-padded)."""

    def __init__(self):
        super().__init__()
        self.model_name = "dummy-diffuse"
        k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        self.register_buffer("k", k.expand(6, 1, 3, 3).clone())

    def forward(self, states: torch.Tensor, doy_sincos: torch.Tensor) -> torch.Tensor:
        last = states[-1][None]
        lap = F.conv2d(F.pad(last, (1, 1, 1, 1), mode="replicate"), self.k, groups=6)
        return (last + 0.1 * lap)[0]


class DoyProbe(nn.Module):
    """Writes atan2(sin, cos) of the received doy into u10 so the test can read it back."""

    def forward(self, states: torch.Tensor, doy_sincos: torch.Tensor) -> torch.Tensor:
        out = states[-1].clone()
        out[2] = torch.atan2(doy_sincos[0], doy_sincos[1])
        return out


class NaNAfterWarmup(nn.Module):
    """Finite while the input looks like the warm-up state (u10 == 2), NaN on real data."""

    def forward(self, states: torch.Tensor, doy_sincos: torch.Tensor) -> torch.Tensor:
        last = states[-1]
        if bool((last[2] == 2.0).all()):
            return last.clone()
        return last * float("nan")


class AlwaysNaN(nn.Module):
    def forward(self, states: torch.Tensor, doy_sincos: torch.Tensor) -> torch.Tensor:
        return states[-1] * float("nan")


class WrongShape(nn.Module):
    def forward(self, states: torch.Tensor, doy_sincos: torch.Tensor) -> torch.Tensor:
        return states[-1][:, :16]


class ConvNet(nn.Module):
    """Realistic-size 3-level conv net (~1.9M params) with normalisation and a residual baked in."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 6, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 6, 1, 1))
        self.e1 = nn.Conv2d(20, 64, 3, padding=1)
        self.e2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.e3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.m1 = nn.Conv2d(256, 256, 3, padding=1)
        self.m2 = nn.Conv2d(256, 256, 3, padding=1)
        self.d2 = nn.Conv2d(256 + 128, 128, 3, padding=1)
        self.d1 = nn.Conv2d(128 + 64, 64, 3, padding=1)
        self.out = nn.Conv2d(64, 6, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, states: torch.Tensor, doy_sincos: torch.Tensor) -> torch.Tensor:
        z = (states - self.mean) / self.std                       # [3,6,32,36]
        doy = doy_sincos.view(1, 2, 1, 1).expand(1, 2, 32, 36)
        x = torch.cat([z.reshape(1, 18, 32, 36), doy], dim=1)
        h1 = F.gelu(self.e1(x))
        h2 = F.gelu(self.e2(h1))
        h3 = F.gelu(self.e3(h2))
        h3 = F.gelu(self.m2(F.gelu(self.m1(h3))))
        u2 = F.interpolate(h3, size=(16, 18), mode="nearest")
        u2 = F.gelu(self.d2(torch.cat([u2, h2], 1)))
        u1 = F.interpolate(u2, size=(32, 36), mode="nearest")
        u1 = F.gelu(self.d1(torch.cat([u1, h1], 1)))
        nxt = z[-1:] + self.out(u1)                               # residual in normalised space
        return (nxt * self.std + self.mean)[0]


def _save(tmp_path: Path, module: nn.Module, name: str = "model.pt") -> Path:
    p = tmp_path / name
    torch.jit.script(module.eval()).save(str(p))
    return p


def test_missing_file_is_persistence(tmp_path):
    t = Twin(weights_path=tmp_path / "nope.pt")
    assert t.backend == "persistence" and t.model_name == DEFAULT_MODEL_NAME
    x = _init()
    out = t.rollout(x, doy_sincos(200), steps=3)
    assert np.array_equal(out, np.repeat(x[-1][None], 3, axis=0))


def test_garbage_file_is_persistence(tmp_path, caplog):
    p = tmp_path / "model.pt"
    p.write_bytes(b"not a torchscript archive")
    t = Twin(weights_path=p)
    assert t.backend == "persistence" and t.load_error
    assert "STAYING ON PERSISTENCE" in caplog.text


def test_lfs_pointer_is_persistence(tmp_path):
    p = tmp_path / "model.pt"
    p.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n")
    t = Twin(weights_path=p)
    assert t.backend == "persistence" and "LFS" in t.load_error


def test_wrong_shape_module_is_persistence(tmp_path):
    t = Twin(weights_path=_save(tmp_path, WrongShape()))
    assert t.backend == "persistence" and "NaN" in t.load_error


def test_dummy_model_shape_name_determinism(tmp_path):
    t = Twin(weights_path=_save(tmp_path, Diffuse()))
    assert t.backend == "model" and t.model_name == "dummy-diffuse"
    x, d = _init(), doy_sincos(200)
    a = t.rollout(x, d, steps=10)
    b = t.rollout(x, d, steps=10)
    z = t.rollout(x, d, steps=10, perturbation=np.zeros(STATE_SHAPE))
    assert a.shape == (10, *STATE_SHAPE) and np.all(np.isfinite(a))
    assert np.array_equal(a, b) and np.array_equal(a, z)
    assert not np.array_equal(a[0], x[-1])           # it is not persistence
    assert a[:, CH["tp"]].min() >= 0
    p = np.zeros(STATE_SHAPE)
    p[CH["t2m"], 10:15, 20:25] = 3.0
    s = t.rollout(x, d, steps=10, perturbation=p)
    diff = np.abs(s - a)[0, CH["t2m"]]
    assert diff.max() > 1.0 and diff[:5].max() < 1e-4    # local response on day 1


def test_model_meta_json_overrides_name(tmp_path):
    p = _save(tmp_path, Diffuse())
    (tmp_path / "model_meta.json").write_text(json.dumps({"model_name": "aether-unet-v1"}))
    assert Twin(weights_path=p).model_name == "aether-unet-v1"


def test_doy_advances_one_day_per_step(tmp_path):
    t = Twin(weights_path=_save(tmp_path, DoyProbe()))
    doy = 360
    out = t.rollout(_init(), doy_sincos(doy), steps=8)
    for k in range(1, 9):
        ang = out[k - 1, CH["u10"], 0, 0]
        want = 2 * math.pi * (doy + k) / 365.0
        assert math.isclose(math.sin(ang), math.sin(want), abs_tol=1e-5)
        assert math.isclose(math.cos(ang), math.cos(want), abs_tol=1e-5)


def test_nan_module_falls_back_to_persistence(tmp_path, caplog):
    t = Twin(weights_path=_save(tmp_path, NaNAfterWarmup()))
    assert t.backend == "model"
    x = _init()
    out = t.rollout(x, doy_sincos(200), steps=4)
    assert np.array_equal(out, np.repeat(x[-1][None], 4, axis=0))
    assert t.nonfinite_fallbacks == 1 and "persistence fallback" in caplog.text


def test_always_nan_module_never_loads(tmp_path):
    t = Twin(weights_path=_save(tmp_path, AlwaysNaN()))
    assert t.backend == "persistence"


def test_realistic_convnet_latency(tmp_path):
    net = ConvNet()
    nparams = sum(p.numel() for p in net.parameters())
    t = Twin(weights_path=_save(tmp_path, net))
    assert t.backend == "model"
    x, d = _init(), doy_sincos(200)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        out = t.rollout(x, d, steps=10)
        times.append((time.perf_counter() - t0) * 1000)
    assert np.array_equal(out, t.rollout(x, d, steps=10))
    ms = sorted(times)[1]
    print(f"\n[latency] ConvNet {nparams / 1e6:.2f}M params, 10-step rollout, {torch.get_num_threads()} threads: "
          f"median {ms:.1f} ms")
    assert ms < 5000
