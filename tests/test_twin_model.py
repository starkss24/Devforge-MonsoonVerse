import os
import time

import numpy as np
import pytest
import torch

from tests.conftest import H, W, doy_sincos_np
from twin_model.twin import Twin, load_checkpoint, save_checkpoint

STEPS = 10


def _roll(twin, win, pert=None, steps=STEPS):
    return twin.rollout(win, doy_sincos_np(), steps, pert)


@pytest.fixture
def twin(tiny_ckpt):
    return Twin(tiny_ckpt)


def test_loaded_not_fallback(twin):
    assert twin.fallback is False


def test_zero_shock(twin, init_window):
    a = _roll(twin, init_window, None)
    b = _roll(twin, init_window, np.zeros((6, H, W), dtype=np.float32))
    assert np.array_equal(a, b)


def test_determinism(twin, init_window):
    assert np.array_equal(_roll(twin, init_window), _roll(twin, init_window))


def test_output_contract(twin, init_window):
    out = _roll(twin, init_window)
    assert out.shape == (STEPS, 6, H, W)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert (out[:, 1] >= 0).all()
    assert out[:, 0].min() >= 233.15 and out[:, 0].max() <= 333.15


def test_clamps_hold_under_huge_inputs(twin, init_window):
    win = init_window.copy()
    win[-1, 0] += 200.0   # absurd temperature
    win[-1, 1] += 5000.0  # absurd rain
    pert = np.full((6, H, W), -1e3, dtype=np.float32)
    out = _roll(twin, win, pert)
    assert np.isfinite(out).all()
    assert (out[:, 1] >= 0).all()
    assert out[:, 0].min() >= 233.15 and out[:, 0].max() <= 333.15


def test_perturbation_applied_to_last_day_and_local(twin, init_window):
    base = _roll(twin, init_window, None, steps=1)
    pert = np.zeros((6, H, W), dtype=np.float32)
    pert[0, 12:20, 14:22] = 15.0  # big local t2m shock
    shocked = _roll(twin, init_window, pert, steps=1)
    diff = np.abs(shocked - base)[0, 0]
    inside = diff[12:20, 14:22].mean()
    mask = np.ones((H, W), bool)
    mask[6:26, 8:28] = False
    far = diff[mask].mean()
    assert inside > 0
    assert inside > 5 * far


def test_perturbation_does_not_mutate_inputs(twin, init_window):
    win = init_window.copy()
    pert = np.random.default_rng(0).normal(size=(6, H, W)).astype(np.float32)
    pert_copy = pert.copy()
    _roll(twin, win, pert)
    assert np.array_equal(win, init_window)
    assert np.array_equal(pert, pert_copy)


@pytest.mark.parametrize("src", ["none", "missing"])
def test_persistence_fallback(src, tmp_path, init_window):
    t = Twin(None if src == "none" else str(tmp_path / "does_not_exist.pt"))
    assert t.fallback is True
    out = _roll(t, init_window)
    assert out.shape == (STEPS, 6, H, W) and out.dtype == np.float32
    last = init_window[-1]
    expected = last.copy()
    expected[1] = np.maximum(expected[1], 0)
    expected[0] = np.clip(expected[0], 233.15, 333.15)
    for k in range(STEPS):
        assert np.allclose(out[k], expected, atol=1e-4)


def test_fallback_carries_perturbation(tmp_path, init_window):
    t = Twin(None)
    pert = np.zeros((6, H, W), dtype=np.float32)
    pert[2] = 3.0
    out = _roll(t, init_window, pert)
    assert np.allclose(out[:, 2], init_window[-1, 2] + 3.0, atol=1e-4)


def test_checkpoint_roundtrip(tmp_path, tiny_ckpt, synth, norm, tiny_config, init_window):
    ck = load_checkpoint(tiny_ckpt)
    for key in ("model", "config", "norm", "static", "static_names", "meta", "raw"):
        assert key in ck
    assert not ck["model"].training
    assert np.allclose(ck["norm"].state_mean, norm.state_mean)
    assert np.asarray(ck["static"]).shape == (2, H, W)

    t1 = Twin(tiny_ckpt)
    # re-save the loaded content and reload it
    from tests.conftest import make_tiny_model
    model = make_tiny_model().eval()
    p2 = str(tmp_path / "again.pt")
    save_checkpoint(p2, model, tiny_config, norm, synth[2],
                    list(synth[3]), {"note": "again"})
    t2 = Twin(p2)
    assert t2.fallback is False
    assert np.array_equal(_roll(t1, init_window), _roll(t2, init_window))


@pytest.mark.slow
def test_latency_default_config(tmp_path, synth, norm, init_window):
    from twin_model.model import UNet
    from tests.conftest import randomize_params
    torch.set_num_threads(4)
    model = randomize_params(UNet(), scale=0.05).eval()
    cfg = dict(base=32, depth=3, in_ch=22, out_ch=6, softplus_beta=0.0)
    p = str(tmp_path / "default.pt")
    save_checkpoint(p, model, cfg, norm, synth[2], list(synth[3]), {})
    t = Twin(p)
    assert t.fallback is False
    _roll(t, init_window)  # warm-up
    times = []
    for _ in range(3):
        s = time.perf_counter()
        _roll(t, init_window)
        times.append(time.perf_counter() - s)
    best = min(times)
    assert best < 1.0, f"10-step rollout took {best:.3f}s"
