import math

import numpy as np
import pytest
import torch

from tests.conftest import H, W, doy_sincos_np
from twin_model.post import Post, clim_slot, doy_from_sincos, qmap, target_slots
from twin_model.twin import Twin

STEPS = 10
Q = 20


def _post(blend=True, qm=True, seed=0):
    rng = np.random.default_rng(seed)
    kw = {}
    if blend:
        kw["blend"] = rng.uniform(0, 0.8, (STEPS, 6)).astype(np.float32)
        clim = np.empty((366, 6, H, W), np.float32)
        base = np.array([300.0, 3.0, 1.0, -1.0, 100500.0, 40.0], np.float32)
        for c in range(6):
            clim[:, c] = base[c] + rng.normal(0, 1, (366, H, W))
        clim[:, 1] = np.abs(clim[:, 1])
        kw["clim"] = clim
    if qm:
        fq = np.concatenate([[0.0], np.cumsum(rng.uniform(0.1, 3, Q - 1))])
        oq = np.concatenate([[0.0], np.cumsum(rng.uniform(0.0, 4, Q - 1))])
        kw["qmap_fq"] = np.tile(fq, (STEPS, 1)).astype(np.float32)
        kw["qmap_oq"] = np.tile(oq, (STEPS, 1)).astype(np.float32)
    return Post(**kw, meta={"note": "test"})


@pytest.fixture
def post_ckpt(tmp_path, tiny_ckpt):
    raw = torch.load(tiny_ckpt, map_location="cpu", weights_only=True)
    raw["post"] = _post().to_dict()
    p = str(tmp_path / "post.pt")
    torch.save(raw, p)
    return p


def test_doy_roundtrip_and_slots():
    for doy in range(1, 366):
        a = 2 * math.pi * doy / 365
        assert doy_from_sincos(np.array([math.sin(a), math.cos(a)], np.float32)) == doy
    assert clim_slot(1) == 0 and clim_slot(59) == 58      # Jan 1, Feb 28
    assert clim_slot(60) == 60 and clim_slot(365) == 365  # Mar 1 skips the Feb 29 slot, Dec 31
    assert clim_slot(366) == 0                            # wraps to Jan 1
    a = 2 * math.pi * 360 / 365
    assert list(target_slots([math.sin(a), math.cos(a)], 7)) == [361, 362, 363, 364, 365, 0, 1]


def test_qmap_monotone_dry_stays_dry():
    p = _post(blend=False)
    x = np.concatenate([[0.0], np.sort(np.random.default_rng(1).uniform(0, 200, 5000))]).astype(np.float32)
    y = qmap(x, p.fq[0], p.oq[0])
    assert y[0] == 0.0
    assert (np.diff(y) >= 0).all()
    assert np.isfinite(y).all() and (y >= 0).all()


def test_invalid_knots_rejected():
    p = _post(blend=False)
    bad = p.fq.copy(); bad[:, 0] = 1.0
    with pytest.raises(ValueError):
        Post(qmap_fq=bad, qmap_oq=p.oq)
    bad = p.oq.copy(); bad[:, 5] = -1.0
    with pytest.raises(ValueError):
        Post(qmap_fq=p.fq, qmap_oq=bad)


def test_blend_weights_zero_is_identity():
    p = _post(qm=False)
    p.blend[:] = 0.0
    y = np.random.default_rng(2).normal(size=(STEPS, 6, H, W)).astype(np.float32)
    assert np.array_equal(p.apply(y, doy_sincos_np()), y)


def test_blend_matches_formula():
    p = _post(qm=False)
    y = np.random.default_rng(3).normal(size=(STEPS, 6, H, W)).astype(np.float32)
    out = p.apply(y, doy_sincos_np(200))
    slots = target_slots(doy_sincos_np(200), STEPS)
    for k in range(STEPS):
        a = p.blend[k].reshape(-1, 1, 1)
        np.testing.assert_allclose(out[k], (1 - a) * y[k] + a * p.clim[slots[k]], rtol=1e-6, atol=1e-3)


def test_twin_post_loaded_and_changes_output(post_ckpt, tiny_ckpt, init_window):
    t_post, t_raw = Twin(post_ckpt), Twin(tiny_ckpt)
    assert t_post.post is not None and t_raw.post is None
    a = t_post.rollout(init_window, doy_sincos_np(), STEPS, None)
    b = t_raw.rollout(init_window, doy_sincos_np(), STEPS, None)
    assert not np.array_equal(a, b)


def test_twin_post_contract_zero_shock_determinism(post_ckpt, init_window):
    tw = Twin(post_ckpt)
    a = tw.rollout(init_window, doy_sincos_np(), STEPS, None)
    b = tw.rollout(init_window, doy_sincos_np(), STEPS, np.zeros((6, H, W), np.float32))
    c = tw.rollout(init_window, doy_sincos_np(), STEPS, None)
    assert np.array_equal(a, b) and np.array_equal(a, c)
    assert a.shape == (STEPS, 6, H, W) and a.dtype == np.float32
    assert np.isfinite(a).all() and (a[:, 1] >= 0).all()
    assert a[:, 0].min() >= 233.15 and a[:, 0].max() <= 333.15


def test_twin_post_more_steps_than_fitted(post_ckpt, init_window):
    out = Twin(post_ckpt).rollout(init_window, doy_sincos_np(), 14, None)
    assert out.shape == (14, 6, H, W) and np.isfinite(out).all()


def test_twin_post_clamps_hold_under_huge_inputs(post_ckpt, init_window):
    win = init_window.copy()
    win[-1, 0] += 200.0
    win[-1, 1] += 5000.0
    out = Twin(post_ckpt).rollout(win, doy_sincos_np(), STEPS, np.full((6, H, W), -1e3, np.float32))
    assert np.isfinite(out).all() and (out[:, 1] >= 0).all()
    assert out[:, 0].min() >= 233.15 and out[:, 0].max() <= 333.15


def test_invalid_post_block_falls_back_to_raw_model(tmp_path, tiny_ckpt, init_window):
    raw = torch.load(tiny_ckpt, map_location="cpu", weights_only=True)
    raw["post"] = {"blend": torch.zeros(STEPS, 5), "clim": torch.zeros(366, 6, H, W)}
    p = str(tmp_path / "bad.pt"); torch.save(raw, p)
    tw = Twin(p)
    assert tw.fallback is False and tw.post is None
    np.testing.assert_array_equal(tw.rollout(init_window, doy_sincos_np(), STEPS, None),
                                  Twin(tiny_ckpt).rollout(init_window, doy_sincos_np(), STEPS, None))


def test_post_roundtrip_weights_only(post_ckpt):
    raw = torch.load(post_ckpt, map_location="cpu", weights_only=True)
    p = Post.from_dict(raw["post"])
    assert p.blend.shape == (STEPS, 6) and p.clim.shape == (366, 6, H, W)
    assert p.fq.shape == (STEPS, Q) and p.fq.dtype == np.float32
