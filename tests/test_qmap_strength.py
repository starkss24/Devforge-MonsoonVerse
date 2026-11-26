"""Optional per-lead rain-calibration strength (twin_model.post.Post qmap_strength)."""
import numpy as np
import pytest

from twin_model.norm import RAIN
from twin_model.post import Post

L, Q = 10, 5
FQ = np.tile(np.array([0.0, 1.0, 5.0, 10.0, 20.0], np.float32), (L, 1))
OQ = np.tile(np.array([0.0, 2.0, 10.0, 25.0, 40.0], np.float32), (L, 1))
DSC = np.array([np.sin(2 * np.pi * 160 / 365), np.cos(2 * np.pi * 160 / 365)], np.float32)


def _y(seed=0):
    r = np.random.default_rng(seed)
    y = r.normal(0, 1, (L, 6, 4, 5)).astype(np.float32)
    y[:, RAIN] = np.abs(r.gamma(0.6, 6.0, (L, 4, 5))).astype(np.float32)
    y[:, RAIN, 0, 0] = 0.0
    return y


def test_absent_strength_equals_full_map():
    y = _y()
    a = Post(qmap_fq=FQ, qmap_oq=OQ).apply(y, DSC)
    b = Post(qmap_fq=FQ, qmap_oq=OQ, qmap_strength=np.ones(L)).apply(y, DSC)
    assert np.array_equal(a, b)


def test_zero_strength_is_identity_on_rain():
    y = _y()
    out = Post(qmap_fq=FQ, qmap_oq=OQ, qmap_strength=np.zeros(L)).apply(y, DSC)
    assert np.allclose(out[:, RAIN], y[:, RAIN])


def test_half_strength_is_midpoint_and_keeps_dry_and_monotone():
    y = _y()
    full = Post(qmap_fq=FQ, qmap_oq=OQ).apply(y, DSC)[:, RAIN]
    half = Post(qmap_fq=FQ, qmap_oq=OQ, qmap_strength=np.full(L, 0.5)).apply(y, DSC)[:, RAIN]
    assert np.allclose(half, 0.5 * (y[:, RAIN] + full), atol=1e-5)
    assert (half[:, 0, 0] == 0).all()
    x = np.linspace(0, 30, 200, dtype=np.float32)
    yy = np.zeros((L, 6, 1, 200), np.float32); yy[:, RAIN, 0] = x
    out = Post(qmap_fq=FQ, qmap_oq=OQ, qmap_strength=np.linspace(1, 0.25, L)).apply(yy, DSC)[:, RAIN, 0]
    assert (np.diff(out, axis=1) >= -1e-6).all()


def test_strength_per_lead_and_roundtrip():
    s = np.linspace(1.0, 0.5, L).astype(np.float32)
    p = Post(qmap_fq=FQ, qmap_oq=OQ, qmap_strength=s)
    p2 = Post.from_dict(p.to_dict())
    y = _y(1)
    assert np.array_equal(p.apply(y, DSC), p2.apply(y, DSC))


@pytest.mark.parametrize("bad", [np.full(L, 1.5), np.full(L, -0.1), np.full(L, np.nan)])
def test_bad_strength_rejected(bad):
    with pytest.raises(ValueError):
        Post(qmap_fq=FQ, qmap_oq=OQ, qmap_strength=bad)


def test_strength_without_qmap_rejected():
    with pytest.raises(ValueError):
        Post(qmap_strength=np.ones(L))
