import numpy as np
import torch

from twin_model.norm import CHANNELS, RAIN, H, W, N_IN, Normalizer, from_model, to_model


def _phys(lead=(4,), seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=lead + (6, H, W)).astype(np.float32)
    x[..., 0, :, :] = 300 + 5 * x[..., 0, :, :]
    x[..., RAIN, :, :] = np.abs(x[..., RAIN, :, :]) * 5
    x[..., RAIN, :3, :3] = 0.0
    return x


def test_constants():
    assert CHANNELS == ["t2m", "tp", "u10", "v10", "msl", "tcwv"]
    assert RAIN == 1 and (H, W) == (32, 36) and N_IN == 3


def test_model_space_roundtrip_numpy():
    x = _phys()
    back = from_model(to_model(x))
    assert np.allclose(back, x, rtol=1e-4, atol=1e-3)
    assert (back[..., RAIN, :, :] >= 0).all()


def test_model_space_roundtrip_torch_and_only_rain_changes():
    x = torch.from_numpy(_phys())
    m = to_model(x)
    assert isinstance(m, torch.Tensor)
    other = [c for c in range(6) if c != RAIN]
    assert torch.allclose(m[..., other, :, :], x[..., other, :, :])
    assert torch.allclose(m[..., RAIN, :, :], torch.log1p(x[..., RAIN, :, :]), atol=1e-6)
    assert torch.allclose(from_model(m), x, rtol=1e-4, atol=1e-3)


def test_negative_rain_clamped():
    x = _phys()
    x[..., RAIN, :, :] = -3.0
    m = to_model(x)
    assert np.allclose(m[..., RAIN, :, :], 0.0)
    assert (from_model(m)[..., RAIN, :, :] >= 0).all()
    neg = np.zeros((6, H, W), dtype=np.float32)
    neg[RAIN] = -2.0
    assert (from_model(neg)[RAIN] >= 0).all()


def test_leading_dims():
    for lead in [(), (2,), (2, 3)]:
        x = _phys(lead)
        assert from_model(to_model(x)).shape == x.shape


def test_standardize_roundtrip(norm, synth):
    xm = to_model(synth[0][:5])
    rt = norm.unstandardize(norm.standardize(xm))
    assert np.allclose(rt, xm, atol=1e-3, rtol=1e-4)


def test_standardize_roundtrip_torch(norm, synth):
    xm = torch.from_numpy(to_model(synth[0][:2]))
    out = norm.unstandardize(norm.standardize(xm))
    assert torch.allclose(out, xm, atol=1e-3, rtol=1e-4)


def test_normalizer_fields(norm):
    for name in ("state_mean", "state_std", "delta_std"):
        a = np.asarray(getattr(norm, name))
        assert a.shape == (6,) and a.dtype == np.float32 and np.isfinite(a).all()
    assert (np.asarray(norm.state_std) > 0).all()
    assert (np.asarray(norm.delta_std) > 0).all()
    for name in ("static_mean", "static_std"):
        a = np.asarray(getattr(norm, name))
        assert a.shape == (2,) and np.isfinite(a).all()


def test_normalizer_dict_roundtrip(norm, synth):
    d = norm.to_dict()
    n2 = Normalizer.from_dict(d)
    for name in ("state_mean", "state_std", "delta_std", "static_mean", "static_std"):
        assert np.allclose(getattr(norm, name), getattr(n2, name))
        assert np.asarray(getattr(n2, name)).dtype == np.float32
    xm = to_model(synth[0][:2])
    assert np.allclose(norm.standardize(xm), n2.standardize(xm))
    assert Normalizer.from_dict(n2.to_dict()).to_dict() == n2.to_dict()
