import math

import torch

from tests.conftest import H, W, make_tiny_model
from twin_model.model import UNet, advance_doy, rollout_std, step
from twin_model.norm import RAIN, from_model


def _inputs(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    win = torch.randn(B, 3, 6, H, W, generator=g)
    static = torch.randn(B, 2, H, W, generator=g)
    ang = torch.rand(B, generator=g) * 2 * math.pi
    doy = torch.stack([ang.sin(), ang.cos()], dim=1)
    return win, static, doy


def test_unet_output_shape():
    model = make_tiny_model().eval()
    out = model(torch.randn(3, 22, H, W))
    assert out.shape == (3, 6, H, W)
    assert torch.isfinite(out).all()


def test_unet_default_config_shape():
    model = UNet().eval()
    assert model(torch.randn(1, 22, H, W)).shape == (1, 6, H, W)


def test_no_circular_padding():
    """A delta at the left edge must not influence the right edge."""
    model = make_tiny_model().eval()
    x = torch.zeros(1, 22, H, W)
    x2 = x.clone()
    x2[:, :, H // 2, 0] = 50.0
    with torch.no_grad():
        d = (model(x2) - model(x)).abs()
    assert d[..., :4].max() > 1e-2, "perturbation should affect the left side (non-vacuous)"
    # GroupNorm leaks a tiny global signal everywhere; wrap-around would be O(left effect).
    left = d[..., 0].max()
    assert d[..., -1].max() < 0.01 * left
    assert d[..., -2].max() < 0.01 * left
    x3 = x.clone()
    x3[:, :, 0, W // 2] = 50.0
    with torch.no_grad():
        d3 = (model(x3) - model(x)).abs()
    assert d3[:, :, -1, :].max() < 0.01 * d3[:, :, 0, :].max()


def test_advance_doy_rotates():
    ang = torch.tensor([0.3, 2.0])
    doy = torch.stack([ang.sin(), ang.cos()], 1)
    out = advance_doy(doy)
    exp = ang + 2 * math.pi / 365
    assert torch.allclose(out, torch.stack([exp.sin(), exp.cos()], 1), atol=1e-5)
    out5 = advance_doy(doy, days=5)
    exp5 = ang + 5 * 2 * math.pi / 365
    assert torch.allclose(out5, torch.stack([exp5.sin(), exp5.cos()], 1), atol=1e-5)
    assert torch.allclose(out.norm(dim=1), torch.ones(2), atol=1e-5)


def test_step_shape_and_rain_nonneg(norm):
    model = make_tiny_model().eval()
    win, static, doy = _inputs()
    with torch.no_grad():
        nxt = step(model, win, static, doy, norm)
    assert nxt.shape == (2, 6, H, W)
    phys = from_model(norm.unstandardize(nxt))
    assert (phys[:, RAIN] >= 0).all()
    assert torch.isfinite(phys).all()


def test_step_rain_nonneg_with_very_negative_inputs(norm):
    """Even when the window is far below the rain mean, next-step rain stays >= 0."""
    model = make_tiny_model().eval()
    win, static, doy = _inputs()
    win[:, :, RAIN] = -20.0
    with torch.no_grad():
        nxt = step(model, win, static, doy, norm)
    assert (from_model(norm.unstandardize(nxt))[:, RAIN] >= 0).all()


def test_rollout_shape_and_grads(norm):
    model = make_tiny_model().train()
    win, static, doy = _inputs()
    out = rollout_std(model, win, static, doy, 3, norm)
    assert out.shape == (2, 3, 6, H, W)
    out.pow(2).mean().backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_rollout_matches_repeated_step(norm):
    model = make_tiny_model().eval()
    win, static, doy = _inputs()
    with torch.no_grad():
        roll = rollout_std(model, win, static, doy, 2, norm)
        n1 = step(model, win, static, doy, norm)
        win2 = torch.cat([win[:, 1:], n1[:, None]], 1)
        n2 = step(model, win2, static, advance_doy(doy), norm)
    assert torch.allclose(roll[:, 0], n1, atol=1e-5)
    assert torch.allclose(roll[:, 1], n2, atol=1e-5)


def test_untrained_unet_is_persistence(norm):
    model = UNet(in_ch=22, out_ch=6, base=4, depth=2).eval()
    win, static, doy = _inputs()
    # keep rain non-negative in physical terms so the floor does not alter it
    rain_floor = float(norm.standardize(torch.zeros(6, 1, 1)).reshape(6)[RAIN])
    win[:, :, RAIN] = win[:, :, RAIN].abs() + rain_floor
    with torch.no_grad():
        nxt = step(model, win, static, doy, norm)
    assert torch.allclose(nxt, win[:, -1], atol=1e-5)


def test_dry_cells_stay_exactly_zero(norm):
    model = UNet(in_ch=22, out_ch=6, base=4, depth=2).eval()
    win, static, doy = _inputs()
    dry = norm.standardize(torch.zeros(6, 1, 1)).reshape(6)[RAIN]
    win[:, :, RAIN] = dry  # rain 0 mm in all 3 input days
    with torch.no_grad():
        nxt = step(model, win, static, doy, norm)
    rain = from_model(norm.unstandardize(nxt))[:, RAIN]
    assert (rain == 0).all() or rain.abs().max() < 1e-6
