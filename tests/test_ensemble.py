"""EnsembleUNet: averaged member deltas, checkpoint round trip, zero shock."""
import numpy as np
import pytest
import torch

from twin_model.model import EnsembleUNet, UNet
from twin_model.twin import Twin, load_checkpoint, save_checkpoint


def _net(seed):
    torch.manual_seed(seed)
    m = UNet(base=4, depth=2)
    for p in m.head.parameters():          # non-zero head so members differ
        torch.nn.init.normal_(p, std=0.05)
    return m.eval()


def test_forward_is_member_mean():
    a, b = _net(0), _net(1)
    x = torch.randn(2, 22, 32, 36)
    with torch.no_grad():
        assert torch.allclose(EnsembleUNet([a, b])(x), (a(x) + b(x)) / 2, atol=1e-6)


def test_config_mismatch_rejected():
    with pytest.raises(ValueError):
        EnsembleUNet([UNet(base=4, depth=2), UNet(base=8, depth=2)])


def test_checkpoint_roundtrip_and_twin(tmp_path, norm, synth):
    static, names = synth[2], synth[3]
    ens = EnsembleUNet([_net(0), _net(1)]).eval()
    p = tmp_path / "ens.pt"
    save_checkpoint(p, ens, None, norm, static, names, {"note": "test"})
    ck = load_checkpoint(p)
    assert isinstance(ck["model"], EnsembleUNet) and ck["config"]["members"] == 2
    x = torch.randn(1, 22, 32, 36)
    with torch.no_grad():
        assert torch.equal(ck["model"](x), ens(x))
    twin = Twin(str(p))
    assert not twin.fallback
    rng = np.random.default_rng(0)
    init = (np.array([295, 4, 2, 1, 100500, 35], np.float32)[None, :, None, None]
            + rng.normal(0, 1, (3, 6, 32, 36)).astype(np.float32))
    init[:, 1] = np.abs(init[:, 1])
    dsc = np.array([0.0, 1.0], np.float32)
    a = twin.rollout(init, dsc, 10, None)
    b = twin.rollout(init, dsc, 10, np.zeros((6, 32, 36), np.float32))
    assert np.array_equal(a, b) and np.isfinite(a).all() and (a[:, 1] >= 0).all()
