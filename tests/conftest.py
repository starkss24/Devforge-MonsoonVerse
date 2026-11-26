import math
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

H, W = 32, 36
TINY = dict(base=4, depth=2)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slower tests (latency)")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture(scope="session")
def synth():
    from training.dataset import make_synthetic
    return make_synthetic(T=60, seed=0)


@pytest.fixture(scope="session")
def norm(synth):
    from twin_model.norm import Normalizer
    states, dates, static, names = synth
    return Normalizer.fit(states, static)


def randomize_params(model, scale=0.1, seed=0):
    """Non-zero init for every parameter (incl. last layer) so tests are not vacuous."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * scale)
    return model


def make_tiny_model(seed=0):
    from twin_model.model import UNet
    torch.manual_seed(seed)
    return randomize_params(UNet(in_ch=22, out_ch=6, **TINY), seed=seed)


@pytest.fixture
def tiny_model():
    return make_tiny_model().eval()


@pytest.fixture
def tiny_config():
    return dict(base=TINY["base"], depth=TINY["depth"], in_ch=22, out_ch=6,
                softplus_beta=0.0)


@pytest.fixture
def tiny_ckpt(tmp_path, synth, norm, tiny_config):
    from twin_model.twin import save_checkpoint
    states, dates, static, names = synth
    model = make_tiny_model().eval()
    path = str(tmp_path / "model.pt")
    save_checkpoint(path, model, tiny_config, norm, static, list(names),
                    {"note": "test"})
    return path


@pytest.fixture
def init_window(synth):
    states = synth[0]
    return np.ascontiguousarray(states[:3]).astype(np.float32)


def doy_sincos_np(doy=200):
    a = 2 * math.pi * doy / 365
    return np.array([math.sin(a), math.cos(a)], dtype=np.float32)
