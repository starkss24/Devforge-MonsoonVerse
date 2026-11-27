"""Anomaly-amplitude loss and the training-side climatology calendar."""
import math

import numpy as np
import torch

from training.losses import anomaly_amplitude_loss
from training.train import target_clim_slots
from twin_model.post import target_slots


def test_slots_match_serving_calendar():
    for doy in [1, 2, 31, 58, 59, 60, 61, 180, 300, 364, 365]:
        sc = torch.tensor([[math.sin(2 * math.pi * doy / 365), math.cos(2 * math.pi * doy / 365)]], dtype=torch.float32)
        got = target_clim_slots(sc, 10)[0].numpy()
        assert np.array_equal(got, target_slots(sc[0].numpy(), 10)), doy


def test_zero_for_perfect_forecast_and_positive_when_damped():
    torch.manual_seed(0)
    clim = torch.randn(2, 3, 6, 8, 9)
    truth = clim + torch.randn(2, 3, 6, 8, 9)
    assert float(anomaly_amplitude_loss(truth, truth, clim)) < 1e-6
    damped = clim + 0.5 * (truth - clim)                  # anomalies shrunk by half
    assert float(anomaly_amplitude_loss(damped, truth, clim)) > 0.05


def test_gradient_pushes_damped_anomaly_outward():
    clim = torch.zeros(1, 1, 1, 4, 4)
    truth = torch.ones(1, 1, 1, 4, 4)
    pred = (0.3 * torch.ones(1, 1, 1, 4, 4)).requires_grad_()
    anomaly_amplitude_loss(pred, truth, clim).backward()
    assert (pred.grad < 0).all()                           # gradient descent increases the anomaly
