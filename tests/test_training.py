import numpy as np
import torch

from tests.conftest import H, W, make_tiny_model
from training.dataset import WindowDataset, make_synthetic, year_split
from training.losses import rollout_loss
from twin_model.model import rollout_std
from twin_model.norm import to_model


def _batch(ds, n):
    items = [ds[i] for i in range(n)]
    return [torch.stack([torch.as_tensor(it[j]) for it in items]).float() for j in range(4)]


def test_synthetic_signature():
    states, dates, static, names = make_synthetic(T=20, seed=1)
    assert states.shape == (20, 6, H, W) and states.dtype == np.float32
    assert len(dates) == 20 and static.shape == (2, H, W)
    assert (states[:, 1] >= 0).all()


def test_k1_rollout_loss_equals_plain_mse():
    g = torch.Generator().manual_seed(0)
    pred = torch.randn(3, 1, 6, H, W, generator=g)
    tgt = torch.randn(3, 1, 6, H, W, generator=g)
    loss = rollout_loss(pred, tgt, rain_weight=1.0, grad_weight=0.0)
    assert torch.allclose(loss, torch.nn.functional.mse_loss(pred, tgt), rtol=1e-5)


def test_k1_loss_from_model_rollout_equals_mse(synth, norm):
    states, dates, static, _ = synth
    ds = WindowDataset(states, dates, static, norm, 1)
    win, st, doy, tgt = _batch(ds, 4)
    model = make_tiny_model().eval()
    with torch.no_grad():
        pred = rollout_std(model, win, st, doy, 1, norm)
        loss = rollout_loss(pred, tgt, rain_weight=1.0, grad_weight=0.0)
    assert pred.shape == tgt.shape
    assert torch.allclose(loss, ((pred - tgt) ** 2).mean(), rtol=1e-5)


def test_window_dataset_item_shapes(synth, norm):
    states, dates, static, _ = synth
    ds = WindowDataset(states, dates, static, norm, 2)
    win, st, doy, tgt = ds[0]
    assert tuple(win.shape) == (3, 6, H, W)
    assert tuple(st.shape) == (2, H, W)
    assert tuple(doy.shape) == (2,)
    assert tuple(tgt.shape) == (2, 6, H, W)
    assert len(ds) == len(states) - 3 - 2 + 1


def test_window_dataset_skips_date_gaps(synth, norm):
    states, dates, static, _ = synth
    T = 30
    states = states[:T].copy()
    dates = np.array(dates[:T], dtype="datetime64[D]").copy()
    gap = 15
    dates[gap:] = dates[gap:] + np.timedelta64(5, "D")  # 5 missing days
    k = 2
    ds = WindowDataset(states, dates, static, norm, k)
    span = 3 + k
    valid = [i for i in range(T - span + 1)
             if np.all(np.diff(dates[i:i + span]) == np.timedelta64(1, "D"))]
    assert len(valid) < T - span + 1  # gap removed some windows
    assert len(ds) == len(valid)
    # every yielded window must be built from consecutive days: match by content
    for idx in range(len(ds)):
        win, _, _, tgt = ds[idx]
        match = False
        for i in valid:
            ref = norm.standardize(to_model(states[i:i + 3]))
            if np.allclose(np.asarray(win), ref, atol=1e-4):
                match = True
                break
        assert match, f"window {idx} does not correspond to a consecutive-day start"


def test_year_split_no_leak():
    dates = np.arange(np.datetime64("2004-01-01"), np.datetime64("2023-01-01"),
                      np.timedelta64(30, "D"))
    out = year_split(dates)
    tr, va = out[0], out[1]
    tr, va = np.asarray(tr), np.asarray(va)
    if tr.dtype == bool:
        tr, va = np.flatnonzero(tr), np.flatnonzero(va)
    years_tr = dates[tr].astype("datetime64[Y]").astype(int) + 1970
    years_va = dates[va].astype("datetime64[Y]").astype(int) + 1970
    assert len(tr) and len(va)
    assert years_tr.min() >= 2005 and years_tr.max() <= 2018
    assert years_va.min() >= 2019 and years_va.max() <= 2021
    assert not set(tr.tolist()) & set(va.tolist())


def test_training_step_reduces_loss(synth, norm):
    states, dates, static, _ = synth
    ds = WindowDataset(states, dates, static, norm, 1)
    win, st, doy, tgt = _batch(ds, 8)
    model = make_tiny_model(seed=3)
    # small init so we start in a sane regime
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(60):
        opt.zero_grad()
        pred = rollout_std(model, win, st, doy, 1, norm)
        loss = rollout_loss(pred, tgt, rain_weight=1.0, grad_weight=0.0)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.isfinite(losses).all()
    assert losses[-1] < 0.8 * losses[0], f"{losses[0]:.4f} -> {losses[-1]:.4f}"
