import numpy as np

from monsoon_r1 import scoring as KS
from training.dataset import make_synthetic, year_split
from training.train import VAL_STEPS, kit_metrics, train_climatology, validate


def _setup(n=4):
    states, dates, static, _ = make_synthetic(T=120, seed=0)
    tr, va = year_split(dates)
    clim = train_climatology(states[tr], dates[tr])
    rng = np.random.default_rng(0)
    starts = np.arange(n) * 5
    T_ = np.stack([states[s + 3:s + 3 + VAL_STEPS] for s in starts])
    L = np.stack([states[s + 2] for s in starts])
    dts = np.array([dates[s + 2] for s in starts])
    return clim, T_, L, dts, rng


def test_climatology_shape_and_train_only():
    states, dates, _, _ = make_synthetic(T=120, seed=0)
    tr, _ = year_split(dates)
    c = train_climatology(states[tr], dates[tr])
    assert c.shape == (366,) + states.shape[1:] and np.isfinite(c).all()


def test_perfect_forecast_saturates_kit_points():
    clim, T_, L, dts, _ = _setup()
    m = kit_metrics(T_.copy(), T_, L, dts, clim)
    assert abs(m["kit_ss"] - 1.0) < 1e-6
    assert m["kit_rmse_pts"] == 30.0
    assert abs(m["kit_blur_pts"] - 10.0) < 1e-6
    assert m["kit_sel"] <= -40.0 - 1e-6  # 30 + 10 + csi pts (>=0)


def test_persistence_forecast_has_nonpositive_skill():
    clim, T_, L, dts, _ = _setup()
    P = np.repeat(L[:, None], VAL_STEPS, 1)
    m = kit_metrics(P, T_, L, dts, clim)
    assert m["kit_ss"] <= 1e-6  # cannot beat the better of persistence/climatology by being persistence


def test_sel_ranks_better_forecast_lower_and_matches_kit_points():
    clim, T_, L, dts, rng = _setup()
    good = T_ + 0.01 * rng.standard_normal(T_.shape).astype(np.float32) * T_.std(axis=(0, 1, 3, 4), keepdims=True)
    bad = np.repeat(L[:, None], VAL_STEPS, 1)
    mg, mb = kit_metrics(good, T_, L, dts, clim), kit_metrics(bad, T_, L, dts, clim)
    assert mg["kit_sel"] < mb["kit_sel"]
    csi_pts = KS.forecast_points([mg["kit_ss"]], [mg["kit_csi_skill"]])["csi_pts"]
    expect = -(min(mg["kit_ss"], KS.SS_REF) / KS.SS_REF * 30 + csi_pts + mg["kit_blur_pts"])
    assert abs(mg["kit_sel"] - expect) < 1e-6


def test_blur_collapsed_forecast_loses_blur_points():
    clim, T_, L, dts, _ = _setup()
    flat = np.broadcast_to(T_.mean(axis=(-2, -1), keepdims=True), T_.shape).copy()
    assert kit_metrics(flat, T_, L, dts, clim)["kit_blur_pts"] < 1.0


def test_validate_returns_kit_and_old_scores(synth, norm, tiny_model):
    from training.dataset import WindowDataset
    states, dates, static, _ = synth
    clim = train_climatology(states, dates)
    ds = WindowDataset(states, dates, static, norm, VAL_STEPS)
    kw = dict(device="cpu", tp_thresh=5.0, max_windows=4, clim=clim)
    mk = validate(tiny_model, ds, norm, static, select="kit", **kw)
    mo = validate(tiny_model, ds, norm, static, select="old", **kw)
    assert mk["score"] == mk["kit_sel"] and mo["score"] == mo["old_score"] == mk["old_score"]
    assert np.isfinite(mk["kit_ss"])
