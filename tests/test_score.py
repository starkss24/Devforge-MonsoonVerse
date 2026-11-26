"""Tests for eval/score.py: hand-computed metrics plus pipeline invariants
(train-only climatology and threshold, lead indexing, persistence ratio == 1)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import CH, CHANNELS, N_CH, N_IN_DAYS, NLAT, NLON, T2M_MAX_K, T2M_MIN_K  # noqa: E402
from eval import score  # noqa: E402


# ---------------------------------------------------------------------------
# Pure metrics, hand-computed
# ---------------------------------------------------------------------------
def test_rmse_per_channel_hand_computed():
    truth = np.zeros((2, 2, 1, 2))           # [N=2, C=2, H=1, W=2]
    pred = np.zeros_like(truth)
    pred[:, 0] = 3.0                         # channel 0: every error 3 -> RMSE 3
    pred[0, 1] = [[1.0, 1.0]]                # channel 1: errors 1,1,0,0 -> sqrt(0.5)
    np.testing.assert_allclose(score.rmse_per_channel(pred, truth), [3.0, np.sqrt(0.5)])


def test_rmse_per_channel_rejects_bad_shapes():
    with pytest.raises(ValueError):
        score.rmse_per_channel(np.zeros((1, 2, 2)), np.zeros((1, 2, 3)))
    with pytest.raises(ValueError):
        score.rmse_per_channel(np.zeros((2, 2)), np.zeros((2, 2)))


def test_heavy_rain_hit_rate_hand_computed():
    truth = np.array([[10.0, 20.0, 1.0, 30.0]])   # 3 events at threshold 10 (>=)
    pred = np.array([[10.0, 5.0, 50.0, 12.0]])    # hits on 10 and 30; false alarm ignored
    assert score.heavy_rain_hit_rate(pred, truth, 10.0) == pytest.approx(2 / 3)
    assert score.heavy_rain_hit_rate(truth, truth, 10.0) == 1.0
    assert np.isnan(score.heavy_rain_hit_rate(pred, truth, 100.0))   # no events


def test_sharpness_ratio_hand_computed():
    truth = np.zeros((2, 2, 1, 2))            # [N=2, C=2, H=1, W=2]
    truth[:, 0] = [[0.0, 2.0]]               # std 1
    truth[:, 1] = [[0.0, 4.0]]               # std 2
    pred = np.zeros_like(truth)
    pred[0, 0] = [[0.0, 1.0]]                # std 0.5 -> 0.5
    pred[1, 0] = [[0.0, 2.0]]                # std 1   -> 1.0   mean 0.75
    pred[:, 1] = 7.0                         # flat forecast -> 0
    np.testing.assert_allclose(score.sharpness_ratio(pred, truth), [0.75, 0.0])
    # single sample [C, H, W] works; constant-truth channel is skipped -> NaN
    t1 = np.stack([[[0.0, 2.0]], [[5.0, 5.0]]])
    out = score.sharpness_ratio(t1, t1)
    assert out[0] == 1.0 and np.isnan(out[1])


def test_validity_counts():
    r = np.zeros((2, N_CH, 2, 2))
    r[:, CH["t2m"]] = 300.0
    assert score.validity_counts(r) == {"nan": 0, "inf": 0, "neg_tp": 0, "t2m_oob": 0}
    r[0, CH["u10"], 0, 0] = np.nan
    r[1, CH["msl"], 1, 1] = np.inf
    r[0, CH["t2m"], 0, 1] = -np.inf            # inf, not counted as t2m_oob
    r[0, CH["t2m"], 1, 0] = np.nan             # nan, not counted as t2m_oob
    r[0, CH["tp"], 0, 0] = -0.1
    r[1, CH["tp"], 1, 0] = -5.0
    r[1, CH["tp"], 0, 1] = 0.0                 # tp == 0 is valid
    r[1, CH["t2m"], 0, 0] = T2M_MIN_K - 0.01
    r[1, CH["t2m"], 0, 1] = T2M_MAX_K + 0.01
    r[1, CH["t2m"], 1, 0] = T2M_MAX_K          # boundary is valid
    assert score.validity_counts(r) == {"nan": 2, "inf": 2, "neg_tp": 2, "t2m_oob": 2}
    with pytest.raises(ValueError):
        score.validity_counts(np.zeros((2, 5, 2, 2)))


def test_persistence_forecast_repeats_last_day():
    x = np.arange(3 * 2 * 1 * 1, dtype=float).reshape(3, 2, 1, 1)
    out = score.persistence_forecast(x, steps=4)
    assert out.shape == (4, 2, 1, 1)
    assert np.all(out == x[-1])


# ---------------------------------------------------------------------------
# Climatology helper
# ---------------------------------------------------------------------------
def _col(v):
    return np.asarray(v, dtype=float).reshape(-1, 1, 1, 1)


def test_climatology_default_window_is_plus_minus_7_days():
    assert score.CLIM_HALF_WINDOW == 7
    doys = np.arange(1, 367)
    vals = np.zeros(366)
    vals[99] = 15.0                          # spike on doy 100
    clim = score.daily_climatology(_col(vals), doys)[:, 0, 0, 0]
    expect = np.zeros(366)
    expect[92:107] = 1.0                     # doys 93..107 (15 days) see 15/15
    np.testing.assert_allclose(clim, expect)


def test_climatology_hand_computed_counts_and_wraparound():
    doys = np.arange(1, 367)
    clim = score.daily_climatology(_col(doys), doys, half_window=1)[:, 0, 0, 0]
    assert clim[9] == pytest.approx(10.0)              # mean(9, 10, 11)
    assert clim[0] == pytest.approx((366 + 1 + 2) / 3)  # circular
    # unequal counts: doy 5 seen twice (0 and 10), doy 4 once (4), doy 6 once (6)
    d = np.concatenate([np.arange(1, 367), [5]])
    v = np.concatenate([np.zeros(366), [10.0]])
    v[3], v[5] = 4.0, 6.0
    c = score.daily_climatology(_col(v), d, half_window=1)[:, 0, 0, 0]
    assert c[4] == pytest.approx((4 + 0 + 10 + 6) / 4)
    # non-leap year only: leap day 366 borrows doy 365 and doy 1
    d = np.arange(1, 366)
    c = score.daily_climatology(_col(d), d, half_window=1)[:, 0, 0, 0]
    assert c[365] == pytest.approx((365 + 1) / 2)


def test_climatology_errors():
    with pytest.raises(ValueError):          # gap wider than the window
        score.daily_climatology(_col([1.0, 2.0]), np.array([1, 100]), half_window=7)
    with pytest.raises(ValueError):
        score.daily_climatology(_col([1.0]), np.array([0]))
    with pytest.raises(ValueError):
        score.daily_climatology(_col([1.0, 2.0]), np.array([1]))


def test_climatology_forecast_indexes_by_doy():
    clim = _col(np.arange(1, 367))
    np.testing.assert_array_equal(score.climatology_forecast(clim, [1, 366, 32])[:, 0, 0, 0],
                                  [1, 366, 32])


# ---------------------------------------------------------------------------
# Pipeline (evaluate) on small synthetic data: train 2018, val 2019
# ---------------------------------------------------------------------------
TRAIN, VAL = (2018, 2018), (2019, 2019)
LEADS = (1, 4, 8)


def _times():
    return np.arange(np.datetime64("2018-01-01"), np.datetime64("2020-01-01"), dtype="datetime64[D]")


def _random_data(seed=0):
    rng = np.random.default_rng(seed)
    t = _times()
    data = rng.normal(0.0, 1.0, (t.size, N_CH, NLAT, NLON)).astype(np.float32)
    data[:, CH["t2m"]] += 300.0
    data[:, CH["tp"]] = rng.gamma(0.8, 8.0, (t.size, NLAT, NLON))
    return data, t


def _persistence(x, d, steps):
    return score.persistence_forecast(x, steps)


def _evaluate(data, times, forecaster, **kw):
    kw.setdefault("max_windows", 6)
    return score.evaluate(data, times, forecaster, leads=LEADS, train_years=TRAIN, val_years=VAL, **kw)


def test_default_year_split():
    assert score.TRAIN_YEARS == (2005, 2018)
    assert score.VAL_YEARS[0] > score.TRAIN_YEARS[1]


def test_persistence_scores_ratio_exactly_one():
    data, t = _random_data()
    res = _evaluate(data, t, _persistence)
    assert len(res["rows"]) == len(LEADS) * N_CH
    for row in res["rows"]:
        assert row["ratio_vs_persistence"] == 1.0, row
        assert row["rmse_model"] == row["rmse_persistence"]
    assert res["validity"] == {"nan": 0, "inf": 0, "neg_tp": 0, "t2m_oob": 0}


def test_lead_k_is_rollout_index_k_minus_1():
    """Every field equals its day index, so the oracle out[k] = last + (k+1) is exact
    only if rollout[k] is scored against day t+k+1."""
    t = _times()
    data = np.broadcast_to(np.arange(t.size, dtype=np.float32)[:, None, None, None],
                           (t.size, N_CH, NLAT, NLON)).copy()

    def oracle(x, d, steps):
        return x[-1][None] + np.arange(1, steps + 1)[:, None, None, None]

    def off_by_one(x, d, steps):
        return x[-1][None] + np.arange(0, steps)[:, None, None, None]

    good = _evaluate(data, t, oracle)
    bad = _evaluate(data, t, off_by_one)
    for g, b in zip(good["rows"], bad["rows"]):
        assert g["rmse_model"] == 0.0
        assert b["rmse_model"] == pytest.approx(1.0)
        assert g["rmse_persistence"] == pytest.approx(g["lead"])   # persistence error = lead days


def test_first_window_uses_last_input_day_in_val_year():
    """Windows lie fully inside val years: first window's input days start 2019-01-01."""
    data, t = _random_data()
    seen = []

    def spy(x, d, steps):
        seen.append(x.copy())
        return score.persistence_forecast(x, steps)

    _evaluate(data, t, spy, max_windows=1)
    i0 = int(np.nonzero(t == np.datetime64("2019-01-01"))[0][0])
    np.testing.assert_array_equal(seen[0], data[i0:i0 + N_IN_DAYS])


def test_validation_years_do_not_influence_climatology_or_threshold(monkeypatch):
    captured = []
    real = score.daily_climatology

    def spy(d, doys, *a, **k):
        out = real(d, doys, *a, **k)
        captured.append((d.shape[0], out))
        return out

    monkeypatch.setattr(score, "daily_climatology", spy)
    data, t = _random_data()
    res1 = _evaluate(data, t, _persistence, max_windows=2)
    data2 = data.copy()
    val = score.year_of(t) == 2019
    data2[val] = data2[val] * 5.0 + 1000.0                # wildly different val years
    res2 = _evaluate(data2, t, _persistence, max_windows=2)

    assert captured[0][0] == 365 and captured[1][0] == 365  # only 2018's days passed in
    np.testing.assert_array_equal(captured[0][1], captured[1][1])
    assert res1["tp_threshold"] == res2["tp_threshold"]
    expect = float(np.percentile(data[~val, CH["tp"]], score.HEAVY_RAIN_PCTL))
    assert res1["tp_threshold"] == pytest.approx(expect)
    # and the threshold really is p95 of train tp, not of all data
    assert res1["tp_threshold"] != pytest.approx(float(np.percentile(data2[:, CH["tp"]], 95.0)))


def test_climatology_forecast_rows_use_train_climatology():
    """If train is a constant field c per channel, climatology RMSE equals RMSE of c vs truth."""
    data, t = _random_data()
    train = score.year_of(t) == 2018
    const = np.arange(N_CH, dtype=np.float32)[:, None, None]
    data[train] = const
    data[train, CH["t2m"]] = 300.0
    res = _evaluate(data, t, _persistence, max_windows=1)
    win0 = int(np.nonzero(t == np.datetime64("2019-01-03"))[0][0])
    for row in res["rows"]:
        c = CHANNELS.index(row["channel"])
        truth = data[win0 + row["lead"], c].astype(np.float64)
        ref = 300.0 if row["channel"] == "t2m" else float(c)
        assert row["rmse_climatology"] == pytest.approx(np.sqrt(np.mean((truth - ref) ** 2)), rel=1e-5)
