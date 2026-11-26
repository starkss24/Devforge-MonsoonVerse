"""Unit tests for impact/ (crop stress, climatology, regions) with HAND-COMPUTED numbers on tiny arrays.

Tiny grid used below: lats = [0, 60] (cos = 1.0 and 0.5), lons = [10, 20, 30].
Whole-grid cos-lat weights: row 0 -> 1.0 each, row 1 -> 0.5 each; sum = 3*1 + 3*0.5 = 4.5.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from impact import climatology as IC  # noqa: E402
from impact.crop_stress import crop_stress, region_weights  # noqa: E402
from impact.regions import REGIONS, region_boxes  # noqa: E402

LATS = np.array([0.0, 60.0])
LONS = np.array([10.0, 20.0, 30.0])
ALL = {"all": (0.0, 60.0, 10.0, 30.0)}
ONE = {"one": (0.0, 0.0, 10.0, 10.0)}          # the single cell (lat 0, lon 10)


def roll(steps, t2m=300.0, tp=0.0):
    """[steps, 6, 2, 3] rollout with constant t2m (K) and tp (mm/day); other channels 0."""
    r = np.zeros((steps, 6, 2, 3))
    r[:, 0] = t2m
    r[:, 1] = tp
    return r


def clim(steps, value):
    return np.full((steps, 2, 3), float(value))


def cs(base, shocked, ctp, ct2, regions=ALL, **kw):
    return crop_stress(base, shocked, ctp, ct2, regions, lats=LATS, lons=LONS, **kw)


# ------------------------------------------------------------------ rain deficit
def test_deficit_sign_and_value():
    # forecast 3 mm/day x 2 days = 6 mm; climatology 5 mm/day x 2 days = 10 mm
    # deficit = 100 * (10 - 6) / 10 = +40 %  (positive = drier than normal)
    r = cs(roll(2, tp=3.0), roll(2, tp=3.0), clim(2, 5.0), clim(2, 300.0))["all"]["baseline"]
    assert r["rain_mm"] == pytest.approx(6.0)
    assert r["clim_rain_mm"] == pytest.approx(10.0)
    assert r["deficit_pct"] == pytest.approx(40.0)
    # wetter than normal: 8 mm/day -> 16 mm vs 10 mm -> 100 * (10 - 16) / 10 = -60 %
    w = cs(roll(2, tp=8.0), roll(2, tp=8.0), clim(2, 5.0), clim(2, 300.0))["all"]["baseline"]
    assert w["deficit_pct"] == pytest.approx(-60.0)


def test_clim_near_zero_guard():
    # climatology 0.04 mm/day x 10 days = 0.4 mm < 1 mm (dry season) -> deficit reported as 0, flagged, no inf/NaN
    r = cs(roll(10, tp=0.0), roll(10, tp=5.0), clim(10, 0.04), clim(10, 300.0))["all"]
    for sc in ("baseline", "shocked"):
        assert r[sc]["clim_dry"] is True
        assert r[sc]["deficit_pct"] == 0.0
        assert np.isfinite(r[sc]["index"])
    assert r["change"]["deficit_pct"] == 0.0
    assert r["change"]["rain_mm"] == pytest.approx(50.0)   # rain itself still reported: 5 x 10 - 0
    # exactly zero climatology too
    z = cs(roll(3, tp=1.0), roll(3, tp=1.0), clim(3, 0.0), clim(3, 300.0))["all"]["baseline"]
    assert z["deficit_pct"] == 0.0 and z["clim_dry"]


# ------------------------------------------------------------------ hot days
def test_hot_day_threshold_is_strict():
    # one cell, so the region mean is the cell value exactly. Climatology 300 K every day.
    # anomalies: 1.9, 2.0, 2.1, 3.0, -5.0 K ; threshold 2.0 K, strictly greater -> 2.1 and 3.0 -> 2 hot days
    b = roll(5)
    b[:, 0] = np.array([301.9, 302.0, 302.1, 303.0, 295.0])[:, None, None]
    r = cs(b, b, clim(5, 1.0), clim(5, 300.0), regions=ONE, hot_anom_k=2.0)["one"]["baseline"]
    assert r["hot_days"] == 2
    # mean anomaly = (1.9 + 2.0 + 2.1 + 3.0 - 5.0) / 5 = 4.0 / 5 = 0.8 K
    assert r["t2m_anom_k"] == pytest.approx(0.8)
    # lowering the threshold to 1.95 adds the 2.0 day -> 3
    assert cs(b, b, clim(5, 1.0), clim(5, 300.0), regions=ONE, hot_anom_k=1.95)["one"]["baseline"]["hot_days"] == 3


# ------------------------------------------------------------------ area weighting and masking
def test_cos_lat_area_weighting():
    # tp = 10 on the lat-0 row, 4 on the lat-60 row, one day
    # weighted mean = (3*1.0*10 + 3*0.5*4) / 4.5 = (30 + 6) / 4.5 = 8.0   (plain mean would be 7.0)
    # climatology 10 mm -> deficit = 100 * (10 - 8) / 10 = 20 %
    b = roll(1)
    b[0, 1, 0, :] = 10.0
    b[0, 1, 1, :] = 4.0
    r = cs(b, b, clim(1, 10.0), clim(1, 300.0))["all"]["baseline"]
    assert r["rain_mm"] == pytest.approx(8.0)
    assert r["deficit_pct"] == pytest.approx(20.0)
    _, W, n = region_weights(ALL, LATS, LONS)
    assert n[0] == 6
    np.testing.assert_allclose(W[0], [[1 / 4.5] * 3, [0.5 / 4.5] * 3])   # 2/9 and 1/9
    assert W[0].sum() == pytest.approx(1.0)


def test_land_mask_drops_sea_cells():
    # same field as above; the lat-60 row is sea (mask 0.0 and exactly 0.5 -> both excluded, rule is > 0.5)
    # -> only the 10 mm row remains -> rain 10 mm, deficit 0 %
    b = roll(1)
    b[0, 1, 0, :] = 10.0
    b[0, 1, 1, :] = 4.0
    land = np.array([[1.0, 0.9, 0.51], [0.0, 0.5, 0.2]])
    r = cs(b, b, clim(1, 10.0), clim(1, 300.0), land_mask=land)["all"]
    assert r["n_cells"] == 3
    assert r["baseline"]["rain_mm"] == pytest.approx(10.0)
    assert r["baseline"]["deficit_pct"] == pytest.approx(0.0)


def test_empty_region_raises_not_nan():
    with pytest.raises(ValueError, match="no cell"):
        cs(roll(1), roll(1), clim(1, 1.0), clim(1, 300.0), regions={"nowhere": (40.0, 50.0, 10.0, 30.0)})
    with pytest.raises(ValueError, match="no land cell"):
        cs(roll(1), roll(1), clim(1, 1.0), clim(1, 300.0), land_mask=np.zeros((2, 3)))


# ------------------------------------------------------------------ index
def test_index_value_by_hand():
    # deficit 40 % (3 vs 5 mm/day), 2 hot days out of 5 (anomalies 3, 3, 0, 0, 0 K > 2 K)
    # index = 100 * (0.6 * 40/100 + 0.4 * 2/5) = 100 * (0.24 + 0.16) = 40
    b = roll(5, tp=3.0)
    b[:2, 0] = 303.0
    r = cs(b, b, clim(5, 5.0), clim(5, 300.0))["all"]["baseline"]
    assert r["deficit_pct"] == pytest.approx(40.0) and r["hot_days"] == 2
    assert r["index"] == pytest.approx(40.0)
    # equal weights: 100 * (0.5 * 0.4 + 0.5 * 0.4) = 40 as well; rain-only: 100 * 0.4 = 40; heat-only: 40
    for w in ((0.5, 0.5), (1.0, 0.0), (0.0, 1.0)):
        assert cs(b, b, clim(5, 5.0), clim(5, 300.0), weights=w)["all"]["baseline"]["index"] == pytest.approx(40.0)


def test_index_bounds():
    # worst case: no rain (deficit 100 %) and every day hot -> 100 * (0.6 * 1 + 0.4 * 1) = 100
    worst = cs(roll(4, t2m=310.0, tp=0.0), roll(4, t2m=310.0, tp=0.0), clim(4, 5.0), clim(4, 300.0))["all"]["baseline"]
    assert worst["index"] == pytest.approx(100.0)
    # best case: three times the normal rain (deficit -200 % clipped to 0) and cool -> 0
    best = cs(roll(4, t2m=290.0, tp=15.0), roll(4, t2m=290.0, tp=15.0), clim(4, 5.0), clim(4, 300.0))["all"]["baseline"]
    assert best["deficit_pct"] == pytest.approx(-200.0) and best["index"] == 0.0
    rng = np.random.default_rng(1)
    for _ in range(20):
        b = roll(6)
        b[:, 0] = rng.uniform(290, 310, (6, 2, 3))
        b[:, 1] = rng.exponential(5, (6, 2, 3))
        r = cs(b, b, rng.uniform(0, 10, (6, 2, 3)), rng.uniform(295, 305, (6, 2, 3)))["all"]["baseline"]
        assert 0.0 <= r["index"] <= 100.0


def test_shocked_equal_baseline_gives_zero_change():
    rng = np.random.default_rng(2)
    b = roll(3)
    b[:, 0] = rng.uniform(295, 305, (3, 2, 3))
    b[:, 1] = rng.exponential(5, (3, 2, 3))
    r = cs(b, b.copy(), clim(3, 4.0), clim(3, 300.0))["all"]["change"]
    assert all(v == 0 for v in r.values()), r


def test_shock_change_by_hand():
    # baseline: 5 mm/day for 2 days (= normal, deficit 0), t2m at normal.
    # shocked: 2 mm/day (4 mm vs 10 mm -> deficit 60 %) and +3 K both days (2 hot days of 2)
    # index baseline 0; shocked 100 * (0.6 * 0.6 + 0.4 * 2/2) = 100 * (0.36 + 0.4) = 76 -> change +76
    base = roll(2, tp=5.0)
    shk = roll(2, t2m=303.0, tp=2.0)
    r = cs(base, shk, clim(2, 5.0), clim(2, 300.0))["all"]
    assert r["baseline"]["index"] == pytest.approx(0.0)
    assert r["shocked"]["deficit_pct"] == pytest.approx(60.0)
    assert r["change"]["hot_days"] == 2
    assert r["change"]["deficit_pct"] == pytest.approx(60.0)
    assert r["change"]["index"] == pytest.approx(76.0)
    assert r["change"]["t2m_anom_k"] == pytest.approx(3.0)


def test_input_validation():
    with pytest.raises(ValueError):
        cs(roll(2), roll(3), clim(2, 1.0), clim(2, 300.0))           # shape mismatch
    with pytest.raises(ValueError):
        cs(roll(2), roll(2), clim(3, 1.0), clim(2, 300.0))           # climatology for the wrong number of days
    with pytest.raises(ValueError):
        cs(roll(2), roll(2), clim(2, 1.0), clim(2, 300.0), weights=(0.7, 0.7))
    bad = roll(2)
    bad[0, 1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        cs(bad, roll(2), clim(2, 1.0), clim(2, 300.0))


def test_deterministic_and_default_grid():
    rng = np.random.default_rng(3)
    b = rng.uniform(0, 10, (10, 6, 32, 36))
    b[:, 0] += 295.0
    c_tp, c_t2 = rng.uniform(0, 10, (10, 32, 36)), rng.uniform(295, 305, (10, 32, 36))
    r1 = crop_stress(b, b + 0.1, c_tp, c_t2, region_boxes())
    r2 = crop_stress(b, b + 0.1, c_tp, c_t2, region_boxes())
    assert r1 == r2 and set(r1) == set(REGIONS)


# ------------------------------------------------------------------ regions
def test_regions_inside_grid_and_have_land():
    from common import LATS as GL, LONS as GO
    land = np.ones((32, 36))
    names, W, n = region_weights(region_boxes(), GL, GO, land)
    assert len(names) == len(REGIONS) >= 5 and (n > 0).all()
    for name, r in REGIONS.items():
        la0, la1, lo0, lo1 = r["box"]
        assert GL.min() <= la0 <= la1 <= GL.max() and GO.min() <= lo0 <= lo1 <= GO.max()
        assert r["note"] and r["short"]
    # the boxes do not overlap
    assert (np.count_nonzero(W, axis=0) <= 1).all()


# ------------------------------------------------------------------ climatology
def test_doy_index_leap_handling():
    d = np.array(["2019-01-01", "2019-02-28", "2019-03-01", "2020-02-29", "2020-03-01", "2019-12-31"],
                 dtype="datetime64[D]")
    np.testing.assert_array_equal(IC.doy_index(d), [0, 58, 60, 59, 60, 365])
    np.testing.assert_array_equal(IC.doy_index(np.array([20200229, 20191231])), [59, 365])


def test_smoothing_by_hand():
    # one non-leap year, value 1 on Jan 1 and 0 elsewhere. Feb 29 slot has no data -> 0 (like the kit).
    # 15-day circular running mean: slots -7..+7 around slot 0 (= 359..365 and 0..7) get 1/15, all others 0.
    dates = np.arange(np.datetime64("2019-01-01"), np.datetime64("2020-01-01"))
    x = np.zeros((dates.size, 1))
    x[0] = 1.0
    c = IC.daily_climatology(x, dates)[:, 0]
    assert c.shape == (366,) and c.dtype == np.float32
    expect = np.zeros(366)
    expect[list(range(359, 366)) + list(range(0, 8))] = 1.0 / 15.0
    np.testing.assert_allclose(c, expect, rtol=1e-6, atol=1e-9)


def test_climatology_matches_kit():
    kit = pytest.importorskip("monsoon_r1.stats")
    dates = np.arange(np.datetime64("2019-01-01"), np.datetime64("2021-01-01"))   # includes 2020-02-29
    rng = np.random.default_rng(4)
    x = rng.normal(size=(dates.size, 2, 3, 4)).astype(np.float32)
    ymd = np.array([int(str(d).replace("-", "")) for d in dates])
    ours = IC.daily_climatology(x, dates)
    theirs = kit.climatology(x, ymd)
    assert ours.shape == theirs.shape == (366, 2, 3, 4) and ours.dtype == theirs.dtype
    np.testing.assert_allclose(ours, theirs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(IC.daily_climatology(x, ymd), theirs, rtol=1e-6, atol=1e-6)


def test_train_climatology_uses_train_years_only():
    dates = np.arange(np.datetime64("2018-01-01"), np.datetime64("2020-01-01"))
    x = np.zeros((dates.size, 6, 1, 1))
    x[:, 0] = 300.0
    x[:, 1] = 2.0
    x[dates >= np.datetime64("2019-01-01")] = 1e6            # test-year values must not leak in
    c = IC.train_climatology(x, dates)
    assert c.shape == (366, 2, 1, 1)
    # slot 59 (Feb 29) is empty in 2018 and pulls its 15 neighbours down by 1/15; check a mid-year slot
    assert c[200, 0, 0, 0] == pytest.approx(300.0) and c[200, 1, 0, 0] == pytest.approx(2.0)
    assert c.max() < 1e5
    with pytest.raises(ValueError):
        IC.train_climatology(x[-10:], dates[-10:])


def test_forecast_dates_and_lookup():
    fd = IC.forecast_dates("2019-07-15", 10)
    assert str(fd[0]) == "2019-07-16" and str(fd[-1]) == "2019-07-25" and fd.size == 10
    c = np.arange(366, dtype=float)[:, None]
    # Jul 16 in the leap-year calendar: 182 (Jul 1) + 15 = 197
    np.testing.assert_array_equal(IC.clim_for_dates(c, fd)[:, 0], np.arange(197, 207))
