"""Tests for eval/score.py: it must compute what the judge computes (monsoon_r1.scoring), with hand-checked
lat weighting, CSI at > 10 mm/day, train-only climatology, lead indexing, invalid-window zeroing and anti-blur."""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monsoon_r1 import scoring as S  # noqa: E402
from monsoon_r1.contract import LAT, LON  # noqa: E402
from monsoon_r1.stats import climatology  # noqa: E402

from common import CH, CHANNELS, N_CH, N_IN_DAYS, NLAT, NLON  # noqa: E402
from eval import score  # noqa: E402

TRAIN, VAL = (2016, 2016), (2017, 2017)     # 2016 is a leap year: every kit climatology slot gets data


def _times():
    return np.arange(np.datetime64("2016-01-01"), np.datetime64("2018-01-01"), dtype="datetime64[D]")


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
    kw.setdefault("n_windows", 6)
    return score.evaluate(data, times, forecaster, train_years=TRAIN, val_years=VAL, judge_draws=20, **kw)


def _row(res, lead, ch):
    return next(r for r in res["rows"] if r["lead"] == lead and r["channel"] == ch)


# ---------------------------------------------------------------------------
# Helpers and loading
# ---------------------------------------------------------------------------
def test_defaults_match_task_and_kit():
    assert score.TRAIN_YEARS == (2005, 2018) and score.VAL_YEARS == (2019, 2021)
    assert score.DEFAULT_STEPS == 10 and score.JUDGE_WINDOWS == 5
    assert S.RAIN_THRESH == 10.0
    assert score.S is S                                   # metrics come from the kit, not a copy


def test_yyyymmdd():
    t = np.array(["2005-01-01", "2020-02-29", "2021-12-31"], dtype="datetime64[D]")
    np.testing.assert_array_equal(score.yyyymmdd(t), [20050101, 20200229, 20211231])


def test_dataset_to_arrays_kit_layout_and_lat_flip():
    xr = pytest.importorskip("xarray")
    t = np.arange(np.datetime64("2019-01-01"), np.datetime64("2019-01-04"), dtype="datetime64[D]")
    arr = np.random.default_rng(0).normal(size=(3, N_CH, NLAT, NLON)).astype(np.float32)
    ds = xr.Dataset({"data": (("time", "channel", "lat", "lon"), arr[:, :, ::-1])},
                    coords={"time": t.astype("datetime64[ns]"), "channel": list(CHANNELS),
                            "lat": LAT[::-1], "lon": LON})
    data, times = score.dataset_to_arrays(ds)
    np.testing.assert_array_equal(data, arr)              # north->south file is flipped back
    np.testing.assert_array_equal(times, t)
    bad = ds.assign_coords(lon=LON + 0.5)
    with pytest.raises(ValueError):
        score.dataset_to_arrays(bad)


def test_load_stats_std_kit_layout(tmp_path):
    p = tmp_path / "stats.json"
    p.write_text(json.dumps({"channels": list(CHANNELS), "mean": [0] * 6, "std": [1, 2, 3, 4, 5, 6]}))
    np.testing.assert_array_equal(score.load_stats_std(str(p)), [1, 2, 3, 4, 5, 6])
    p.write_text(json.dumps({"channels": list(CHANNELS[::-1]), "std": [1] * 6}))
    with pytest.raises(ValueError):
        score.load_stats_std(str(p))


# ---------------------------------------------------------------------------
# Hand-checked judge pieces
# ---------------------------------------------------------------------------
def test_rmse_is_latitude_weighted_by_hand():
    """Error 1 on the southernmost row only: RMSE = sqrt(w_south / NLAT) with w = cos(lat) / mean(cos(lat))."""
    t = _times()
    data = np.zeros((t.size, N_CH, NLAT, NLON), np.float32)
    data[:, CH["t2m"]] = 300.0

    def south_err(x, d, steps):
        out = score.persistence_forecast(x, steps).astype(np.float64)
        out[:, CH["u10"], 0, :] += 1.0
        return out

    def north_err(x, d, steps):
        out = score.persistence_forecast(x, steps).astype(np.float64)
        out[:, CH["u10"], -1, :] += 1.0
        return out

    w = np.cos(np.deg2rad(np.arange(6.5, 38.0, 1.0)))
    w = w / w.mean()
    rs = _row(_evaluate(data, t, south_err, n_windows=2), 4, "u10")["rmse_model"]
    rn = _row(_evaluate(data, t, north_err, n_windows=2), 4, "u10")["rmse_model"]
    assert rs == pytest.approx(np.sqrt(w[0] / NLAT), rel=1e-6)
    assert rn == pytest.approx(np.sqrt(w[-1] / NLAT), rel=1e-6)
    assert rs > np.sqrt(1 / NLAT) > rn                     # unweighted RMSE would be identical


def test_csi_threshold_is_strictly_above_10mm_by_hand():
    truth = np.array([10.0, 10.5, 20.0, 3.0])
    pred = np.array([10.5, 9.9, 12.0, 11.0])
    # pred>10: T F T T ; truth>10: F T T F -> hits 1, misses 1, false alarms 2
    assert S.csi(pred, truth) == pytest.approx(1 / 4)
    assert np.isnan(S.csi(np.full(4, 10.0), np.full(4, 10.0)))   # exactly 10 is not heavy rain


def test_pipeline_csi_matches_hand_count():
    data, t = _random_data()
    res = _evaluate(data, t, _persistence, n_windows=5)
    idx = [int(np.nonzero(t == np.datetime64(d))[0][0]) for d in res["window_dates"]]
    for lead in (1, 4, 8):
        pred = np.stack([data[i, CH["tp"]] for i in idx])          # persistence = last input day
        tru = np.stack([data[i + lead, CH["tp"]] for i in idx])
        h = np.sum((pred > 10) & (tru > 10)); m = np.sum((pred <= 10) & (tru > 10))
        f = np.sum((pred > 10) & (tru <= 10))
        r = _row(res, lead, "tp")
        assert r["csi_model"] == pytest.approx(h / (h + m + f))
        assert r["csi_persist"] == r["csi_model"]
        assert r["csi_skill"] <= 0.0 + 1e-12


def test_climatology_is_kit_15_day_from_train_years_only():
    data, t = _random_data()
    clim = score.train_climatology(data, t, TRAIN)
    tr = score.year_of(t) == 2016
    np.testing.assert_array_equal(clim, climatology(data[tr], score.yyyymmdd(t[tr])))
    data2 = data.copy()
    data2[~tr] = data2[~tr] * 5.0 + 1000.0                         # wildly different val year
    np.testing.assert_array_equal(clim, score.train_climatology(data2, t, TRAIN))
    # hand check of the 15-day smoothing: a spike on one train day spreads as 1/15 over +-7 calendar days
    d0 = np.zeros((366, N_CH, NLAT, NLON), np.float32)
    d0[100] = 15.0
    c = score.train_climatology(d0, t[:366], TRAIN)[:, 0, 0, 0]
    expect = np.zeros(366)
    expect[93:108] = 1.0
    np.testing.assert_allclose(c, expect, atol=1e-6)


def test_climatology_rows_use_train_climatology():
    """If the train year is a constant field per channel, climatology RMSE is RMSE of that constant vs truth."""
    data, t = _random_data()
    tr = score.year_of(t) == 2016
    const = np.arange(N_CH, dtype=np.float32)[:, None, None]
    data[tr] = const
    data[tr, CH["t2m"]] = 300.0
    res = _evaluate(data, t, _persistence, n_windows=1)
    i = int(np.nonzero(t == np.datetime64(res["window_dates"][0]))[0][0])
    w = np.cos(np.deg2rad(LAT)); w = (w / w.mean())[:, None]
    for row in res["rows"]:
        c = CHANNELS.index(row["channel"])
        ref = 300.0 if row["channel"] == "t2m" else float(c)
        err = (data[i + row["lead"], c].astype(np.float64) - ref) ** 2
        assert row["rmse_clim"] == pytest.approx(np.sqrt((err * w).mean()), rel=1e-5)


def test_lead_k_is_rollout_index_k_minus_1():
    t = _times()
    data = np.broadcast_to((290.0 + 0.05 * np.arange(t.size, dtype=np.float32))[:, None, None, None],
                           (t.size, N_CH, NLAT, NLON)).copy()

    def oracle(x, d, steps):
        return x[-1][None] + 0.05 * np.arange(1, steps + 1)[:, None, None, None]

    def off_by_one(x, d, steps):
        return x[-1][None] + 0.05 * np.arange(0, steps)[:, None, None, None]

    good, bad = _evaluate(data, t, oracle), _evaluate(data, t, off_by_one)
    for g, b in zip(good["rows"], bad["rows"]):
        assert g["rmse_model"] == pytest.approx(0.0, abs=1e-4)
        assert b["rmse_model"] == pytest.approx(0.05, rel=1e-3)
        assert g["rmse_persist"] == pytest.approx(0.05 * g["lead"], rel=1e-3)


def test_windows_inside_val_years_and_sampling():
    data, t = _random_data()
    all_w = score.build_windows(t, score.year_of(t) == 2017)
    assert t[all_w[0] - N_IN_DAYS + 1] == np.datetime64("2017-01-01")
    assert t[all_w[-1] + 10] == np.datetime64("2017-12-31")
    a, b = score.sample_windows(all_w, 7, seed=3), score.sample_windows(all_w, 7, seed=3)
    np.testing.assert_array_equal(a, b)
    assert len(set(a.tolist())) == 7 and np.all(np.isin(a, all_w))
    np.testing.assert_array_equal(score.sample_windows(all_w, 0), all_w)


# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------
def test_persistence_scores_zero_vs_persistence():
    data, t = _random_data()
    res = _evaluate(data, t, _persistence)
    assert len(res["rows"]) == 3 * N_CH
    for r in res["rows"]:
        assert r["ss_persist"] == 0.0 and r["rmse_model"] == r["rmse_persist"]
        assert r["ss"] <= 0.0
    assert all(s <= 0.0 for s in res["window_ss"])
    assert res["points"]["rmse_pts"] == 0.0 and res["points"]["csi_pts"] == 0.0
    assert not res["gate_day4_beats_persistence"] and res["failures"] == []


def _perfect(data, t):
    lookup = {data[i].tobytes(): i for i in range(t.size)}

    def f(x, d, steps):
        i = lookup[np.asarray(x[-1], np.float32).tobytes()]
        return data[i + 1:i + 1 + steps].astype(np.float64)
    return f


def test_perfect_forecast_gets_full_points_and_judge5_agrees():
    data, t = _random_data()
    res = _evaluate(data, t, _perfect(data, t), n_windows=5, judge_windows=5)
    assert res["mean_ss"] == pytest.approx(1.0)
    assert res["points"] == {"rmse_pts": 30.0, "csi_pts": 15.0, "anti_blur_pts": 10.0}
    assert res["judge5"]["total"]["mean"] == pytest.approx(55.0)       # k == n -> same as the full score


def test_invalid_window_scores_zero_not_skipped():
    data, t = _random_data()
    perfect = _perfect(data, t)
    calls = {"n": 0}

    def flaky(x, d, steps):
        calls["n"] += 1
        out = perfect(x, d, steps)
        if calls["n"] % 2 == 0:
            out[0, CH["tp"], 0, 0] = -1.0          # negative rain -> invalid window
        if calls["n"] == 3:
            out[2, CH["msl"], 5, 5] = np.nan
        return out

    res = _evaluate(data, t, flaky, n_windows=6)
    assert len(res["failures"]) == 4
    assert sorted(res["window_ss"]) == [0.0] * 4 + [pytest.approx(1.0)] * 2
    assert res["mean_ss"] == pytest.approx(2 / 6)


def test_anti_blur_flat_forecast_loses_points():
    data, t = _random_data()

    def flat(x, d, steps):
        out = score.persistence_forecast(x, steps).astype(np.float64)
        return np.broadcast_to(out.mean(axis=(-2, -1), keepdims=True), out.shape).copy()

    res = _evaluate(data, t, flat)
    assert res["points"]["anti_blur_pts"] == pytest.approx(0.0, abs=1e-6)     # float32 roundoff only
    assert all(v == pytest.approx(0.0, abs=1e-6) for v in res["day8_var_ratio"].values())
    assert _evaluate(data, t, _persistence)["points"]["anti_blur_pts"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_synthetic_writes_csv_and_summary(tmp_path, capsys):
    out, summ = tmp_path / "s.csv", tmp_path / "s.json"
    assert score.main(["--synthetic", "--twin", "--windows", "8", "--out", str(out),
                       "--summary-out", str(summ)]) == 0
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == score.CSV_COLUMNS and len(rows) == 18
    assert rows[1]["channel"] == "tp" and rows[1]["csi_model"] != "" and rows[0]["csi_model"] == ""
    js = json.loads(summ.read_text())
    assert set(js["points"]) == {"rmse_pts", "csi_pts", "anti_blur_pts"} and js["n_windows"] == 8
    assert "SUMMARY" in capsys.readouterr().out
