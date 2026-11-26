"""data_tools: split, windows, normalisation (synthetic) + checks on the real archive (skipped without data/)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import doy_sincos  # noqa: E402
from data_tools.loader import (WindowDataset, day_of_year, denormalize, normalize, split_years,  # noqa: E402
                               window_starts)

NC = ROOT / "data" / "india_climate_train.nc"
STATS = ROOT / "data" / "stats.json"
needs_data = pytest.mark.skipif(not (NC.exists() and STATS.exists()), reason="data/ not present")


def synthetic(year0=2016, year1=2020):
    dates = np.arange(np.datetime64(f"{year0}-01-01"), np.datetime64(f"{year1 + 1}-01-01")).astype("datetime64[D]")
    data = np.zeros((dates.size, 6, 32, 36), np.float32)
    data += np.arange(dates.size, dtype=np.float32)[:, None, None, None]   # value = day index
    return data, dates


def test_split_is_disjoint_and_by_year():
    _, dates = synthetic()
    tr, va = split_years(dates, train=(2016, 2018), val=(2019, 2020))
    assert not (tr & va).any() and (tr | va).all()
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    assert years[tr].max() == 2018 and years[va].min() == 2019
    with pytest.raises(ValueError):
        split_years(dates, train=(2016, 2019), val=(2019, 2020))


@pytest.mark.parametrize("k", [1, 4, 10])
def test_windows_never_cross_split(k):
    data, dates = synthetic()
    tr, va = split_years(dates, train=(2016, 2018), val=(2019, 2020))
    for mask in (tr, va):
        ds = WindowDataset(data, dates, mask, k=k)
        assert len(ds) == int(mask.sum()) - 3 - k + 1   # one contiguous block of years
        for i in (0, len(ds) - 1):
            init, _, tgt = ds[i]
            idx = np.concatenate([init[:, 0, 0, 0], tgt[:, 0, 0, 0]]).astype(int)
            assert mask[idx].all()


def test_windows_skip_date_gaps():
    data, dates = synthetic(2019, 2019)
    keep = np.ones(dates.size, bool)
    keep[100] = False                                   # drop one day from the file
    w = window_starts(dates[keep], np.ones(keep.sum(), bool), k=2)
    d = dates[keep].astype(np.int64)
    for t in w:
        assert d[t + 2] - d[t - 2] == 4


def test_target_alignment_and_doy_of_last_input_day():
    data, dates = synthetic()
    ds = WindowDataset(data, dates, np.ones(dates.size, bool), k=5)
    for i in (0, 37, len(ds) - 1):
        t = int(ds.index[i])
        init, sc, tgt = ds[i]
        assert init.shape == (3, 6, 32, 36) and tgt.shape == (5, 6, 32, 36) and sc.shape == (2,)
        np.testing.assert_array_equal(init[:, 0, 0, 0], [t - 2, t - 1, t])
        np.testing.assert_array_equal(tgt[:, 3, 5, 7], t + 1 + np.arange(5))   # target[k] = day k+1 after last input
        np.testing.assert_allclose(sc, doy_sincos(day_of_year(dates[t:t + 1])[0]), atol=1e-6)
        assert ds.last_input_date(i) == dates[t]


def test_day_of_year_is_one_based():
    d = np.array(["2019-01-01", "2019-12-31", "2020-12-31"], dtype="datetime64[D]")
    np.testing.assert_array_equal(day_of_year(d), [1, 365, 366])


def test_normalize_round_trip():
    rng = np.random.default_rng(0)
    stats = {"channels": ["t2m", "tp", "u10", "v10", "msl", "tcwv"],
             "mean": np.array([292.0, 3.6, 0.9, 0.2, 101122.0, 30.0]), "std": np.array([13.5, 7.5, 2.9, 2.4, 619.0, 20.5])}
    x = (stats["mean"][:, None, None] + stats["std"][:, None, None] * rng.standard_normal((4, 6, 32, 36)))
    z = normalize(x, stats)
    assert abs(float(z.mean())) < 0.05 and abs(float(z.std()) - 1) < 0.05
    np.testing.assert_allclose(denormalize(z, stats), x, rtol=1e-6, atol=1e-2)


def test_torch_wrapper_if_available():
    torch = pytest.importorskip("torch")
    from data_tools.loader import TorchWindowDataset
    data, dates = synthetic(2019, 2019)
    ds = TorchWindowDataset(WindowDataset(data, dates, np.ones(dates.size, bool), k=2))
    init, sc, tgt = ds[0]
    assert isinstance(init, torch.Tensor) and tuple(tgt.shape) == (2, 6, 32, 36) and sc.dtype == torch.float32


# ---------------------------------------------------------------------------
# real archive
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def archive():
    from data_tools.loader import load_archive
    return load_archive(NC)


@needs_data
def test_archive_matches_stats_and_grid(archive):
    from data_tools.loader import check_against_stats, check_grid, load_stats
    assert archive.data.shape == (6209, 6, 32, 36)
    assert check_grid(archive.lat, archive.lon, archive.channels) == []
    assert check_against_stats(archive.data, load_stats(STATS), verbose=False) == []
    assert archive.static_names == ("land_sea_mask", "orography_m")


@needs_data
def test_val_window_count_matches_eval(archive):
    _, va = split_years(archive.dates)
    assert window_starts(archive.dates, va, k=10).size == 1084    # same as eval.score --windows 0


@needs_data
def test_baselines_match_eval_baseline_md(archive):
    pytest.importorskip("monsoon_r1")
    from data_tools.baselines import compute_baselines
    rows, n = compute_baselines(archive.data, archive.dates, max_lead=4)
    assert n == 1096 - 3 - 4 + 1                      # all val windows with 4 target days
    r = {(x["lead"], x["channel"]): x for x in rows}
    assert r[(1, "t2m")]["best_ref"] == "persist" and r[(4, "t2m")]["best_ref"] == "clim"
    assert r[(1, "t2m")]["rmse_lat_persist"] == pytest.approx(1.03, abs=0.02)
    assert r[(4, "tp")]["rmse_lat_clim"] == pytest.approx(6.74, abs=0.05)
