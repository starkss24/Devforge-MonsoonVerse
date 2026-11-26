"""Hand-checked tests for dashboard/metrics.py (pure numpy helpers behind Shock Lab and Ensemble)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "dashboard"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import metrics as M  # noqa: E402
from common import BOXES, LATS, LONS, box_mask  # noqa: E402


def test_near_mask_tiny_diamond():
    m = np.zeros((7, 7), bool)
    m[3, 3] = True
    n = M.near_mask(m, dilate=2)
    ii, jj = np.indices(m.shape)
    assert np.array_equal(n, (np.abs(ii - 3) + np.abs(jj - 3)) <= 2)   # 4-neighbour growth = diamond
    assert n.sum() == 13


@pytest.mark.parametrize("name", ["warm_ocean", "monsoon_stall"])
def test_near_mask_matches_kit(name):
    S = pytest.importorskip("monsoon_r1.scoring")
    box = BOXES[name]
    assert np.array_equal(M.near_mask(M.mask_from_box(box)), S.near_mask(box))


def test_ring_distance_chebyshev():
    m = np.zeros((5, 5), bool)
    m[2, 2] = True
    d = M.ring_distance(m)
    assert d[2, 2] == 0 and d[1, 1] == 1 and d[0, 0] == 2 and d[0, 2] == 2 and d[2, 3] == 1


def test_energy_and_region_sums():
    std = np.array([1.0, 2.0, 1, 1, 1, 1])
    d = np.zeros((2, 6, 9, 9))
    d[0, 0, 4, 4] = 3.0          # inside: 9
    d[0, 1, 4, 6] = 2.0          # 2 cells right of the box -> near: (2/2)^2 = 1
    d[1, 0, 0, 0] = 1.0          # far corner (Manhattan 8) -> outside: 1
    e = M.energy(d, std)
    assert e[0, 4, 4] == 9.0 and e[0, 4, 6] == 1.0
    m = np.zeros((9, 9), bool)
    m[4, 4] = True
    r = M.region_sums(e, m)
    assert r["inside"].tolist() == [9.0, 0.0]
    assert r["near"].tolist() == [1.0, 0.0]
    assert r["outside"].tolist() == [0.0, 1.0]
    assert r["total"].tolist() == [10.0, 1.0]
    assert M.near_frac_day1(e, m) == pytest.approx(1.0)


def test_radial_profile_and_reach():
    e = np.zeros((3, 9, 9))
    e[0, 4, 4] = 1.0                     # all inside
    e[1, 4, 4] = 0.5; e[1, 4, 7] = 0.5   # half at ring 3
    m = np.zeros((9, 9), bool)
    m[4, 4] = True
    p = M.radial_profile(e, m, max_ring=4)
    assert p[0].tolist() == [1, 0, 0, 0, 0]
    assert p[1].tolist() == [0.5, 0, 0, 0.5, 0]
    assert p[2].tolist() == [0, 0, 0, 0, 0]
    assert M.reach(e, m, max_ring=4).tolist() == [0, 3, -1]


def test_box_from_perturbation():
    pert = np.zeros((6, 32, 36))
    pert[0, 3, 5] = 1.0
    pert[5, 6, 2] = -1.0
    box = M.box_from_perturbation(pert)
    assert box == (LATS[3], LATS[6], LONS[2], LONS[5])
    assert M.bbox_mask(pert).sum() == 4 * 4
    assert M.box_from_perturbation(np.zeros((6, 32, 36))) is None
    full = np.zeros((6, 32, 36)); full[0][box_mask(*BOXES["warm_ocean"])] = 3.0
    assert np.array_equal(M.bbox_mask(full), box_mask(*BOXES["warm_ocean"]))


def test_local_checks_match_kit_on_persistent_box_response():
    S = pytest.importorskip("monsoon_r1.scoring")
    box = BOXES["warm_ocean"]
    std = np.array([8.0, 8.0, 3.5, 2.5, 450.0, 20.0])
    base = np.zeros((10, 6, 32, 36))
    shocked = base.copy()
    shocked[:, 0][:, box_mask(*box)] = 3.0
    kit = S.causality_checks(base, shocked, box, std)
    loc = M.causality_checks_local(base, shocked, M.mask_from_box(box), std)
    for k in ("nonzero", "localized", "propagates", "stable", "no_teleport"):
        assert kit[k] == loc[k], k
    assert loc["near_frac_day1"] == pytest.approx(kit["near_frac_day1"])


def test_noise_and_spread():
    x = np.ones((3, 6, 4, 4)); x[:, 1] = 0.0
    std = np.ones(6)
    a = M.add_input_noise(x, std, 0.01, seed=1)
    b = M.add_input_noise(x, std, 0.01, seed=1)
    assert np.array_equal(a, b) and (a[:, 1] >= 0).all() and not np.array_equal(a, x)
    mem = np.zeros((2, 1, 6, 4, 4)); mem[1] = 2.0
    assert np.allclose(M.spread_by_day(mem, std * 2), 0.5)   # std of {0,2} = 1, / 2
