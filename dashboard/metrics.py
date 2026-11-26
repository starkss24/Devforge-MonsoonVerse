"""Pure numpy helpers for the Shock Lab and Ensemble tabs. No UI imports, no server access.

All "energy" numbers follow the organisers' kit (monsoon_r1.scoring._energy): the response to a shock is
d = shocked - base, and its energy per cell is sum over channels of (d / channel_std)^2, i.e. std units squared.
The "near" neighbourhood copies the kit's near_mask: the box grown by DILATE = 3 steps of 4-neighbour dilation.
Rings use Chebyshev distance from the box (0 = inside, 1..8 = rings), as eval/shock_test.ring_distance does.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common import LATS, LONS, NLAT, NLON, box_mask

DILATE = 3                     # kit scoring.DILATE
LOCAL_MIN_FRAC = 0.60          # kit scoring.LOCAL_MIN_FRAC
NONZERO_MIN = 1e-3             # kit scoring.NONZERO_MIN
STABLE_MAX_STD = 5.0           # kit scoring.STABLE_MAX_STD
MAX_SPEED_CELLS_PER_DAY = 8.0  # kit scoring.MAX_SPEED_CELLS_PER_DAY
MAX_RING = 8
REACH_FRAC = 0.05


# ---------------------------------------------------------------- boxes
def mask_from_box(box) -> np.ndarray:
    """(lat_min, lat_max, lon_min, lon_max) -> bool [32, 36] mask (cell centres inside, inclusive)."""
    return box_mask(*box)


def box_from_perturbation(pert: np.ndarray) -> tuple[float, float, float, float] | None:
    """Bounding box (lat_min, lat_max, lon_min, lon_max) of every cell where any channel is nonzero, or None."""
    nz = np.any(np.asarray(pert) != 0, axis=0)
    if not nz.any():
        return None
    ii, jj = np.where(nz)
    return float(LATS[ii.min()]), float(LATS[ii.max()]), float(LONS[jj.min()]), float(LONS[jj.max()])


def bbox_mask(pert: np.ndarray) -> np.ndarray:
    """Filled bounding-box mask of the perturbation's nonzero cells (all False if the perturbation is zero)."""
    box = box_from_perturbation(pert)
    return np.zeros((NLAT, NLON), bool) if box is None else mask_from_box(box)


def near_mask(mask: np.ndarray, dilate: int = DILATE) -> np.ndarray:
    """Kit-identical neighbourhood: `dilate` rounds of 4-neighbour growth of the box mask."""
    out = np.asarray(mask, bool).copy()
    for _ in range(dilate):
        o = out.copy()
        o[1:] |= out[:-1]
        o[:-1] |= out[1:]
        o[:, 1:] |= out[:, :-1]
        o[:, :-1] |= out[:, 1:]
        out = o
    return out


def ring_distance(mask: np.ndarray) -> np.ndarray:
    """Chebyshev distance in cells from the mask: 0 inside, k for the k-th ring. Large value if mask is empty."""
    mask = np.asarray(mask, bool)
    big = np.iinfo(np.int32).max
    if not mask.any():
        return np.full(mask.shape, big, dtype=np.int64)
    ii, jj = np.indices(mask.shape)
    bi, bj = np.where(mask)
    d = np.maximum(np.abs(ii[..., None] - bi), np.abs(jj[..., None] - bj)).min(axis=-1)
    return d.astype(np.int64)


def euclid_distance(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance in cells from the nearest mask cell (the kit's teleport metric)."""
    ii, jj = np.indices(mask.shape)
    bi, bj = np.where(mask)
    return np.sqrt((ii[..., None] - bi) ** 2 + (jj[..., None] - bj) ** 2).min(axis=-1)


# ---------------------------------------------------------------- energy
def energy(delta: np.ndarray, std: np.ndarray) -> np.ndarray:
    """[steps, 6, H, W] -> [steps, H, W] summed squared response in channel-std units (kit _energy)."""
    std = np.asarray(std, dtype=np.float64)
    return ((np.asarray(delta, dtype=np.float64) / std[None, :, None, None]) ** 2).sum(axis=1)


def region_sums(e: np.ndarray, mask: np.ndarray) -> dict:
    """Per-day energy inside the box, in the near ring (near minus box) and outside the near zone."""
    mask = np.asarray(mask, bool)
    near = near_mask(mask)
    return {
        "inside": e[:, mask].sum(axis=1),
        "near": e[:, near & ~mask].sum(axis=1),
        "outside": e[:, ~near].sum(axis=1),
        "total": e.reshape(e.shape[0], -1).sum(axis=1),
    }


def near_frac_day1(e: np.ndarray, mask: np.ndarray) -> float:
    """Share of day-1 energy within the kit's near zone (box + 3 cells). 0 when there is no response."""
    tot = float(e[0].sum())
    return float(e[0][near_mask(mask)].sum() / tot) if tot > 0 else 0.0


def radial_profile(e: np.ndarray, mask: np.ndarray, max_ring: int = MAX_RING) -> np.ndarray:
    """[steps, max_ring + 1] fraction of each day's energy at ring 0 (inside) .. max_ring. Rows sum to <= 1
    (energy beyond max_ring is left out). Days with zero energy give a zero row."""
    dist = ring_distance(mask)
    tot = e.reshape(e.shape[0], -1).sum(axis=1)
    prof = np.stack([e[:, dist == k].sum(axis=1) for k in range(max_ring + 1)], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tot[:, None] > 0, prof / np.where(tot[:, None] > 0, tot[:, None], 1.0), 0.0)


def reach(e: np.ndarray, mask: np.ndarray, frac: float = REACH_FRAC, max_ring: int = MAX_RING) -> np.ndarray:
    """[steps] furthest ring (0..max_ring) holding >= frac of that day's energy; -1 for a day with no response."""
    prof = radial_profile(e, mask, max_ring)
    out = np.full(prof.shape[0], -1, dtype=int)
    for k, row in enumerate(prof):
        idx = np.where(row >= frac)[0]
        if idx.size:
            out[k] = int(idx.max())
    return out


# ---------------------------------------------------------------- kit-style checks (fallback)
def causality_checks_local(base: np.ndarray, shocked: np.ndarray, mask: np.ndarray, std: np.ndarray) -> dict:
    """Same checks and thresholds as monsoon_r1.scoring.causality_checks, for when the kit is not importable."""
    d = np.asarray(shocked, np.float64) - np.asarray(base, np.float64)
    res = {"finite": bool(np.isfinite(d).all())}
    if not res["finite"]:
        return {**res, "nonzero": False, "localized": False, "propagates": False, "stable": False,
                "near_frac_day1": 0.0, "far_frac_max": 1.0, "no_teleport": False}
    std = np.asarray(std, np.float64)
    z = np.abs(d / std[None, :, None, None])
    res["nonzero"] = bool(z.max() >= NONZERO_MIN)
    res["stable"] = bool(z.max() <= STABLE_MAX_STD)
    e = energy(d, std)
    steps = e.shape[0]
    near = near_mask(mask)
    res["near_frac_day1"] = near_frac_day1(e, mask)
    res["localized"] = bool(e[0].sum() > 0 and res["near_frac_day1"] >= LOCAL_MIN_FRAC)
    out_e = np.array([e[k][~near].sum() for k in range(steps)])
    late = out_e[3:8].mean() if steps >= 8 else out_e[-1]
    grows = bool(late > 1.5 * out_e[0] + 1e-12 and late > 0.02 * max(e.sum(axis=(1, 2)).max(), 1e-12))
    dist = euclid_distance(mask) if np.asarray(mask).any() else np.zeros(mask.shape)
    tele = 0.0
    for k in range(steps):
        far = dist > (DILATE + MAX_SPEED_CELLS_PER_DAY * (k + 1))
        tele = max(tele, float(e[k][far].sum() / max(e[k].sum(), 1e-12)))
    res["far_frac_max"] = tele
    res["no_teleport"] = bool(tele < 0.05)
    res["propagates"] = bool(grows and res["no_teleport"])
    return res


# ---------------------------------------------------------------- ensemble
def add_input_noise(init_states: np.ndarray, std: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """init_states [3,6,H,W] + Gaussian noise of frac * channel std (fixed seed). Rain is kept >= 0."""
    rng = np.random.default_rng(seed)
    x = np.asarray(init_states, np.float64)
    noise = rng.standard_normal(x.shape) * (frac * np.asarray(std, np.float64))[None, :, None, None]
    out = x + noise
    out[:, 1] = np.maximum(out[:, 1], 0.0)
    return out


def spread_by_day(members: np.ndarray, std: np.ndarray) -> np.ndarray:
    """members [M, steps, 6, H, W] -> [steps, 6] mean over cells of ensemble std, divided by channel std."""
    s = members.std(axis=0)                         # [steps, 6, H, W]
    return s.mean(axis=(-2, -1)) / np.asarray(std, np.float64)[None, :]
