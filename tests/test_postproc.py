"""postproc.py (numpy port of twin_model/post.py::Post.apply) and its wiring into root twin.py.

The model lane's Post is imported here, in the TEST only, as the reference implementation."""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import CH, STATE_SHAPE, doy_sincos, enforce_validity  # noqa: E402
from postproc import POST_FILE, PostProc, clim_slot, doy_from_sincos, save_npz, target_slots  # noqa: E402
from twin import Twin  # noqa: E402
from twin_model.post import Post  # noqa: E402  (reference, test-only import)
from twin_model.post import target_slots as ref_target_slots  # noqa: E402

H, W = 32, 36
L, Q = 10, 40


def _params(seed=0, blend=True, qm=True, strength=False, L=L, Q=Q):
    rng = np.random.default_rng(seed)
    kw = {}
    if blend:
        b = rng.uniform(0.0, 0.9, (L, 6))
        b[0, :] = 0.0                       # a lead with no blend (Post skips it)
        b[1, 3] = 1.3                       # out of range: Post clips to [0, 1]
        kw["blend"] = b.astype(np.float32)
        base = np.array([300.0, 3.0, 1.0, -1.0, 100500.0, 40.0])
        scale = np.array([5.0, 4.0, 3.0, 3.0, 600.0, 15.0])
        clim = base[None, :, None, None] + scale[None, :, None, None] * rng.normal(size=(366, 6, H, W))
        clim[:, 1] = np.abs(clim[:, 1])
        kw["clim"] = clim.astype(np.float32)
    if qm:
        fq = np.concatenate([np.zeros((L, 1)), np.cumsum(rng.uniform(0.05, 3.0, (L, Q - 1)), axis=1)], axis=1)
        oq = np.concatenate([np.zeros((L, 1)), np.cumsum(rng.uniform(0.0, 4.0, (L, Q - 1)), axis=1)], axis=1)
        kw["qmap_fq"], kw["qmap_oq"] = fq.astype(np.float32), oq.astype(np.float32)
        if strength:
            s = rng.uniform(0.2, 1.0, L)
            s[2] = 1.0
            kw["qmap_strength"] = s.astype(np.float32)
    return kw


def _rollout(steps, seed=0):
    rng = np.random.default_rng(100 + seed)
    y = np.zeros((steps, 6, H, W))
    y[:, 0] = 300 + rng.normal(0, 8, y[:, 0].shape)
    rain = rng.gamma(0.5, 15.0, y[:, 1].shape)
    rain[rng.uniform(size=rain.shape) < 0.3] = 0.0           # exactly dry cells
    rain.flat[:20] = 500.0                                    # beyond the last knot
    y[:, 1] = rain
    y[:, 2] = rng.normal(2, 4, y[:, 2].shape)
    y[:, 3] = rng.normal(1, 4, y[:, 3].shape)
    y[:, 4] = rng.normal(100500, 500, y[:, 4].shape)
    y[:, 5] = rng.uniform(5, 70, y[:, 5].shape)
    return y.astype(np.float32).astype(np.float64)            # twin.py's rollout: float64 of float32 values


def _dsc(day: dt.date) -> np.ndarray:
    """Contract encoding of the last input day (1-based doy / 365), float32 as twin_model/twin.py passes it."""
    return doy_sincos(day.timetuple().tm_yday).astype(np.float32)


DATES = [dt.date(2023, 2, 28), dt.date(2024, 2, 28), dt.date(2024, 2, 29), dt.date(2023, 3, 1),
         dt.date(2024, 3, 1), dt.date(2023, 12, 31), dt.date(2024, 12, 31), dt.date(2023, 12, 25),
         dt.date(2024, 12, 30), dt.date(2023, 1, 1), dt.date(2024, 7, 15)]


@pytest.mark.parametrize("variant", ["full", "strength", "blend_only", "qmap_only", "short_L"])
def test_matches_post_apply(tmp_path, variant):
    kw = {"full": dict(), "strength": dict(strength=True), "blend_only": dict(qm=False),
          "qmap_only": dict(blend=False), "short_L": dict(L=4)}[variant]
    params = _params(seed=len(variant), **kw)
    ref = Post(**params)
    save_npz(tmp_path / POST_FILE, **params, meta={"variant": variant})
    ours = PostProc.load(tmp_path / POST_FILE)
    assert ours.meta == {"variant": variant}
    worst = 0.0
    for i, day in enumerate(DATES):
        for steps in (1, 10, 14):
            y = _rollout(steps, seed=i)
            d = _dsc(day)
            a, b = ours.apply(y, d), ref.apply(y, d)
            assert a.dtype == np.float32 and a.shape == (steps, 6, H, W)
            worst = max(worst, float(np.abs(a.astype(np.float64) - b).max()))
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=0)
    print(f"\n[postproc] {variant}: max |ours - Post.apply| = {worst:.3e}")
    assert worst == 0.0      # same float32 ops in the same order -> bit-identical


def test_calendar_matches_post_and_kit():
    stats = pytest.importorskip("monsoon_r1.stats")
    for start in (dt.date(2023, 1, 1), dt.date(2024, 1, 1)):
        for n in range(366):
            day = start + dt.timedelta(days=n)
            d = _dsc(day)
            assert np.array_equal(target_slots(d, 14), ref_target_slots(d, 14))
            if day.year % 4 and (day + dt.timedelta(days=14)).year % 4:   # non-leap: exact kit calendar
                want = [stats._doy_index(day + dt.timedelta(days=k)) for k in range(1, 15)]
                assert list(target_slots(d, 14)) == want, day
    assert clim_slot(59) == 58 and clim_slot(60) == 60 and clim_slot(366) == 0      # Feb 28, Mar 1, wrap
    assert doy_from_sincos(doy_sincos(366)) == 1                                    # leap Dec 31 (Post's convention)
    assert list(target_slots(_dsc(dt.date(2023, 12, 31)), 2)) == [0, 1]             # Dec 31 -> Jan 1, 2


@pytest.mark.parametrize("bad", ["clim_shape", "blend_shape", "knots", "zero_knot", "unknown_key", "lfs",
                                 "garbage", "nan", "strength", "version", "empty", "pickle"])
def test_malformed_rejected(tmp_path, bad):
    p = tmp_path / POST_FILE
    params = _params(strength=True)
    if bad == "lfs":
        p.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n")
    elif bad == "garbage":
        p.write_bytes(b"definitely not a zip")
    else:
        arrs = {k: v for k, v in params.items()}
        if bad == "clim_shape":
            arrs["clim"] = arrs["clim"][:365]
        elif bad == "blend_shape":
            arrs["blend"] = arrs["blend"][:, :5]
        elif bad == "knots":
            arrs["qmap_fq"][3, 5] = arrs["qmap_fq"][3, 4]
        elif bad == "zero_knot":
            arrs["qmap_oq"][:, 0] = 0.5
        elif bad == "unknown_key":
            arrs["fq"] = arrs.pop("qmap_fq")
        elif bad == "nan":
            arrs["clim"][7, 2, 3, 4] = np.nan
        elif bad == "strength":
            arrs["qmap_strength"][0] = 1.5
        elif bad == "version":
            arrs["version"] = np.array(2)
        elif bad == "empty":
            arrs = {"version": np.array(1)}
        elif bad == "pickle":
            arrs["meta"] = np.array([{"a": 1}], dtype=object)
        np.savez(p, **arrs)
    with pytest.raises(Exception):
        PostProc.load(p)


# ---- wiring into twin.py ----------------------------------------------------------------------------------
@pytest.fixture
def onnx_dir(tmp_path):
    from tests.test_twin import Diffuse, _save_onnx
    _save_onnx(tmp_path, Diffuse())
    return tmp_path


def _init_state():
    from tests.test_twin import _init
    return _init(3)


def test_end_to_end_twin_matches_post(onnx_dir):
    params = _params(seed=7, strength=True)
    save_npz(onnx_dir / POST_FILE, **params)
    t = Twin(weights_path=onnx_dir)
    raw = Twin(weights_path=onnx_dir, post_path=onnx_dir / "none.npz")
    assert t.backend == "onnx" and t.post_status == "loaded" and raw.post_status == "absent"
    ref = Post(**params)
    x = _init_state()
    for day in (dt.date(2024, 2, 28), dt.date(2023, 12, 30), dt.date(2023, 7, 1)):
        for steps in (1, 10, 14):
            d = doy_sincos(day.timetuple().tm_yday)
            got = t.rollout(x, d, steps=steps)
            want = enforce_validity(ref.apply(raw.rollout(x, d, steps=steps), d.astype(np.float32)))
            assert np.array_equal(got, want)
            assert not np.array_equal(got, raw.rollout(x, d, steps=steps))
            assert np.isfinite(got).all() and got[:, CH["tp"]].min() >= 0
    d = doy_sincos(200)
    a = t.rollout(x, d, steps=10)
    assert np.array_equal(a, t.rollout(x, d, steps=10, perturbation=np.zeros(STATE_SHAPE)))
    assert np.array_equal(a, t.rollout(x, d, steps=10))


def test_absent_file_is_unchanged(onnx_dir):
    t = Twin(weights_path=onnx_dir)
    assert t.post is None and t.post_status == "absent"
    x, d = _init_state(), doy_sincos(150)
    # exactly the pre-post-processing path: model loop, then the clamps
    assert np.array_equal(t.rollout(x, d, steps=10), enforce_validity(t._model_rollout(x, d, 10)))


@pytest.mark.parametrize("content", ["lfs", "shape"])
def test_malformed_file_skipped_and_reported(onnx_dir, content, caplog):
    p = onnx_dir / POST_FILE
    if content == "lfs":
        p.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n")
    else:
        np.savez(p, blend=np.zeros((10, 5), np.float32), clim=np.zeros((366, 6, H, W), np.float32))
    t = Twin(weights_path=onnx_dir)
    raw = Twin(weights_path=onnx_dir, post_path=onnx_dir / "none.npz")
    assert t.backend == "onnx" and t.post is None and t.post_status.startswith("error:")
    assert "FAILED to load post-processing" in caplog.text
    x, d = _init_state(), doy_sincos(150)
    assert np.array_equal(t.rollout(x, d, steps=10), raw.rollout(x, d, steps=10))
    from fastapi.testclient import TestClient
    from serve import create_app
    h = TestClient(create_app(t)).post("/health").json()
    assert h["status"] == "ready" and h["backend"] == "onnx" and h["post"].startswith("error:")


def test_health_reports_loaded(onnx_dir):
    save_npz(onnx_dir / POST_FILE, **_params())
    from fastapi.testclient import TestClient
    from serve import create_app
    h = TestClient(create_app(Twin(weights_path=onnx_dir))).post("/health").json()
    assert h["backend"] == "onnx" and h["post"] == "loaded" and h["post_failures"] == 0


def test_persistence_backend_never_post_processed(tmp_path, onnx_dir, monkeypatch):
    nomodel = tmp_path / "post_only"
    nomodel.mkdir()
    save_npz(nomodel / POST_FILE, **_params())           # post.npz but no model at all
    x, d = _init_state(), doy_sincos(150)
    t = Twin(weights_path=nomodel / "model.pt")
    t2 = Twin(weights_path=nomodel)
    monkeypatch.setenv("TWIN_BACKEND", "persistence")    # model + post present, persistence forced
    save_npz(onnx_dir / POST_FILE, **_params())
    t3 = Twin(weights_path=onnx_dir)
    for tw in (t, t2, t3):
        assert tw.backend == "persistence" and tw.post is None
        out = tw.rollout(x, d, steps=10)
        assert np.array_equal(out, enforce_validity(np.repeat(x[-1][None], 10, axis=0)))
    assert t.post_status.startswith("skipped") and t3.post_status.startswith("skipped")


def test_rain_nonnegative_and_t2m_bounded_after_post(onnx_dir):
    params = _params(seed=3, qm=False)
    params["blend"][:] = 1.0
    params["clim"][:, 1] = -5.0          # nonsense negative rain climatology
    params["clim"][:, 0] = 400.0         # and absurd heat: the clamps still hold
    save_npz(onnx_dir / POST_FILE, **params)
    t = Twin(weights_path=onnx_dir)
    assert t.post_status == "loaded"
    out = t.rollout(_init_state(), doy_sincos(150), steps=10)
    assert np.isfinite(out).all() and out[:, CH["tp"]].min() >= 0
    assert out[:, CH["t2m"]].max() <= 333.15


def test_save_npz_format(tmp_path):
    save_npz(tmp_path / POST_FILE, **_params(strength=True), meta={"model": "x"})
    with np.load(tmp_path / POST_FILE, allow_pickle=False) as z:
        assert int(z["version"]) == 1 and json.loads(str(z["meta"])) == {"model": "x"}
        assert z["clim"].shape == (366, 6, H, W) and z["clim"].dtype == np.float32
        assert z["blend"].shape == (L, 6) and z["qmap_fq"].shape == (L, Q) and z["qmap_strength"].shape == (L,)
