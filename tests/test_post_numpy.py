"""twin_model.post_numpy: the serving-path copy of the post-processing (numpy only)."""
import subprocess
import sys

import numpy as np
import pytest

from twin_model import post_numpy as PN
from twin_model.post import Post

L, Q, H, W = 10, 6, 4, 5
rng = np.random.default_rng(0)
BLEND = rng.uniform(0, 0.9, (L, 6)).astype(np.float32)
CLIM = (rng.normal(0, 1, (366, 6, H, W)) + 5).astype(np.float32)
FQ = np.tile(np.array([0, 1, 3, 8, 15, 30], np.float32), (L, 1))
OQ = np.tile(np.array([0, 1.5, 5, 14, 28, 50], np.float32), (L, 1))
DSC = np.array([np.sin(2 * np.pi * 200 / 365), np.cos(2 * np.pi * 200 / 365)], np.float32)


def _y(steps=10):
    y = rng.normal(0, 1, (steps, 6, H, W)).astype(np.float32)
    y[:, 1] = rng.gamma(0.6, 6.0, (steps, H, W)).astype(np.float32)
    y[:, 1, 0, 0] = 0.0
    return y


def test_imports_without_torch():
    code = ("import sys; sys.modules['torch'] = None; "
            "import twin_model.post_numpy as p; print(p.N_SLOTS)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "366"


@pytest.mark.parametrize("steps", [1, 10, 14])
def test_matches_torch_side_post(steps):
    y = _y(steps)
    p = PN.validate({"blend": BLEND, "clim": CLIM, "qmap_fq": FQ, "qmap_oq": OQ})
    ref = Post(BLEND, CLIM, FQ, OQ).apply(y, DSC)
    assert np.array_equal(PN.apply(y, DSC, p), ref)


def test_npz_roundtrip_and_dry_stays_dry(tmp_path):
    f = tmp_path / "post.npz"
    np.savez_compressed(f, blend=np.zeros((L, 6), np.float32), clim=CLIM, qmap_fq=FQ, qmap_oq=OQ)
    p = PN.load(f)
    out = PN.apply(_y(), DSC, p)
    assert (out[:, 1, 0, 0] == 0).all()
    assert np.isfinite(out).all()


def test_blend_only_and_qmap_only():
    y = _y()
    a = PN.apply(y, DSC, PN.validate({"blend": BLEND, "clim": CLIM}))
    assert np.array_equal(a, Post(BLEND, CLIM).apply(y, DSC))
    b = PN.apply(y, DSC, PN.validate({"qmap_fq": FQ, "qmap_oq": OQ}))
    assert np.array_equal(b, Post(qmap_fq=FQ, qmap_oq=OQ).apply(y, DSC))


@pytest.mark.parametrize("bad", [
    {"blend": BLEND},                                   # blend without clim
    {"qmap_fq": FQ},                                    # fq without oq
    {"qmap_fq": FQ, "qmap_oq": OQ[:, ::-1].copy()},     # decreasing
    {"qmap_fq": FQ + 1, "qmap_oq": OQ},                 # 0 -> nonzero
    {"blend": BLEND, "clim": CLIM[:10]},                # wrong clim shape
])
def test_bad_params_rejected(bad):
    with pytest.raises(ValueError):
        PN.validate(bad)
