"""Stdlib-only pre-check client for serve.py (the kit's schema validator is the real judge).

Usage:  python tools/validate.py --url http://localhost:8000
Exit code 0 = all checks pass (WARNs allowed), 1 = any FAIL.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
import urllib.error
import urllib.request

# Mirrors common.py (kept stdlib-only so this runs without numpy). Shape [3,6,32,36].
N_IN, N_CH, NLAT, NLON = 3, 6, 32, 36
T2M_MIN_K, T2M_MAX_K = -40.0 + 273.15, 60.0 + 273.15
FAIL_S, WARN_S = 5.0, 1.0

failures: list[str] = []


def report(ok: bool, name: str, info: str = "", warn: bool = False) -> None:
    tag = "PASS" if ok else ("WARN" if warn else "FAIL")
    print(f"[{tag}] {name}{(' - ' + info) if info else ''}")
    if not ok and not warn:
        failures.append(name)


def post(url: str, payload, timeout: float = 120.0):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data if data is not None else b"{}", method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def synthetic_init(seed: int = 0) -> list:
    """Plausible monsoon-season fields: t2m warmer south, tp >= 0, winds SW, msl ~1005 hPa."""
    rng = random.Random(seed)
    days = []
    for _ in range(N_IN):
        chans = [[[0.0] * NLON for _ in range(NLAT)] for _ in range(N_CH)]
        for i in range(NLAT):
            lat = 6.5 + i
            for j in range(NLON):
                chans[0][i][j] = 303.0 - 0.3 * (lat - 6.5) + rng.gauss(0, 1.0)        # t2m K
                chans[1][i][j] = max(0.0, rng.gammavariate(0.8, 8.0))                 # tp mm/day
                chans[2][i][j] = 5.0 + rng.gauss(0, 2.0)                              # u10
                chans[3][i][j] = 3.0 + rng.gauss(0, 2.0)                              # v10
                chans[4][i][j] = 100500.0 + 30.0 * (lat - 20) + rng.gauss(0, 100.0)   # msl Pa
                chans[5][i][j] = max(1.0, 50.0 - 0.8 * (lat - 6.5) + rng.gauss(0, 5)) # tcwv
        days.append(chans)
    return days


def flat(x):
    if isinstance(x, list):
        for v in x:
            yield from flat(v)
    else:
        yield x


def shape(x) -> list:
    s = []
    while isinstance(x, list):
        s.append(len(x))
        x = x[0] if x else None
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()
    base = args.url.rstrip("/")
    if "//localhost" in base:
        # On Windows "localhost" tries IPv6 ::1 first and adds ~2 s per request
        # before falling back to IPv4; the server binds 0.0.0.0, so go direct.
        base = base.replace("//localhost", "//127.0.0.1")
        print(f"note: using {base} (avoids IPv6 localhost delay)")
    steps = args.steps

    try:
        st, h = post(base + "/health", None)
        report(st == 200 and h.get("status") == "ready", "health", json.dumps(h))
    except (urllib.error.URLError, OSError, ValueError) as e:
        report(False, "health", str(e))
        return 1

    init = synthetic_init()
    doy = 200.0
    a = 2 * math.pi * doy / 365.0
    req = {"init_states": init, "doy_sincos": [math.sin(a), math.cos(a)], "steps": steps,
           "perturbation": None}
    try:
        st, r = post(base + "/simulate", req)
    except (urllib.error.URLError, OSError, ValueError) as e:
        report(False, "simulate", str(e))
        return 1
    roll = r.get("rollout")
    shp = shape(roll)
    report(shp == [steps, N_CH, NLAT, NLON], "rollout shape", str(shp))
    if shp != [steps, N_CH, NLAT, NLON]:
        return 1
    report(isinstance(r.get("inference_ms"), (int, float)), "inference_ms present",
           str(r.get("inference_ms")))
    vals = list(flat(roll))
    report(all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals), "all finite")
    tp = [v for s in roll for v in flat(s[1])]
    t2m = [v for s in roll for v in flat(s[0])]
    report(min(tp) >= 0.0, "tp >= 0", f"min={min(tp):.4g}")
    report(T2M_MIN_K <= min(t2m) and max(t2m) <= T2M_MAX_K, "t2m in bounds",
           f"[{min(t2m):.2f}, {max(t2m):.2f}] K")

    zero = [[[0.0] * NLON for _ in range(NLAT)] for _ in range(N_CH)]
    _, rz = post(base + "/simulate", dict(req, perturbation=zero))
    diff = max(abs(a_ - b_) for a_, b_ in zip(flat(rz["rollout"]), vals))
    report(diff == 0.0, "zero perturbation == null perturbation", f"max|diff|={diff:.3g}")

    # Bad input must be rejected with 422, not 500.
    try:
        post(base + "/simulate", dict(req, init_states=init[:2]))
        report(False, "bad shape -> 422", "server accepted [2,6,32,36]")
    except urllib.error.HTTPError as e:
        report(e.code == 422, "bad shape -> 422", f"HTTP {e.code}")

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        post(base + "/simulate", req)
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    info = f"median of 5 = {med * 1000:.1f} ms (wall, incl. JSON + HTTP)"
    if med > FAIL_S:
        report(False, "latency", info)
    elif med > WARN_S:
        report(False, "latency", info, warn=True)
    else:
        report(True, "latency", info)

    print("RESULT:", "FAIL " + ", ".join(failures) if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
