"""Data loading, synthetic data, year split and consecutive-day window dataset."""
from __future__ import annotations

import warnings

import numpy as np
import torch

from twin_model.norm import CHANNELS, RAIN, H, W, N_IN, Normalizer, to_model

LAT0, LAT1, LON0, LON1 = 6.5, 37.5, 65.5, 100.5

# tolerant name maps (lower-cased candidates)
_VAR_ALIASES = {
    "t2m": ["t2m", "2m_temperature", "t2", "temp", "temperature", "air_temperature", "tas"],
    "tp": ["tp", "total_precipitation", "precip", "precipitation", "pr", "rain", "prcp"],
    "u10": ["u10", "10m_u_component_of_wind", "u", "uas", "u_wind"],
    "v10": ["v10", "10m_v_component_of_wind", "v", "vas", "v_wind"],
    "msl": ["msl", "mean_sea_level_pressure", "mslp", "slp", "psl"],
    "tcwv": ["tcwv", "total_column_water_vapour", "total_column_water_vapor", "pwat", "prw"],
}
_TIME = ["time", "date", "valid_time", "datetime"]
_LAT = ["lat", "latitude", "y"]
_LON = ["lon", "longitude", "x"]


def _find(names, candidates):
    low = {str(n).lower(): n for n in names}
    for c in candidates:
        if c in low:
            return low[c]
    return None


def _loud(msg):
    print("!" * 80 + f"\n[load_nc] WARNING: {msg}\n" + "!" * 80)
    warnings.warn(msg)


def load_nc(path, tp_scale: float = 1.0):
    """Load the India climate NetCDF -> (states [T,6,H,W] f32, dates datetime64[D], static [2,H,W], static_names).

    Names are matched tolerantly. Latitude is sorted ascending (6.5..37.5). tp units are NOT
    guessed: if it looks like metres or per-hour a LOUD warning is printed (use tp_scale to fix).
    """
    import xarray as xr

    ds = xr.open_dataset(path)
    allnames = list(ds.variables)
    print(f"[load_nc] {path}: data_vars={list(ds.data_vars)} coords={list(ds.coords)} dims={dict(ds.sizes)}")
    tname = _find(ds.coords, _TIME) or _find(allnames, _TIME)
    latn = _find(ds.dims, _LAT) or _find(allnames, _LAT)
    lonn = _find(ds.dims, _LON) or _find(allnames, _LON)
    assert tname and latn and lonn, f"cannot find time/lat/lon in {allnames}"
    ds = ds.sortby(latn)  # ascending latitude
    lat = ds[latn].values
    lon = ds[lonn].values
    print(f"[load_nc] time={tname} lat={latn}[{lat[0]:.2f}..{lat[-1]:.2f}] lon={lonn}[{lon[0]:.2f}..{lon[-1]:.2f}]")
    assert abs(lat[0] - LAT0) < 0.6 and abs(lat[-1] - LAT1) < 0.6, f"unexpected lat range {lat[0]}..{lat[-1]}"
    assert np.all(np.diff(lon) > 0), "lon must be ascending"

    kit_layout = "data" in ds.data_vars and "channel" in ds["data"].dims
    if kit_layout:
        # Official kit layout (monsoon_r1.build_dataset.export_nc): data(time, channel, lat, lon)
        # + static(static_channel, lat, lon) with static_channel = [land_sea_mask, orography_m].
        names = [str(c) for c in ds["channel"].values]
        assert names == list(CHANNELS), f"kit channel order {names} != contract {list(CHANNELS)}"
        states = ds["data"].transpose(tname, "channel", latn, lonn).values.astype(np.float32)
        used = {c: f"data[{c}]" for c in CHANNELS}
        print(f"[load_nc] kit layout: data{tuple(ds['data'].dims)} channels={names}")
    else:
        chans, used = [], {}
        for c in CHANNELS:
            v = _find(ds.data_vars, _VAR_ALIASES[c])
            assert v is not None, f"channel {c}: none of {_VAR_ALIASES[c]} in {list(ds.data_vars)}"
            used[c] = v
            da = ds[v]
            assert set(da.dims) == {tname, latn, lonn}, f"{v} has unexpected dims {da.dims}"
            chans.append(da.transpose(tname, latn, lonn).values.astype(np.float32))
        print(f"[load_nc] channel mapping: {used}")
        states = np.stack(chans, axis=1)
    if tp_scale != 1.0:
        states[:, RAIN] *= tp_scale
        print(f"[load_nc] applied tp_scale={tp_scale}")
    assert states.shape[1:] == (6, H, W), f"bad shape {states.shape}, expected [T,6,{H},{W}]"
    nan_frac = float(np.isnan(states).mean())
    if nan_frac > 0:
        _loud(f"{nan_frac:.2%} NaNs in states")

    # static maps: non-time variables with lat/lon dims, not among the 6 channels
    static_names, static = [], []
    if kit_layout:
        assert "static" in ds.data_vars, "kit layout without 'static' variable"
        sn = [str(s) for s in ds["static_channel"].values]
        assert sn == ["land_sea_mask", "orography_m"], f"unexpected static order {sn}"
        sa = ds["static"].transpose("static_channel", latn, lonn).values.astype(np.float32)
        static_names, static = sn, [sa[0], sa[1]]
    for n in ([] if kit_layout else ds.data_vars):
        if n in used.values():
            continue
        da = ds[n]
        if tname in da.dims or latn not in da.dims or lonn not in da.dims:
            continue
        static_names.append(str(n))
        static.append(da.transpose(latn, lonn).values.astype(np.float32))
    assert len(static) >= 2, f"need two static maps (lsm, orography); found {static_names}"
    if len(static) > 2:
        def pick(keys):
            for i, n in enumerate(static_names):
                if any(k in n.lower() for k in keys):
                    return i
            return None
        i_l, i_o = pick(["lsm", "land", "mask"]), pick(["z", "oro", "elev", "height", "geopot"])
        assert i_l is not None and i_o is not None and i_l != i_o, f"ambiguous static maps {static_names}"
        static_names = [static_names[i_l], static_names[i_o]]
        static = [static[i_l], static[i_o]]
    static = np.stack(static).astype(np.float32)
    assert static.shape == (2, H, W), static.shape
    print(f"[load_nc] static maps (order kept): {static_names}")

    dates = ds[tname].values.astype("datetime64[D]")
    assert states.shape[0] == len(dates)
    order = np.argsort(dates, kind="stable")
    states, dates = states[order], dates[order]
    d = np.diff(dates).astype(int)
    print(f"[load_nc] T={len(dates)} {dates[0]}..{dates[-1]}; gaps(>1 day)={int((d > 1).sum())}, dup={int((d == 0).sum())}")

    tp = states[:, RAIN]
    print(f"[load_nc] tp stats: mean={np.nanmean(tp):.4g} p99={np.nanpercentile(tp, 99):.4g} "
          f"max={np.nanmax(tp):.4g} min={np.nanmin(tp):.4g}")
    if np.nanmax(tp) < 1.0 or np.nanmean(tp) < 0.05:
        _loud("tp LOOKS LIKE METRES (or per-hour): values are tiny. Contract wants mm/day. NOT auto-converting; "
              "rerun with --tp-scale 1000 (m/day->mm/day) or 24000 (m/hr->mm/day) after confirming.")
    elif np.nanmean(tp) > 30:
        _loud("tp mean > 30 mm/day: units look wrong (cumulative?). Check.")
    if np.nanmin(tp) < 0:
        print("[load_nc] note: negative tp values present (clamped to 0 in model space)")
    if np.nanmean(states[:, 0]) < 100:
        _loud("t2m mean < 100: looks like Celsius, contract wants Kelvin")
    if np.nanmean(states[:, 4]) < 2000:
        _loud("msl mean < 2000: looks like hPa, contract wants Pa")
    return states, dates, static, static_names


def _smooth_noise(rng, shape, scale):
    """Spatially smooth gaussian field (FFT low-pass), unit std per leading index. shape=(..., H, W)."""
    z = rng.standard_normal(shape)
    f = np.fft.rfft2(z)
    ky = np.fft.fftfreq(shape[-2])[:, None]
    kx = np.fft.rfftfreq(shape[-1])[None, :]
    f *= np.exp(-(kx ** 2 + ky ** 2) * (np.pi * scale) ** 2 * 2)
    out = np.fft.irfft2(f, s=shape[-2:])
    return out / (out.std(axis=(-2, -1), keepdims=True) + 1e-9)


def _shift(field, dy, dx):
    """Fractional periodic shift of [..,H,W] via Fourier phase."""
    f = np.fft.rfft2(field)
    ky = np.fft.fftfreq(field.shape[-2])[:, None]
    kx = np.fft.rfftfreq(field.shape[-1])[None, :]
    f = f * np.exp(-2j * np.pi * (ky * dy + kx * dx))
    return np.fft.irfft2(f, s=field.shape[-2:])


def make_synthetic(T: int = 1360, seed: int = 0, years=(2005, 2021)):
    """Synthetic data, same return signature as load_nc.

    One block of consecutive days per year (start doy cycling through the year) so dates span
    2005..2021 but the set stays small. Smooth advected fields + seasonal cycle, skewed rain with zeros.
    """
    rng = np.random.default_rng(seed)
    nb = int(min(years[1] - years[0] + 1, max(1, T // 40)))
    yrs = [int(y) for y in np.linspace(years[0], years[1], nb).round()]
    blks = [T // nb + (1 if i < T % nb else 0) for i in range(nb)]  # exactly T days in total
    lat = np.linspace(LAT0, LAT1, H)[:, None] * np.ones((1, W))
    lsm = ((_smooth_noise(rng, (H, W), 4.0) + 0.3 * (lat - 22) / 10) > 0).astype(np.float32)
    oro = np.clip(_smooth_noise(rng, (H, W), 3.0) * 600 + 400 + 3000 * np.exp(-((lat - 32) / 5) ** 2), 0, 6000)
    static = np.stack([lsm, oro]).astype(np.float32)

    states, dates = [], []
    for bi, y in enumerate(yrs):
        blk = blks[bi]
        start = np.datetime64(f"{y}-01-01") + np.timedelta64(int((bi * 71 + 20) % max(1, 365 - blk)), "D")
        d = start + np.arange(blk)
        doy = (d - d.astype("datetime64[Y]").astype("datetime64[D]")).astype(int)
        vel = rng.uniform(-1.0, 1.0, 2) + np.array([0.0, 0.8])  # cells/day
        pat = _smooth_noise(rng, (6, H, W), 3.5)
        pat2 = _smooth_noise(rng, (6, H, W), 3.5)
        for ti in range(blk):
            mix = 0.5 - 0.5 * np.cos(2 * np.pi * ti / 40.0)
            p = _shift((1 - mix) * pat + mix * pat2, vel[0] * ti, vel[1] * ti)
            p = p / (p.std(axis=(-2, -1), keepdims=True) + 1e-9)
            s = np.sin(2 * np.pi * (doy[ti] - 110) / 365)
            monsoon = max(np.sin(2 * np.pi * (doy[ti] - 150) / 365), 0.0)
            t2m = 300.0 - 0.9 * (lat - 6.5) + 7 * s * (lat - 6.5) / 31 + 3 * s - 5 * oro / 3000 + 2.5 * p[0]
            rz = 0.9 * p[1] + 0.4 * p[3] - 0.7 + 1.5 * monsoon
            tp = 14.0 * np.clip(rz, 0, None) ** 1.6 * (0.4 + 0.6 * lsm)
            u10 = 1.0 + 4 * monsoon + 2.0 * p[2]
            v10 = 0.5 + 1.5 * p[3] + 1.0 * monsoon
            msl = 101300 - 450 * monsoon - 150 * (lat - 20) + 350 * p[4]
            tcwv = 18 + 22 * monsoon * (0.5 + 0.5 * lsm) + 6 * p[5] - 0.2 * (lat - 6.5)
            tcwv = np.clip(tcwv + 4 * p[1] * (tp > 0), 1, None)
            states.append(np.stack([t2m, tp, u10, v10, msl, tcwv]))
        dates.append(d)
    states = np.asarray(states, dtype=np.float32)
    dates = np.concatenate(dates).astype("datetime64[D]")
    return states, dates, static, ["lsm", "orography"]


def year_split(dates, train=(2005, 2018), val=(2019, 2021)):
    """Split by calendar year -> (train_idx, val_idx) integer index arrays into the day axis."""
    years = np.asarray(dates).astype("datetime64[Y]").astype(int) + 1970
    tr = np.where((years >= train[0]) & (years <= train[1]))[0]
    va = np.where((years >= val[0]) & (years <= val[1]))[0]
    return tr, va


class WindowDataset:
    """Windows of 3 input days + k target days, only over consecutive calendar days.

    Items: (window_std [3,6,H,W], static_std [2,H,W], doy_sincos [2], target_std [k,6,H,W]).
    Also exposes get_batch(idx) for fast batched access and phys_batch(idx) for validation.
    """

    def __init__(self, states, dates, static, norm: Normalizer, k: int = 1):
        self.k = k
        self.norm = norm
        self.phys = np.asarray(states, dtype=np.float32)
        self.dates = np.asarray(dates).astype("datetime64[D]")
        std = norm.standardize(to_model(self.phys))
        self.std = torch.as_tensor(np.asarray(std, dtype=np.float32))
        st = norm.standardize_static(np.asarray(static, np.float32))
        self.static_std = torch.as_tensor(np.asarray(st, dtype=np.float32))
        n = N_IN + k
        d = self.dates.astype("int64")
        T = len(d)
        starts = [i for i in range(T - n + 1) if d[i + n - 1] - d[i] == n - 1]
        self.starts = np.asarray(starts, dtype=np.int64)
        doy = (self.dates - self.dates.astype("datetime64[Y]").astype("datetime64[D]")).astype(int) + 1
        ang = 2 * np.pi * doy / 365.0
        self.doy = torch.as_tensor(np.stack([np.sin(ang), np.cos(ang)], -1).astype(np.float32))

    def __len__(self):
        return len(self.starts)

    def get_batch(self, idx):
        s = torch.as_tensor(self.starts[np.asarray(idx)])
        ar = torch.arange(N_IN + self.k)
        seq = self.std[s[:, None] + ar[None, :]]  # [B,3+k,6,H,W]
        stat = self.static_std[None].expand(len(s), -1, -1, -1)
        return seq[:, :N_IN], stat, self.doy[s + N_IN - 1], seq[:, N_IN:]

    def phys_batch(self, idx):
        """Physical [B,3+k,6,H,W] numpy for the same windows."""
        s = self.starts[np.asarray(idx)]
        return self.phys[s[:, None] + np.arange(N_IN + self.k)[None, :]]

    def __getitem__(self, i):
        win, stat, doy, tgt = self.get_batch([i])
        return win[0], stat[0], doy[0], tgt[0]
