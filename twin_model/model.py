"""Residual U-Net and the step / rollout functions in standardized model space."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import N_IN, RAIN, Normalizer

PAD_MODE = "replicate"  # regional box: never circular, never zero


def _groups(ch: int) -> int:
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    """conv3x3-GN-GELU-conv3x3-GN + skip, then GELU. Replicate padding."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, padding_mode=PAD_MODE)
        self.norm1 = nn.GroupNorm(_groups(cout), cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, padding_mode=PAD_MODE)
        self.norm2 = nn.GroupNorm(_groups(cout), cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class UNet(nn.Module):
    """Residual U-Net: [B,in_ch,H,W] -> raw normalized delta [B,out_ch,H,W].

    Spatial dims are replicate-padded up to a multiple of 2**depth and cropped back.
    The output conv is initialised near zero so the untrained model ~ persistence.
    """

    def __init__(self, in_ch: int = 22, out_ch: int = 6, base: int = 32, depth: int = 3,
                 softplus_beta: float = 0.0):
        super().__init__()
        self.in_ch, self.out_ch, self.base, self.depth = in_ch, out_ch, base, depth
        self.softplus_beta = float(softplus_beta)
        chs = [base * 2 ** i for i in range(depth + 1)]

        self.stem = ResBlock(in_ch, chs[0])
        self.downs = nn.ModuleList()
        self.enc = nn.ModuleList()
        for i in range(depth):
            self.downs.append(nn.Conv2d(chs[i], chs[i], 2, stride=2))  # learned 2x2 downsample
            self.enc.append(ResBlock(chs[i], chs[i + 1]))
        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in reversed(range(depth)):
            self.ups.append(nn.Conv2d(chs[i + 1], chs[i], 1))
            self.dec.append(ResBlock(2 * chs[i], chs[i]))
        self.head = nn.Conv2d(chs[0], out_ch, 1)
        nn.init.zeros_(self.head.weight)  # exact persistence at init; grads still reach the head
        nn.init.zeros_(self.head.bias)

    def config(self) -> dict:
        return {"in_ch": self.in_ch, "out_ch": self.out_ch, "base": self.base,
                "depth": self.depth, "softplus_beta": self.softplus_beta}

    def forward(self, x):
        Hh, Ww = x.shape[-2:]
        m = 2 ** self.depth
        ph, pw = (-Hh) % m, (-Ww) % m
        if ph or pw:
            x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), mode=PAD_MODE)
        h = self.stem(x)
        skips = []
        for down, enc in zip(self.downs, self.enc):
            skips.append(h)
            h = enc(down(h))
        for up, dec in zip(self.ups, self.dec):
            s = skips.pop()
            h = F.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False)
            h = dec(torch.cat([up(h), s], dim=1))
        out = self.head(h)
        if ph or pw:
            out = out[..., ph // 2: ph // 2 + Hh, pw // 2: pw // 2 + Ww]
        return out


def build_input(window_std, static_std, doy_sincos):
    """[B,3,6,H,W], [B,2,H,W], [B,2] -> [B,22,H,W] (oldest day first, static, sin, cos)."""
    B, T, C, Hh, Ww = window_std.shape
    doy = doy_sincos.to(window_std.dtype).view(B, 2, 1, 1).expand(B, 2, Hh, Ww)
    return torch.cat([window_std.reshape(B, T * C, Hh, Ww), static_std.to(window_std.dtype), doy], dim=1)


def step(model: UNet, window_std, static_std, doy_sincos, norm: Normalizer, softplus_beta=None):
    """One day ahead in standardized model space: next = last + raw * delta_std.

    ``doy_sincos`` is the day-of-year encoding of the LAST day in ``window_std``.
    The rain channel is floored at 0 in log1p space. Default (beta <= 0): straight-through
    ReLU — forward is exactly max(x, 0) so dry cells stay dry with no per-step drift,
    backward is identity so wrongly-dry cells still get gradient. beta > 0 selects a
    softplus instead (biases dry cells upward by ln2/beta per step; kept for experiments).
    """
    beta = float(softplus_beta if softplus_beta is not None else getattr(model, "softplus_beta", 0.0))
    st = norm.torch_stats(window_std.device, window_std.dtype)
    raw = model(build_input(window_std, static_std, doy_sincos))
    nxt = window_std[:, -1] + raw * st["delta_std"].view(1, -1, 1, 1)
    m, s = st["state_mean"][RAIN], st["state_std"][RAIN]
    x = nxt[:, RAIN] * s + m
    if beta > 0:
        rain_log = F.softplus(x, beta=beta)
    else:
        rain_log = x + (torch.relu(x) - x).detach()
    rain_std = (rain_log - m) / s
    return torch.cat([nxt[:, :RAIN], rain_std.unsqueeze(1), nxt[:, RAIN + 1:]], dim=1)


def advance_doy(doy_sincos, days: int = 1):
    """Rotate the (sin, cos) day-of-year angle by 2*pi*days/365. [B,2] -> [B,2]."""
    ang = torch.atan2(doy_sincos[..., 0], doy_sincos[..., 1]) + 2.0 * math.pi * days / 365.0
    return torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)


def rollout_std(model: UNet, window_std, static_std, doy_sincos, steps: int, norm: Normalizer):
    """Autoregressive rollout -> [B,steps,6,H,W]. Differentiable (no no_grad inside)."""
    if window_std.shape[1] != N_IN:
        raise ValueError(f"window_std must have {N_IN} days, got {window_std.shape[1]}")
    window, doy, outs = window_std, doy_sincos, []
    for _ in range(steps):
        nxt = step(model, window, static_std, doy, norm)
        outs.append(nxt)
        window = torch.cat([window[:, 1:], nxt.unsqueeze(1)], dim=1)
        doy = advance_doy(doy)
    if not outs:
        B, _, C, Hh, Ww = window_std.shape
        return window_std.new_zeros((B, 0, C, Hh, Ww))
    return torch.stack(outs, dim=1)
