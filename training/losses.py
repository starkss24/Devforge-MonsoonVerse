"""Rollout losses in standardized model space."""
from __future__ import annotations

import torch

from twin_model.norm import RAIN


def _grads(x):
    return x[..., :, 1:] - x[..., :, :-1], x[..., 1:, :] - x[..., :-1, :]


def rollout_loss(pred_std, target_std, rain_weight: float = 2.0, grad_weight: float = 0.1,
                 grad_mode: str = "match"):
    """pred/target [B,k,6,H,W] -> scalar.

    MSE averaged over steps (and batch, space) with channel weights (rain extra), plus an optional
    spatial-gradient term against blur:
      - "match": MSE between forecast and truth gradients. Still rewards hedging under positional
        uncertainty (a smeared gradient minimises it), so it is a weak anti-blur term.
      - "mag": per-sample, per-step, per-channel mean gradient MAGNITUDE of forecast vs truth
        (squared difference). Penalises fields that are too smooth without penalising a sharp
        feature placed slightly wrong.
    """
    if grad_mode == "mag" and grad_weight > 0:
        w = torch.ones(pred_std.shape[-3], device=pred_std.device, dtype=pred_std.dtype)
        w[RAIN] = rain_weight
        w = (w / w.mean()).view(1, 1, -1)
        loss = (w.view(1, 1, -1, 1, 1) * (pred_std - target_std) ** 2).mean()
        (pgx, pgy), (tgx, tgy) = _grads(pred_std), _grads(target_std)
        pm = pgx.abs().mean(dim=(-2, -1)) + pgy.abs().mean(dim=(-2, -1))   # [B,k,6]
        tm = tgx.abs().mean(dim=(-2, -1)) + tgy.abs().mean(dim=(-2, -1))
        return loss + grad_weight * (w * (pm - tm) ** 2).mean()
    w = torch.ones(pred_std.shape[-3], device=pred_std.device, dtype=pred_std.dtype)
    w[RAIN] = rain_weight
    w = (w / w.mean()).view(1, 1, -1, 1, 1)
    loss = (w * (pred_std - target_std) ** 2).mean()
    if grad_weight > 0:
        (pgx, pgy), (tgx, tgy) = _grads(pred_std), _grads(target_std)
        gl = (w * (pgx - tgx) ** 2).mean() + (w * (pgy - tgy) ** 2).mean()
        loss = loss + grad_weight * gl
    return loss
