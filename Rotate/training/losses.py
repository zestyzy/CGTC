# -*- coding: utf-8 -*-
# TPAC/training/losses.py
from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.nn as nn

def make_scaler(enabled: bool, device: torch.device):
    return torch.cuda.amp.GradScaler(enabled=enabled and (device.type == "cuda"))

def make_point_weights(
    xyz: torch.Tensor, mode: str = "interior_focus",
    sigma: float = 0.15, gamma: float = 2.0,
    eps: float = 1e-8, alpha: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    xyz: (B,3,N) in [0,1]
    return:
      w: (B,1,N) with mean=1 per-batch
      d: (B,1,N) distance proxy to wall
    """
    B, C, N = xyz.shape
    assert C == 3
    x, y, z = xyz[:,0:1,:], xyz[:,1:2,:], xyz[:,2:3,:]
    d = torch.minimum(torch.minimum(x, 1 - x),
        torch.minimum(torch.minimum(y, 1 - y), torch.minimum(z, 1 - z)))

    if mode == "wall_focus" or (mode == "mixed" and (alpha is None or alpha >= 0.999)):
        w = 1.0 + torch.exp(-d / max(sigma, eps))
    elif mode == "interior_focus":
        dmax = torch.amax(d, dim=2, keepdim=True).clamp_min(eps)
        w = 1.0 + (d / dmax) ** gamma
    elif mode == "uniform":
        w = torch.ones_like(d)
    elif mode == "mixed":
        w_wall = 1.0 + torch.exp(-d / max(sigma, eps))
        a = float(alpha if alpha is not None else 1.0)
        a = max(0.0, min(1.0, a))
        w = a * w_wall + (1.0 - a) * 1.0
    else:
        w = torch.ones_like(d)

    w = w / (w.mean(dim=2, keepdim=True).clamp_min(eps))
    return w, d

def compute_weighted_losses(
    pred: torch.Tensor, labels: torch.Tensor, xyz: torch.Tensor,
    focus_mode: str, channel_weights, alpha: Optional[float]=None,
    wall_sigma: float=0.08
):
    if isinstance(channel_weights, (list, tuple)):
        w_ch = torch.tensor(channel_weights, device=pred.device, dtype=pred.dtype)
    else:
        w_ch = channel_weights.to(device=pred.device, dtype=pred.dtype)
    assert w_ch.numel() == pred.shape[1], "channel_weights size mismatch"

    w_pts, d = make_point_weights(
        xyz, mode=focus_mode, sigma=wall_sigma, alpha=alpha
    )
    w_ch = w_ch.view(1, pred.shape[1], 1)
    e = pred - labels
    denom = (w_pts * w_ch).sum().clamp_min(1e-8)
    mse = ((e**2) * w_pts * w_ch).sum() / denom
    mae = (e.abs()  * w_pts * w_ch).sum() / denom
    return mse, mae, d

def losses_per_channel(pred: torch.Tensor, target: torch.Tensor, w_pts: torch.Tensor):
    e = pred - target
    denom = (w_pts.expand_as(e)).sum(dim=(0,2)).clamp_min(1e-8)
    mae_c = (e.abs() * w_pts).sum(dim=(0,2)) / denom
    mse_c = ((e**2) * w_pts).sum(dim=(0,2)) / denom
    return mae_c, mse_c
