# -*- coding: utf-8 -*-
# TPAC/eval/metrics.py
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn

@torch.no_grad()
def compute_region_metrics(pred: torch.Tensor, labels: torch.Tensor, d: torch.Tensor,
                           q_wall: float = 0.3, q_interior: float = 0.70):
    B, C, N = pred.shape
    thr_wall     = torch.quantile(d, q_wall, dim=2, keepdim=True)
    thr_interior = torch.quantile(d, q_interior, dim=2, keepdim=True)
    mask_wall = (d <= thr_wall)
    mask_int  = (d >= thr_interior)
    e = pred - labels
    denom_w = mask_wall.sum().clamp_min(1)
    mae_w = (e.abs() * mask_wall).sum() / denom_w
    mse_w = ((e**2) * mask_wall).sum() / denom_w
    denom_i = mask_int.sum().clamp_min(1)
    mae_i = (e.abs() * mask_int).sum() / denom_i
    mse_i = ((e**2) * mask_int).sum() / denom_i
    return {
        "mae_wall": mae_w.mean().item(),
        "mse_wall": mse_w.mean().item(),
        "mae_interior": mae_i.mean().item(),
        "mse_interior": mse_i.mean().item(),
    }
