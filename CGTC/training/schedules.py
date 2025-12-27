# -*- coding: utf-8 -*-
from __future__ import annotations
import math

def base_lambda_cont(
    epoch: int, use_pinn: bool,
    warmup: int, ramp: int,
    lmin: float, lmax: float,
    schedule: str="cosine", mode: str="down",
):
    if not use_pinn: return 0.0
    if epoch <= warmup: return 0.0
    t = epoch - warmup
    r = max(0.0, min(1.0, t / max(1, ramp)))
    if schedule == "linear": s = r
    elif schedule == "cosine": s = 0.5 - 0.5 * math.cos(math.pi * r)
    elif schedule == "exp": s = (math.exp(5*r) - 1.0) / (math.exp(5) - 1.0)
    else: s = r
    if mode.lower() == "down":
        base = lmin + (lmax - lmin) * (1.0 - s)
        if t >= ramp: base = lmin
    else:
        base = lmin + (lmax - lmin) * s
        if t >= ramp: base = lmax
    return float(base)

def current_alpha(
    epoch: int, use_mixed: bool, start: float, end: float,
    decay_start: int, decay_epochs: int, schedule: str="cosine"
):
    if not use_mixed: return 1.0
    if epoch < decay_start: return float(start)
    t = epoch - decay_start
    if t >= max(1, decay_epochs): return float(end)
    r = t / max(1, decay_epochs)
    if schedule == "linear": w = 1.0 - r
    elif schedule == "cosine": w = 0.5 * (1.0 + math.cos(math.pi * r))
    elif schedule == "exp": w = math.exp(-5*r)
    else: w = 1.0 - r
    return float(end + (start - end) * w)

def teacher_consis_weight(epoch: int, use_teacher: bool, warmup: int, max_w: float, decay_eps: int):
    if not use_teacher: return 0.0
    if epoch <= warmup: return 0.0
    t = min(max(epoch - warmup, 0), decay_eps)
    r = t / max(1, decay_eps)
    return float(0.5 * (1 + math.cos(math.pi * r)) * max_w)
