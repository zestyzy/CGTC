#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import OrderedDict
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import yaml
"""
在能运行的基础上继续上次的跑
    python Rotate/wo.py \
  --root /mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task2_Rotate/Rotate \
  --case ICA_norm \
  --tag 1205 \
  --only-pt \
  --resume \
  --device cuda:0 --pts 4096 --batch 2 --workers 4

    python Rotate/wo.py \
  --root /mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task2_Rotate/Rotate \
  --case ICA_norm \
  --tag 1205_all \
  --device cuda:0 --pts 4096 --batch 2 --workers 4


"""
# =============================================================================
# 实验定义（Baseline → CGTC-Core → Core/Robust → CGTC-Enhance → Enhance/Robust）
# =============================================================================

EXPERIMENTS: "OrderedDict[str, Dict[str, object]]" = OrderedDict([
    # -------------------------------------------------------------------------
    # 0) Anchors & Baselines (PointNet++ default)
    # -------------------------------------------------------------------------
    (
        "supervised_only",
        {
            "description": "Pure supervision: no teacher / spatial / PINN / mixed / adapt (reference anchor)",
            "overrides": {
                "teacher.use": False,
                "teacher.max_ratio": 0.0,
                "teacher.spatial.use": False,
                "teacher.spatial.weight": 0.0,
                "teacher.spatial.max_ratio": 0.0,
                "pinn.use": False,
                "pinn.max_loss_ratio": 0.0,
                "pinn.max_ratio_vs_spatial": 0.0,
                "pinn.lambda_mom": 0.0,
                "mixed.use": False,
                "adapt.enable": False,
                "pinn.adapt": False,
                "pinn.auto_calib": False,
                "guard.disable": True,
                "pinn.cooling_window": 0,
            },
            "flags": {"family": "baseline", "type": "sup_only"},
        },
    ),
    (
        "baseline_equal",
        {
            "description": (
                "Baseline-Equal: L_sup + L_phys + spatial, "
                "no guard, no mixed, no adapt (fixed weights)"
            ),
            "overrides": {
                "mixed.use": False,
                "adapt.enable": False,
                "pinn.adapt": False,
                "pinn.auto_calib": False,

                "guard.disable": True,
                "pinn.max_loss_ratio": 0.0,
                "pinn.max_ratio_vs_spatial": 0.0,
                "pinn.guard_gamma": 1.0,
                "pinn.cooling_window": 0,
                "teacher.max_ratio": 0.0,
                "teacher.spatial.max_ratio": 0.0,
                "teacher.spatial.gamma": 1.0,
                "teacher.spatial.cooling_window": 0,
            },
            "flags": {"family": "baseline", "type": "equal"},
        },
    ),
    (
        "baseline_adapt",
        {
            "description": (
                "Baseline-Adaptive: L_sup + L_phys + spatial, "
                "no guard, but enable simple adaptive weights (adapt + λ_cont adapt)"
            ),
            "overrides": {
                "mixed.use": False,

                "adapt.enable": True,
                "pinn.adapt": True,
                "pinn.auto_calib": False,

                "guard.disable": True,
                "pinn.max_loss_ratio": 0.0,
                "pinn.max_ratio_vs_spatial": 0.0,
                "pinn.guard_gamma": 1.0,
                "pinn.cooling_window": 0,
                "teacher.max_ratio": 0.0,
                "teacher.spatial.max_ratio": 0.0,
                "teacher.spatial.gamma": 1.0,
                "teacher.spatial.cooling_window": 0,
            },
            "flags": {"family": "baseline", "type": "adaptive"},
        },
    ),

    # -------------------- 新增：PCGrad / CAGrad baseline（PointNet） --------------------
    (
        "baseline_pcgrad",
        {
            "description": (
                "Baseline-PCGrad: L_sup + L_phys + spatial with PCGrad "
                "(no guard, no mixed, no adapt)"
            ),
            "overrides": {
                "mixed.use": False,
                "adapt.enable": False,
                "pinn.adapt": False,
                "pinn.auto_calib": False,

                "guard.disable": True,
                "pinn.max_loss_ratio": 0.0,
                "pinn.max_ratio_vs_spatial": 0.0,
                "pinn.guard_gamma": 1.0,
                "pinn.cooling_window": 0,
                "teacher.max_ratio": 0.0,
                "teacher.spatial.max_ratio": 0.0,
                "teacher.spatial.gamma": 1.0,
                "teacher.spatial.cooling_window": 0,

                "mo.solver": "pcgrad",
            },
            "flags": {"family": "baseline", "type": "pcgrad"},
        },
    ),
    (
        "baseline_cagrad",
        {
            "description": (
                "Baseline-CAGrad: L_sup + L_phys + spatial with CAGrad "
                "(no guard, no mixed, no adapt)"
            ),
            "overrides": {
                "mixed.use": False,
                "adapt.enable": False,
                "pinn.adapt": False,
                "pinn.auto_calib": False,

                "guard.disable": True,
                "pinn.max_loss_ratio": 0.0,
                "pinn.max_ratio_vs_spatial": 0.0,
                "pinn.guard_gamma": 1.0,
                "pinn.cooling_window": 0,
                "teacher.max_ratio": 0.0,
                "teacher.spatial.max_ratio": 0.0,
                "teacher.spatial.gamma": 1.0,
                "teacher.spatial.cooling_window": 0,

                "mo.solver": "cagrad",
            },
            "flags": {"family": "baseline", "type": "cagrad"},
        },
    ),

    # -------------------------------------------------------------------------
    # 1) CGTC-Core on clean data
    # -------------------------------------------------------------------------
    (
        "core_full",
        {
            "description": "CGTC-Core full (ID): guard + cooling + EMA teacher + spatial + PINN, no mixed/adapt",
            "overrides": {
                "guard.disable": False,
                "mixed.use": False,
                "adapt.enable": False,
                "pinn.adapt": False,
                "pinn.auto_calib": False,
                "teacher.use": True,
                "teacher.spatial.use": True,
                "pinn.use": True,
            },
            "flags": {"family": "cgtc_core", "variant": "full"},
        },
    ),
    (
        "core_no_pinn",
        {
            "description": "CGTC-Core abl. — no PINN physics (teacher + spatial only)",
            "overrides": {
                "guard.disable": False,
                "mixed.use": False,
                "adapt.enable": False,
                "pinn.use": False,
                "pinn.max_loss_ratio": 0.0,
                "pinn.max_ratio_vs_spatial": 0.0,
                "pinn.lambda_mom": 0.0,
                "pinn.adapt": False,
                "pinn.auto_calib": False,
            },
            "flags": {"family": "cgtc_core", "variant": "no_pinn"},
        },
    ),
    (
        "core_no_spatial",
        {
            "description": "CGTC-Core abl. — disable spatial consistency regularisation",
            "overrides": {
                "guard.disable": False,
                "mixed.use": False,
                "adapt.enable": False,
                "teacher.spatial.use": False,
                "teacher.spatial.weight": 0.0,
                "teacher.spatial.max_ratio": 0.0,
            },
            "flags": {"family": "cgtc_core", "variant": "no_spatial"},
        },
    ),
    (
        "core_no_teacher",
        {
            "description": "CGTC-Core abl. — no EMA teacher, no spatial consistency",
            "overrides": {
                "guard.disable": False,
                "mixed.use": False,
                "adapt.enable": False,
                "teacher.use": False,
                "teacher.max_ratio": 0.0,
                "teacher.spatial.use": False,
                "teacher.spatial.weight": 0.0,
                "teacher.spatial.max_ratio": 0.0,
            },
            "flags": {"family": "cgtc_core", "variant": "no_teacher"},
        },
    ),
    (
        "core_no_guard",
        {
            "description": "CGTC-Core abl. — disable ratio guard & cooling (unconstrained multi-loss, still PINN+teacher)",
            "overrides": {
                "mixed.use": False,
                "adapt.enable": False,
                "guard.disable": True,
                "pinn.max_loss_ratio": 0.0,
                "pinn.max_ratio_vs_spatial": 0.0,
                "pinn.guard_gamma": 1.0,
                "pinn.cooling_window": 0,
                "teacher.max_ratio": 0.0,
                "teacher.spatial.max_ratio": 0.0,
                "teacher.spatial.gamma": 1.0,
                "teacher.spatial.cooling_window": 0,
            },
            "flags": {"family": "cgtc_core", "variant": "no_guard"},
        },
    ),

    # -------------------------------------------------------------------------
    # 2) CGTC-Core / Robustness
    # -------------------------------------------------------------------------
    (
        "core_noise_med",
        {
            "description": "CGTC-Core full under medium field noise",
            "overrides": {
                "guard.disable": False,
                "mixed.use": False,
                "adapt.enable": False,
                "robust.field_noise_sigma": 0.5,
                "robust.apply_to_val": True,
                "robust.apply_to_test": True,
            },
            "flags": {"family": "cgtc_core_robust", "distortion": "noise_med", "role": "core"},
        },
    ),
    (
        "base_equal_noise_med",
        {
            "description": "Baseline-Equal under medium field noise",
            "overrides": {
                "robust.field_noise_sigma": 0.5,
                "robust.apply_to_val": True,
                "robust.apply_to_test": True,

                "mixed.use": False,
                "adapt.enable": False,
                "pinn.adapt": False,
                "pinn.auto_calib": False,
                "guard.disable": True,
                "pinn.max_loss_ratio": 0.0,
                "pinn.max_ratio_vs_spatial": 0.0,
                "pinn.guard_gamma": 1.0,
                "pinn.cooling_window": 0,
                "teacher.max_ratio": 0.0,
                "teacher.spatial.max_ratio": 0.0,
                "teacher.spatial.gamma": 1.0,
                "teacher.spatial.cooling_window": 0,
            },
            "flags": {"family": "baseline_robust", "base": "equal", "distortion": "noise_med"},
        },
    ),
    (
        "base_adapt_noise_med",
        {
            "description": "Baseline-Adaptive under medium field noise",
            "overrides": {
                "robust.field_noise_sigma": 0.5,
                "robust.apply_to_val": True,
                "robust.apply_to_test": True,

                "mixed.use": False,
                "adapt.enable": True,
                "pinn.adapt": True,
                "pinn.auto_calib": False,
                "guard.disable": True,
                "pinn.max_loss_ratio": 0.0,
                "pinn.max_ratio_vs_spatial": 0.0,
                "pinn.guard_gamma": 1.0,
                "pinn.cooling_window": 0,
                "teacher.max_ratio": 0.0,
                "teacher.spatial.max_ratio": 0.0,
                "teacher.spatial.gamma": 1.0,
                "teacher.spatial.cooling_window": 0,
            },
            "flags": {"family": "baseline_robust", "base": "adaptive", "distortion": "noise_med"},
        },
    ),
    (
        "core_noise_high",
        {
            "description": "CGTC-Core full under high field noise",
            "overrides": {
                "guard.disable": False,
                "mixed.use": False,
                "adapt.enable": False,
                "robust.field_noise_sigma": 1.0,
                "robust.apply_to_val": True,
                "robust.apply_to_test": True,
            },
            "flags": {"family": "cgtc_core_robust", "distortion": "noise_high", "role": "core"},
        },
    ),
    (
        "base_equal_noise_high",
        {
            "description": "Baseline-Equal under high field noise",
            "overrides": {
                "robust.field_noise_sigma": 1.0,
                "robust.apply_to_val": True,
                "robust.apply_to_test": True,

                "mixed.use": False,
                "adapt.enable": False,
                "pinn.adapt": False,
                "pinn.auto_calib": False,
                "guard.disable": True,
                "pinn.max_loss_ratio": 0.0,
                "pinn.max_ratio_vs_spatial": 0.0,
                "pinn.guard_gamma": 1.0,
                "pinn.cooling_window": 0,
                "teacher.max_ratio": 0.0,
                "teacher.spatial.max_ratio": 0.0,
                "teacher.spatial.gamma": 1.0,
                "teacher.spatial.cooling_window": 0,
            },
            "flags": {"family": "baseline_robust", "base": "equal", "distortion": "noise_high"},
        },
    ),
    (
        "base_adapt_noise_high",
        {
            "description": "Baseline-Adaptive under high field noise",
            "overrides": {
                "robust.field_noise_sigma": 1.0,
                "robust.apply_to_val": True,
                "robust.apply_to_test": True,

                "mixed.use": False,
                "adapt.enable": True,
                "pinn.adapt": True,
                "pinn.auto_calib": False,
                "guard.disable": True,
                "pinn.max_loss_ratio": 0.0,
                "pinn.max_ratio_vs_spatial": 0.0,
                "pinn.guard_gamma": 1.0,
                "pinn.cooling_window": 0,
                "teacher.max_ratio": 0.0,
                "teacher.spatial.max_ratio": 0.0,
                "teacher.spatial.gamma": 1.0,
                "teacher.spatial.cooling_window": 0,
            },
            "flags": {"family": "baseline_robust", "base": "adaptive", "distortion": "noise_high"},
        },
    ),

    # --- 稀疏采样扰动 ---
    (
        "core_sparse_0.5",
        {
            "description": "CGTC-Core full under 50% subsampling",
            "overrides": {
                "guard.disable": False,
                "mixed.use": False,
                "adapt.enable": False,
                "robust.subsample_ratio": 0.5,
                "robust.apply_to_val": True,
                "robust.apply_to_test": True,
            },
            "flags": {"family": "cgtc_core_robust", "distortion": "sparse_0.5", "role": "core"},
        },
    ),
    (
        "base_equal_sparse_0.5",
        {
            "description": "Baseline-Equal under 50% subsampling",
            "overrides": {
                "robust.subsample_ratio": 0.5,
                "robust.apply_to_val": True,
                "robust.apply_to_test": True,

                "mixed.use": False,
                "adapt.enable": False,
                "pinn.adapt": False,
                "pinn.auto_calib": False,
                "guard.disable": True,
                "pinn.max_loss_ratio": 0.0,
                "pinn.max_ratio_vs_spatial": 0.0,
                "pinn.guard_gamma": 1.0,
                "pinn.cooling_window": 0,
                "teacher.max_ratio": 0.0,
                "teacher.spatial.max_ratio": 0.0,
                "teacher.spatial.gamma": 1.0,
                "teacher.spatial.cooling_window": 0,
            },
            "flags": {"family": "baseline_robust", "base": "equal", "distortion": "sparse_0.5"},
        },
    ),
    (
        "base_adapt_sparse_0.5",
        {
            "description": "Baseline-Adaptive under 50% subsampling",
            "overrides": {
                "robust.subsample_ratio": 0.5,
                "robust.apply_to_val": True,
                "robust.apply_to_test": True,

                "mixed.use": False,
                "adapt.enable": True,
                "pinn.adapt": True,
                "pinn.auto_calib": False,
                "guard.disable": True,
                "pinn.max_loss_ratio": 0.0,
                "pinn.max_ratio_vs_spatial": 0.0,
                "pinn.guard_gamma": 1.0,
                "pinn.cooling_window": 0,
                "teacher.max_ratio": 0.0,
                "teacher.spatial.max_ratio": 0.0,
                "teacher.spatial.gamma": 1.0,
                "teacher.spatial.cooling_window": 0,
            },
            "flags": {"family": "baseline_robust", "base": "adaptive", "distortion": "sparse_0.5"},
        },
    ),
    (
        "core_sparse_0.25",
        {
            "description": "CGTC-Core full under 25% subsampling",
            "overrides": {
                "guard.disable": False,
                "mixed.use": False,
                "adapt.enable": False,
                "robust.subsample_ratio": 0.25,
                "robust.apply_to_val": True,
                "robust.apply_to_test": True,
            },
            "flags": {"family": "cgtc_core_robust", "distortion": "sparse_0.25", "role": "core"},
        },
    ),
    (
        "base_equal_sparse_0.25",
        {
            "description": "Baseline-Equal under 25% subsampling",
            "overrides": {
                "robust.subsample_ratio": 0.25,
                "robust.apply_to_val": True,
                "robust.apply_to_test": True,

                "mixed.use": False,
                "adapt.enable": False,
                "pinn.adapt": False,
                "pinn.auto_calib": False,
                "guard.disable": True,
                "pinn.max_loss_ratio": 0.0,
                "pinn.max_ratio_vs_spatial": 0.0,
                "pinn.guard_gamma": 1.0,
                "pinn.cooling_window": 0,
                "teacher.max_ratio": 0.0,
                "teacher.spatial.max_ratio": 0.0,
                "teacher.spatial.gamma": 1.0,
                "teacher.spatial.cooling_window": 0,
            },
            "flags": {"family": "baseline_robust", "base": "equal", "distortion": "sparse_0.25"},
        },
    ),
    (
        "base_adapt_sparse_0.25",
        {
            "description": "Baseline-Adaptive under 25% subsampling",
            "overrides": {
                "robust.subsample_ratio": 0.25,
                "robust.apply_to_val": True,
                "robust.apply_to_test": True,

                "mixed.use": False,
                "adapt.enable": True,
                "pinn.adapt": True,
                "pinn.auto_calib": False,
                "guard.disable": True,
                "pinn.max_loss_ratio": 0.0,
                "pinn.max_ratio_vs_spatial": 0.0,
                "pinn.guard_gamma": 1.0,
                "pinn.cooling_window": 0,
                "teacher.max_ratio": 0.0,
                "teacher.spatial.max_ratio": 0.0,
                "teacher.spatial.gamma": 1.0,
                "teacher.spatial.cooling_window": 0,
            },
            "flags": {"family": "baseline_robust", "base": "adaptive", "distortion": "sparse_0.25"},
        },
    ),

    # --- 旋转 OOD 扰动 ---
    (
        "core_rot_45",
        {
            "description": "CGTC-Core full under ±45° random OOD rotation",
            "overrides": {
                "guard.disable": False,
                "mixed.use": False,
                "adapt.enable": False,
                "robust.ood_rotation.enabled": True,
                "robust.ood_rotation.max_deg": 45.0,
                "robust.apply_to_val": True,
                "robust.apply_to_test": True,
            },
            "flags": {"family": "cgtc_core_robust", "distortion": "rot_45", "role": "core"},
        },
    ),
    (
        "base_equal_rot_45",
        {
            "description": "Baseline-Equal under ±45° random OOD rotation",
            "overrides": {
                "robust.ood_rotation.enabled": True,
                "robust.ood_rotation.max_deg": 45.0,
                "robust.apply_to_val": True,
                "robust.apply_to_test": True,

                "mixed.use": False,
                "adapt.enable": False,
                "pinn.adapt": False,
                "pinn.auto_calib": False,
                "guard.disable": True,
                "pinn.max_loss_ratio": 0.0,
                "pinn.max_ratio_vs_spatial": 0.0,
                "pinn.guard_gamma": 1.0,
                "pinn.cooling_window": 0,
                "teacher.max_ratio": 0.0,
                "teacher.spatial.max_ratio": 0.0,
                "teacher.spatial.gamma": 1.0,
                "teacher.spatial.cooling_window": 0,
            },
            "flags": {"family": "baseline_robust", "base": "equal", "distortion": "rot_45"},
        },
    ),
    (
        "base_adapt_rot_45",
        {
            "description": "Baseline-Adaptive under ±45° random OOD rotation",
            "overrides": {
                "robust.ood_rotation.enabled": True,
                "robust.ood_rotation.max_deg": 45.0,
                "robust.apply_to_val": True,
                "robust.apply_to_test": True,

                "mixed.use": False,
                "adapt.enable": True,
                "pinn.adapt": True,
                "pinn.auto_calib": False,
                "guard.disable": True,
                "pinn.max_loss_ratio": 0.0,
                "pinn.max_ratio_vs_spatial": 0.0,
                "pinn.guard_gamma": 1.0,
                "pinn.cooling_window": 0,
                "teacher.max_ratio": 0.0,
                "teacher.spatial.max_ratio": 0.0,
                "teacher.spatial.gamma": 1.0,
                "teacher.spatial.cooling_window": 0,
            },
            "flags": {"family": "baseline_robust", "base": "adaptive", "distortion": "rot_45"},
        },
    ),

    # -------------------------------------------------------------------------
    # 3) CGTC-Enhance & Enhance/Robustness
    # -------------------------------------------------------------------------
    (
        "enh_full",
        {
            "description": "CGTC-Enhance full (ID): CGTC-Core + boundary-aware mixed focus + adaptive weights",
            "overrides": {
                "guard.disable": False,
                "mixed.use": True,
                "adapt.enable": True,
                "pinn.adapt": True,
                "pinn.auto_calib": True,
                "teacher.use": True,
                "teacher.spatial.use": True,
                "pinn.use": True,
            },
            "flags": {"family": "cgtc_enh", "variant": "full"},
        },
    ),
    (
        "enh_no_mixed",
        {
            "description": "Enhance abl. — disable boundary-focused mixed curriculum (should be close to Core)",
            "overrides": {
                "guard.disable": False,
                "mixed.use": False,
                "adapt.enable": True,
                "pinn.adapt": True,
                "pinn.auto_calib": True,
            },
            "flags": {"family": "cgtc_enh", "variant": "no_mixed"},
        },
    ),

    # --- Enhance / 噪声鲁棒性 ---
    (
        "enh_noise_med",
        {
            "description": "CGTC-Enhance full under medium field noise",
            "overrides": {
                "guard.disable": False,
                "mixed.use": True,
                "adapt.enable": True,
                "pinn.adapt": True,
                "pinn.auto_calib": True,
                "robust.field_noise_sigma": 0.5,
                "robust.apply_to_val": True,
                "robust.apply_to_test": True,
            },
            "flags": {"family": "cgtc_enh_robust", "distortion": "noise_med", "role": "enhance"},
        },
    ),
    (
        "enh_noise_high",
        {
            "description": "CGTC-Enhance full under high field noise",
            "overrides": {
                "guard.disable": False,
                "mixed.use": True,
                "adapt.enable": True,
                "pinn.adapt": True,
                "pinn.auto_calib": True,
                "robust.field_noise_sigma": 1.0,
                "robust.apply_to_val": True,
                "robust.apply_to_test": True,
            },
            "flags": {"family": "cgtc_enh_robust", "distortion": "noise_high", "role": "enhance"},
        },
    ),

    # --- Enhance / 稀疏采样鲁棒性 ---
    (
        "enh_sparse_0.5",
        {
            "description": "CGTC-Enhance full under 50% subsampling",
            "overrides": {
                "guard.disable": False,
                "mixed.use": True,
                "adapt.enable": True,
                "pinn.adapt": True,
                "pinn.auto_calib": True,
                "robust.subsample_ratio": 0.5,
                "robust.apply_to_val": True,
                "robust.apply_to_test": True,
            },
            "flags": {"family": "cgtc_enh_robust", "distortion": "sparse_0.5", "role": "enhance"},
        },
    ),
    (
        "enh_sparse_0.25",
        {
            "description": "CGTC-Enhance full under 25% subsampling",
            "overrides": {
                "guard.disable": False,
                "mixed.use": True,
                "adapt.enable": True,
                "pinn.adapt": True,
                "pinn.auto_calib": True,
                "robust.subsample_ratio": 0.25,
                "robust.apply_to_val": True,
                "robust.apply_to_test": True,
            },
            "flags": {"family": "cgtc_enh_robust", "distortion": "sparse_0.25", "role": "enhance"},
        },
    ),

    # --- Enhance / 旋转 OOD 鲁棒性 ---
    (
        "enh_rot_45",
        {
            "description": "CGTC-Enhance full under ±45° random OOD rotation",
            "overrides": {
                "guard.disable": False,
                "mixed.use": True,
                "adapt.enable": True,
                "pinn.adapt": True,
                "pinn.auto_calib": True,
                "robust.ood_rotation.enabled": True,
                "robust.ood_rotation.max_deg": 45.0,
                "robust.apply_to_val": True,
                "robust.apply_to_test": True,
            },
            "flags": {"family": "cgtc_enh_robust", "distortion": "rot_45", "role": "enhance"},
        },
    ),
])

# =============================================================================
# 在 wo 层保证：除 PCGrad / CAGrad 消融外，其他实验默认使用 mo.solver = "sum"
# =============================================================================

def ensure_default_mo_sum(experiments: "OrderedDict[str, Dict[str, object]]") -> None:
    for _name, exp in experiments.items():
        overrides = exp.get("overrides", {})
        if not isinstance(overrides, dict):
            overrides = {}
        if "mo.solver" not in overrides:
            overrides = dict(overrides)
            overrides["mo.solver"] = "sum"
            exp["overrides"] = overrides

ensure_default_mo_sum(EXPERIMENTS)

# =============================================================================
# PointTransformer 全量镜像：做完所有 PointNet 消融后，再完整做一遍
# =============================================================================

def _clone_for_pointtransformer(
    base_items: List[Tuple[str, Dict[str, object]]]
) -> "OrderedDict[str, Dict[str, object]]":
    pt_exps: "OrderedDict[str, Dict[str, object]]" = OrderedDict()
    for base_name, base_exp in base_items:
        pt_name = f"pt_{base_name}"
        pt_exp = deepcopy(base_exp)

        desc = str(pt_exp.get("description", ""))
        pt_exp["description"] = f"[PointTransformer] {desc}" if desc else "[PointTransformer]"

        overrides = dict(pt_exp.get("overrides", {}))

        # ✅ 切换 backbone
        overrides["models.backbone.name"] = "pointtransformer"

        # ✅ 关键修复：彻底清空 PointNet++ 的分层采样/半径等 args
        # 这样 apply_overrides 会用 {} 覆盖 base_cfg 中原来的 npoint1/...
        overrides["models.backbone.args"] = {}

        pt_exp["overrides"] = overrides

        flags = dict(pt_exp.get("flags", {}))
        flags["backbone"] = "pointtransformer"
        pt_exp["flags"] = flags

        pt_exps[pt_name] = pt_exp
    return pt_exps

_BASE_ITEMS_SNAPSHOT: List[Tuple[str, Dict[str, object]]] = list(EXPERIMENTS.items())
PT_EXPERIMENTS = _clone_for_pointtransformer(_BASE_ITEMS_SNAPSHOT)

ensure_default_mo_sum(PT_EXPERIMENTS)

EXPERIMENTS.update(PT_EXPERIMENTS)

# =============================================================================
# Extra: CGTC + (PCGrad / CAGrad) on both backbones
# IMPORTANT:
#   - Append at the END to keep existing global order stable for --resume.
#   - We explicitly define PT variants instead of relying on the auto-mirror.
# =============================================================================

EXTRA_MO_CGTC_EXPS: "OrderedDict[str, Dict[str, object]]" = OrderedDict([
    # --------------------
    # PointNet++ (default)
    # --------------------
    (
        "core_pcgrad",
        {
            "description": "CGTC-Core full (ID) + PCGrad (PointNet++)",
            "overrides": {
                # same intent as core_full
                "guard.disable": False,
                "mixed.use": False,
                "adapt.enable": False,
                "pinn.adapt": False,
                "pinn.auto_calib": False,
                "teacher.use": True,
                "teacher.spatial.use": True,
                "pinn.use": True,

                # key difference
                "mo.solver": "pcgrad",
            },
            "flags": {"family": "cgtc_core", "variant": "full", "mo": "pcgrad"},
        },
    ),
    (
        "core_cagrad",
        {
            "description": "CGTC-Core full (ID) + CAGrad (PointNet++)",
            "overrides": {
                "guard.disable": False,
                "mixed.use": False,
                "adapt.enable": False,
                "pinn.adapt": False,
                "pinn.auto_calib": False,
                "teacher.use": True,
                "teacher.spatial.use": True,
                "pinn.use": True,

                "mo.solver": "cagrad",
            },
            "flags": {"family": "cgtc_core", "variant": "full", "mo": "cagrad"},
        },
    ),
    (
        "enh_pcgrad",
        {
            "description": "CGTC-Enhance full (ID) + PCGrad (PointNet++)",
            "overrides": {
                # same intent as enh_full
                "guard.disable": False,
                "mixed.use": True,
                "adapt.enable": True,
                "pinn.adapt": True,
                "pinn.auto_calib": True,
                "teacher.use": True,
                "teacher.spatial.use": True,
                "pinn.use": True,

                "mo.solver": "pcgrad",
            },
            "flags": {"family": "cgtc_enh", "variant": "full", "mo": "pcgrad"},
        },
    ),
    (
        "enh_cagrad",
        {
            "description": "CGTC-Enhance full (ID) + CAGrad (PointNet++)",
            "overrides": {
                "guard.disable": False,
                "mixed.use": True,
                "adapt.enable": True,
                "pinn.adapt": True,
                "pinn.auto_calib": True,
                "teacher.use": True,
                "teacher.spatial.use": True,
                "pinn.use": True,

                "mo.solver": "cagrad",
            },
            "flags": {"family": "cgtc_enh", "variant": "full", "mo": "cagrad"},
        },
    ),

    # --------------------
    # PointTransformer
    # --------------------
    (
        "pt_core_pcgrad",
        {
            "description": "[PointTransformer] CGTC-Core full (ID) + PCGrad",
            "overrides": {
                "models.backbone.name": "pointtransformer",
                "models.backbone.args": {},

                "guard.disable": False,
                "mixed.use": False,
                "adapt.enable": False,
                "pinn.adapt": False,
                "pinn.auto_calib": False,
                "teacher.use": True,
                "teacher.spatial.use": True,
                "pinn.use": True,

                "mo.solver": "pcgrad",
            },
            "flags": {"family": "cgtc_core", "variant": "full", "mo": "pcgrad", "backbone": "pointtransformer"},
        },
    ),
    (
        "pt_core_cagrad",
        {
            "description": "[PointTransformer] CGTC-Core full (ID) + CAGrad",
            "overrides": {
                "models.backbone.name": "pointtransformer",
                "models.backbone.args": {},

                "guard.disable": False,
                "mixed.use": False,
                "adapt.enable": False,
                "pinn.adapt": False,
                "pinn.auto_calib": False,
                "teacher.use": True,
                "teacher.spatial.use": True,
                "pinn.use": True,

                "mo.solver": "cagrad",
            },
            "flags": {"family": "cgtc_core", "variant": "full", "mo": "cagrad", "backbone": "pointtransformer"},
        },
    ),
    (
        "pt_enh_pcgrad",
        {
            "description": "[PointTransformer] CGTC-Enhance full (ID) + PCGrad",
            "overrides": {
                "models.backbone.name": "pointtransformer",
                "models.backbone.args": {},

                "guard.disable": False,
                "mixed.use": True,
                "adapt.enable": True,
                "pinn.adapt": True,
                "pinn.auto_calib": True,
                "teacher.use": True,
                "teacher.spatial.use": True,
                "pinn.use": True,

                "mo.solver": "pcgrad",
            },
            "flags": {"family": "cgtc_enh", "variant": "full", "mo": "pcgrad", "backbone": "pointtransformer"},
        },
    ),
    (
        "pt_enh_cagrad",
        {
            "description": "[PointTransformer] CGTC-Enhance full (ID) + CAGrad",
            "overrides": {
                "models.backbone.name": "pointtransformer",
                "models.backbone.args": {},

                "guard.disable": False,
                "mixed.use": True,
                "adapt.enable": True,
                "pinn.adapt": True,
                "pinn.auto_calib": True,
                "teacher.use": True,
                "teacher.spatial.use": True,
                "pinn.use": True,

                "mo.solver": "cagrad",
            },
            "flags": {"family": "cgtc_enh", "variant": "full", "mo": "cagrad", "backbone": "pointtransformer"},
        },
    ),
])

EXPERIMENTS.update(EXTRA_MO_CGTC_EXPS)

# =============================================================================
# NEW: baseline_new — 强制“从头到尾三者共训 / 同口径 co-training from epoch 1”
#   目标：让 baseline_equal/adapt/pcgrad/cagrad 不再走 warm->multi 的控制语义，
#        而是从 epoch=1 就启用 multi-objective（sup + spatial + PINN）。
# =============================================================================

BASELINE_NEW_START_OVERRIDES = {
    # 你在 trainer.py 里新加的 backward-compatible 入口语义：
    # warm_patience<=0 且 dynamic_warmup<=0 => 直接进入 warmup(=multi) 阶段
    "train.start_stage": "warmup",
    "early.patience": 0,
    "pinn.warmup": 0,
}

def _mk_baseline_new(base_key: str, new_key: str, *, pt: bool) -> Dict[str, object]:
    """
    base_key: 现有实验名（如 baseline_equal / baseline_adapt / baseline_pcgrad / baseline_cagrad / supervised_only）
    new_key : 新实验名
    pt      : 是否生成 PointTransformer 版本（pt=True 会覆盖 backbone）
    """
    if base_key not in EXPERIMENTS:
        raise KeyError(f"[baseline_new] base_key not found in EXPERIMENTS: {base_key}")

    exp = deepcopy(EXPERIMENTS[base_key])

    # ---- description ----
    desc = str(exp.get("description", ""))
    prefix = "[PointTransformer] " if pt else ""
    exp["description"] = f"{prefix}{desc} | baseline_new: co-train from epoch 1 (no warm stage)"

    # ---- overrides ----
    overrides = dict(exp.get("overrides", {}))
    overrides.update(BASELINE_NEW_START_OVERRIDES)

    if pt:
        # 与 PT 镜像保持一致：切 backbone + 清空 PointNet++ args
        overrides["models.backbone.name"] = "pointtransformer"
        overrides["models.backbone.args"] = {}

    exp["overrides"] = overrides

    # ---- flags ----
    flags = dict(exp.get("flags", {}))
    flags["family"] = "baseline_new"
    flags["base"] = base_key
    flags["backbone"] = "pointtransformer" if pt else "pointnetpp"
    exp["flags"] = flags

    return exp

BASELINE_NEW_EXPS: "OrderedDict[str, Dict[str, object]]" = OrderedDict([
    # --------------------
    # PointNet++ (default)
    # --------------------
    ("baseline_new_supervised_only", _mk_baseline_new("supervised_only", "baseline_new_supervised_only", pt=False)),
    ("baseline_new_equal",           _mk_baseline_new("baseline_equal",   "baseline_new_equal",           pt=False)),
    ("baseline_new_adapt",           _mk_baseline_new("baseline_adapt",   "baseline_new_adapt",           pt=False)),
    ("baseline_new_pcgrad",          _mk_baseline_new("baseline_pcgrad",  "baseline_new_pcgrad",          pt=False)),
    ("baseline_new_cagrad",          _mk_baseline_new("baseline_cagrad",  "baseline_new_cagrad",          pt=False)),

    # --------------------
    # PointTransformer
    # --------------------
    ("pt_baseline_new_supervised_only", _mk_baseline_new("supervised_only", "pt_baseline_new_supervised_only", pt=True)),
    ("pt_baseline_new_equal",           _mk_baseline_new("baseline_equal",   "pt_baseline_new_equal",           pt=True)),
    ("pt_baseline_new_adapt",           _mk_baseline_new("baseline_adapt",   "pt_baseline_new_adapt",           pt=True)),
    ("pt_baseline_new_pcgrad",          _mk_baseline_new("baseline_pcgrad",  "pt_baseline_new_pcgrad",          pt=True)),
    ("pt_baseline_new_cagrad",          _mk_baseline_new("baseline_cagrad",  "pt_baseline_new_cagrad",          pt=True)),
])

# 追加到末尾：不影响既有实验的全局序号，保证 --resume 的“顺序稳定性”假设
EXPERIMENTS.update(BASELINE_NEW_EXPS)


# =============================================================================
# 一些通用工具
# =============================================================================

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _set_nested(config: MutableMapping[str, object], key: str, value: object) -> None:
    parts = key.split(".")
    cursor: MutableMapping[str, object] = config
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, MutableMapping):
            child = {}
            cursor[part] = child
        cursor = child  # type: ignore[assignment]
    cursor[parts[-1]] = value


def apply_overrides(base: Mapping[str, object], overrides: Mapping[str, object]) -> Dict[str, object]:
    updated = deepcopy(base)
    for key, value in overrides.items():
        _set_nested(updated, key, value)
    return updated


def write_yaml(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


# =============================================================================
# metrics 解析 / 总结
# =============================================================================

def _split_metadata_and_csv(path: Path) -> Tuple[Dict[str, object], pd.DataFrame]:
    with path.open("r", encoding="utf-8-sig") as handle:
        raw = handle.read().splitlines()
    meta: Dict[str, object] = {}
    csv_lines: List[str] = []
    for line in raw:
        if line.startswith("#"):
            payload = line[1:].strip()
            if "=" in payload:
                key, value = payload.split("=", 1)
                key = key.strip()
                value = value.strip()
                try:
                    meta[key] = json.loads(value)
                except json.JSONDecodeError:
                    meta[key] = value
        else:
            csv_lines.append(line)
    if not csv_lines:
        raise ValueError(f"No CSV content found in {path}")
    data = pd.read_csv(StringIO("\n".join(csv_lines)))
    return meta, data


def _float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_guard_hit_rate(train_df: pd.DataFrame) -> Optional[float]:
    if "guard_phys_triggered" not in train_df.columns:
        return None
    if train_df.empty:
        return None
    series = train_df["guard_phys_triggered"].astype(bool)
    return float(series.mean())


def _compute_cooling_mean(metadata: Dict[str, object], df: pd.DataFrame) -> Optional[float]:
    events = metadata.get("cooling_events")
    if not isinstance(events, list):
        return None
    train_df = df[df["split"].str.lower() == "train"].copy()
    if train_df.empty:
        return None
    if {"global_step", "epoch"}.issubset(train_df.columns):
        step_to_epoch = train_df.drop_duplicates("global_step")["epoch"].to_dict()
    else:
        step_to_epoch = {}
    active: Dict[str, Dict[str, object]] = {}
    durations: List[float] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        term = event.get("term")
        step = int(event.get("step", 0))
        epoch = float(step_to_epoch.get(step, step))
        kind = event.get("event")
        if term != "pinn":
            continue
        if kind == "start":
            active[term] = {"step": step, "epoch": epoch}
        elif kind == "end":
            start = active.pop(term, None)
            if start is None:
                continue
            dur = float(epoch - start.get("epoch", epoch))
            durations.append(dur if math.isfinite(dur) else 0.0)
    if not durations:
        return None
    return float(sum(durations) / len(durations))


def _mean_epoch_time(val_df: pd.DataFrame) -> Optional[float]:
    if "epoch_time" not in val_df.columns:
        return None
    values = pd.to_numeric(val_df["epoch_time"], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def summarise_experiment(
    metrics_path: Path,
    physics_summary: Optional[pd.DataFrame],
    exp_flags: Mapping[str, object],
) -> Dict[str, object]:
    metadata, df = _split_metadata_and_csv(metrics_path)
    train_df = df[df["split"].str.lower() == "train"].copy()
    val_df = df[df["split"].str.lower() == "val"].copy()
    if val_df.empty:
        raise ValueError(f"No validation rows in metrics file: {metrics_path}")
    last_val = val_df.iloc[-1]

    sup_mae = _float(last_val.get("L_sup_mae"))
    sup_rmse = _float(last_val.get("L_sup_rmse"))
    if sup_rmse is None and "L_sup" in last_val:
        v = _float(last_val.get("L_sup"))
        sup_rmse = math.sqrt(v) if v is not None else None

    guard_hit_rate = _compute_guard_hit_rate(train_df)
    avg_cooling = _compute_cooling_mean(metadata, df)
    time_per_epoch = _mean_epoch_time(val_df)
    seed = metadata.get("seed")

    record: Dict[str, object] = {
        "flags_json": json.dumps(exp_flags, sort_keys=True),
        "sup_MAE": sup_mae,
        "sup_RMSE": sup_rmse,
        "guard_hit_rate": guard_hit_rate,
        "avg_cooling_len": avg_cooling,
        "time_per_epoch": time_per_epoch,
        "seed": seed,
    }

    if physics_summary is not None and not physics_summary.empty:
        row = physics_summary.iloc[-1]
        record.update({
            "div_P95": _float(row.get("div_P95")),
            "mom_P95": _float(row.get("mom_P95")),
        })
    else:
        record.update({"div_P95": None, "mom_P95": None})

    return record


# =============================================================================
# 进程执行
# =============================================================================

def run_process(cmd: Sequence[str], *, cwd: Optional[Path] = None) -> None:
    print("[RUN]", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TPAC ablation + robustness runner (wo.py)")
    parser.add_argument("--root", type=Path, required=True, help="Project root (same as train_tpac --root)")
    parser.add_argument("--case", required=True, help="Dataset case identifier (e.g., C1)")
    parser.add_argument("--tag", required=True, help="Base experiment tag; ablation suffixes are appended")
    parser.add_argument(
        "--base-cfg",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Baseline YAML config (relative to root if not absolute)",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        choices=list(EXPERIMENTS.keys()),
        default=None,
        help="Subset of experiments to execute (default: all)",
    )
    parser.add_argument(
        "--only-pt",
        action="store_true",
        help="Run only PointTransformer mirrored experiments (pt_*) while preserving global order numbers",
    )
    parser.add_argument(
        "--only-pn",
        action="store_true",
        help="Run only PointNet++ experiments (non-pt_*) while preserving global order numbers",
    )

    parser.add_argument("--train-script", type=Path, default=Path("train_tpac.py"))
    parser.add_argument("--draw-script", type=Path, default=Path("draw_picture.py"))
    parser.add_argument("--eval-script", type=Path, default=Path("eval/eval_physics.py"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pts", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to train_tpac (after --)",
    )

    parser.add_argument("--eval-split", type=str, default="test", help="Split for physics evaluation")
    parser.add_argument("--eval-m-samples", type=int, default=4096)
    parser.add_argument("--eval-k", type=int, default=24)
    parser.add_argument("--eval-max-samples", type=int, default=-1)

    # ✅ 续跑/断点
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip training when train_val_metrics.csv already exists for this exp_tag",
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Re-run physics evaluation even if physics_eval.csv exists",
    )
    parser.add_argument(
        "--no-draw",
        action="store_true",
        help="Skip draw_picture step",
    )

    return parser


# =============================================================================
# 主入口
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    root = args.root.resolve()

    base_cfg_path = args.base_cfg
    if not base_cfg_path.is_absolute():
        base_cfg_path = (root / base_cfg_path).resolve()

    train_script = args.train_script
    if not train_script.is_absolute():
        train_script = (root / train_script).resolve()

    draw_script = args.draw_script
    if not draw_script.is_absolute():
        draw_script = (root / draw_script).resolve()

    eval_script = args.eval_script
    if not eval_script.is_absolute():
        eval_script = (root / eval_script).resolve()

    if not base_cfg_path.exists():
        parser.error(f"Baseline config not found: {base_cfg_path}")
    if not train_script.exists():
        parser.error(f"Training script not found: {train_script}")
    if not draw_script.exists():
        parser.error(f"draw_picture.py not found: {draw_script}")
    if not eval_script.exists():
        parser.error(f"eval_physics.py not found: {eval_script}")

    with base_cfg_path.open("r", encoding="utf-8") as handle:
        base_cfg = yaml.safe_load(handle)

    # ✅ 全局顺序编号表（保证只跑子集也不改变 order 编号）
    global_order = {name: idx for idx, name in enumerate(EXPERIMENTS.keys(), start=1)}

    selected_names = args.experiments if args.experiments else list(EXPERIMENTS.keys())

    if args.only_pt and args.only_pn:
        parser.error("Cannot set both --only-pt and --only-pn.")

    if args.only_pt:
        selected_names = [n for n in selected_names if n.startswith("pt_")]
    elif args.only_pn:
        selected_names = [n for n in selected_names if not n.startswith("pt_")]

    # 组装 (order, name, exp)，并按全局顺序排序
    experiments: List[Tuple[int, str, Dict[str, object]]] = []
    for name in selected_names:
        exp = EXPERIMENTS.get(name)
        if exp is None:
            continue
        experiments.append((global_order[name], name, exp))
    experiments.sort(key=lambda x: x[0])

    if not experiments:
        parser.error("No experiments selected.")

    wo_root = root / "runs" / "wo" / args.case / args.tag
    cfg_out_dir = wo_root / "configs"
    _ensure_dir(cfg_out_dir)

    summary_records: List[Dict[str, object]] = []

    for order, name, exp in experiments:
        overrides = exp.get("overrides", {})
        if not isinstance(overrides, dict):
            overrides = {}

        cfg_payload = apply_overrides(base_cfg, overrides)

        cfg_filename = f"{args.tag}_{order:02d}_{name}.yaml"
        cfg_path = cfg_out_dir / cfg_filename
        write_yaml(cfg_path, cfg_payload)

        exp_tag = f"{args.tag}_{order:02d}_{name}"
        temp_dir = root / "results" / "temp_results" / args.case / exp_tag
        _ensure_dir(temp_dir)

        metrics_path = temp_dir / "train_val_metrics.csv"

        # ------------------------------------------------------------------
        # 1) 训练（支持 resume）
        # ------------------------------------------------------------------
        if args.resume and metrics_path.exists():
            print(f"[SKIP] resume enabled, metrics exists: {exp_tag}")
        else:
            cmd = [
                sys.executable,
                str(train_script.relative_to(root)) if train_script.is_relative_to(root) else str(train_script),
                "--cfg",
                str(cfg_path.relative_to(root)) if cfg_path.is_relative_to(root) else str(cfg_path),
                "--root",
                str(root),
                "--case",
                args.case,
                "--tag",
                exp_tag,
                "--device",
                args.device,
                "--pts",
                str(args.pts),
                "--batch",
                str(args.batch),
                "--workers",
                str(args.workers),
            ]
            if args.extra:
                cmd.extend(args.extra)

            try:
                run_process(cmd, cwd=root)
            except Exception as e:
                print(f"[ERR] training failed for {exp_tag}: {e}")
                continue

        if not metrics_path.exists():
            print(f"[WARN] metrics file missing for {exp_tag}: {metrics_path}")
            continue

        # ------------------------------------------------------------------
        # 2) 画诊断曲线（可关闭）
        # ------------------------------------------------------------------
        if not args.no_draw:
            diagnostics_dir = temp_dir / "diagnostics"
            draw_cmd = [
                sys.executable,
                str(draw_script.relative_to(root)) if draw_script.is_relative_to(root) else str(draw_script),
                "--metrics",
                str(metrics_path.relative_to(root)) if metrics_path.is_relative_to(root) else str(metrics_path),
                "--out-dir",
                str(diagnostics_dir.relative_to(root)) if diagnostics_dir.is_relative_to(root) else str(diagnostics_dir),
            ]
            try:
                run_process(draw_cmd, cwd=root)
            except Exception as e:
                print(f"[WARN] draw failed for {exp_tag}: {e}")

        # ------------------------------------------------------------------
        # 3) 物理评估
        #    ✅ 用当前 cfg_path，避免 PT 时仍加载 base_cfg 的 PN++ 参数
        # ------------------------------------------------------------------
        ckpt_path = temp_dir / "weight" / "final_reco.pth"
        physics_summary: Optional[pd.DataFrame] = None

        if ckpt_path.exists():
            physics_out = temp_dir / "physics_eval.csv"

            need_eval = args.force_eval or (not physics_out.exists())
            if need_eval:
                eval_cmd = [
                    sys.executable,
                    str(eval_script.relative_to(root)) if eval_script.is_relative_to(root) else str(eval_script),
                    "--cfg",
                    str(cfg_path.relative_to(root)) if cfg_path.is_relative_to(root) else str(cfg_path),
                    "--root",
                    str(root),
                    "--case",
                    args.case,
                    "--split",
                    args.eval_split,
                    "--pts",
                    str(args.pts),
                    "--ckpt",
                    str(ckpt_path.relative_to(root)) if ckpt_path.is_relative_to(root) else str(ckpt_path),
                    "--out",
                    str(physics_out.relative_to(root)) if physics_out.is_relative_to(root) else str(physics_out),
                    "--device",
                    args.device,
                    "--m-samples",
                    str(args.eval_m_samples),
                    "--k",
                    str(args.eval_k),
                ]
                if args.eval_max_samples > 0:
                    eval_cmd.extend(["--max-samples", str(args.eval_max_samples)])
                try:
                    run_process(eval_cmd, cwd=root)
                except Exception as e:
                    print(f"[WARN] physics eval failed for {exp_tag}: {e}")

            if physics_out.exists():
                try:
                    physics_summary = pd.read_csv(physics_out)
                except Exception:
                    physics_summary = None
        else:
            print(f"[WARN] final checkpoint missing for {exp_tag}: {ckpt_path}")

        # ------------------------------------------------------------------
        # 4) 汇总
        # ------------------------------------------------------------------
        try:
            summary = summarise_experiment(metrics_path, physics_summary, exp.get("flags", {}))
        except Exception as e:
            print(f"[WARN] summarise failed for {exp_tag}: {e}")
            continue

        summary.update({
            "exp_name": name,
            "description": exp.get("description", ""),
            "tag": exp_tag,
        })
        summary_records.append(summary)

    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_path = wo_root / "wo_summary.csv"
        _ensure_dir(summary_path.parent)
        summary_df.to_csv(summary_path, index=False)
        print(f"[OK] summary written to {summary_path}")
    else:
        print("[WARN] No summaries generated – check earlier warnings.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
