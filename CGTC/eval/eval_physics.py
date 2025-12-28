# -*- coding: utf-8 -*-
"""
CGTC/eval/eval_physics.py

- 支持两类模型前向：
  1) 经典模型:  forward((B,3,N)) -> (B,4,N)  (通道顺序 [p,u,v,w])
  2) TPAC 适配器: forward((B,N,3)) -> dict{"p","u","v","w"} 各 (B,N)

- 评估内容：
  * 随机采样 M 点的预测/真值散度
  * 预测流场的动量残差 r = ρ (u·∇)u + ∇p - ρν ∇²u
  * 统计与直方图（可选保存）
"""
from __future__ import annotations
import argparse
import json
import math
import pickle
import random
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import yaml

_THIS_DIR = Path(__file__).resolve()
_CODE_ROOT = _THIS_DIR.parents[1]
_REPO_ROOT = _THIS_DIR.parents[2]
for _p in (str(_CODE_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:  # pragma: no cover - expected path when package installed
    from Rotate.data.dataset import pointdata, build_out_csv_from_dir, norm_data
    from Rotate.models.backbone import build_backbone
except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution
    from data.dataset import pointdata, build_out_csv_from_dir, norm_data
    from models.backbone import build_backbone


def _dict_to_ns(payload):
    if isinstance(payload, dict):
        return NS(**{k: _dict_to_ns(v) for k, v in payload.items()})
    return payload


def _ns_to_dict(ns_obj):
    if isinstance(ns_obj, NS):
        return {k: _ns_to_dict(getattr(ns_obj, k)) for k in vars(ns_obj)}
    return ns_obj

# ================== 前向适配（自动适配 TPAC/backbone 与老模型） ==================
@torch.no_grad()
def _stack_dict_to_b4n(preds: dict) -> torch.Tensor:
    # 输入: {"p","u","v","w"} 各 (B,N)
    p = preds["p"].unsqueeze(1)  # (B,1,N)
    u = preds["u"].unsqueeze(1)
    v = preds["v"].unsqueeze(1)
    w = preds["w"].unsqueeze(1)
    return torch.cat([p, u, v, w], dim=1).contiguous()  # (B,4,N)

@torch.no_grad()
def _forward_to_b4n(model, X_b3n: torch.Tensor) -> torch.Tensor:
    """
    统一把模型输出转为 (B,4,N) with [p,u,v,w]
    - 若模型接受 (B,3,N) -> (B,4,N)，直接返回
    - 若是 TPAC 适配器，接受 (B,N,3) -> dict，做拼接
    """
    try:
        X_bnc = X_b3n.transpose(1, 2).contiguous()  # (B,N,3)
        out = model(X_bnc)
        if isinstance(out, dict) and all(k in out for k in ("p","u","v","w")):
            return _stack_dict_to_b4n(out)
    except Exception:
        pass

    out = model(X_b3n)
    assert out.dim() == 3 and out.size(1) == 4, "model must output (B,4,N) with [p,u,v,w]"
    return out

# ================== 基础工具（kNN + 局部最小二乘，一阶导） ==================
@torch.no_grad()
def _pairwise_cdist(x, y):
    # x: (B,M,3), y: (B,N,3) -> (B,M,N)
    return torch.cdist(x, y, p=2)

@torch.no_grad()
def _knn_indices(query_xyz, ref_xyz, k: int):
    """
    更稳健的 kNN：下限>=6；上限严格<N（避开自身）。
    """
    d = _pairwise_cdist(query_xyz, ref_xyz)  # (B,M,N)
    N = int(ref_xyz.shape[1])
    if N <= 1:
        B, M, _ = d.shape
        return torch.zeros(B, M, 1, dtype=torch.long, device=d.device)
    k_lower = 6
    k_upper = max(1, N - 1)
    k_eff = min(k_upper, max(k_lower, int(k)))
    _, idx = torch.topk(d, k=k_eff, dim=-1, largest=False, sorted=False)
    return idx  # (B,M,k_eff)

@torch.no_grad()
def _gather_neighbors(t, idx):
    """
    t: (B,N,C) or (B,N) ; idx: (B,M,k)
    -> (B,M,k,C) or (B,M,k)
    """
    if t.dim() == 2:
        B, N = t.shape
        B2, M, K = idx.shape
        assert B == B2
        bidx = torch.arange(B, device=idx.device).view(B,1,1).expand(-1,M,K)
        return t[bidx, idx]
    elif t.dim() == 3:
        B, N, C = t.shape
        B2, M, K = idx.shape
        assert B == B2
        bidx = torch.arange(B, device=idx.device).view(B,1,1).expand(-1,M,K)
        return t[bidx, idx, :]
    else:
        raise ValueError("t must be (B,N) or (B,N,C)")

@torch.no_grad()
def _least_squares_grad(
    query_xyz, ref_xyz, f, idx, ridge: float = 1e-6, ridge_retry: int = 3
):
    """
    稳健的一阶 MLS 梯度估计：
      - 解 (X^T X + λI) g = X^T (f_nbr - f_q)
      - cholesky_ex 失败则增大 λ 重试；最后 pinv 兜底
      - f_q 用“k 内最近邻”近似，避免再做全 N 的 cdist 与自邻居 0 距离
    输入:
      query_xyz: (B,M,3) ; ref_xyz: (B,N,3) ; f: (B,N) ; idx: (B,M,k)
    输出:
      grad: (B,M,3), f_q: (B,M)
    """
    device = query_xyz.device
    dtype  = query_xyz.dtype

    nbr_xyz = _gather_neighbors(ref_xyz, idx)                      # (B,M,k,3)
    f_nbr   = _gather_neighbors(f.unsqueeze(-1), idx).squeeze(-1)  # (B,M,k)

    q  = query_xyz.unsqueeze(2)                                    # (B,M,1,3)
    dq = (nbr_xyz - q)                                             # (B,M,k,3)

    # “k 内最近邻”近似 f(q)
    d_loc = torch.linalg.norm(dq, dim=-1)                          # (B,M,k)
    min_k = torch.argmin(d_loc, dim=-1)                            # (B,M)
    B, M, K = d_loc.shape
    bidx = torch.arange(B, device=device)[:, None].expand(B, M)
    midx = torch.arange(M, device=device)[None, :].expand(B, M)
    f_q  = f_nbr[bidx, midx, min_k]                                # (B,M)

    df = f_nbr - f_q.unsqueeze(-1)                                 # (B,M,k)

    XT  = dq.transpose(-1, -2)                                     # (B,M,3,k)
    XTX = XT @ dq                                                  # (B,M,3,3)
    XTy = XT @ df.unsqueeze(-1)                                    # (B,M,3,1)

    I = torch.eye(3, device=device, dtype=torch.float32).view(1,1,3,3)
    base = (XTX.abs().mean(dim=(-1,-2), keepdim=True) + 1e-12)

    ridge_list = [float(ridge)]
    for _ in range(max(0, int(ridge_retry))):
        ridge_list.append(ridge_list[-1] * 10.0)
    ridge_list = [max(1e-12, r) for r in ridge_list] + [1e-3]

    g = None; solved = False
    for lam in ridge_list:
        A = (XTX.to(torch.float32) + (lam * base) * I)
        L, info = torch.linalg.cholesky_ex(A)
        if (info == 0).all():
            g = torch.cholesky_solve(XTy.to(torch.float32), L).squeeze(-1)
            solved = True; break

    if not solved:
        A = (XTX.to(torch.float32) + (1e-3*base) * I)
        A_pinv = torch.linalg.pinv(A)
        g = (A_pinv @ XTy.to(torch.float32)).squeeze(-1)

    return g.to(dtype), f_q.to(dtype)

# ================== 散度估计 ==================
@torch.no_grad()
def estimate_divergence_for_sample(
    xyz_np: np.ndarray,
    vec_np: np.ndarray,             # (3,N) -> [u,v,w]
    m_samples: int = 4096,
    k_neighbors: int = 16,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    assert xyz_np.shape[0] == 3 and vec_np.shape[0] == 3
    N = xyz_np.shape[1]
    M = int(min(max(1, m_samples), N))

    xyz = torch.from_numpy(xyz_np.T).to(device=device, dtype=torch.float32).unsqueeze(0)  # (1,N,3)
    U   = torch.from_numpy(vec_np.T).to(device=device, dtype=torch.float32).unsqueeze(0)  # (1,N,3)

    idx_sub = torch.randperm(N, device=device)[:M]
    qxyz = xyz[:, idx_sub, :]   # (1,M,3)
    rxyz = xyz                   # (1,N,3)

    idx_knn = _knn_indices(qxyz, rxyz, k_neighbors)   # (1,M,k)

    u = U[..., 0]; v = U[..., 1]; w = U[..., 2]
    gu, _ = _least_squares_grad(qxyz, rxyz, u, idx_knn)
    gv, _ = _least_squares_grad(qxyz, rxyz, v, idx_knn)
    gw, _ = _least_squares_grad(qxyz, rxyz, w, idx_knn)

    div = gu[...,0] + gv[...,1] + gw[...,2]
    return div.squeeze(0)  # (M,)

# ================== 图拉普拉斯（用于粘性项） ==================
@torch.no_grad()
def _graph_laplacian_scalar_at_queries(
    query_xyz: torch.Tensor, ref_xyz: torch.Tensor, f: torch.Tensor,
    idx_knn: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    nbr_xyz = _gather_neighbors(ref_xyz, idx_knn)      # (B,M,K,3)
    q = query_xyz.unsqueeze(2)                         # (B,M,1,3)
    dqsq = ((nbr_xyz - q)**2).sum(dim=-1) + eps        # (B,M,K)

    sigma2 = dqsq.mean(dim=-1, keepdim=True)           # (B,M,1)
    w = torch.exp(-0.5 * dqsq / (sigma2 + eps))        # (B,M,K)

    f_nbr = _gather_neighbors(f.unsqueeze(-1), idx_knn).squeeze(-1)   # (B,M,K)
    d_all = _pairwise_cdist(query_xyz, ref_xyz) + eps
    minidx = torch.argmin(d_all, dim=-1)              # (B,M)
    f_i = f.gather(1, minidx)                         # (B,M)

    num = (w * (f_nbr - f_i.unsqueeze(-1))).sum(dim=-1)
    den = (w.sum(dim=-1) * (sigma2.squeeze(-1) + eps))
    return num / (den + eps)

@torch.no_grad()
def _graph_laplacian_vector_at_queries(query_xyz, ref_xyz, U, idx_knn):
    Lu = _graph_laplacian_scalar_at_queries(query_xyz, ref_xyz, U[...,0], idx_knn)
    Lv = _graph_laplacian_scalar_at_queries(query_xyz, ref_xyz, U[...,1], idx_knn)
    Lw = _graph_laplacian_scalar_at_queries(query_xyz, ref_xyz, U[...,2], idx_knn)
    return torch.stack([Lu, Lv, Lw], dim=-1)  # (B,M,3)

# ================== 单样本：评估 div 与 动量残差 ==================
@torch.no_grad()
def eval_physics_one(
    model: nn.Module,
    xyz_norm: np.ndarray,
    label_norm: np.ndarray,
    stats_np: np.ndarray,
    device: torch.device,
    m_samples: int = 4096,
    k_neighbors: int = 16,
    rho: float = 1.0,
    nu_eff: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Evaluate divergence and momentum residual in physical units for one sample."""

    stats_np = np.asarray(stats_np, dtype=np.float64).reshape(-1)
    x_min = stats_np[0:3]
    x_max = stats_np[3:6]
    y_min = stats_np[6:10]
    y_max = stats_np[10:14]
    x_rng = np.clip(x_max - x_min, 1e-12, None)
    y_rng = np.clip(y_max - y_min, 1e-12, None)

    xyz_norm = np.asarray(xyz_norm, dtype=np.float64)
    label_norm = np.asarray(label_norm, dtype=np.float64)

    xyz_phys = xyz_norm * x_rng[:, None] + x_min[:, None]
    label_phys = label_norm * y_rng[:, None] + y_min[:, None]

    # forward pass using normalised coordinates, then反归一化输出
    X_b3n = torch.from_numpy(xyz_norm).unsqueeze(0).to(device, dtype=torch.float32)
    pred_norm = _forward_to_b4n(model, X_b3n).detach().cpu().numpy()[0]
    pred_phys = pred_norm * y_rng[:, None] + y_min[:, None]

    p_pred = pred_phys[0:1, :]
    uvw_pred = pred_phys[1:4, :]
    uvw_true = label_phys[1:4, :]

    # convert to torch tensors in physical units
    xyz = torch.from_numpy(xyz_phys.T).to(device, dtype=torch.float32).unsqueeze(0)
    U = torch.from_numpy(uvw_pred.T).to(device, dtype=torch.float32).unsqueeze(0)
    P = torch.from_numpy(p_pred.T).to(device, dtype=torch.float32).unsqueeze(0)

    N = xyz_phys.shape[1]
    M = int(min(max(1, m_samples), N))
    idx_sub = torch.randperm(N, device=device)[:M]
    qxyz = xyz[:, idx_sub, :]
    rxyz = xyz

    idx_knn = _knn_indices(qxyz, rxyz, k_neighbors)

    div_pred = estimate_divergence_for_sample(xyz_phys, uvw_pred, m_samples, k_neighbors, device).cpu().numpy()
    div_true = estimate_divergence_for_sample(xyz_phys, uvw_true, m_samples, k_neighbors, device).cpu().numpy()

    u = U[..., 0]
    v = U[..., 1]
    w = U[..., 2]
    gu, u_q = _least_squares_grad(qxyz, rxyz, u, idx_knn)
    gv, v_q = _least_squares_grad(qxyz, rxyz, v, idx_knn)
    gw, w_q = _least_squares_grad(qxyz, rxyz, w, idx_knn)

    p_field = P.squeeze(-1)
    gp, _ = _least_squares_grad(qxyz, rxyz, p_field, idx_knn)

    adv_x = u_q * gu[..., 0] + v_q * gu[..., 1] + w_q * gu[..., 2]
    adv_y = u_q * gv[..., 0] + v_q * gv[..., 1] + w_q * gv[..., 2]
    adv_z = u_q * gw[..., 0] + v_q * gw[..., 1] + w_q * gw[..., 2]
    adv = torch.stack([adv_x, adv_y, adv_z], dim=-1)

    lapU = _graph_laplacian_vector_at_queries(qxyz, rxyz, U, idx_knn)

    mom_res = rho * adv + gp - (rho * nu_eff) * lapU
    mom_res = mom_res.squeeze(0).cpu().numpy()
    mom_norm = np.linalg.norm(mom_res, axis=-1)

    return {
        "div_pred": div_pred,
        "div_true": div_true,
        "mom_res_xyz": mom_res,
        "mom_res_norm": mom_norm,
    }

# ================== 统计与可视化 ==================
def _stats(arr: np.ndarray) -> dict:
    arr = arr.astype(np.float64)
    return {
        "mean": float(arr.mean()),
        "abs_mean": float(np.mean(np.abs(arr))),
        "rms": float(np.sqrt(np.mean(arr**2))),
        "p95_abs": float(np.percentile(np.abs(arr), 95)),
        "max_abs": float(np.max(np.abs(arr))),
        "count": int(arr.size),
    }

@torch.no_grad()
def compute_physics_on_dataset(
    model,
    dataset,
    device: torch.device,
    max_items: int = 6,
    m_samples: int = 4096,
    k_neighbors: int = 16,
    rho: float = 1.0,
    nu_eff: float = 0.0,
    save_hist_png_div: Optional[Path] = None,
    save_hist_png_mom: Optional[Path] = None,
    save_csv: Optional[Path] = None
) -> Tuple[Dict[str, float], "pd.DataFrame"]:
    """
    返回 overall(dict) 与 per-sample DataFrame
    """
    model.eval()
    import numpy as _np
    idxs = _np.random.choice(len(dataset), size=min(max_items, len(dataset)), replace=False)
    rows = []
    all_pred_div, all_true_div, all_mom_norm = [], [], []

    for _, idx in enumerate(idxs):
        sample = dataset[idx]
        if isinstance(sample, (tuple, list)) and len(sample) >= 3:
            xyz_np, label_np, stats_np = sample[0], sample[1], sample[2]
        else:
            raise RuntimeError("dataset item must provide (xyz, label, stats)")
        out = eval_physics_one(
            model, xyz_np, label_np, stats_np, device,
            m_samples=m_samples, k_neighbors=k_neighbors,
            rho=rho, nu_eff=nu_eff
        )
        div_pred, div_true = out["div_pred"], out["div_true"]
        mom_norm = out["mom_res_norm"]

        s_pred = _stats(div_pred); s_true = _stats(div_true); s_mom = _stats(mom_norm)
        rows.append({
            "sample_idx": int(idx),
            **{f"pred_div_{k}": v for k, v in s_pred.items()},
            **{f"true_div_{k}": v for k, v in s_true.items()},
            **{f"mom_{k}": v for k, v in s_mom.items()},
        })

        all_pred_div.append(div_pred); all_true_div.append(div_true); all_mom_norm.append(mom_norm)

    import pandas as pd
    df = pd.DataFrame(rows)

    overall = {
        "N_samples": int(df.shape[0]),
        "pred_div_abs_mean_avg": float(df["pred_div_abs_mean"].mean()),
        "true_div_abs_mean_avg": float(df["true_div_abs_mean"].mean()),
        "pred_div_rms_avg": float(df["pred_div_rms"].mean()),
        "true_div_rms_avg": float(df["true_div_rms"].mean()),
        "mom_abs_mean_avg": float(df["mom_abs_mean"].mean()),
        "mom_rms_avg": float(df["mom_rms"].mean()),
        "mom_p95_abs_avg": float(df["mom_p95_abs"].mean()),
    }

    if save_csv is not None:
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
        print(f"[PHYS] per-sample stats saved -> {save_csv}")

    if save_hist_png_div is not None:
        save_hist_png_div.parent.mkdir(parents=True, exist_ok=True)
        pred_all = np.concatenate(all_pred_div) if len(all_pred_div) else np.array([])
        true_all = np.concatenate(all_true_div) if len(all_true_div) else np.array([])

        fig = plt.figure(figsize=(9, 4), dpi=150, constrained_layout=True)
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1,1])
        ax1 = fig.add_subplot(gs[0,0]); ax2 = fig.add_subplot(gs[0,1])
        ax1.hist(true_all, bins=60, alpha=0.75, label="True div(u)")
        ax1.hist(pred_all, bins=60, alpha=0.75, label="Pred div(u)")
        ax1.set_title("Divergence Histogram (linear)")
        ax1.set_xlabel("div(u)"); ax1.set_ylabel("count"); ax1.legend()
        ax2.hist(np.abs(true_all), bins=60, alpha=0.75, label="|True div(u)|")
        ax2.hist(np.abs(pred_all), bins=60, alpha=0.75, label="|Pred div(u)|")
        ax2.set_yscale("log")
        ax2.set_title("Abs Divergence Histogram (log y)")
        ax2.set_xlabel("|div(u)|"); ax2.set_ylabel("count"); ax2.legend()
        plt.savefig(save_hist_png_div, dpi=300, bbox_inches="tight"); plt.close(fig)
        print(f"[PHYS] divergence histogram saved -> {save_hist_png_div}")

    if save_hist_png_mom is not None:
        save_hist_png_mom.parent.mkdir(parents=True, exist_ok=True)
        mom_all = np.concatenate(all_mom_norm) if len(all_mom_norm) else np.array([])
        fig = plt.figure(figsize=(4.8, 4), dpi=150, constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.hist(mom_all, bins=60, alpha=0.85, label="||Momentum residual||")
        ax.set_yscale("log")
        ax.set_title("Momentum Residual Norm (log y)")
        ax.set_xlabel("||r||"); ax.set_ylabel("count"); ax.legend()
        plt.savefig(save_hist_png_mom, dpi=300, bbox_inches="tight"); plt.close(fig)
        print(f"[PHYS] momentum-residual histogram saved -> {save_hist_png_mom}")

    print("[PHYS] Overall (averaged over samples):")
    for k, v in overall.items():
        print(f"  - {k}: {v:.6e}" if isinstance(v, float) else f"  - {k}: {v}")

    return overall, df


def _ks_distance(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    if sample_a.size == 0 or sample_b.size == 0:
        return float("nan")
    a = np.sort(sample_a.astype(np.float64))
    b = np.sort(sample_b.astype(np.float64))
    i = j = 0
    n = a.size
    m = b.size
    d = 0.0
    cdf_a = cdf_b = 0.0
    while i < n and j < m:
        if a[i] <= b[j]:
            i += 1
            cdf_a = i / n
        else:
            j += 1
            cdf_b = j / m
        d = max(d, abs(cdf_a - cdf_b))
    if i < n:
        d = max(d, abs(1.0 - cdf_b))
    if j < m:
        d = max(d, abs(cdf_a - 1.0))
    return float(d)


def _prepare_dataset(root: Path, case: str, split: str, points_per_sample: int) -> Tuple[pointdata, dict]:
    txt_dir = root / "data" / "dataset" / case / split
    if not txt_dir.exists():
        raise FileNotFoundError(f"Missing dataset split: {txt_dir}")
    tmp_dir = root / "results" / "temp_results" / case / "eval" / split
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tmp_dir / "out.csv"
    if not out_csv.exists():
        build_out_csv_from_dir(txt_dir, out_csv)
    info_path = tmp_dir / "data.pkl"
    if not info_path.exists():
        min_val, max_val, num_data, num_points = norm_data(str(txt_dir))
        payload = {
            "input_min": min_val,
            "input_max": max_val,
            "num_data": num_data,
            "num_points": num_points,
        }
        with info_path.open("wb") as handle:
            pickle.dump(payload, handle)
    with info_path.open("rb") as handle:
        info = pickle.load(handle)
    dataset = pointdata(str(txt_dir), str(out_csv), info, int(points_per_sample))
    return dataset, info


def _load_config(cfg_path: Path) -> NS:
    with cfg_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return _dict_to_ns(payload)


def _build_model_from_cfg(cfg: NS, device: torch.device) -> nn.Module:
    mcfg = NS()
    mcfg.models = cfg.models
    if hasattr(mcfg.models.backbone, "args") and isinstance(mcfg.models.backbone.args, NS):
        mcfg.models.backbone.args = _ns_to_dict(mcfg.models.backbone.args)
    model = build_backbone(mcfg).to(device)
    return model


def _collect_physics_arrays(
    model: nn.Module,
    dataset: pointdata,
    device: torch.device,
    *,
    m_samples: int,
    k_neighbors: int,
    rho: float,
    nu_eff: float,
    max_items: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    model.eval()
    div_preds: List[np.ndarray] = []
    div_trues: List[np.ndarray] = []
    mom_norms: List[np.ndarray] = []
    total_points = 0
    length = len(dataset) if max_items is None or max_items < 0 else min(len(dataset), max_items)
    for idx in range(length):
        xyz_np, label_np, stats_np = dataset[idx]
        out = eval_physics_one(
            model,
            xyz_np,
            label_np,
            stats_np,
            device,
            m_samples=m_samples,
            k_neighbors=k_neighbors,
            rho=rho,
            nu_eff=nu_eff,
        )
        div_preds.append(out["div_pred"])
        div_trues.append(out["div_true"])
        mom_norms.append(out["mom_res_norm"])
        total_points += int(out["div_pred"].size)
    div_pred_all = np.concatenate(div_preds) if div_preds else np.zeros(0, dtype=np.float64)
    div_true_all = np.concatenate(div_trues) if div_trues else np.zeros(0, dtype=np.float64)
    mom_norm_all = np.concatenate(mom_norms) if mom_norms else np.zeros(0, dtype=np.float64)
    return div_pred_all, div_true_all, mom_norm_all, total_points


def _summarise_arrays(
    div_pred: np.ndarray,
    div_true: np.ndarray,
    mom_norm: np.ndarray,
    *,
    k_neighbors: int,
    m_samples: int,
    rho: float,
    nu_eff: float,
    n_points: int,
) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    if div_pred.size > 0:
        abs_div = np.abs(div_pred)
        summary.update(
            {
                "div_P50": float(np.percentile(abs_div, 50)),
                "div_P90": float(np.percentile(abs_div, 90)),
                "div_P95": float(np.percentile(abs_div, 95)),
                "div_mean": float(np.mean(div_pred)),
                "div_std": float(np.std(div_pred)),
            }
        )
    else:
        summary.update({"div_P50": math.nan, "div_P90": math.nan, "div_P95": math.nan, "div_mean": math.nan, "div_std": math.nan})

    if mom_norm.size > 0:
        summary.update(
            {
                "mom_P50": float(np.percentile(mom_norm, 50)),
                "mom_P90": float(np.percentile(mom_norm, 90)),
                "mom_P95": float(np.percentile(mom_norm, 95)),
                "mom_mean": float(np.mean(mom_norm)),
                "mom_std": float(np.std(mom_norm)),
            }
        )
    else:
        summary.update({"mom_P50": math.nan, "mom_P90": math.nan, "mom_P95": math.nan, "mom_mean": math.nan, "mom_std": math.nan})

    summary["KS_div"] = _ks_distance(np.abs(div_pred), np.abs(div_true)) if div_pred.size and div_true.size else float("nan")
    summary["KS_mom"] = _ks_distance(mom_norm, np.zeros_like(mom_norm)) if mom_norm.size else float("nan")
    summary["n_points"] = float(n_points)
    summary["knn_k"] = float(k_neighbors)
    summary["m_samples"] = float(m_samples)
    summary["rho"] = float(rho)
    summary["nu_eff"] = float(nu_eff)
    summary["mls_params"] = json.dumps({"m_samples": m_samples, "k_neighbors": k_neighbors})
    return summary


def _save_histograms(out_dir: Path, tag: str, div_pred: np.ndarray, div_true: np.ndarray, mom_norm: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    div_linear = out_dir / f"{tag}_divergence.png"
    div_log = out_dir / f"{tag}_divergence_log.png"
    mom_log = out_dir / f"{tag}_momentum.png"

    fig = plt.figure(figsize=(9, 4), dpi=150, constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax1.hist(div_true, bins=60, alpha=0.75, label="True div")
    ax1.hist(div_pred, bins=60, alpha=0.75, label="Pred div")
    ax1.set_title("Divergence (linear)")
    ax1.set_xlabel("div")
    ax1.set_ylabel("count")
    ax1.legend()
    ax2.hist(np.abs(div_true), bins=60, alpha=0.75, label="|True div|")
    ax2.hist(np.abs(div_pred), bins=60, alpha=0.75, label="|Pred div|")
    ax2.set_title("|Divergence| (log y)")
    ax2.set_xlabel("|div|")
    ax2.set_ylabel("count")
    ax2.set_yscale("log")
    ax2.legend()
    fig.savefig(div_linear, bbox_inches="tight")
    fig.savefig(div_log, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(4.8, 4), dpi=150, constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.hist(mom_norm, bins=60, alpha=0.85, label="||Momentum residual||")
    ax.set_yscale("log")
    ax.set_title("Momentum residual norm (log y)")
    ax.set_xlabel("||r||")
    ax.set_ylabel("count")
    ax.legend()
    fig.savefig(mom_log, bbox_inches="tight")
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate TPAC physics metrics")
    parser.add_argument("--cfg", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--pts", type=int, default=16384)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--ckpt-dir", dest="ckpt_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--m-samples", dest="m_samples", type=int, default=4096)
    parser.add_argument("--k", dest="k_neighbors", type=int, default=24)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--nu-eff", dest="nu_eff", type=float, default=0.0)
    parser.add_argument("--max-samples", dest="max_samples", type=int, default=-1,
                        help="Limit the number of dataset samples evaluated (default: all)")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--hist-dir", type=Path, help="Optional directory to store histograms per checkpoint")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    ckpt_paths: List[Path] = []
    if args.ckpt:
        ckpt_paths.append(args.ckpt)
    if args.ckpt_dir:
        ckpt_paths.extend(sorted(p for p in args.ckpt_dir.glob("*.pth")))
    if not ckpt_paths:
        raise ValueError("No checkpoint specified. Use --ckpt or --ckpt-dir.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device_str = args.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    cfg = _load_config(args.cfg)
    dataset, _ = _prepare_dataset(args.root, args.case, args.split, args.pts)
    model = _build_model_from_cfg(cfg, device)

    summary_rows: List[Dict[str, float]] = []
    hist_dir = args.hist_dir

    for ckpt_path in ckpt_paths:
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)

        div_pred, div_true, mom_norm, n_points = _collect_physics_arrays(
            model,
            dataset,
            device,
            m_samples=args.m_samples,
            k_neighbors=args.k_neighbors,
            rho=args.rho,
            nu_eff=args.nu_eff,
            max_items=args.max_samples if args.max_samples > 0 else None,
        )
        summary = _summarise_arrays(
            div_pred,
            div_true,
            mom_norm,
            k_neighbors=args.k_neighbors,
            m_samples=args.m_samples,
            rho=args.rho,
            nu_eff=args.nu_eff,
            n_points=n_points,
        )
        summary.update({
            "ckpt_path": str(ckpt_path),
            "split": args.split,
        })
        summary_rows.append(summary)

        if hist_dir is not None:
            tag = ckpt_path.stem
            _save_histograms(hist_dir, tag, div_pred, div_true, mom_norm)

    df_summary = pd.DataFrame(summary_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(args.out, index=False)
    print(f"[OK] Physics summary saved -> {args.out}")


if __name__ == "__main__":
    main()
