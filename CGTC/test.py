# -*- coding: utf-8 -*-
"""
test.py 
"""

from __future__ import annotations

import argparse
import pickle
import re
import inspect
from types import SimpleNamespace as NS
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List, Sequence

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.gridspec import GridSpec

# ===== 内部依赖 =====
try:  # pragma: no cover
    from Rotate.models.backbone import build_backbone
except ModuleNotFoundError:  # pragma: no cover
    from models.backbone import build_backbone

from training.schedules import base_lambda_cont as _sched_base_lc
from training.schedules import current_alpha as _sched_alpha
from training.utils import ensure_dir

# 物理评估位置兼容
try:
    from eval.eval_physics import compute_physics_on_dataset
except Exception:
    from eval_physics import compute_physics_on_dataset

from data.dataset import pointdata, norm_data, build_out_csv_from_dir


# ----------------- 小工具：Namespace <-> dict -----------------
def dict2ns(d):
    if isinstance(d, dict):
        return NS(**{k: dict2ns(v) for k, v in d.items()})
    return d


def ns2dict(x):
    if isinstance(x, dict):
        return {k: ns2dict(v) for k, v in x.items()}
    if isinstance(x, NS):
        return {k: ns2dict(getattr(x, k)) for k in vars(x)}
    return x


# ----------------- YAML -----------------
def load_cfg(yaml_path: Optional[str]) -> Any:
    import yaml
    if yaml_path is None:
        # 可运行的默认配置（与训练默认等价）
        d = {
            "train": {"epochs": 300, "lr": 1e-4, "weight_decay": 1e-4, "eta_min": 1e-5,
                      "grad_clip": 1.0, "device": "cuda:0"},
            "loss": {"channel_weights": [1.0, 0.5, 0.5, 0.5], "phys_alpha": 0.5},
            "curriculum": {"split": 0.8},
            "mixed": {"use": True, "alpha_start": 1.0, "alpha_end": 0.80,
                      "decay_start": 1, "decay_epochs": 300, "schedule": "cosine",
                      "wall_sigma": 0.10},
            "pinn": {"use": True, "samples": 4096, "k": 20, "rho": 1.0, "nu_eff": 0.003,
                     "lambda_mom": 5e-4, "warmup": 120, "ramp": 60,
                     "lmax": 0.02, "lmin": 3e-4, "schedule": "cosine", "mode": "down",
                     "adapt": True, "lcont_min_mult": 0.5, "lcont_max_mult": 1.8,
                     "lcont_mult_init": 1.0, "up_rate": 1.05, "down_rate": 0.985,
                     "div_target": 0.05, "div_tol": 0.20, "adapt_after": 121, "adapt_interval": 3,
                     "auto_calib": True, "calib_mult": 0.6},
            "teacher": {"use": True, "ema_beta": 0.995, "max_w": 0.20, "decay_eps": 200, "use_best_warm": True},
            "p_grad": {"release_epoch": 130, "scale": 0.05},
            "early": {"patience": 15, "delta": 1.0e-4},
            "save": {"constrained": True, "ok_margin": 0.40, "consec_ok_epochs": 2, "metric": "val_loss"},
            "models": {
                "backbone": {
                    "name": "pointnet2pp_ssg",
                    "args": {
                        "out_channels": 4,
                        "p_drop": 0.3,
                        "npoint1": 2048, "npoint2": 512, "npoint3": 128,
                        "k1": 16, "k2": 16, "k3": 16,
                        "c1": 128, "c2": 256, "c3": 512,
                        "groups_gn": 32, "use_pe": False, "use_msg": False, "in_ch0": 0,
                        "p_head_ch": [128, 64], "uvw_head_ch": [128, 64],
                        "use_graph_refine": True, "refine_k": 16, "refine_layers": 2,
                        "refine_hidden": 128, "refine_residual": True
                    }
                }
            }
        }
        return dict2ns(d)

    with open(yaml_path, "r") as f:
        return dict2ns(yaml.safe_load(f))


# ----------------- 路径与数据准备 -----------------
def build_paths(root: Path, case: str, tag: str) -> Dict[str, Path]:
    tmp_root = root / "results" / "temp_results" / case / tag
    data_root = root / "data" / "dataset" / case
    paths = {
        "test_txt": data_root / "test",
        "tmp_test": tmp_root / "dataset_temp_results" / "test",
        "val_csv":  tmp_root / "dataset_temp_results" / "test" / "out_test.csv",
        "val_pkl":  tmp_root / "dataset_temp_results" / "test" / "test_data.pkl",
        "weight_dir": tmp_root / "weight",
        "curve_dir":  tmp_root / "curves",

        # legacy single-sample outputs
        "pred_img":   tmp_root / "pred_vs_real_puvw.png",
        "pinn_txt":   tmp_root / "pred_vs_real_puvw.txt",

        # physics outputs
        "div_png":    tmp_root / "div_hist.png",
        "mom_png":    tmp_root / "mom_hist.png",
        "phys_csv":   tmp_root / "phys_stats.csv",

        # legacy triplets dir
        "triplet_dir": tmp_root / "triplets",

        # new: per-index deterministic visualization root
        "samples_vis_dir": tmp_root / "samples_vis",
    }
    return {k: Path(v) for k, v in paths.items()}


def ensure_test_info(txt_dir: Path, pkl_path: Path):
    if not pkl_path.exists():
        min_val, max_val, num_data, num_points = norm_data(str(txt_dir))
        info = {"input_min": min_val, "input_max": max_val, "num_data": num_data, "num_points": num_points}
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(info, open(pkl_path, "wb"))


def load_info(pkl_path: Path) -> Dict[str, Any]:
    return pickle.load(open(pkl_path, "rb"))


# ----------------- PINN 归一化配置（对齐训练侧口径） -----------------
def extract_pinn_norm_cfg(info: Dict[str, Any], cfg) -> Dict[str, Any]:
    """
    基于训练侧 _extract_norm_cfg 逻辑的轻量复刻：
    从 data.pkl 的 input_min/max 中取：
      - x_min/x_max: 3D 坐标
      - y_min/y_max: 4 通道物理量 (p,u,v,w)
    """
    base = {
        "rho": float(getattr(cfg.pinn, "rho", 1.0)),
        "nu_eff": float(getattr(cfg.pinn, "nu_eff", 0.0)),
    }
    if not info:
        return base

    inp_min = info.get("input_min")
    inp_max = info.get("input_max")
    if inp_min is None or inp_max is None:
        return base

    inp_min = [float(x) for x in inp_min]
    inp_max = [float(x) for x in inp_max]

    if len(inp_min) >= 7 and len(inp_max) >= 7:
        base.update({
            "x_min": inp_min[0:3],
            "x_max": inp_max[0:3],
            "y_min": inp_min[3:7],
            "y_max": inp_max[3:7],
        })
    return base


def resolve_pinn_norm_mode(cfg, pinn_norm_cfg: Optional[Dict[str, Any]]) -> str:
    """
    训练侧默认：有 norm_cfg 时倾向 denorm_physical。
    这里尽量模仿该决策。
    """
    cfg_mode = getattr(getattr(cfg, "pinn", NS()), "norm_mode", None)
    if cfg_mode is not None:
        return str(cfg_mode).lower()

    has_xy = False
    if pinn_norm_cfg:
        has_xy = all(k in pinn_norm_cfg for k in ("x_min", "x_max", "y_min", "y_max"))
    return "denorm_physical" if has_xy else "none"


# ----------------- 权重/模型工具 -----------------
def strip_module_prefix(state_dict: dict) -> dict:
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def model_param_count(m: nn.Module) -> str:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return f"params: {total/1e6:.2f}M (trainable {trainable/1e6:.2f}M)"


# ----------------- 从 ckpt 名推断 epoch -> λ_cont 与 α(t) -----------------
def parse_epoch_from_ckpt_name(p: Path) -> Optional[int]:
    m = re.search(r"_ep(\d+)\.pth$", p.name)
    if m:
        return int(m.group(1))
    m2 = re.search(r"epoch(\d+)", p.name)
    return int(m2.group(1)) if m2 else None


def sched_lambda_cont_from_ckpt_epoch(ep: Optional[int], cfg) -> float:
    if not getattr(cfg.pinn, "use", False):
        return 0.0
    if ep is None:
        return float(cfg.pinn.lmin) if str(cfg.pinn.mode).lower() == "down" else float(cfg.pinn.lmax)
    return _sched_base_lc(
        ep, cfg.pinn.use, cfg.pinn.warmup, cfg.pinn.ramp,
        cfg.pinn.lmin, cfg.pinn.lmax, cfg.pinn.schedule, cfg.pinn.mode
    )


def sched_alpha_from_ckpt_epoch(ep: Optional[int], cfg) -> float:
    if not getattr(cfg.mixed, "use", False):
        return 1.0
    if ep is None:
        return float(cfg.mixed.alpha_end)
    return _sched_alpha(
        ep, cfg.mixed.use, cfg.mixed.alpha_start, cfg.mixed.alpha_end,
        cfg.mixed.decay_start, cfg.mixed.decay_epochs, cfg.mixed.schedule
    )


# ----------------- 可视化工具 -----------------
def _set_equal_3d(ax, X, Y, Z):
    rng = max(X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min())
    xm, ym, zm = (X.max() + X.min()) / 2, (Y.max() + Y.min()) / 2, (Z.max() + Z.min()) / 2
    r = 0.5 * rng
    ax.set_xlim(xm - r, xm + r)
    ax.set_ylim(ym - r, ym + r)
    ax.set_zlim(zm - r, zm + r)


def _get_field(y_np, field):
    idx_map = {'p': 0, 'u': 1, 'v': 2, 'w': 3}
    return y_np[idx_map[field]]


def plot_fields_grid(
    XYZ, y_true, y_pred, save_path: Path,
    fields=('p', 'u', 'v', 'w'), s=14, cmap_name='coolwarm',
    max_points_plot: Optional[int] = 50000
):
    """
    单样本 p/uvw 网格图（左真实，右预测），每个通道一行。
    """
    x, y, z = XYZ[0].copy(), XYZ[1].copy(), XYZ[2].copy()
    y_true = y_true.copy()
    y_pred = y_pred.copy()

    if max_points_plot is not None:
        N = x.shape[0]
        Np = min(N, int(max_points_plot))
        if Np < N:
            idx = np.random.choice(N, Np, replace=False)
            x, y, z = x[idx], y[idx], z[idx]
            y_true = y_true[:, idx]
            y_pred = y_pred[:, idx]

    n_rows = len(fields)
    fig = plt.figure(figsize=(10, 2.8 * n_rows), dpi=400, constrained_layout=True)
    gs = GridSpec(n_rows, 3, figure=fig, width_ratios=[1, 1, 0.05], wspace=0.05, hspace=0.05)
    cmap = plt.colormaps.get_cmap(cmap_name)

    for r, fld in enumerate(fields):
        c_true, c_pred = _get_field(y_true, fld), _get_field(y_pred, fld)
        vmin, vmax = float(min(c_true.min(), c_pred.min())), float(max(c_true.max(), c_pred.max()))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        axL = fig.add_subplot(gs[r, 0], projection='3d')
        axL.scatter(x, y, z, c=c_true, s=s, marker='.', lw=0, cmap=cmap, norm=norm)
        axL.set_title(f"{fld.upper()} • Real", pad=6)
        axL.set_axis_off()
        _set_equal_3d(axL, x, y, z)

        axR = fig.add_subplot(gs[r, 1], projection='3d')
        axR.scatter(x, y, z, c=c_pred, s=s, marker='.', lw=0, cmap=cmap, norm=norm)
        axR.set_title(f"{fld.upper()} • Predicted", pad=6)
        axR.set_axis_off()
        _set_equal_3d(axR, x, y, z)

        cax = fig.add_subplot(gs[r, 2])
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        plt.colorbar(sm, cax=cax).set_label(fld.upper())

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 保存图像 → {save_path}")


def save_raw_arrays_for_vis(
    XYZ_np: np.ndarray,
    Y_np: np.ndarray,
    pred_b4n: np.ndarray,
    out_dir: Path,
    idx: int,
) -> None:
    """
    保存参与可视化的原始点云数据（用于后续统一色阶绘图）：

    - NPZ:  out_dir / vis_raw_idxXXXX.npz
      * idx: int
      * xyz: (N, 3)
      * y_label: (N, 4)   # p,u,v,w
      * y_pred:  (N, 4)
      * y_err:   (N, 4)   # |pred - label|
      * fields: ["p","u","v","w"]

    - CSV:  out_dir / vis_raw_idxXXXX.csv
      列顺序:
      x,y,z,
      p_label,u_label,v_label,w_label,
      p_pred,u_pred,v_pred,w_pred,
      p_err,u_err,v_err,w_err
    """
    ensure_dir(out_dir)

    # 转成 (N, C) 方便后续处理
    xyz = np.asarray(XYZ_np, dtype=np.float32).T   # (N, 3)
    y_label = np.asarray(Y_np, dtype=np.float32).T  # (N, 4)
    y_pred = np.asarray(pred_b4n, dtype=np.float32).T  # (N, 4)
    y_err = np.abs(y_pred - y_label)  # (N, 4)

    npz_path = out_dir / f"vis_raw_idx{idx:04d}.npz"
    np.savez_compressed(
        npz_path,
        idx=np.int32(idx),
        xyz=xyz,
        y_label=y_label,
        y_pred=y_pred,
        y_err=y_err,
        fields=np.array(["p", "u", "v", "w"], dtype="U1"),
    )

    csv_path = out_dir / f"vis_raw_idx{idx:04d}.csv"
    stacked = np.concatenate([xyz, y_label, y_pred, y_err], axis=1)  # (N, 3+4+4+4)
    header_cols = [
        "x", "y", "z",
        "p_label", "u_label", "v_label", "w_label",
        "p_pred", "u_pred", "v_pred", "w_pred",
        "p_err", "u_err", "v_err", "w_err",
    ]
    np.savetxt(
        csv_path,
        stacked,
        delimiter=",",
        header=",".join(header_cols),
        comments="",
    )

    print(f"[OK] 保存可视化原始点云 → {npz_path}")
    print(f"[OK] 保存可视化原始点云 CSV → {csv_path}")


# ----------------- 前向封装（适配 TPAC 输出 dict -> (B,4,N)） -----------------
@torch.no_grad()
def _stack_preds(preds_dict: dict) -> torch.Tensor:
    p = preds_dict["p"].unsqueeze(1)
    u = preds_dict["u"].unsqueeze(1)
    v = preds_dict["v"].unsqueeze(1)
    w = preds_dict["w"].unsqueeze(1)
    return torch.cat([p, u, v, w], dim=1).contiguous()


@torch.no_grad()
def forward_backbone_to_b4n(backbone, x_b3n: torch.Tensor) -> torch.Tensor:
    x_bnc = x_b3n.transpose(1, 2).contiguous()
    preds_dict = backbone(x_bnc)
    return _stack_preds(preds_dict)  # (B,4,N)


# ----------------- PINN 类导入与安全构建 -----------------
def import_pinn_class():
    try:
        from Rotate.models.tpac_pinn import SteadyIncompressiblePINN
        return SteadyIncompressiblePINN
    except Exception:
        pass
    try:
        from models.tpac_pinn import SteadyIncompressiblePINN
        return SteadyIncompressiblePINN
    except Exception:
        pass
    try:
        from TPAC.models.tpac_pinn import SteadyIncompressiblePINN  # type: ignore
        return SteadyIncompressiblePINN
    except Exception as e:
        raise ImportError("Cannot import SteadyIncompressiblePINN from known paths.") from e


def build_pinn_for_eval(
    cfg,
    lambda_cont_now: float,
    alpha_now: float,
    pinn_norm_mode: str = "none",
    pinn_norm_cfg: Optional[Dict[str, Any]] = None,
):
    """
    尝试以“新签名”构建（含 norm_mode/norm_cfg），
    若当前项目中类版本不支持则自动降级。
    """
    SteadyIncompressiblePINN = import_pinn_class()

    kwargs = dict(
        k_neighbors=int(cfg.pinn.k),
        m_samples=int(cfg.pinn.samples),
        lambda_cont=float(lambda_cont_now),
        lambda_mom=float(cfg.pinn.lambda_mom),
        rho=float(cfg.pinn.rho),
        nu_eff=float(cfg.pinn.nu_eff),
        residual_weight_mode=("mixed" if cfg.mixed.use else "uniform"),
        residual_alpha=(float(alpha_now) if cfg.mixed.use else None),
        residual_sigma=float(cfg.mixed.wall_sigma),
    )

    if pinn_norm_mode:
        kwargs["norm_mode"] = str(pinn_norm_mode)
    if pinn_norm_cfg:
        kwargs["norm_cfg"] = pinn_norm_cfg

    try:
        return SteadyIncompressiblePINN(**kwargs)
    except TypeError:
        kwargs.pop("norm_mode", None)
        kwargs.pop("norm_cfg", None)
        return SteadyIncompressiblePINN(**kwargs)


# ----------------- 单样本/指定样本：可视化 + PINN 指标 -----------------
def vis_and_pinn_for_index(
    model,
    dataset,
    device,
    out_dir: Path,
    idx: int,
    lambda_cont_now: float,
    alpha_now: float,
    cfg,
    pinn_norm_mode: str = "none",
    pinn_norm_cfg: Optional[Dict[str, Any]] = None,
):
    """
    对指定 idx：
      - 生成 pred_vs_real_puvw.png
      - 生成 pinn_summary.txt
      - 保存该样本的原始点云 / label / pred / error（npz + csv）
    """
    model.eval()
    XYZ, Y = dataset[idx]

    XYZ_np = XYZ.detach().cpu().numpy() if torch.is_tensor(XYZ) else np.asarray(XYZ)
    Y_np = Y.detach().cpu().numpy() if torch.is_tensor(Y) else np.asarray(Y)

    X = torch.from_numpy(XYZ_np).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1,3,N)
    with torch.no_grad():
        pred_b4n = forward_backbone_to_b4n(model, X).detach().cpu().numpy()[0].astype(np.float32)

    # ✅ 保存可视化所用的原始数据（统一色阶用）
    save_raw_arrays_for_vis(
        XYZ_np=XYZ_np,
        Y_np=Y_np,
        pred_b4n=pred_b4n,
        out_dir=out_dir,
        idx=idx,
    )

    ensure_dir(out_dir)
    img_path = out_dir / "pred_vs_real_puvw.png"
    txt_path = out_dir / "pinn_summary.txt"

    plot_fields_grid(XYZ_np, Y_np, pred_b4n, img_path)

    lines = []
    if getattr(cfg.pinn, "use", False) and lambda_cont_now > 0.0:
        pinn = build_pinn_for_eval(
            cfg,
            lambda_cont_now=lambda_cont_now,
            alpha_now=alpha_now,
            pinn_norm_mode=pinn_norm_mode,
            pinn_norm_cfg=pinn_norm_cfg,
        )

        y_pred = torch.from_numpy(pred_b4n).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1,4,N)
        with torch.no_grad():
            out = pinn(X, y_pred)

        loss_pinn = float(out.get("loss_pinn", torch.tensor(0.)).item())
        loss_cont_w = float(out.get("loss_cont", torch.tensor(0.)).item())
        loss_mom_w = float(out.get("loss_mom", torch.tensor(0.)).item())
        cont_raw = float(out.get("cont_raw_mean", torch.tensor(0.)).item())
        mom_raw = float(out.get("mom_raw_mean", torch.tensor(0.)).item())

        lines.append(
            f"[PINN] sample#{idx} | λ_cont(base)={lambda_cont_now:.6g} | α={alpha_now:.3f} "
            f"| norm_mode={pinn_norm_mode}"
        )
        lines.append(f"  weighted : cont={loss_cont_w:.6e}, mom={loss_mom_w:.6e}, pinn={loss_pinn:.6e}")
        lines.append(f"  raw(mean): cont={cont_raw:.6e}, mom={mom_raw:.6e}")
    else:
        lines.append(
            f"[PINN] sample#{idx} | λ_cont(base)={lambda_cont_now:.6g} (未启用或=0) "
            f"| α={alpha_now:.3f} | norm_mode={pinn_norm_mode}"
        )

    msg = "\n".join(lines)
    with open(str(txt_path), "w") as f:
        f.write(msg + "\n")
    print(msg)
    print(f"[OK] PINN 指标保存 → {txt_path}")


# ----------------- 三联图（Label / Pred / Error） -----------------
def _save_triplet_and_singles(
    XYZ_np: np.ndarray,
    label_vals: np.ndarray,
    pred_vals: np.ndarray,
    field_name: str,
    out_dir: Path,
    sample_label: str,
    dpi: int = 400,
    cmap_name: str = "coolwarm",
    s: float = 4.0
):
    """
    为单个通道保存：
      - 三联图：Label / Pred / Error
      - 三张子图：Label-only, Pred-only, Error-only
    共 4 张图。
    """
    ensure_dir(out_dir)
    x, y, z = XYZ_np[0], XYZ_np[1], XYZ_np[2]
    err_vals = np.abs(pred_vals - label_vals)

    vmin = float(min(label_vals.min(), pred_vals.min()))
    vmax = float(max(label_vals.max(), pred_vals.max()))
    emin = float(err_vals.min())
    emax = float(err_vals.max())

    cmap = plt.colormaps.get_cmap(cmap_name)

    # ---- 三联图 ----
    fig = plt.figure(figsize=(18, 6), dpi=dpi)
    ax1 = fig.add_subplot(131, projection='3d')
    sc1 = ax1.scatter(x, y, z, c=label_vals, s=s, cmap=cmap)
    sc1.set_clim(vmin, vmax)
    ax1.set_title(f"{field_name.upper()} Label")
    ax1.set_axis_off()
    _set_equal_3d(ax1, x, y, z)
    cb1 = fig.colorbar(sc1, ax=ax1, fraction=0.03, pad=0.05)
    cb1.set_label(field_name.upper())

    ax2 = fig.add_subplot(132, projection='3d')
    sc2 = ax2.scatter(x, y, z, c=pred_vals, s=s, cmap=cmap)
    sc2.set_clim(vmin, vmax)
    ax2.set_title(f"{field_name.upper()} Pred")
    ax2.set_axis_off()
    _set_equal_3d(ax2, x, y, z)
    cb2 = fig.colorbar(sc2, ax=ax2, fraction=0.03, pad=0.05)
    cb2.set_label(field_name.upper())

    ax3 = fig.add_subplot(133, projection='3d')
    sc3 = ax3.scatter(x, y, z, c=err_vals, s=s, cmap=cmap)
    sc3.set_clim(emin, emax)
    ax3.set_title(f"{field_name.upper()} Error = |Pred-Label|")
    ax3.set_axis_off()
    _set_equal_3d(ax3, x, y, z)
    cb3 = fig.colorbar(sc3, ax=ax3, fraction=0.03, pad=0.05)
    cb3.set_label(f"|Δ{field_name.upper()}|")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    triplet_path = out_dir / f"{sample_label}_{field_name}_triplet.png"
    fig.savefig(triplet_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # ---- Label-only ----
    figL = plt.figure(figsize=(6, 6), dpi=dpi)
    axL = figL.add_subplot(111, projection='3d')
    scL = axL.scatter(x, y, z, c=label_vals, s=s, cmap=cmap)
    scL.set_clim(vmin, vmax)
    axL.set_title(f"{field_name.upper()} Label")
    axL.set_axis_off()
    _set_equal_3d(axL, x, y, z)
    cbL = figL.colorbar(scL, fraction=0.03, pad=0.05)
    cbL.set_label(field_name.upper())
    label_path = out_dir / f"{sample_label}_{field_name}_label.png"
    figL.savefig(label_path, dpi=dpi, bbox_inches="tight")
    plt.close(figL)

    # ---- Pred-only ----
    figP = plt.figure(figsize=(6, 6), dpi=dpi)
    axP = figP.add_subplot(111, projection='3d')
    scP = axP.scatter(x, y, z, c=pred_vals, s=s, cmap=cmap)
    scP.set_clim(vmin, vmax)
    axP.set_title(f"{field_name.upper()} Pred")
    axP.set_axis_off()
    _set_equal_3d(axP, x, y, z)
    cbP = figP.colorbar(scP, fraction=0.03, pad=0.05)
    cbP.set_label(field_name.upper())
    pred_path = out_dir / f"{sample_label}_{field_name}_pred.png"
    figP.savefig(pred_path, dpi=dpi, bbox_inches="tight")
    plt.close(figP)

    # ---- Error-only ----
    figE = plt.figure(figsize=(6, 6), dpi=dpi)
    axE = figE.add_subplot(111, projection='3d')
    scE = axE.scatter(x, y, z, c=err_vals, s=s, cmap=cmap)
    scE.set_clim(emin, emax)
    axE.set_title(f"{field_name.upper()} Error")
    axE.set_axis_off()
    _set_equal_3d(axE, x, y, z)
    cbE = figE.colorbar(scE, fraction=0.03, pad=0.05)
    cbE.set_label(f"|Δ{field_name.upper()}|")
    error_path = out_dir / f"{sample_label}_{field_name}_error.png"
    figE.savefig(error_path, dpi=dpi, bbox_inches="tight")
    plt.close(figE)


def draw_triplets_for_index(
    model,
    dataset,
    device,
    out_dir: Path,
    idx: int,
    fields=('p', 'u', 'v', 'w'),
    dpi: int = 400,
    cmap_name: str = "coolwarm",
    s: float = 4.0
):
    """
    对指定 idx 绘制四通道 triplet + 单图，输出到 out_dir。
    （原始点云的保存由 vis_and_pinn_for_index 完成）
    """
    ensure_dir(out_dir)
    model.eval()

    XYZ, Y = dataset[idx]
    XYZ_np = XYZ.detach().cpu().numpy() if torch.is_tensor(XYZ) else np.asarray(XYZ)
    Y_np = Y.detach().cpu().numpy() if torch.is_tensor(Y) else np.asarray(Y)

    X = torch.from_numpy(XYZ_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        pred_b4n = forward_backbone_to_b4n(model, X).detach().cpu().numpy()[0].astype(np.float32)

    idx_map = {'p': 0, 'u': 1, 'v': 2, 'w': 3}
    sample_label = f"idx{idx:04d}"

    for fld in fields:
        ch_idx = idx_map[fld]
        label_vals = Y_np[ch_idx]
        pred_vals = pred_b4n[ch_idx]
        _save_triplet_and_singles(
            XYZ_np, label_vals, pred_vals,
            field_name=fld,
            out_dir=out_dir,
            sample_label=sample_label,
            dpi=dpi,
            cmap_name=cmap_name,
            s=s
        )

    print(f"[triplet] sample#{idx} → {out_dir}")


# ----------------- 主流程 -----------------
def parse_indices(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            continue
    return out if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--root",   type=str, required=True, help="项目根目录（含 data/ 和 results/）")
    ap.add_argument("--case",   type=str, required=True, help="如 C1 / ICA_norm")
    ap.add_argument("--tag",    type=str, required=True, help="与训练时使用的 tag 一致，如 1205_01_xxx")
    ap.add_argument("--ckpt",   type=str, default=None,  help="权重路径；不提供则使用 ROOT/results/.../final_reco.pth")
    ap.add_argument("--cfg",    type=str, default=None,  help="YAML 配置文件路径（若不提供则使用内置默认）")
    ap.add_argument("--pts",    type=int, default=16384, help="测试采样点数")
    ap.add_argument("--force_lambda_cont", type=float, default=None)
    ap.add_argument("--force_alpha",       type=float, default=None)

    # ✅ 新增：物理评估样本数控制
    ap.add_argument("--phys_max_items", type=int, default=6,
                    help="物理评估最多样本数；-1 表示全量 test")

    # ✅ 可选：固定可追溯样本可视化
    ap.add_argument("--vis_indices", type=str, default=None,
                    help="可视化样本索引列表（逗号分隔），如 0,5,12；提供后将按固定索引输出到 samples_vis/idx_XXXX/")
    ap.add_argument("--vis_num_random", type=int, default=1,
                    help="未提供 --vis_indices 时，随机可视化样本个数（默认 1）")

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    cfg = load_cfg(args.cfg)

    # 构建路径
    ROOT = Path(args.root)
    paths = build_paths(ROOT, args.case, args.tag)

    # 选择权重
    ckpt_path = Path(args.ckpt) if args.ckpt is not None else (paths["weight_dir"] / "final_reco.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"[ERR] 权重不存在: {ckpt_path}")

    # 准备 test 的 out.csv / data.pkl
    ensure_dir(paths["tmp_test"])
    if not paths["val_csv"].exists():
        build_out_csv_from_dir(paths["test_txt"], paths["val_csv"])
    ensure_test_info(paths["test_txt"], paths["val_pkl"])
    data_info = load_info(paths["val_pkl"])

    # ===== PINN 归一化对齐训练 =====
    pinn_norm_cfg = extract_pinn_norm_cfg(data_info, cfg)
    pinn_norm_mode = resolve_pinn_norm_mode(cfg, pinn_norm_cfg)
    print(f"[INFO] PINN norm_mode={pinn_norm_mode} | norm_cfg keys={list(pinn_norm_cfg.keys())}")

    # ===== 构建模型并加载权重 =====
    if hasattr(cfg, "models") and hasattr(cfg.models, "backbone"):
        mcfg = NS()
        mcfg.models = NS()
        bb_name = getattr(cfg.models.backbone, "name", "pointnet2pp_ssg")
        bb_args = getattr(cfg.models.backbone, "args", {})
        bb_args = ns2dict(bb_args)
        mcfg.models.backbone = NS(name=bb_name, args=bb_args)
    else:
        mcfg = NS()
        mcfg.models = NS()
        mcfg.models.backbone = NS(
            name="pointnet2pp_ssg",
            args=dict(
                out_channels=4, p_drop=0.3,
                npoint1=2048, npoint2=512, npoint3=128,
                k1=16, k2=16, k3=16,
                c1=128, c2=256, c3=512,
                groups_gn=32, use_pe=False, use_msg=False, in_ch0=0,
                p_head_ch=(128, 64), uvw_head_ch=(128, 64),
                use_graph_refine=True, refine_k=16, refine_layers=2,
                refine_hidden=128, refine_residual=True
            )
        )

    model = build_backbone(mcfg).to(device)
    print(f"[INFO] {model_param_count(model)}")

    # 加载 state_dict（兼容 DataParallel 保存）
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = strip_module_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > 0:
        print("[WARN] missing keys:", missing)
    if len(unexpected) > 0:
        print("[WARN] unexpected keys:", unexpected)

    POINTS_PER_SAMPLE = int(args.pts)

    # ===== 可视化用 dataset：只返回 (x,y) 的 numpy 视图 =====
    class _TestView:
        def __init__(self, txt_dir: str, csv_path: str, info: dict, pts: int):
            self.inner = pointdata(txt_dir, csv_path, info, pts)

        def __len__(self):
            return len(self.inner)

        def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
            out = self.inner[i]   # 期望 (3,N), (4,N) (+ stats)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                x, y = out[0], out[1]
            else:
                raise RuntimeError("pointdata must return (x,y,stats)")
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            if torch.is_tensor(y):
                y = y.detach().cpu().numpy()
            return x, y

    dataset_vis = _TestView(str(paths["test_txt"]), str(paths["val_csv"]), data_info, POINTS_PER_SAMPLE)

    # ===== 从 ckpt 文件名解析 epoch -> 对齐训练侧 λ_cont(base) 与 α(t) =====
    ep = parse_epoch_from_ckpt_name(ckpt_path.resolve())
    lambda_cont_now = sched_lambda_cont_from_ckpt_epoch(ep, cfg)
    alpha_now = sched_alpha_from_ckpt_epoch(ep, cfg)

    if args.force_lambda_cont is not None:
        lambda_cont_now = float(args.force_lambda_cont)
    if args.force_alpha is not None:
        alpha_now = float(args.force_alpha)

    print(f"[INFO] λ_cont(base)={lambda_cont_now:.6g} | α={alpha_now:.3f} | ckpt epoch={ep}")

    # ===== 可视化：固定 indices 或随机 =====
    vis_indices = parse_indices(args.vis_indices)

    if vis_indices is not None:
        # 过滤非法索引
        n_vis = len(dataset_vis)
        vis_indices = [i for i in vis_indices if 0 <= i < n_vis]
        if not vis_indices:
            print("[WARN] --vis_indices provided but none valid; fallback to random mode.")
            vis_indices = None

    if vis_indices is None:
        # ------- 兼容旧行为：随机单样本 + 随机 triplet -------
        idx = np.random.randint(0, len(dataset_vis))
        print(f"[INFO] Random vis sample idx={idx}")

        # legacy single-sample outputs + 保存原始数据
        vis_and_pinn_for_index(
            model, dataset_vis, device,
            out_dir=paths["pred_img"].parent,  # will save new names if same folder
            idx=idx,
            lambda_cont_now=lambda_cont_now,
            alpha_now=alpha_now,
            cfg=cfg,
            pinn_norm_mode=pinn_norm_mode,
            pinn_norm_cfg=pinn_norm_cfg,
        )

        # 为保持 legacy 文件名兼容：复制/重存一份到老路径名
        try:
            XYZ, Y = dataset_vis[idx]
            XYZ_np = np.asarray(XYZ)
            Y_np = np.asarray(Y)
            X = torch.from_numpy(XYZ_np).unsqueeze(0).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                pred_b4n = forward_backbone_to_b4n(model, X).detach().cpu().numpy()[0].astype(np.float32)
            plot_fields_grid(XYZ_np, Y_np, pred_b4n, paths["pred_img"])
            with open(str(paths["pinn_txt"]), "w") as f:
                f.write(f"[INFO] legacy summary for random idx={idx}\n")
        except Exception as e:
            print(f"[WARN] legacy single outputs failed: {e}")

        # legacy triplets
        ensure_dir(paths["triplet_dir"])
        num_rand = max(1, int(args.vis_num_random))
        for _ in range(num_rand):
            ridx = np.random.randint(0, len(dataset_vis))
            draw_triplets_for_index(
                model, dataset_vis, device,
                out_dir=paths["triplet_dir"],
                idx=ridx
            )
    else:
        # ------- 新行为：固定 indices，每个样本独立文件夹 -------
        ensure_dir(paths["samples_vis_dir"])
        print(f"[INFO] Fixed vis indices={vis_indices}")
        print(f"[INFO] samples_vis_dir={paths['samples_vis_dir']}")

        for idx in vis_indices:
            sample_dir = paths["samples_vis_dir"] / f"idx_{idx:04d}"
            trip_dir = sample_dir / "triplets"

            vis_and_pinn_for_index(
                model, dataset_vis, device,
                out_dir=sample_dir,
                idx=idx,
                lambda_cont_now=lambda_cont_now,
                alpha_now=alpha_now,
                cfg=cfg,
                pinn_norm_mode=pinn_norm_mode,
                pinn_norm_cfg=pinn_norm_cfg,
            )

            draw_triplets_for_index(
                model, dataset_vis, device,
                out_dir=trip_dir,
                idx=idx
            )

    # ===== 批量物理评估：用原始 pointdata（带 stats） =====
    dataset_phys = pointdata(
        str(paths["test_txt"]),
        str(paths["val_csv"]),
        data_info,
        POINTS_PER_SAMPLE
    )

    # --- 解析 phys_max_items ---
    phys_max = int(args.phys_max_items)
    if phys_max < 0:
        phys_max = len(dataset_phys)
    else:
        phys_max = min(phys_max, len(dataset_phys))

    print(f"[INFO] Physics eval max_items={phys_max} / test_size={len(dataset_phys)}")

    # 兼容不同版本的 compute_physics_on_dataset 签名：
    try:
        sig = inspect.signature(compute_physics_on_dataset)
        if "norm_mode" in sig.parameters or "norm_cfg" in sig.parameters:
            _overall, _df = compute_physics_on_dataset(
                model, dataset_phys, device,
                max_items=phys_max,
                m_samples=int(cfg.pinn.samples),
                k_neighbors=int(cfg.pinn.k),
                rho=float(cfg.pinn.rho),
                nu_eff=float(cfg.pinn.nu_eff),
                norm_mode=pinn_norm_mode,
                norm_cfg=pinn_norm_cfg,
                save_hist_png_div=paths["div_png"],
                save_hist_png_mom=paths["mom_png"],
                save_csv=paths["phys_csv"]
            )
        else:
            _overall, _df = compute_physics_on_dataset(
                model, dataset_phys, device,
                max_items=phys_max,
                m_samples=int(cfg.pinn.samples),
                k_neighbors=int(cfg.pinn.k),
                rho=float(cfg.pinn.rho),
                nu_eff=float(cfg.pinn.nu_eff),
                save_hist_png_div=paths["div_png"],
                save_hist_png_mom=paths["mom_png"],
                save_csv=paths["phys_csv"]
            )
    except Exception:
        _overall, _df = compute_physics_on_dataset(
            model, dataset_phys, device,
            max_items=phys_max,
            m_samples=int(cfg.pinn.samples),
            k_neighbors=int(cfg.pinn.k),
            rho=float(cfg.pinn.rho),
            nu_eff=float(cfg.pinn.nu_eff),
            save_hist_png_div=paths["div_png"],
            save_hist_png_mom=paths["mom_png"],
            save_csv=paths["phys_csv"]
        )

    print(
        "[OK] 评估完成：\n"
        f"  - samples_vis_dir(若启用固定样本): {paths['samples_vis_dir']}\n"
        f"  - legacy 单样本图: {paths['pred_img']}\n"
        f"  - legacy triplets: {paths['triplet_dir']}\n"
        f"  - div 直方图: {paths['div_png']}\n"
        f"  - 动量残差直方图: {paths['mom_png']}\n"
        f"  - 统计CSV: {paths['phys_csv']}\n"
        f"  - phys_max_items: {phys_max}"
    )


if __name__ == "__main__":
    main()
