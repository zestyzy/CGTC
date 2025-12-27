# -*- coding: utf-8 -*-
# TPAC/eval/posthoc_runner.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import pandas as pd

from eval.eval_physics import compute_physics_on_dataset
from training.losses import compute_weighted_losses, make_point_weights

@torch.no_grad()
def _forward_to_b4n_maybe_adapter(model, x_b3n: torch.Tensor) -> torch.Tensor:
    """
    统一前向到 (B,4,N)。兼容：
      - 经典骨干:  (B,3,N) -> (B,4,N)
      - TPAC 适配器: (B,N,3) -> dict -> (B,4,N)
    """
    # 尝试 TPAC 适配器路径
    try:
        out = model(x_b3n.transpose(1,2).contiguous())  # (B,N,3)
        if isinstance(out, dict) and all(k in out for k in ("p","u","v","w")):
            p = out["p"].unsqueeze(1)
            u = out["u"].unsqueeze(1)
            v = out["v"].unsqueeze(1)
            w = out["w"].unsqueeze(1)
            return torch.cat([p,u,v,w], dim=1).contiguous()
    except Exception:
        pass
    # 回退到经典路径
    out = model(x_b3n)
    assert out.dim()==3 and out.size(1)==4, "model must output (B,4,N) with [p,u,v,w]"
    return out

@torch.no_grad()
def _supervised_eval_epoch(model, dl, device: str, focus_mode: str="uniform",
                           channel_weights=(1,1,1,1), alpha: float|None=None, wall_sigma: float=0.08
                           ) -> Tuple[float, float]:
    """
    与 Trainer.evaluate 中监督部分对齐：输出 (mse_avg, mae_avg)
    """
    model.eval()
    mse_sum = mae_sum = 0.0
    n = 0
    for x, labels in dl:
        x = x.to(device, dtype=torch.float32)          # (B,3,N)
        labels = labels.to(device, dtype=torch.float32)  # (B,4,N)
        pred = _forward_to_b4n_maybe_adapter(model, x) # (B,4,N)
        mse, mae, _ = compute_weighted_losses(
            pred, labels, x, focus_mode, channel_weights, alpha=alpha, wall_sigma=wall_sigma
        )
        mse_sum += float(mse); mae_sum += float(mae); n += 1
    n = max(1, n)
    return mse_sum/n, mae_sum/n

def run_posthoc(
    model,
    dl,
    weight_path: Path,
    device: str = "cuda",
    # ---- 监督评估参数（与训练保持一致的可选项）----
    focus_mode: str = "uniform",
    channel_weights = (1,1,1,1),
    alpha: float | None = None,
    wall_sigma: float = 0.08,
    # ---- 物理评估参数 ----
    dataset_for_phys: Any | None = None,     # 若提供，将调用 compute_physics_on_dataset
    phys_max_items: int = 6,
    phys_m_samples: int = 4096,
    phys_k_neighbors: int = 16,
    rho: float = 1.0,
    nu_eff: float = 0.0,
    # ---- 输出路径（可选）----
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    加载权重 -> 对 dl 做监督评估(可选) -> 对 dataset 做物理评估(可选)
    返回一个结果字典，并在 out_dir 下落地：
      - train_val_metrics_posthoc.csv（仅包含汇总一行）
      - physics_stats.csv / div_hist.png / mom_hist.png（若传入 dataset_for_phys）
    """
    # 1) 加载与设备
    sd = torch.load(weight_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    print(f"[POSTHOC] Loaded weights: {weight_path}")

    # 输出目录
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # 2) 监督评估（若传入 dl）
    sup_mse = sup_mae = None
    if dl is not None:
        sup_mse, sup_mae = _supervised_eval_epoch(
            model, dl, device, focus_mode=focus_mode,
            channel_weights=channel_weights, alpha=alpha, wall_sigma=wall_sigma
        )
        print(f"[POSTHOC] Supervised Eval -> MSE: {sup_mse:.6f} | MAE: {sup_mae:.6f}")

    # 3) 物理评估（若提供 dataset_for_phys）
    phys_overall = None; phys_df = None
    div_png = mom_png = phys_csv = None
    if dataset_for_phys is not None:
        if out_dir is not None:
            div_png  = out_dir / "div_hist.png"
            mom_png  = out_dir / "mom_hist.png"
            phys_csv = out_dir / "physics_stats.csv"
        phys_overall, phys_df = compute_physics_on_dataset(
            model, dataset_for_phys, torch.device(device),
            max_items=phys_max_items, m_samples=phys_m_samples, k_neighbors=phys_k_neighbors,
            rho=rho, nu_eff=nu_eff,
            save_hist_png_div=div_png, save_hist_png_mom=mom_png, save_csv=phys_csv
        )

    # 4) 汇总与保存总览 CSV（可选）
    summary = {
        "weight": str(weight_path),
        "supervised_mse": None if sup_mse is None else float(sup_mse),
        "supervised_mae": None if sup_mae is None else float(sup_mae),
    }
    if phys_overall is not None:
        summary.update({f"phys_{k}": v for k, v in phys_overall.items()})

    if out_dir is not None:
        df_sum = pd.DataFrame([summary])
        df_sum.to_csv(out_dir / "train_val_metrics_posthoc.csv", index=False)
        print(f"[POSTHOC] Summary saved -> {out_dir/'train_val_metrics_posthoc.csv'}")

    return {"summary": summary, "phys_df": phys_df}
