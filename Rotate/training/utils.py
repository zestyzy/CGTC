# -*- coding: utf-8 -*-
# TPAC/training/utils.py
from __future__ import annotations
from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_state(state_dict, path: Path):
    ensure_dir(path.parent)
    torch.save(state_dict, path)

def symlink_or_copy(src: Path, dst: Path):
    ensure_dir(dst.parent)
    try:
        if dst.exists(): dst.unlink()
        dst.symlink_to(src)
    except Exception:
        import shutil
        shutil.copyfile(src, dst)

def plot_training_curves(hist: pd.DataFrame, curve_dir: Path, curve_png: Path,
                         calib_epoch: int, calib_mult: float, div_target: float):
    if hist is None or len(hist) == 0:
        raise ValueError("Training history is empty; cannot plot curves. Check that the training loop produced metrics.")
    ensure_dir(curve_dir)
    fig, axes = plt.subplots(8, 1, figsize=(9, 24), dpi=150, sharex=True)
    ep = hist["epoch"].to_numpy()

    # 1 Loss
    axes[0].plot(ep, hist["train_loss"], label="Train Loss")
    axes[0].plot(ep, hist["val_loss"], label="Val Loss")
    axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(True, ls="--", alpha=.5)

    # 2 MAE
    axes[1].plot(ep, hist["val_mae"], label="Val MAE")
    axes[1].plot(ep, hist["train_mae"], label="Train MAE")
    axes[1].set_ylabel("MAE"); axes[1].legend(); axes[1].grid(True, ls="--", alpha=.5)

    # 3 Cont RAW
    axes[2].plot(ep, hist["pinn_cont_raw"], label="Cont RAW (E[div^2])")
    if len(hist) >= 1:
        target_line = (float(hist["pinn_cont_raw"].iloc[max(0, calib_epoch-1)]) * calib_mult
                       if len(hist) >= max(1, calib_epoch) else div_target)
        axes[2].axhline(target_line, ls="--", label=f"target≈{target_line:.4f}")
    axes[2].legend(); axes[2].grid(True, ls="--", alpha=.5)

    # 4 Phys weighted
    axes[3].plot(ep, hist["pinn_cont_w"], label="Cont Weighted")
    axes[3].plot(ep, hist["pinn_mom_w"], label="Mom Weighted")
    axes[3].legend(); axes[3].grid(True, ls="--", alpha=.5)

    # 5 Combo
    axes[4].plot(ep, hist["combo_score"], label="combo")
    axes[4].legend(); axes[4].grid(True, ls="--", alpha=.5)

    # 6 Channel weights
    for k in ["w_p","w_u","w_v","w_w"]:
        axes[5].plot(ep, hist[k], label=k)
    axes[5].legend(); axes[5].grid(True, ls="--", alpha=.5)

    # 7 alpha
    axes[6].plot(ep, hist["alpha_now"], label="alpha (mixed)")
    axes[6].legend(); axes[6].grid(True, ls="--", alpha=.5)

    # 8 λ multiplier
    axes[7].plot(ep, hist["lcont_mult"], label="λ_mult")
    axes[7].set_xlabel("Epoch")
    axes[7].legend(); axes[7].grid(True, ls="--", alpha=.5)

    plt.tight_layout(); plt.savefig(curve_png, dpi=300, bbox_inches="tight"); plt.close()
