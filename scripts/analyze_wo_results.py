#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate and visualise results from ``wo.sh``/``wo.py`` ablations.

适配新版 Rotate/wo.py：
- flags_json 中包含 family / type / variant / distortion / role；
- baseline 可以通过 variant 或 type 来选择（例如 sup_only / equal / full 等）。
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

# optional: Spearman correlation; fall back to Pearson if SciPy not installed
try:
    from scipy.stats import spearmanr as _spearmanr  # type: ignore
except Exception:  # pragma: no cover
    _spearmanr = None

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# 路径 & 导入 Rotate.wo 中的底层解析函数
# ---------------------------------------------------------------------------

def _ensure_repo_on_path() -> Path:
    """Return repository root (Rotate dir) and ensure it is importable."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    rotate_pkg = repo_root / "Rotate"
    if rotate_pkg.exists():
        import sys
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        return rotate_pkg
    # 兜底：当前目录本身是 Rotate
    return script_dir


REPO_ROOT = _ensure_repo_on_path()

from Rotate import wo as wo_module  # noqa: E402


# ------------------------- 要做归一化的一些列 -------------------------

ENRICHED_NORMALISE_TARGETS = [
    "sup_MAE_final",
    "sup_MAE_best",
    "sup_RMSE_final",
    "div_P95",
    "mom_P95",
    "guard_hit_rate",
    "avg_cooling_len",
    "time_per_epoch",
]

GROUPED_NORMALISE_TARGETS = [
    "sup_MAE_final_mean",
    "sup_MAE_best_mean",
    "sup_RMSE_final_mean",
    "div_P95_mean",
    "mom_P95_mean",
    "guard_hit_rate_mean",
    "avg_cooling_len_mean",
    "time_per_epoch_mean",
]


# ------------------------------ dataclasses ------------------------------

@dataclass
class ExperimentPaths:
    tag: str
    metrics: Path
    physics: Optional[Path]
    diagnostics: Optional[Path]


def _default_root() -> Path:
    # 对于你的项目，Rotate 本身就是 root
    return REPO_ROOT


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def discover_experiments(root: Path, case: str, tag_prefix: str) -> Dict[str, ExperimentPaths]:
    """在 results/temp_results/<case> 下面找 tag 前缀匹配的所有实验."""
    base = root / "results" / "temp_results" / case
    experiments: Dict[str, ExperimentPaths] = {}
    if not base.exists():
        return experiments
    for child in base.iterdir():
        if not child.is_dir() or not child.name.startswith(f"{tag_prefix}_"):
            continue
        metrics = child / "train_val_metrics.csv"
        physics_path = child / "physics_eval.csv"
        diagnostics_dir = child / "diagnostics"
        experiments[child.name] = ExperimentPaths(
            tag=child.name,
            metrics=metrics,
            physics=physics_path if physics_path.exists() else None,
            diagnostics=diagnostics_dir if diagnostics_dir.exists() else None,
        )
    return experiments


def load_summary(root: Path, case: str, tag: str) -> Optional[pd.DataFrame]:
    """读取 wo.py 写出的 wo_summary.csv（如果存在）."""
    summary_path = root / "runs" / "wo" / case / tag / "wo_summary.csv"
    if not summary_path.exists():
        return None
    df = pd.read_csv(summary_path)
    df["flags"] = df.get("flags_json", "").apply(_parse_flags)
    return df


# ---------------------------------------------------------------------------
# Metric extraction utilities
# ---------------------------------------------------------------------------

def _parse_flags(value: object) -> Dict[str, object]:
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return {}


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


@dataclass
class MetricSummary:
    sup_mae_final: Optional[float]
    sup_rmse_final: Optional[float]
    sup_mae_best: Optional[float]
    sup_rmse_best: Optional[float]
    sup_mae_best_epoch: Optional[float]
    sup_rmse_best_epoch: Optional[float]
    guard_hit_rate: Optional[float]
    avg_cooling: Optional[float]
    time_per_epoch: Optional[float]
    total_epochs: Optional[float]
    metadata: Mapping[str, object]


def extract_metrics(metrics_path: Path) -> MetricSummary:
    """直接用 wo_module 的底层解析函数，从 train_val_metrics.csv 中抽指标."""
    metadata, df = wo_module._split_metadata_and_csv(metrics_path)
    train_df = df[df["split"].str.lower() == "train"].copy()
    val_df = df[df["split"].str.lower() == "val"].copy()

    sup_mae_final = sup_rmse_final = None
    sup_mae_best = sup_rmse_best = None
    sup_mae_best_epoch = sup_rmse_best_epoch = None

    if not val_df.empty:
        last_row = val_df.iloc[-1]
        sup_mae_final = wo_module._float(last_row.get("L_sup_mae"))
        sup_rmse_final = wo_module._float(last_row.get("L_sup_rmse"))
        if sup_rmse_final is None and "L_sup" in val_df.columns:
            sup_val = wo_module._float(last_row.get("L_sup"))
            sup_rmse_final = math.sqrt(sup_val) if sup_val is not None else None

        # best MAE / RMSE over epochs
        for col_name, dest in (("L_sup_mae", "mae"), ("L_sup_rmse", "rmse")):
            if col_name in val_df.columns:
                numeric = _safe_numeric(val_df[col_name])
                if numeric.notna().any():
                    idx = int(numeric.idxmin())
                    epoch = wo_module._float(val_df.loc[idx, "epoch"]) if "epoch" in val_df.columns else None
                    value = wo_module._float(numeric.min())
                    if dest == "mae":
                        sup_mae_best = value
                        sup_mae_best_epoch = epoch
                    else:
                        sup_rmse_best = value
                        sup_rmse_best_epoch = epoch

    guard_hit_rate = wo_module._compute_guard_hit_rate(train_df)
    avg_cooling = wo_module._compute_cooling_mean(metadata, df)
    time_per_epoch = wo_module._mean_epoch_time(val_df)
    total_epochs = (
        wo_module._float(val_df["epoch"].iloc[-1])
        if not val_df.empty and "epoch" in val_df.columns
        else None
    )

    return MetricSummary(
        sup_mae_final=sup_mae_final,
        sup_rmse_final=sup_rmse_final,
        sup_mae_best=sup_mae_best,
        sup_rmse_best=sup_rmse_best,
        sup_mae_best_epoch=sup_mae_best_epoch,
        sup_rmse_best_epoch=sup_rmse_best_epoch,
        guard_hit_rate=guard_hit_rate,
        avg_cooling=avg_cooling,
        time_per_epoch=time_per_epoch,
        total_epochs=total_epochs,
        metadata=metadata,
    )


@dataclass
class PhysicsSummary:
    div_p95: Optional[float]
    mom_p95: Optional[float]


def extract_physics(physics_path: Path | None) -> PhysicsSummary:
    if not physics_path or not physics_path.exists():
        return PhysicsSummary(div_p95=None, mom_p95=None)
    df = pd.read_csv(physics_path)
    if df.empty:
        return PhysicsSummary(div_p95=None, mom_p95=None)
    row = df.iloc[-1]
    return PhysicsSummary(
        div_p95=wo_module._float(row.get("div_P95")),
        mom_p95=wo_module._float(row.get("mom_P95")),
    )


# ---------------------------------------------------------------------------
# Aggregation utilities
# ---------------------------------------------------------------------------

def enrich_summary(
    summary_df: Optional[pd.DataFrame],
    experiments: Mapping[str, ExperimentPaths],
    *,
    tag: str,
) -> pd.DataFrame:
    """把 metrics + physics + wo_summary 合并成一张 enriched 表."""
    records: List[Dict[str, object]] = []
    for exp_tag, paths in experiments.items():
        if not paths.metrics.exists():
            continue
        metric_summary = extract_metrics(paths.metrics)
        physics_summary = extract_physics(paths.physics)

        base_record: Dict[str, object] = {
            "tag": exp_tag,
            "sup_MAE_final": metric_summary.sup_mae_final,
            "sup_RMSE_final": metric_summary.sup_rmse_final,
            "sup_MAE_best": metric_summary.sup_mae_best,
            "sup_RMSE_best": metric_summary.sup_rmse_best,
            "sup_MAE_best_epoch": metric_summary.sup_mae_best_epoch,
            "sup_RMSE_best_epoch": metric_summary.sup_rmse_best_epoch,
            "guard_hit_rate": metric_summary.guard_hit_rate,
            "avg_cooling_len": metric_summary.avg_cooling,
            "time_per_epoch": metric_summary.time_per_epoch,
            "total_epochs": metric_summary.total_epochs,
            "div_P95": physics_summary.div_p95,
            "mom_P95": physics_summary.mom_p95,
            "seed": metric_summary.metadata.get("seed"),
        }

        # 如果 wo_summary.csv 存在，先合并 flags_json / description 等信息
        if summary_df is not None:
            match = summary_df[summary_df["tag"] == exp_tag]
            if not match.empty:
                merged = match.iloc[0].to_dict()
                merged.update(base_record)  # metrics 以最新计算为准
                base_record = merged

        records.append(_postprocess_record(base_record, tag_prefix=tag))

    enriched = pd.DataFrame(records)
    if not enriched.empty:
        enriched.sort_values(by=["order", "exp_name"], inplace=True)
        enriched.reset_index(drop=True, inplace=True)
    return enriched


def _postprocess_record(record: Dict[str, object], *, tag_prefix: str) -> Dict[str, object]:
    """根据 tag 提取 order / exp_name，并从 flags 中展开 family / type / variant 等."""
    tag = record.get("tag")
    if isinstance(tag, str):
        suffix = tag.removeprefix(f"{tag_prefix}_")
        try:
            order_str, name = suffix.split("_", 1)
            record.setdefault("order", int(order_str))
            record.setdefault("exp_name", name)
        except ValueError:
            record.setdefault("order", None)
            record.setdefault("exp_name", suffix)
    else:
        record.setdefault("order", None)
        record.setdefault("exp_name", None)

    flags = record.get("flags") or {}
    if isinstance(flags, Mapping):
        # family / type / variant / distortion / role 适配你新的 wo.py
        record.setdefault("family", flags.get("family"))
        record.setdefault("type", flags.get("type"))
        # 如果 variant 为空，就用 type 兜底（例如 sup_only / equal / adaptive / full）
        record.setdefault("variant", flags.get("variant") or flags.get("type"))
        record.setdefault("distortion", flags.get("distortion"))
        record.setdefault("role", flags.get("role"))

        # description 原本就存一份在 wo_summary 里，这里只在缺失时兜底
        record.setdefault("description", flags.get("description"))

        # 旧版 ratio sweep 的字段（若不存在就保持 None）
        record.setdefault("rho_spat", flags.get("rho_spat"))
        record.setdefault("rho_phys", flags.get("rho_phys"))
        record.setdefault("gamma", flags.get("gamma"))

    return record


def summarise_by_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """按 exp_name 聚合 (mean±std)，并保留 family / variant / type 等 meta."""
    if df.empty:
        return df
    aggregations = {
        "sup_MAE_final": ["mean", "std"],
        "sup_MAE_best": ["mean", "std"],
        "sup_RMSE_final": ["mean", "std"],
        "div_P95": ["mean", "std"],
        "mom_P95": ["mean", "std"],
        "guard_hit_rate": ["mean"],
        "avg_cooling_len": ["mean"],
        "time_per_epoch": ["mean"],
    }
    grouped = df.groupby("exp_name").agg(aggregations)
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped.reset_index(inplace=True)

    unique_rows = df.drop_duplicates("exp_name")

    # meta 信息一并带上
    lookup = {row["exp_name"]: row for _, row in unique_rows.iterrows()}
    for col in ["variant", "family", "type", "description", "distortion", "role"]:
        grouped[col] = grouped["exp_name"].map({k: v.get(col) for k, v in lookup.items()})

    counts = df.groupby("exp_name")["tag"].count()
    grouped["count"] = grouped["exp_name"].map(counts)
    return grouped


# ---------- baseline 选择: variant 或 type 都可以匹配 ----------

def _baseline_mask(df: pd.DataFrame, baseline_variant: str) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    if "variant" in df.columns:
        mask |= (df["variant"] == baseline_variant)
    if "type" in df.columns:
        mask |= (df["type"] == baseline_variant)
    return mask


def compute_baseline_delta(
    df: pd.DataFrame,
    *,
    metric: str,
    variant: str,
    column: Optional[str] = None,
) -> pd.Series:
    value_col = column or f"{metric}_mean"
    if value_col not in df.columns:
        return pd.Series([math.nan] * len(df), index=df.index)
    baseline_rows = df[_baseline_mask(df, variant)]
    if baseline_rows.empty:
        return pd.Series([math.nan] * len(df), index=df.index)
    baseline_value = baseline_rows[value_col].mean()
    return df[value_col] - baseline_value


# ---------- 归一化 & util ----------

def _min_max_normalise(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series([math.nan] * len(series), index=series.index)
    min_val = valid.min()
    max_val = valid.max()
    if math.isclose(min_val, max_val):
        return pd.Series([0.0] * len(series), index=series.index)
    scaled = (numeric - min_val) / (max_val - min_val)
    return scaled.reindex(series.index)


def add_normalised_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[f"{column}_norm"] = _min_max_normalise(result[column])
    return result


# ---------------------------------------------------------------------------
# Visualisation utilities
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_metric_bar(
    df: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    normalised: bool = False,
) -> None:
    if df.empty:
        return
    base_col = metric if metric in df.columns else f"{metric}_mean"
    metric_col = f"{base_col}_norm" if normalised else base_col
    clean_df = df[df[metric_col].notna()]
    if clean_df.empty:
        return
    order = clean_df.sort_values(by=metric_col)["exp_name"].tolist()
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(order)), 4.5))
    values = clean_df.set_index("exp_name").loc[order]
    ax.bar(
        x=np.arange(len(order)),
        height=values[metric_col],
        yerr=None if normalised else values.get(f"{metric}_std"),
        alpha=0.85,
    )
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _robust_center_limits(arr: np.ndarray, lower_q: float = 5, upper_q: float = 95) -> tuple[float, float]:
    vals = arr[np.isfinite(arr)]
    lo = np.nanpercentile(vals, lower_q)
    hi = np.nanpercentile(vals, upper_q)
    lim = float(max(abs(lo), abs(hi)))
    if lim == 0:
        lim = 1e-6
    return -lim, +lim


def plot_ratio_heatmaps(df: pd.DataFrame, *, out_dir: Path, baseline_variant: str = "full") -> None:
    """
    Δ heatmap vs baseline (variant==baseline_variant) per gamma.
    蓝色=更好 (loss 更低)，红色=更差。
    对于你当前 wo.py，rho_* / gamma 本身可能全是 NaN，这里会自动 no-op。
    """
    if not {"rho_spat", "rho_phys", "gamma"}.issubset(df.columns):
        return

    ratio_df = df[df[["rho_spat", "rho_phys", "gamma"]].notna().all(axis=1)].copy()
    if ratio_df.empty:
        return

    target_col = (
        "sup_MAE_final_mean_norm"
        if "sup_MAE_final_mean_norm" in ratio_df.columns
        else "sup_MAE_final_mean"
    )
    label_unit = " (normalised)" if target_col.endswith("_norm") else ""

    for gamma, group in ratio_df.groupby("gamma"):
        base_rows = group[_baseline_mask(group, baseline_variant)]
        if base_rows.empty:
            continue
        baseline_val = base_rows[target_col].mean()

        g = group.copy()
        g["delta"] = g[target_col] - baseline_val  # negative = better

        pivot = g.pivot_table(index="rho_spat", columns="rho_phys", values="delta", aggfunc="mean")
        if pivot.isna().all().all():
            continue
        vals = pivot.values
        vmin, vmax = _robust_center_limits(vals)

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax),
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": f"Δ Supervision MAE{label_unit} vs {baseline_variant}"},
            ax=ax,
        )
        ax.set_title(f"Ratio sweep Δ vs {baseline_variant} (gamma={gamma:.3g})")
        ax.set_xlabel("rho_phys (max)")
        ax.set_ylabel("rho_spat (max)")
        fig.tight_layout()
        out_path = out_dir / f"ratio_heatmap_delta_gamma_{gamma:.3g}.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


def _spearman_xy(x: np.ndarray, y: np.ndarray) -> tuple[float, float, str]:
    """Compute Spearman ρ if available; else Pearson r. Return (rho, pval, label)."""
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return float("nan"), float("nan"), "ρ"
    if _spearmanr is not None:
        rho, p = _spearmanr(x, y)
        return float(rho), float(p), "ρ"
    # fallback: Pearson
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0, float("nan"), "r"
    r = float(np.corrcoef(x, y)[0, 1])
    return r, float("nan"), "r"


def plot_guard_scatter(df: pd.DataFrame, *, out_path: Path, baseline_variant: str = "sup_only") -> None:
    """
    x: guard_hit_rate_mean
    y: Δ supervision MAE vs baseline (负数更好).
    baseline 可以是 sup_only / equal / full 等（匹配 variant/type 由 _baseline_mask 决定）。
    """
    if df.empty or "guard_hit_rate_mean" not in df.columns or df["guard_hit_rate_mean"].notna().sum() == 0:
        return

    target_col = (
        "sup_MAE_final_mean_norm"
        if "sup_MAE_final_mean_norm" in df.columns
        else "sup_MAE_final_mean"
    )
    if target_col not in df.columns:
        return

    work = df[["exp_name", "variant", "guard_hit_rate_mean", target_col]].dropna(
        subset=["guard_hit_rate_mean", target_col]
    ).copy()
    if work.empty:
        return

    # baseline 选择：用你已有的 _baseline_mask
    base_rows = work[_baseline_mask(work, baseline_variant)]
    if base_rows.empty:
        return
    baseline_val = base_rows[target_col].mean()

    work["delta"] = work[target_col] - baseline_val  # 负数 = 比 baseline 更好

    rho, pval, sym = _spearman_xy(
        work["guard_hit_rate_mean"].values, work["delta"].values
    )
    rho_str = f"{sym}={rho:.3f}" + (f", p={pval:.2g}" if np.isfinite(pval) else "")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    # 回归线
    sns.regplot(
        data=work,
        x="guard_hit_rate_mean",
        y="delta",
        scatter=False,
        ci=None,
        line_kws={"linewidth": 2, "alpha": 0.6},
        truncate=True,
        ax=ax,
    )

    # 散点：颜色 & 形状都按 variant 来，这样 legend 里的形状就和图里一致
    sns.scatterplot(
        data=work,
        x="guard_hit_rate_mean",
        y="delta",
        hue="variant",
        style="variant",
        s=80,
        edgecolor="black",
        linewidth=0.3,
        ax=ax,
    )

    ax.axhline(0.0, color="k", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Guard trigger rate (train)")
    ax.set_ylabel(
        f"Δ Supervision MAE{' (norm.)' if target_col.endswith('_norm') else ''} "
        "vs baseline (← better)"
    )
    ax.set_title(f"Guard vs Supervision Δ — {rho_str}")
    ax.grid(True, linestyle="--", alpha=0.3)

    # ---- 清理 legend：去掉那条 label 为 'variant' 的冗余项，并放在右上角 ----
    handles, labels = ax.get_legend_handles_labels()
    new_handles, new_labels = [], []
    for h, lab in zip(handles, labels):
        if lab == "variant":  # seaborn 自动加的变量名，跳过
            continue
        new_handles.append(h)
        new_labels.append(lab)

    leg = ax.legend(
        new_handles,
        new_labels,
        title="variant",
        loc="upper right",
        bbox_to_anchor=(1.01, 1.01),
        borderaxespad=0.0,
        frameon=True,
        fontsize=8,
        title_fontsize=8,
    )
    leg.get_frame().set_alpha(0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_tradeoff_mae_div(df: pd.DataFrame, *, out_path: Path, baseline_variant: str = "sup_only") -> None:
    """
    Δ Supervision MAE vs Δ Divergence(P95) relative to baseline.
    这里 Δ = method - baseline，负数更好 → 左下角是“最好”区域.
    """
    if df.empty:
        return

    target_col = (
        "sup_MAE_final_mean_norm"
        if "sup_MAE_final_mean_norm" in df.columns
        else "sup_MAE_final_mean"
    )
    if target_col not in df.columns or "div_P95_mean" not in df.columns:
        return

    # baseline 选择
    base = df[_baseline_mask(df, baseline_variant)].copy()
    base = base.dropna(subset=[target_col, "div_P95_mean"])
    if base.empty:
        return

    mae_base = base[target_col].mean()
    div_base = base["div_P95_mean"].mean()

    work = df[["exp_name", "variant", target_col, "div_P95_mean"]].dropna(
        subset=[target_col, "div_P95_mean"]
    ).copy()
    if work.empty:
        return

    work["dMAE"] = work[target_col] - mae_base   # 负数: MAE 比 baseline 小 → 更好
    work["dDIV"] = work["div_P95_mean"] - div_base  # 负数: 物理误差更小 → 更好

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    sns.scatterplot(
        data=work,
        x="dMAE",
        y="dDIV",
        hue="variant",
        style="variant",     # 形状也按 variant
        s=90,
        edgecolor="k",
        linewidth=0.3,
        ax=ax,
    )

    ax.axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel(
        f"Δ Supervision MAE{' (norm.)' if target_col.endswith('_norm') else ''} "
        "(← better)"
    )
    ax.set_ylabel("Δ Divergence P95 (→ better)")
    ax.set_title("Trade-off vs baseline")
    ax.grid(True, linestyle="--", alpha=0.3)

    # ---- 同样清理 legend，多余的 'variant' label 去掉，右上角摆放 ----
    handles, labels = ax.get_legend_handles_labels()
    new_handles, new_labels = [], []
    for h, lab in zip(handles, labels):
        if lab == "variant":
            continue
        new_handles.append(h)
        new_labels.append(lab)

    leg = ax.legend(
        new_handles,
        new_labels,
        title="variant",
        loc="upper right",
        bbox_to_anchor=(1.01, 1.01),
        borderaxespad=0.0,
        frameon=True,
        fontsize=8,
        title_fontsize=8,
    )
    leg.get_frame().set_alpha(0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)



# ---------------------------------------------------------------------------
# Reporting utilities
# ---------------------------------------------------------------------------

def write_markdown_report(
    out_path: Path,
    grouped: pd.DataFrame,
    *,
    baseline_variant: str,
    intro: str,
) -> None:
    if grouped.empty:
        return

    mae_delta_raw = compute_baseline_delta(
        grouped, metric="sup_MAE_final", variant=baseline_variant, column="sup_MAE_final_mean"
    )
    norm_col = (
        "sup_MAE_final_mean_norm"
        if "sup_MAE_final_mean_norm" in grouped.columns
        else "sup_MAE_final_mean"
    )
    mae_delta_norm = compute_baseline_delta(
        grouped, metric="sup_MAE_final", variant=baseline_variant, column=norm_col
    )
    sorted_df = (
        grouped.assign(mae_delta_raw=mae_delta_raw, mae_delta_norm=mae_delta_norm)
        .sort_values(by=norm_col)
        .reset_index(drop=True)
    )

    lines = ["# WO Ablation Analysis", ""]
    if intro:
        lines.extend(textwrap.dedent(intro).strip().splitlines())
        lines.append("")

    lines.append("## Ranked supervision performance")
    lines.append("")
    for _, row in sorted_df.iterrows():
        desc = row.get("description") or "(no description)"
        delta_norm = row.get("mae_delta_norm")
        delta_str = f" ({delta_norm:+.3f} vs. {baseline_variant}, normalised)" if pd.notna(delta_norm) else ""
        delta_raw = row.get("mae_delta_raw")
        if pd.notna(delta_raw):
            delta_str += f" [{delta_raw:+.4f} raw]"
        guard_val = row.get("guard_hit_rate_mean")
        guard_str = f"{guard_val:.2%}" if pd.notna(guard_val) else "n/a"
        sup_mae_mean = row.get("sup_MAE_final_mean")
        sup_str = f"{sup_mae_mean:.4f}" if pd.notna(sup_mae_mean) else "n/a"
        sup_norm = row.get("sup_MAE_final_mean_norm")
        sup_norm_str = f"{sup_norm:.3f}" if pd.notna(sup_norm) else "n/a"
        fam = row.get("family") or "-"
        var = row.get("variant") or "-"
        lines.append(
            f"* **{row['exp_name']}** `[family={fam}, variant={var}]` – "
            f"normalised MAE={sup_norm_str}{delta_str}; "
            f"raw MAE={sup_str}; guard={guard_str} – {desc}"
        )
    lines.append("")

    physics_cols = [col for col in sorted_df.columns if col.startswith("div_P95") or col.startswith("mom_P95")]
    if physics_cols:
        lines.append("## Physics diagnostics (mean ± std)")
        lines.append("")
        lines.append("| Experiment | div_P95 | mom_P95 |")
        lines.append("|---|---|---|")
        for _, row in sorted_df.iterrows():
            div_mean = row.get("div_P95_mean")
            div_std = row.get("div_P95_std")
            mom_mean = row.get("mom_P95_mean")
            mom_std = row.get("mom_P95_std")
            lines.append(
                f"| {row['exp_name']} | {format_with_pm(div_mean, div_std)} | {format_with_pm(mom_mean, mom_std)} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def format_with_pm(mean: Optional[float], std: Optional[float]) -> str:
    if pd.isna(mean):
        return "n/a"
    if pd.isna(std) or std == 0:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate wo.py ablation runs")
    parser.add_argument("--root", type=Path, default=_default_root(), help="Project root (contains Rotate package)")
    parser.add_argument("--case", required=True, help="Dataset case identifier (e.g. C1)")
    parser.add_argument("--tag", required=True, help="Base experiment tag prefix")
    parser.add_argument("--out", type=Path, help="Directory for analysis outputs")
    parser.add_argument(
        "--baseline-variant",
        default="sup_only",
        help="Label used as baseline (matches flags.variant or flags.type, e.g. sup_only / equal / full)",
    )
    parser.add_argument("--intro", default="", help="Optional Markdown snippet inserted at the top of the report")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    root = args.root.resolve()
    out_dir = args.out or (root / "runs" / "wo" / args.case / args.tag / "analysis")
    out_dir = out_dir.resolve()
    _ensure_dir(out_dir)

    experiments = discover_experiments(root, args.case, args.tag)
    if not experiments:
        parser.error(
            f"No experiments found under {root / 'results' / 'temp_results' / args.case} "
            f"with prefix {args.tag}."
        )

    summary_df = load_summary(root, args.case, args.tag)
    enriched = enrich_summary(summary_df, experiments, tag=args.tag)
    enriched = add_normalised_columns(enriched, ENRICHED_NORMALISE_TARGETS)
    if enriched.empty:
        parser.error("Unable to build enriched summary – check that metrics CSV files exist.")

    enriched_csv = out_dir / "analysis_enriched.csv"
    enriched.to_csv(enriched_csv, index=False)

    grouped = summarise_by_experiment(enriched)
    grouped = add_normalised_columns(grouped, GROUPED_NORMALISE_TARGETS)
    grouped_csv = out_dir / "analysis_grouped.csv"
    grouped.to_csv(grouped_csv, index=False)

    markdown_path = out_dir / "analysis_report.md"
    write_markdown_report(markdown_path, grouped, baseline_variant=args.baseline_variant, intro=args.intro)

    # --- plots ---
    plot_metric_bar(
        grouped,
        metric="sup_MAE_final",
        ylabel="Normalised validation supervision MAE",
        title="Supervision MAE by experiment (normalised)",
        out_path=out_dir / "sup_mae_bar.png",
        normalised=True,
    )
    plot_metric_bar(
        grouped,
        metric="div_P95",
        ylabel="Normalised divergence P95",
        title="Physics divergence (P95, normalised)",
        out_path=out_dir / "divergence_bar.png",
        normalised=True,
    )

    # guard 相关散点（默认 baseline 是 sup_only，可以自己改参数）
    plot_guard_scatter(grouped, out_path=out_dir / "guard_vs_sup_delta.png", baseline_variant=args.baseline_variant)

    # ratio sweep（如果你后面又加了 rho_spat / rho_phys / gamma，这里会自动生效；现在则 no-op）
    ratio_lookup = (
        enriched.groupby("exp_name")[["rho_spat", "rho_phys", "gamma"]].mean()
        if {"rho_spat", "rho_phys", "gamma"}.issubset(enriched.columns)
        else pd.DataFrame()
    )
    if not ratio_lookup.empty:
        plot_ratio_heatmaps(
            grouped.join(ratio_lookup, on="exp_name"),
            out_dir=out_dir,
            baseline_variant=args.baseline_variant,
        )

    # MAE vs Div 的 trade-off 图
    plot_tradeoff_mae_div(
        grouped,
        out_path=out_dir / "tradeoff_mae_vs_div.png",
        baseline_variant=args.baseline_variant,
    )

    print(f"[OK] Wrote enriched table to {enriched_csv}")
    print(f"[OK] Wrote grouped table to {grouped_csv}")
    print(f"[OK] Wrote markdown report to {markdown_path}")
    print(f"[OK] Plots saved under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
