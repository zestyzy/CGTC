#!/usr/bin/env python3
"""Visual diagnostics for TPAC training runs.

This utility consumes the step-level ``train_val_metrics.csv`` written by the
trainer and renders a multi-panel diagnostic figure that highlights the
supervision→spatial→physics hierarchy, guard activity, cooling windows and
gradient-cosine feedback.  Metadata emitted in the CSV header (prefixed with
``#``) is parsed to recover stage transitions, cooling events and robustness
settings.

Example::

    python draw_picture.py \
        --metrics results/temp_results/C1/exp/train_val_metrics.csv \
        --out-dir results/temp_results/C1/exp/diagnostics
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Metrics:
    data: pd.DataFrame
    metadata: Dict[str, object]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_metadata(lines: Iterable[str]) -> Dict[str, object]:
    meta: Dict[str, object] = {}
    for line in lines:
        line = line.strip()
        if not line or not line.startswith("#"):
            continue
        payload = line[1:].strip()
        if "=" not in payload:
            continue
        key, value = payload.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            meta[key] = json.loads(value)
        except json.JSONDecodeError:
            meta[key] = value
    return meta


def load_metrics(path: Path) -> Metrics:
    with path.open("r", encoding="utf-8-sig") as handle:
        raw = handle.read()
    lines = raw.splitlines()
    meta = _parse_metadata(lines)
    content = "\n".join(line for line in lines if not line.startswith("#"))
    if not content.strip():
        raise ValueError(f"No metric rows found in {path}")
    data = pd.read_csv(StringIO(content))
    if "phase" not in data.columns:
        data["phase"] = "warm"
    return Metrics(data=data, metadata=meta)


# ---------------------------------------------------------------------------
# Stage & cooling utilities
# ---------------------------------------------------------------------------


def _epoch_from_step(step: int, step_map: Dict[int, float]) -> float:
    if step in step_map:
        return float(step_map[step])
    if not step_map:
        return float(step)
    # fallback to closest step
    keys = np.array(list(step_map.keys()), dtype=float)
    idx = int(np.argmin(np.abs(keys - step)))
    return float(step_map[int(keys[idx])])


def compute_stage_spans(train_df: pd.DataFrame, metadata: Dict[str, object]) -> Tuple[List[Tuple[str, float, float]], List[float]]:
    spans: List[Tuple[str, float, float]] = []
    markers: List[float] = []
    events = metadata.get("stage_events", [])
    if not isinstance(events, list):
        return spans, markers
    step_map = (
        train_df.drop_duplicates("global_step")["epoch"].to_dict()
        if {"global_step", "epoch"}.issubset(train_df.columns)
        else {}
    )
    ordered = sorted(events, key=lambda x: x.get("global_step", 0))
    current_stage: Optional[str] = None
    start_epoch: Optional[float] = None
    last_epoch = float(train_df["epoch"].max()) if not train_df.empty else 0.0
    for event in ordered:
        phase = event.get("phase")
        gstep = int(event.get("global_step", 0))
        epoch = float(event.get("epoch", _epoch_from_step(gstep, step_map)))
        if phase in {"warm", "warmup"}:
            if current_stage is not None and start_epoch is not None:
                spans.append((current_stage, start_epoch, epoch))
            current_stage = phase
            start_epoch = epoch
        elif phase == "rollback":
            markers.append(epoch)
    if current_stage is not None and start_epoch is not None:
        spans.append((current_stage, start_epoch, last_epoch))
    return spans, markers


def compute_cooling_windows(
    train_df: pd.DataFrame,
    metadata: Dict[str, object],
    term: str,
) -> List[Tuple[float, float, str]]:
    windows: List[Tuple[float, float, str]] = []
    events = metadata.get("cooling_events", [])
    if not isinstance(events, list):
        return windows
    step_map = (
        train_df.drop_duplicates("global_step")["epoch"].to_dict()
        if {"global_step", "epoch"}.issubset(train_df.columns)
        else {}
    )
    active_start: Optional[float] = None
    reason: str = ""
    for event in events:
        if event.get("term") != term:
            continue
        step = int(event.get("step", 0))
        epoch = float(_epoch_from_step(step, step_map))
        if event.get("event") == "start":
            active_start = epoch
            reason = str(event.get("reason", ""))
        elif event.get("event") == "end" and active_start is not None:
            windows.append((active_start, epoch, reason))
            active_start = None
    return windows


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _ax_stage_background(ax, spans: List[Tuple[str, float, float]], palette: Dict[str, str]) -> None:
    for stage, start, end in spans:
        color = palette.get(stage, "#E5E7EB")
        if math.isfinite(start) and math.isfinite(end) and end >= start:
            ax.axvspan(start, end, color=color, alpha=0.18, linewidth=0)


def _ax_stage_markers(ax, markers: List[float], label: str, color: str = "#111827") -> None:
    added = False
    for epoch in markers:
        if not math.isfinite(epoch):
            continue
        if not added:
            ax.axvline(epoch, color=color, linestyle="--", linewidth=1.5, label=label)
            added = True
        else:
            ax.axvline(epoch, color=color, linestyle="--", linewidth=1.0)


def _scatter_triggers(ax, epochs: np.ndarray, values: np.ndarray, mask: np.ndarray, label: str, color: str) -> None:
    if mask.any():
        ax.scatter(epochs[mask], values[mask], s=18, color=color, label=label, zorder=5)


def _cooling_spans(ax, windows: List[Tuple[float, float, str]], color: str) -> None:
    for start, end, reason in windows:
        if math.isfinite(start) and math.isfinite(end) and end > start:
            ax.axvspan(start, end, color=color, alpha=0.15)
            if reason:
                ax.text(
                    (start + end) / 2,
                    ax.get_ylim()[1],
                    reason,
                    ha="center",
                    va="top",
                    fontsize=8,
                    color=color,
                    alpha=0.7,
                )


def _fill_neg_cos(ax, epochs: np.ndarray, cos_values: np.ndarray, color: str) -> None:
    mask = np.isfinite(cos_values) & (cos_values < 0)
    if mask.any():
        ax.fill_between(epochs, cos_values, 0.0, where=mask, color=color, alpha=0.2)


# ---------------------------------------------------------------------------
# Main plotting routine
# ---------------------------------------------------------------------------


def render_diagnostics(metrics: Metrics, out_dir: Path) -> None:
    df = metrics.data.copy()
    meta = metrics.metadata
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = df[df["split"].str.lower() == "train"].copy()
    if train_df.empty:
        raise ValueError("Metrics file contains no train split entries; cannot render diagnostics.")
    train_df.sort_values(["epoch", "global_step"], inplace=True)

    def _float_column(name: str, fill: float = math.nan) -> np.ndarray:
        if name in train_df.columns:
            return pd.to_numeric(train_df[name], errors="coerce").to_numpy(dtype=float)
        return np.full(len(train_df), fill, dtype=float)

    def _bool_column(name: str) -> np.ndarray:
        if name in train_df.columns:
            return train_df[name].astype(bool).to_numpy()
        return np.zeros(len(train_df), dtype=bool)

    spans, markers = compute_stage_spans(train_df, meta)
    pinn_windows = compute_cooling_windows(train_df, meta, term="pinn")
    spatial_windows = compute_cooling_windows(train_df, meta, term="spatial")

    epochs = _float_column("epoch")
    sup = _float_column("L_sup")
    spat_cap = _float_column("L_spat_cap")
    phys_cap = _float_column("L_phys_cap")

    spat_scale = _float_column("guard_spat_scale", fill=1.0)
    phys_scale = _float_column("guard_phys_scale", fill=1.0)

    pinn_blend = _float_column("pinn_blend", fill=0.0)
    spat_blend = _float_column("spat_blend", fill=0.0)

    cos_spat = _float_column("cos_sup_spat", fill=math.nan)
    cos_phys = _float_column("cos_sup_phys", fill=math.nan)

    guard_spat_mask = _bool_column("guard_spat_triggered")
    guard_phys_mask = _bool_column("guard_phys_triggered")

    fig, axes = plt.subplots(4, 1, figsize=(11, 14), sharex=True)
    palette = {"warm": "#FDE68A", "warmup": "#BFDBFE"}

    # Panel 1: hierarchy
    ax = axes[0]
    _ax_stage_background(ax, spans, palette)
    _ax_stage_markers(ax, markers, label="rollback")
    ax.plot(epochs, sup, label="Supervision", color="#1f77b4", linewidth=2.0)
    ax.plot(epochs, spat_cap, label="Spatial (capped)", color="#2ca02c", linewidth=2.0)
    ax.plot(epochs, phys_cap, label="Physics (capped)", color="#d62728", linewidth=2.0)
    _scatter_triggers(ax, epochs, spat_cap, guard_spat_mask, label="Spatial guard", color="#2ca02c")
    _scatter_triggers(ax, epochs, phys_cap, guard_phys_mask, label="Physics guard", color="#d62728")
    _cooling_spans(ax, pinn_windows, color="#d62728")
    _cooling_spans(ax, spatial_windows, color="#2ca02c")
    ax.set_ylabel("Loss magnitude")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    ax.set_title("Loss hierarchy and guard triggers")

    # Panel 2: guard scales
    ax = axes[1]
    _ax_stage_background(ax, spans, palette)
    ax.plot(epochs, spat_scale, label="Spatial scale", color="#2ca02c", linewidth=2.0)
    ax.plot(epochs, phys_scale, label="Physics scale", color="#d62728", linewidth=2.0)
    ax.set_ylabel("Guard scale")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    ax.set_title("Soft projection scales")

    # Panel 3: blend & cooling
    ax = axes[2]
    _ax_stage_background(ax, spans, palette)
    _cooling_spans(ax, pinn_windows, color="#d62728")
    _cooling_spans(ax, spatial_windows, color="#2ca02c")
    ax.plot(epochs, pinn_blend, label="pinn_blend", color="#d62728", linewidth=2.0)
    ax.plot(epochs, spat_blend, label="spatial_blend", color="#2ca02c", linewidth=2.0)
    ax.set_ylabel("Blend")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    ax.set_title("Blend factors and cooling windows")

    # Panel 4: gradient cosines
    ax = axes[3]
    _ax_stage_background(ax, spans, palette)
    ax.axhline(0.0, color="#111827", linewidth=1.0, linestyle=":")
    ax.plot(epochs, cos_spat, label="cos(supervision, spatial)", color="#2ca02c", linewidth=1.8)
    ax.plot(epochs, cos_phys, label="cos(supervision, physics)", color="#d62728", linewidth=1.8)
    _fill_neg_cos(ax, epochs, cos_spat, color="#2ca02c")
    _fill_neg_cos(ax, epochs, cos_phys, color="#d62728")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    ax.set_title("Gradient cosine diagnostics")

    plt.tight_layout()
    pdf_path = out_dir / "curves.pdf"
    png_path = out_dir / "curves.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    # Backwards-compatible PNGs for quick inspection
    single_dir = out_dir
    _plot_single_series(single_dir / "loss_hierarchy.png", epochs, sup, spat_cap, phys_cap, spans)
    _plot_guard_series(single_dir / "guard_scaling.png", epochs, spat_scale, phys_scale, spans)
    _plot_scheduler_series(single_dir / "scheduler.png", epochs, pinn_blend, spat_blend, spans)


def _plot_single_series(path: Path, epochs: np.ndarray, sup: np.ndarray, spat: np.ndarray, phys: np.ndarray, spans):
    fig, ax = plt.subplots(figsize=(10, 6))
    _ax_stage_background(ax, spans, {"warm": "#FDE68A", "warmup": "#BFDBFE"})
    ax.plot(epochs, sup, label="Supervision", linewidth=2)
    ax.plot(epochs, spat, label="Spatial", linewidth=2)
    ax.plot(epochs, phys, label="Physics", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Magnitude")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_title("Loss hierarchy")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_guard_series(path: Path, epochs: np.ndarray, spat_scale: np.ndarray, phys_scale: np.ndarray, spans):
    fig, ax = plt.subplots(figsize=(10, 5))
    _ax_stage_background(ax, spans, {"warm": "#FDE68A", "warmup": "#BFDBFE"})
    ax.plot(epochs, spat_scale, label="Spatial scale", linewidth=2)
    ax.plot(epochs, phys_scale, label="Physics scale", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Scale")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_title("Guard scale factors")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_scheduler_series(path: Path, epochs: np.ndarray, pinn_blend: np.ndarray, spat_blend: np.ndarray, spans):
    fig, ax = plt.subplots(figsize=(10, 5))
    _ax_stage_background(ax, spans, {"warm": "#FDE68A", "warmup": "#BFDBFE"})
    ax.plot(epochs, pinn_blend, label="pinn_blend", linewidth=2)
    ax.plot(epochs, spat_blend, label="spatial_blend", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Blend")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_title("Scheduler signals")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True, help="Path to train_val_metrics.csv")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for the generated figures (defaults to metrics directory)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    metrics_path: Path = args.metrics
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    out_dir = args.out_dir or metrics_path.parent
    metrics = load_metrics(metrics_path)
    render_diagnostics(metrics, out_dir)
    print(f"[OK] Diagnostics saved to {out_dir}")


if __name__ == "__main__":
    main()
