# -*- coding: utf-8 -*-
"""
train_tpac.py —— 训练入口（支持 YAML 配置，和 test_tpac 对齐）

功能要点
- 读取 YAML（默认 Rotate/configs/default.yaml），并允许命令行覆盖 device / batch / workers / 采样点数等。
- 自动构建数据集（train / test），若缺少 out.csv 或 data.pkl 自动生成。
- 统一构建 backbone（从 cfg.models.backbone 读取；自动把 args 规范为 dict，避免 SimpleNamespace 导致的报错）。
- 构建 teacher（可选，取决于 cfg.teacher.use）。
- 使用 Trainer.run() 完整训练，输出：
  * results/temp_results/{case}/{tag}/weight/ 下多种“最佳”权重和 final_reco.pth（符号链接或拷贝）
  * 训练过程 metrics CSV 与 TXT 日志
  * 训练曲线曲线图（loss、MAE、连续性 raw/weighted、combo、alpha、λ_mult 等）

用法示例
CUDA_VISIBLE_DEVICES=0 python train_tpac.py \
  --cfg   Rotate/configs/default.yaml \
  --root  /data/.../Rotate \
  --case  C1 \
  --tag   tpac_test \
  --device cuda:0 \
  --pts 16384 --batch 2 --workers 4

训练结束后，推荐权重位于：
  /.../results/temp_results/{case}/{tag}/weight/final_reco.pth
"""

from __future__ import annotations
import os
import yaml
import torch
import pickle
import random
import numpy as np
from types import SimpleNamespace as NS
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

# --- 工程内依赖 ---
try:  # pragma: no cover - prefer absolute import when available
    from Rotate.models.backbone import build_backbone
except ModuleNotFoundError:  # pragma: no cover - fallback for script execution
    from models.backbone import build_backbone
from training.trainer import Trainer
from training.utils import ensure_dir, plot_training_curves
from data.dataset import pointdata, norm_data, build_out_csv_from_dir


DEFAULT_CFG_PATH = Path(__file__).resolve().parent / "configs" / "default.yaml"


# ----------------- 配置读取与规范化 -----------------
def dict2ns(d):
    """递归 dict -> SimpleNamespace；保持非 dict 类型原样。"""
    if isinstance(d, dict):
        return NS(**{k: dict2ns(v) for k, v in d.items()})
    return d


def ns2dict(ns):
    """递归 SimpleNamespace -> dict；其余类型原样返回。"""
    if isinstance(ns, NS):
        return {k: ns2dict(getattr(ns, k)) for k in vars(ns)}
    return ns


def load_cfg(yaml_path: str) -> Any:
    with open(yaml_path, "r") as f:
        cfg = dict2ns(yaml.safe_load(f))
    # 规范化：确保 models.backbone.args 为 dict，避免 build_backbone 中 dict() 报错
    if hasattr(cfg, "models") and hasattr(cfg.models, "backbone"):
        if hasattr(cfg.models.backbone, "args"):
            args_obj = cfg.models.backbone.args
            if isinstance(args_obj, NS):
                cfg.models.backbone.args = ns2dict(args_obj)
    return cfg


# ----------------- 路径组织 -----------------
def make_paths(root: Path, case: str, tag: str) -> NS:
    tmp_root = root / "results" / "temp_results" / case / tag
    data_root = root / "data" / "dataset" / case
    paths = NS(
        # 数据
        train_txt=data_root / "train",
        val_txt=data_root / "test",
        train_tmp=tmp_root / "dataset_temp_results" / "train",
        val_tmp=tmp_root / "dataset_temp_results" / "test",
        train_csv=tmp_root / "dataset_temp_results" / "train" / "out.csv",
        val_csv=tmp_root / "dataset_temp_results" / "test" / "out_test.csv",
        train_pkl=tmp_root / "dataset_temp_results" / "train" / "data.pkl",
        val_pkl=tmp_root / "dataset_temp_results" / "test" / "test_data.pkl",
        # 输出
        save_dir=tmp_root / "weight",
        weight_dir=tmp_root / "weight",
        final_reco=tmp_root / "weight" / "final_reco.pth",
        csv_path=tmp_root / "train_val_metrics.csv",
        curve_dir=tmp_root / "curves",
        curve_png=tmp_root / "curves" / "loss_curve.png",
    )
    # 创建基础目录
    for d in [paths.train_tmp, paths.val_tmp, paths.save_dir, paths.curve_dir]:
        ensure_dir(d)
    return paths


# ----------------- 数据准备 -----------------
def ensure_info(txt_dir: Path, pkl_path: Path):
    if not pkl_path.exists():
        min_val, max_val, num_data, num_points = norm_data(str(txt_dir))
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(
            {
                "input_min": min_val,
                "input_max": max_val,
                "num_data": num_data,
                "num_points": num_points,
            },
            open(pkl_path, "wb"),
        )


def build_datasets(
    paths: NS, points_per_sample: int, *, perturb_train=None, perturb_val=None
):
    # out.csv
    if not paths.train_csv.exists():
        build_out_csv_from_dir(paths.train_txt, paths.train_csv)
    if not paths.val_csv.exists():
        build_out_csv_from_dir(paths.val_txt, paths.val_csv)
    # data.pkl
    ensure_info(paths.train_txt, paths.train_pkl)
    ensure_info(paths.val_txt, paths.val_pkl)
    # 载入 info
    train_info = pickle.load(open(paths.train_pkl, "rb"))
    val_info = pickle.load(open(paths.val_pkl, "rb"))

    # 数据集
    ds_train = pointdata(
        str(paths.train_txt),
        str(paths.train_csv),
        train_info,
        int(points_per_sample),
        perturb=perturb_train,
    )
    ds_val = pointdata(
        str(paths.val_txt),
        str(paths.val_csv),
        val_info,
        int(points_per_sample),
        perturb=perturb_val,
    )
    return ds_train, ds_val, train_info, val_info


# ----------------- 随机种子 -----------------
def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _make_worker_init_fn(base_seed: int):
    def _worker_init(worker_id: int):
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _worker_init


# ----------------- 主流程 -----------------
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default=str(DEFAULT_CFG_PATH),
                    help="YAML 配置文件路径")
    ap.add_argument("--root", type=str, required=True,
                    help="项目根目录（包含 data/ 和 results/）")
    ap.add_argument("--case", type=str, required=True, help="例如 C1")
    ap.add_argument("--tag", type=str, required=True, help="例如 tpac_test")
    # 覆盖项
    ap.add_argument("--device", type=str, default=None, help="例如 cuda:0")
    ap.add_argument("--pts", type=int, default=None, help="每样本点数（覆盖 YAML）")
    ap.add_argument("--batch", type=int, default=2, help="batch size")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    ap.add_argument("--seed", type=int, default=2025, help="随机种子")

    # ========= 多目标梯度 solver（PCGrad / CAGrad）相关 =========
    ap.add_argument(
        "--mo_solver",
        type=str,
        default=None,
        choices=["sum", "pcgrad", "cagrad"],
        help="multi-objective gradient solver（默认 sum=普通加权）",
    )
    ap.add_argument(
        "--cagrad_alpha",
        type=float,
        default=None,
        help="CAGrad 中 mean grad 与 min-norm grad 的插值系数 (0~1，默认 0.5)",
    )

    # 鲁棒 / OOD 扰动控制
    ap.add_argument("--subsample_ratio", type=float, default=None)
    ap.add_argument("--coord_noise_sigma", type=float, default=None)
    ap.add_argument("--field_noise_sigma", type=float, default=None)
    ap.add_argument("--occlusion_ratio", type=float, default=None)
    ap.add_argument("--occlusion_min", type=float, default=None)
    ap.add_argument("--occlusion_max", type=float, default=None)
    ap.add_argument("--scale_factor", type=float, default=None)
    ap.add_argument("--ood_rotation", action="store_true", help="对训练数据启用分布外旋转")
    ap.add_argument("--ood_rot_deg", type=float, default=None)
    # 验证/测试独立控制
    ap.add_argument(
        "--apply_to_val",
        action="store_true",
        help="将训练扰动同样应用于验证集",
    )
    ap.add_argument(
        "--apply_to_test",
        action="store_true",
        help="将训练扰动同样应用于测试集",
    )
    ap.add_argument("--val_subsample_ratio", type=float, default=None)
    ap.add_argument("--val_coord_noise_sigma", type=float, default=None)
    ap.add_argument("--val_field_noise_sigma", type=float, default=None)
    ap.add_argument("--val_occlusion_ratio", type=float, default=None)
    ap.add_argument("--val_occlusion_min", type=float, default=None)
    ap.add_argument("--val_occlusion_max", type=float, default=None)
    ap.add_argument("--val_scale_factor", type=float, default=None)
    ap.add_argument("--val_ood_rotation", action="store_true", help="验证集 OOD 旋转")
    ap.add_argument("--val_ood_rot_deg", type=float, default=None)
    ap.add_argument("--test_subsample_ratio", type=float, default=None)
    ap.add_argument("--test_coord_noise_sigma", type=float, default=None)
    ap.add_argument("--test_field_noise_sigma", type=float, default=None)
    ap.add_argument("--test_occlusion_ratio", type=float, default=None)
    ap.add_argument("--test_occlusion_min", type=float, default=None)
    ap.add_argument("--test_occlusion_max", type=float, default=None)
    ap.add_argument("--test_scale_factor", type=float, default=None)
    ap.add_argument("--test_ood_rotation", action="store_true", help="测试集 OOD 旋转")
    ap.add_argument("--test_ood_rot_deg", type=float, default=None)
    args = ap.parse_args()

    # 读取配置
    cfg = load_cfg(args.cfg)

    # 覆盖 device/pts
    if args.device is not None:
        cfg.train.device = args.device
    if args.pts is not None:
        # 若 cfg 未定义 data_points，这里仅用于 main 内部传参
        cfg.data_points = int(args.pts)

    # ========= 多目标梯度 solver 配置写入 cfg.mo =========
    if not hasattr(cfg, "mo"):
        cfg.mo = NS()

    # solver 类型：sum / pcgrad / cagrad
    if args.mo_solver is not None:
        cfg.mo.solver = args.mo_solver
    else:
        if not hasattr(cfg.mo, "solver"):
            cfg.mo.solver = "sum"

    # CAGrad 的 alpha（0~1，越大越“避冲突”）
    if args.cagrad_alpha is not None:
        cfg.mo.cagrad_alpha = float(args.cagrad_alpha)
    else:
        if not hasattr(cfg.mo, "cagrad_alpha"):
            cfg.mo.cagrad_alpha = 0.5

    # 鲁棒/扰动参数解析
    robust_cfg = {}
    if hasattr(cfg, "robust"):
        robust_cfg = ns2dict(cfg.robust)
    perturb_keys = [
        "subsample_ratio",
        "coord_noise_sigma",
        "field_noise_sigma",
        "occlusion_ratio",
        "occlusion_min",
        "occlusion_max",
        "scale_factor",
    ]

    base_rotation = robust_cfg.get("ood_rotation", {}) if isinstance(robust_cfg, dict) else {}
    base_apply_val = bool(robust_cfg.get("apply_to_val", False)) if isinstance(robust_cfg, dict) else False
    base_apply_test = bool(robust_cfg.get("apply_to_test", False)) if isinstance(robust_cfg, dict) else False

    def build_split(prefix: str, *, inherit: dict | None, apply_flag: bool) -> dict:
        target: dict = {}
        if inherit:
            target.update(inherit)
        # 叠加默认配置
        for key in perturb_keys:
            if prefix == "":
                base_val = robust_cfg.get(key) if isinstance(robust_cfg, dict) else None
            else:
                base_val = (
                    target.get(key) if apply_flag
                    else robust_cfg.get(key) if isinstance(robust_cfg, dict)
                    else None
                )
            cli_name = f"{prefix}{key}" if prefix else key
            cli_val = getattr(args, cli_name, None)
            if cli_val is not None:
                target[key] = cli_val
            elif key not in target and base_val is not None:
                target[key] = base_val
        # 旋转
        if prefix == "":
            rot_enabled = bool(base_rotation.get("enabled", False))
            if getattr(args, "ood_rotation"):
                rot_enabled = True
            rot_deg = float(base_rotation.get("max_deg", 0.0))
            if args.ood_rot_deg is not None:
                rot_deg = args.ood_rot_deg
        else:
            inherited = inherit.get("ood_rotation_enabled", False) if inherit else False
            rot_enabled = bool(inherited) if apply_flag else bool(base_rotation.get("enabled", False))
            if getattr(args, f"{prefix}ood_rotation"):
                rot_enabled = True
            rot_deg = inherit.get("ood_rot_deg", base_rotation.get("max_deg", 0.0)) if inherit else base_rotation.get("max_deg", 0.0)
            cli_deg = getattr(args, f"{prefix}ood_rot_deg", None)
            if cli_deg is not None:
                rot_deg = cli_deg
        target["ood_rotation_enabled"] = bool(rot_enabled)
        target["ood_rot_deg"] = float(rot_deg)
        return target

    train_perturb = build_split("", inherit={}, apply_flag=True)

    val_override_present = any(
        getattr(args, f"val_{k}") is not None for k in perturb_keys
    ) or args.val_ood_rotation or (args.val_ood_rot_deg is not None)
    apply_to_val = base_apply_val or args.apply_to_val or val_override_present
    val_inherit = train_perturb if apply_to_val else {}
    val_perturb = build_split("val_", inherit=val_inherit, apply_flag=apply_to_val)
    if not apply_to_val and not val_override_present:
        val_perturb = {}

    test_override_present = any(
        getattr(args, f"test_{k}") is not None for k in perturb_keys
    ) or args.test_ood_rotation or (args.test_ood_rot_deg is not None)
    apply_to_test = base_apply_test or args.apply_to_test or test_override_present
    test_inherit = train_perturb if apply_to_test else {}
    test_perturb = build_split("test_", inherit=test_inherit, apply_flag=apply_to_test)
    if not apply_to_test and not test_override_present:
        test_perturb = {}

    # 定位与准备
    ROOT = Path(args.root)
    paths = make_paths(ROOT, args.case, args.tag)

    # 随机种子
    set_seed(args.seed)

    # DataLoader 选项
    PTS = int(args.pts if args.pts is not None else 16384)
    BATCH_SIZE = int(args.batch)
    NUM_WORKERS = int(args.workers)
    pin = (str(cfg.train.device).startswith("cuda") and torch.cuda.is_available())

    # 数据集
    ds_train, ds_val, train_info, val_info = build_datasets(
        paths,
        PTS,
        perturb_train=train_perturb if train_perturb else None,
        perturb_val=val_perturb if val_perturb else None,
    )

    loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=_make_worker_init_fn(args.seed),
    )
    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = 2
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

    # 构建骨干 & teacher（注意把 args 保持为 dict）
    robust_meta = {
        "train": train_perturb,
        "val": val_perturb,
        "test": test_perturb,
    }
    if not hasattr(cfg, "runtime"):
        cfg.runtime = NS()
    cfg.runtime.robust = dict2ns(robust_meta)
    cfg.runtime.seed = args.seed

    mcfg = NS()
    mcfg.models = cfg.models
    if hasattr(mcfg.models.backbone, "args") and isinstance(mcfg.models.backbone.args, NS):
        # 双保险：若 load_cfg 未把 args 还原为 dict，这里再转一遍
        def _ns2dict(ns):
            return {
                k: _ns2dict(getattr(ns, k)) if isinstance(getattr(ns, k), NS) else getattr(ns, k)
                for k in vars(ns)
            }

        mcfg.models.backbone.args = _ns2dict(mcfg.models.backbone.args)

    model = build_backbone(mcfg).to(cfg.train.device)
    teacher = build_backbone(mcfg).to(cfg.train.device) if getattr(cfg.teacher, "use", False) else None

    print(f"[INFO] device={cfg.train.device}")
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[INFO] params: {total:.2f}M (trainable {trainable:.2f}M)")

    # Trainer
    def _extract_norm_cfg(info):
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
            base.update(
                {
                    "x_min": inp_min[0:3],
                    "x_max": inp_max[0:3],
                    "y_min": inp_min[3:7],
                    "y_max": inp_max[3:7],
                }
            )
        return base

    pinn_norm_cfg = _extract_norm_cfg(train_info)
    trainer = Trainer(model, teacher, cfg, pinn_norm_cfg=pinn_norm_cfg, robust_meta=robust_meta)
    hist = trainer.run(dl_train, dl_val, paths, cfg)

    # 训练曲线
    plot_training_curves(
        hist,
        paths.curve_dir,
        paths.curve_png,
        calib_epoch=max(5, cfg.pinn.warmup + getattr(cfg.pinn, "calib_delay", 5)),
        calib_mult=cfg.pinn.calib_mult,
        div_target=cfg.pinn.div_target,
    )

    print("\n[OK] 训练完成：")
    print(f"  - metrics CSV : {paths.csv_path}")
    print(f"  - curves PNG  : {paths.curve_png}")
    print(f"  - weights dir : {paths.save_dir}")
    print(f"  - final_reco  : {paths.final_reco}")


if __name__ == "__main__":
    # 确保包根在 PYTHONPATH 中（便于绝对导入）
    os.environ.setdefault("PYTHONPATH", str(Path(__file__).resolve().parent))
    main()
