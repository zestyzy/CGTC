# CGTC：点云流场监督 + PINN 约束训练

本仓库实现了一个面向点云流场重建的训练/评估流程：以监督损失为主线，叠加连续性与动量残差（PINN）约束，并提供教师 EMA、层级守卫（guard）、冷却（cooling）与多目标梯度求解器等稳定训练机制。

## 功能概览

- **监督学习主干**：以 PointNet++ / PointTransformer 回归点云上的 `p,u,v,w`。
- **物理约束（PINN）**：基于 kNN + 最小二乘梯度估计的连续性/动量残差。
- **训练稳定机制**：
  - 梯度/损失比率 guard（可切换，支持自动阈值）。
  - 冷却窗口（cooling）与 guard-backoff。
  - 教师 EMA + 一致性损失。
  - 多目标梯度求解（sum / PCGrad / CAGrad）。
- **鲁棒与 OOD 扰动**：训练/验证/测试可独立配置采样、噪声、遮挡、缩放、旋转。
- **评估与可视化**：
  - `test.py` 生成预测/误差可视化与物理统计。
  - `eval/eval_physics.py` 批量 physics 统计与可选直方图。
  - `eval/posthoc_runner.py` 对指定权重做后验监督 + 物理评估。

## 目录结构

- `CGTC/train.py`：训练入口，构建数据、模型、Trainer，并保存权重与曲线。
- `CGTC/test.py`：测试与可视化入口，支持固定样本可视化与物理评估。
- `CGTC/eval/eval_physics.py`：批量物理评估（散度 + 动量残差）并输出 CSV/直方图。
- `CGTC/eval/posthoc_runner.py`：后验评估封装（加载权重 -> 监督评估 -> 物理评估）。
- `CGTC/training/trainer.py`：完整训练循环，包含 guard/cooling/EMA/多目标求解。
- `CGTC/models/`：PointNet2/PointTransformer 主干与 PINN 计算。
- `CGTC/configs/default.yaml`：默认训练配置。

## 数据与依赖

训练/评估依赖 `data.dataset` 模块（`pointdata`, `norm_data`, `build_out_csv_from_dir`）。该模块不在本仓库中，需要在 `PYTHONPATH` 中提供；数据目录结构约定如下：

```
<ROOT>/
  data/
    dataset/
      <CASE>/
        train/   # 训练点云样本
        test/    # 验证/测试点云样本
  results/
    temp_results/
      <CASE>/<TAG>/
        weight/  # 权重输出
        curves/  # 曲线输出
```

## 快速开始

### 训练

```bash
python CGTC/train.py \
  --cfg CGTC/configs/default.yaml \
  --root /path/to/project \
  --case C1 \
  --tag demo \
  --device cuda:0 \
  --pts 4096 \
  --batch 8 \
  --workers 4
```

训练输出位于：

```
results/temp_results/<CASE>/<TAG>/
  weight/final_reco.pth
  train_val_metrics.csv
  curves/loss_curve.png
```

### 训练时鲁棒/OOD 扰动

```bash
python CGTC/train.py \
  --cfg CGTC/configs/default.yaml \
  --root /path/to/project \
  --case C1 \
  --tag robust_demo \
  --subsample_ratio 0.5 \
  --coord_noise_sigma 0.002 \
  --occlusion_ratio 0.2 \
  --ood_rotation \
  --ood_rot_deg 60
```

### 测试与可视化

```bash
python CGTC/test.py \
  --root /path/to/project \
  --case C1 \
  --tag demo \
  --device cuda:0 \
  --pts 4096 \
  --vis_indices 0,5,12 \
  --phys_max_items 6
```

输出包括：

- `pred_vs_real_puvw.png`：旧版单样本对比图（随机样本）。
- `samples_vis/idx_XXXX/`：固定索引的可追溯可视化。
- `div_hist.png` / `mom_hist.png` / `phys_stats.csv`：物理统计与直方图。

### 批量物理评估

```bash
python CGTC/eval/eval_physics.py \
  --cfg CGTC/configs/default.yaml \
  --root /path/to/project \
  --case C1 \
  --split test \
  --pts 4096 \
  --ckpt-dir results/temp_results/C1/demo/weight \
  --out results/temp_results/C1/demo/physics_eval.csv
```

### 后验评估（可选）

```bash
python - <<'PY'
from pathlib import Path
from CGTC.eval.posthoc_runner import run_posthoc
from CGTC.models.backbone import build_backbone
from types import SimpleNamespace as NS

cfg = NS(models=NS(backbone=NS(name="pointnet2pp_ssg", args={})))
model = build_backbone(cfg)

run_posthoc(
    model,
    dl=None,  # 可传入 DataLoader 进行监督评估
    weight_path=Path("results/temp_results/C1/demo/weight/final_reco.pth"),
    device="cuda:0",
    out_dir=Path("results/temp_results/C1/demo/posthoc"),
)
PY
```

## 训练流程要点

- **阶段式训练**：先 warm（纯监督）再进入 PINN 约束阶段；支持回滚与 teacher 冻结。
- **守卫与冷却**：基于 loss 或 grad 比率限制物理项；触发时进入 cooling 窗口并降低物理权重。
- **多目标求解**：当监督 + 空间/物理项冲突时，可启用 PCGrad/CAGrad。
- **鲁棒扰动**：训练/验证/测试均支持独立扰动配置，并记录到运行元数据。

更多参数与默认值请查看 `CGTC/configs/default.yaml` 和各模块的注释说明。
