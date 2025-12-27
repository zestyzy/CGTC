# TPAC 1train2step 代码评审

## 1. warm 阶段仍缺少“独立监督模型”出入口
- `train_tpac.py` 在主流程里每次都会重新实例化学生与教师骨干；教师权重立即被学生当前参数覆盖，既不会尝试加载历史 warm-only checkpoint，也没有配置项允许直接跳过 warm 阶段复用既有教师模型。【F:train_tpac.py†L199-L240】【F:training/trainer.py†L121-L126】
- 如果用户已经离线训练出最优监督模型，希望 warmup 只消耗 PINN + 一致性正则，当前脚本没有办法从磁盘读入该监督模型当作教师，从而阻断了“warm 完毕 → warmup 接力”的工作流。

## 2. 缺乏真正的 warm→warmup 状态机
- `Trainer.run` 的主循环只依赖 `epoch > cfg.pinn.warmup` 这一硬编码阈值来打开 PINN，完全没有按照 `cfg.early.patience` 触发的早停、保存或阶段切换逻辑；即便监督指标已经长期不再改进，也会继续跑满 `cfg.pinn.warmup` 轮后才进入物理阶段。【F:training/trainer.py†L331-L520】【F:configs/default.yaml†L70-L95】
- warm 阶段的最优权重仅用于复制给教师，学生本身会沿着最后一次监督 epoch 的参数直接进入物理阶段，缺失“回滚到 warm 最优再出发”的环节，这与“warm 阶段得到的监督模型指导 warmup”这一需求相矛盾。

## 3. 监督与物理阶段的随机性仍不可控
- `pointdata.__getitem__` 在读取 CFD 点云后会就地 `np.random.shuffle`，但 `set_seed` 只设置了 Python 与 Torch 的随机数种子，NumPy 仍是非确定性；因此 warm/warmup 两个阶段每次迭代看到的点顺序可能不同，监督与物理残差的对齐会随运行漂移。【F:data/dataset.py†L180-L226】【F:train_tpac.py†L172-L200】
- 当前默认点数恰好等于样本总点数确实能缓解问题，但一旦调整采样密度或做数据增广，该不确定性就会重新暴露，建议至少在入口显式设置 NumPy 种子。

## 4. 物理正则的尺度对齐已经就绪
- 训练脚本会把 `data.pkl` 中的全局 min/max 传给 Trainer，并在创建 PINN 时启用 `denorm_physical`，保证监督和物理项共享相同的量纲，这一部分与需求一致。【F:train_tpac.py†L217-L240】【F:training/trainer.py†L145-L212】
- 但在缺少上述阶段管理之前，物理正则虽能正确计算，也无法确保 warm 教师与学生之间形成稳固的教师-学生配合。

> 综上，当前版本依旧无法自动完成“两阶段：warm 监督 → warmup 物理正则”的闭环；需要补充 checkpoint 载入、阶段状态机与随机性控制后，才能达到目标流程。
