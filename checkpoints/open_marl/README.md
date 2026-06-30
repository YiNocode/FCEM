# OPEN MARL Checkpoint

**v3 奖励/课程修复**（2026-06）：若此前训练 strict_capture 始终为 0%，请删除旧 checkpoint 并用下方命令重训。

此目录存放 OPEN MARL baseline 的预训练权重，**需自行训练生成**。

## 训练（两阶段，论文 Sec IV-D）

```bash
pip install -r requirements-rl.txt

# 快速 smoke（数分钟）
python experiments/train_open_marl.py --stage 1 --total-steps 5000 --n-envs 8

# Stage 1：无 smoothness，课程学习 + AEG（约 6–8h，GPU 推荐）
python experiments/train_open_marl.py --stage 1 --device cuda --n-envs 64

# Stage 2：从 best.pt 续训，加入 smoothness（约 2–4h）
python experiments/train_open_marl.py --stage 2 --resume checkpoints/open_marl/best.pt --device cuda

# 中断续训（自动从 checkpoint 的 global_step 继续）
python experiments/train_open_marl.py --stage 1 --resume checkpoints/open_marl/latest.pt --device cuda --n-envs 64
```

输出：
- `checkpoints/open_marl/default.pt` — 训练结束权重
- `checkpoints/open_marl/latest.pt` — 每次 eval 自动保存（中断续训用）
- `checkpoints/open_marl/best.pt` — 评估 **FCEM 严格捕获率**最高时保存（推荐用于对比）

超参见 [`config/baselines/open_marl.yaml`](../../config/baselines/open_marl.yaml)。

## 对比实验

```bash
python experiments/run_comparison_2d.py --methods open_marl --trials 50 --scenarios free
```

可在 `config/default.yaml` 的 `baselines.open_marl.checkpoint_path` 中修改权重路径。

**注意**：v2 checkpoint 与旧版不兼容，需用上述命令重新训练。
