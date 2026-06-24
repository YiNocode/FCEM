# OPEN MARL Checkpoint

此目录存放 OPEN MARL baseline 的预训练权重，**需自行训练生成**，仓库不附带 `default.pt`。

## 训练

```bash
pip install -r requirements-rl.txt

# 快速试跑（约数分钟）
python experiments/train_open_marl.py --total-steps 50000 --n-envs 16

# 论文级（建议 GPU，约 50 万步）
python experiments/train_open_marl.py --total-steps 500000 --n-envs 32
```

默认输出：`checkpoints/open_marl/default.pt`

超参见 [`config/baselines/open_marl.yaml`](../../config/baselines/open_marl.yaml)。

## 对比实验

训练完成后：

```bash
python experiments/run_comparison_2d.py --methods open_marl --trials 1 --scenarios free
```

可在 `config/default.yaml` 的 `baselines.open_marl.checkpoint_path` 中修改权重路径。
