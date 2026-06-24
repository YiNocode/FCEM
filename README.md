# FCEM — Flow-Constrained Encirclement Manifold

面向 TRO 论文的多追捕者围捕实验框架。支持四段式实验流水线（Setup / Layer-wise / Comparative / Ablation），2D 批量验证，以及 PyFlyt / PX4（WSL2）可选扩展。

## 环境要求

- **Windows / Linux**：2D 实验（`numpy`, `matplotlib`, `pyyaml`）
- **WSL2**（可选）：PyFlyt 2.5D、PX4 + Gazebo SITL + Mighty + ROS2

```bash
pip install -r requirements.txt
```

## 实时可视化（Vue）

浏览器内实时观看 2D 围捕过程，可调算法、消融层、速度、场景与障碍密度。

**终端 1 — Python 后端：**

```bash
pip install fastapi "uvicorn[standard]"
python scripts/run_viz_server.py --reload
```

默认端口 **18888**（若被占用可用 `--port 19090` 等）。

**终端 2 — Vue 前端（开发模式）：**

```bash
cd viz
npm install
npm run dev
```

浏览器打开 `http://localhost:5173`。

| 可调参数 | 说明 |
|----------|------|
| 追捕算法 | FCEM + 全部 baseline（10 种，含 OPEN MARL） |
| 场景 | 空旷 / 随机障碍 / 单出口 U 形 |
| 障碍物数量 | 随机场景下圆柱个数（0–24） |
| 场景边长 | 正方形场地边长（m） |
| v_p / v_e | 追捕者、逃逸者最大速度 |
| FCEM 层消融 | 勾选 L1–L4 逐层移除（仅 FCEM） |
| 逃逸策略 | `game` 微分博弈 / `apf` 势场 |

生产部署：在 `viz/` 下执行 `npm run build`，再只启动后端；`viz/dist` 会由 FastAPI 静态托管。

## 实验结构（论文 Section VI）

### 1. Experimental Setup

| Tier | 论文环境 | 代码入口 | 状态 |
|------|----------|----------|------|
| T1 | 2D point-mass | `envs/sim2d.py` | 可跑 |
| T2 | PyBullet / PyFlyt 2.5D | `envs/pyflyt_env.py`, `run_pyflyt.py` | 接口占位 |
| T3 | Gazebo + PX4 SITL | `envs/px4_env.py`, `run_px4.sh` | 接口占位 |

- **Baselines**：`fcem`, `pure_pursuit`, `fixed_ring`，以及文献对标方法（见下）
- **场景**：`free`, `random_obstacles`, `single_exit`
- **指标**：success rate, time-to-capture, D_ang, C_cov, G_max（定义见 `metrics/structure.py`）
- **配置**：[`config/experiments/setup.yaml`](config/experiments/setup.yaml)

### 2. Layer-wise Validation（E1–E4）

逐层移除验证，映射表见 [`config/experiments/layers.yaml`](config/experiments/layers.yaml)：

| Layer | Experiment | 模块 |
|-------|------------|------|
| L1 | E1 | 逃逸预测 |
| L2 | E2 | 多候选流形 |
| L3 | E3 | 可执行性分配 |
| L4 | E4 | 槽位速度前馈 + 低层跟踪（门控收缩默认关闭） |

```bash
python experiments/run_layer_validation_2d.py --trials 1 --scenarios free
python scripts/generate_layer_table.py
```

### 3. Comparative Evaluation

默认 `comparison.yaml` 含 **10 方法**（FCEM + 全部 baseline，含 OPEN MARL）× **3 场景**，动力学 v_e/v_p = 2.5（见 `setup.yaml` → `dynamics/evader_faster.yaml`）。

| 方法 ID | 文献 |
|---------|------|
| `fcem` / `pure_pursuit` / `fixed_ring` | 本文 / APF / 固定环 |
| `deghat_circumnavigation` | Deghat et al., IROS 2012 |
| `kou_xiang_fencing` | Kou & Xiang, 自动化学报 2022 |
| `fang_relay_2022` | Fang et al., IEEE T-Cybernetics 2022 |
| `relay_pursuit` | 中继追捕 / Voronoi+Apollonius 切换 |
| `liao_mpc` | Liao et al., ICRA 2021 |
| `yu_consensus` | Yu et al., Automatica 2018 |
| `open_marl` | Chen et al., OPEN (MARL, 2D 质点适配) |

```bash
# smoke（全方法）
python experiments/run_comparison_2d.py --trials 1 --scenarios free

# 子集快速试跑
python experiments/run_comparison_2d.py --trials 1 --methods fcem fixed_ring --scenarios free

# 论文统计
python experiments/run_comparison_2d.py --trials 50
python scripts/aggregate_results.py --results-dir results/comparison
```

`literature_comparison.yaml` 与 `comparison.yaml` 方法列表相同，仅输出目录为 `comparison_literature/`（兼容旧路径）。

### OPEN MARL baseline（需自行训练）

OPEN（[Chen et al.](https://arxiv.org/abs/2409.15866)）为 EPN + MAPPO 的多智能体强化学习基线，已在 FCEM 内做 **2D 质点适配**（不含 Isaac Sim / AEG）。仓库**不附带**预训练权重。

```bash
pip install -r requirements-rl.txt

# 训练（输出 checkpoints/open_marl/default.pt）
python experiments/train_open_marl.py --total-steps 500000 --n-envs 32

# 训练完成后参与对比
python experiments/run_comparison_2d.py --methods open_marl --trials 1 --scenarios free
```

超参见 [`config/baselines/open_marl.yaml`](config/baselines/open_marl.yaml)；checkpoint 路径可在 `config/default.yaml` → `baselines.open_marl.checkpoint_path` 修改。

### 4. Ablation Study

层间递进叠加 + 超参敏感性：

```bash
python experiments/run_combination_ablation_2d.py --trials 1 --scenarios free
python experiments/run_sensitivity_2d.py --trials 1 --scenarios free
```

**Legacy**（组件级消融，附录/调试用）：`run_ablation_2d.py` + `ablation_components.yaml`

### 一键运行

```bash
# smoke test（comparison 段含 9 方法 × 3 场景，建议先限场景）
python experiments/run_all_experiments.py --trials 1 --sections comparison --scenarios free

# 论文统计（耗时较长）
python experiments/run_all_experiments.py --trials 50
```

### 动画演示

```bash
python fcem_demo.py
```

## 项目结构

```
common/          # 动力学、障碍物、逃逸者 APF、捕获判定
metrics/         # 结构度量 D_ang, C_cov, G_max, C_col；实验日志
fcem/            # 逃逸预测、流形生成、槽位分配、PD 跟踪
envs/            # sim2d, pyflyt_env, px4_env
baselines/       # pure_pursuit, fixed_ring
config/experiments/
  setup.yaml              # §1 实验设置
  layers.yaml               # L1–L4 定义
  layer_validation.yaml     # §2
  comparison.yaml           # §3
  ablation_combination.yaml # §4 层叠加
  ablation_sensitivity.yaml # §4 超参
experiments/     # 各段运行器 + run_all_experiments.py
scripts/         # 聚合、汇总、作图
paper/           # 论文大纲
```

## 指标

| 指标 | 说明 |
|------|------|
| success rate | 捕获成功 trial 比例 |
| time-to-capture | 成功 trial 的 `capture_step × dt`（秒） |
| D_ang | 角分布均匀度 |
| C_cov | 角覆盖度 |
| G_max | 最大逃逸角间隙（度） |
| C_sync | 同步到达一致性 |
| pre_capture_* | 捕获前最后 K 步（默认 K=10）的 D_ang、C_cov、G_max、C_sync 均值；仅统计成功 trial |

`pre_capture_window` 可在 `config/default.yaml` 或 `config/experiments/setup.yaml` 的 `metrics` 段配置。

## 结果分析

```bash
python scripts/aggregate_results.py --results-dir results
python scripts/summarize_experiments.py --csv results/aggregated.csv
python scripts/compare_methods.py --csv results/aggregated.csv --section comparison
python scripts/plot_comparison_bar.py
python scripts/plot_comparison_radar.py
python scripts/plot_layer_drop.py
python scripts/plot_sensitivity.py
# 单次 trial
python scripts/visualize_run.py results/comparison/fcem/free/fcem_free_t000.json
# 文件夹内全部 trial（默认递归；输出为同目录 *.viz.png）
python scripts/visualize_run.py results/comparison/fcem/free
python scripts/visualize_run.py --dir results/comparison --out-dir results/figures/trajectories
```

结果目录：

```
results/
  layer_validation/     # §2
  comparison/           # §3
  ablation/combination/ # §4
  ablation/sensitivity/
  summary/              # 分组汇总 CSV
  figures/              # 论文图表
```

## 算法流程（每控制步）

1. 读取状态 → 预测逃逸方向 → 预测流形中心
2. 生成候选流形 → 采样槽位 → 评估可执行性
3. 槽位分配 → 打分 → 选最优 → 结构度量 → 固定速率收缩 R → 槽位速度 → 低层跟踪

## 逃逸者策略

默认使用 **微分博弈逃逸**（`evader_policy: game`）：

- **Minimax 清距**：在离散航向上网格搜索，最大化“追捕者纯追击预测后”的最小距离（Isaacs 离散近似）
- **最大角隙突破**：朝追捕者包围圈的最大空角平分方向逃逸
- 保留障碍/边界势场约束

切换回旧版 APF：`evader_policy: apf`（或场景 YAML 中覆盖，如 `fixed_ring_failure` 演示场景）。

## 默认参数

| 参数 | 值 |
|------|-----|
| 逃逸者策略 | `game`（minimax + 角隙突破） |
| 实验套件 v_e/v_p | 2.5（`dynamics/evader_faster.yaml`） |
| 门控收缩 | 默认关闭（`ablate_no_guarded_contraction: true`）；附录可用 `with_guarded_contraction` 重开 |
| R_init / R_terminal | 12.0 / 1.2 m |
| capture_radius | 1.8 m |
| G_max_allowed | 140° |
| dt / n_trials | 0.10 s / 50 |

## PyFlyt / PX4（WSL2）

PyFlyt 实验与 2D 套件共用 `setup.yaml` 中的 `dynamics_file`（默认 v_e/v_p = 2.5）：

```bash
python experiments/run_pyflyt.py --scenario free --method fcem
python experiments/run_pyflyt.py --config config/experiments/comparison.yaml --trials 1
bash experiments/run_px4.sh
```

## 引用

若使用本代码，请引用 FCEM TRO 论文（待发表）。
