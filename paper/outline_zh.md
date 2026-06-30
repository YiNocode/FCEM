# FCEM 论文大纲（中文）

## 标题
面向杂乱环境的流形约束围捕（FCEM）多机协同捕获方法

## 摘要
- 问题：多追捕者在障碍环境中围捕更快逃逸目标
- 方法：逃逸感知动态流形、可执行性评估分配、门控收缩
- 结果：三机、40×40 m 场地，2D 主实验 + 四层验证 + 基线对比 + 消融；PyFlyt/PX4 作为扩展接口

## 1. 引言
- 多智能体追逃在机器人中的应用
- 纯追踪与固定环形的局限
- 贡献：FCEM 四层流水线、结构度量、门控收缩、四段式实验框架

## 2. 相关工作
- 追逃、人工势场、编队控制、任务分配

### Baseline 映射（2D 质点适配）

| 代码 ID | 类别 | 说明 |
|---------|------|------|
| `pure_pursuit` | 简单基线 | 纯追踪 + Hermite 局部规划 |
| `ac_baseline` | 几何追逃 | 阿波罗尼斯圆安全区域 $S_E$ 面积收缩 |
| `liao_mpc` | MPC 协同狩猎 | Liao et al., ICRA 2021 |
| `open_marl` | MARL（EPN + MAPPO） | Chen et al., OPEN — 可选 baseline，需自行训练 checkpoint |

实验配置：[`config/experiments/comparison.yaml`](../config/experiments/comparison.yaml)

## 3. 问题建模
- 质点二阶动力学、工作空间、障碍
- 捕获条件：默认 `escape_sector`，所有追捕者进入 capture_radius 且可行逃逸扇区满足 `G_esc ≤ 30°`, `C_esc ≥ 0.85`；`G_max ≤ 140°` 保留为 full-circle 结构指标/兼容模式

## 4. FCEM 方法
### 4.1 L1 / E1：逃逸者预测（逃逸方向、流形中心）
### 4.2 L2 / E2：候选流形生成（相位/半径变体、障碍感知半径）
### 4.3 L3 / E3：可执行性 rollout + 槽位分配 + 打分
### 4.4 L4 / E4：门控收缩（D_ang, C_cov, G_max）+ 低层 PD 与槽位速度前馈

## 5. 实验（四段结构）

### VI-A Experimental Setup
- 仿真环境：2D（主统计）；PyBullet/PyFlyt 2.5D、Gazebo+PX4 SITL 为扩展接口
- Baselines：默认 FCEM、pure_pursuit、ac_baseline、liao_mpc；open_marl 需 checkpoint 后手动加入
- Metrics：success rate, time-to-capture, D_ang, C_cov, G_max, C_sync, C_esc, G_esc

### VI-B Layer-wise Validation（E1–E4）
- Tab. X：Layer → Experiment 映射（`layers.yaml`）
- 逐层移除 w/o L1…L4，报告相对 full 的性能下降
- Fig. layer-drop waterfall

### VI-C Comparative Evaluation
- 四方法 × 三场景（默认 comparison；open_marl 单独可选）
- Fig. 多场景柱状图（success rate + time-to-capture）
- Fig. 雷达图（归一化多维指标）

### VI-D Ablation Study
- 层间递进：L1 → L1+L2 → L1+L2+L3 → full
- 超参敏感性：G_max_allowed, R_init, contraction_rate, lookahead_time
- Fig. 敏感性折线图

## 6. 仿真扩展
- PyFlyt 2.5D（WSL2，接口路径）
- PX4 + Gazebo SITL + Mighty 桥接（桩实现）

## 7. 结论

## 图表清单
- 算法流程图（四层）
- Tab. layer mapping
- Fig. 轨迹叠加
- Fig. 对比柱状图 / 雷达图
- Fig. layer-drop waterfall
- Fig. 超参敏感性
- Fig. 结构度量时序曲线
