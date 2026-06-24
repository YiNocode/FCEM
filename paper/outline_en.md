# FCEM Paper Outline (English)

## Title
Flow-Constrained Encirclement Manifold (FCEM) for Multi-Pursuer Capture in Cluttered Environments

## Abstract
- Problem: cooperative encirclement and capture of a faster-evading target with obstacles
- Method: escape-aware dynamic manifold, executability-aware assignment, guarded contraction
- Results: 3 pursuers, 40×40 m arena, four-section evaluation vs baselines and ablations

## 1. Introduction
- Multi-agent pursuit–evasion in robotics
- Limitations of pure pursuit and fixed-ring formations
- Contributions: four-layer FCEM pipeline, structure metrics, guarded contraction, structured experiments

## 2. Related Work
- Pursuit–evasion, APF, formation control, assignment

## 3. Problem Formulation
- Point-mass dynamics, workspace, obstacles
- Capture condition: all pursuers within capture_radius AND G_max ≤ G_max_allowed

## 4. FCEM Method
### 4.1 L1 / E1: Evader prediction (escape direction, manifold center)
### 4.2 L2 / E2: Candidate manifold generation
### 4.3 L3 / E3: Executability rollout + slot assignment + scoring
### 4.4 L4 / E4: Guarded contraction + low-level PD with slot velocity feedforward

## 5. Experiments (four sections)

### VI-A Experimental Setup
- Environments: 2D (main stats), PyBullet/PyFlyt 2.5D, Gazebo+PX4 SITL (showcase)
- Baselines: FCEM, pure pursuit APF, fixed ring APF
- Metrics: success rate, time-to-capture, D_ang, C_cov, G_max

### VI-B Layer-wise Validation (E1–E4)
- Tab. X: Layer → Experiment mapping (`layers.yaml`)
- Remove-one-layer variants w/o L1…L4 vs full FCEM
- Fig. layer-drop waterfall

### VI-C Comparative Evaluation
- 3 methods × 3 scenarios
- Fig. grouped bar chart (success rate + time-to-capture)
- Fig. normalized radar chart

### VI-D Ablation Study
- Progressive stacks: L1 → L1+L2 → L1+L2+L3 → full
- Sensitivity: G_max_allowed, R_init, contraction_rate, lookahead_time
- Fig. sensitivity line plots

## 6. Sim2Real
- PyFlyt 2.5D (WSL2)
- PX4 + Gazebo SITL + Mighty bridge (stub)

## 7. Conclusion

## Figures (planned)
- Four-layer pipeline diagram
- Tab. layer mapping
- Trajectory overlays per scenario
- Comparative bar / radar charts
- Layer-drop waterfall
- Hyperparameter sensitivity curves
- Structure metric time series
