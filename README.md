# FCEM - Flow-Constrained Encirclement Manifold

FCEM is a pursuit-evasion research codebase for structure-preserving multi-pursuer capture in cluttered 2D environments. The current reproducible path is the 2D point-mass benchmark; PyFlyt and PX4 interfaces are present as extension stubs.

## Setup

```bash
pip install -r requirements.txt
```

Optional OPEN MARL training/inference requires PyTorch:

```bash
pip install -r requirements-rl.txt
```

## What Is Implemented

| Area | Status | Entry points |
| --- | --- | --- |
| 2D FCEM simulation | Runnable | `envs/sim2d.py`, `experiments/run_comparison_2d.py` |
| Vue visualization | Runnable | `scripts/run_viz_server.py`, `viz/` |
| Baseline comparison | Runnable | `fcem`, `pure_pursuit`, `liao_mpc`, `ac_baseline` |
| OPEN MARL baseline | Optional | requires a local checkpoint under `checkpoints/open_marl/` |
| PyFlyt / PX4 | Interface stubs | `envs/pyflyt_env.py`, `envs/px4_env.py` |

Removed legacy baselines are no longer registered or configured. The active method registry is [`baselines/registry.py`](baselines/registry.py).

## Run Experiments

Main comparison, using the default faster evader dynamics (`v_e/v_p = 2.0`, `v_p=4.0`, `v_e=8.0`) and differential-game evader:

```bash
python experiments/run_comparison_2d.py --trials 1 --scenarios free
python experiments/run_comparison_2d.py --trials 50
```

Default comparison methods are:

```text
fcem pure_pursuit liao_mpc ac_baseline
```

To include OPEN MARL after training a checkpoint:

```bash
python experiments/run_comparison_2d.py --methods open_marl --trials 50
```

Speed-pressure sweep:

```bash
python experiments/run_speed_pressure_2d.py --trials 1 --ratios 1.0 2.0 3.0
python experiments/run_speed_pressure_2d.py --trials 50
```

Layer validation and ablations:

```bash
python experiments/run_layer_validation_2d.py --trials 1 --scenarios free
python experiments/run_combination_ablation_2d.py --trials 1 --scenarios free
python experiments/run_sensitivity_2d.py --trials 1 --scenarios free
```

Focused DG layer ablation (random_obstacles + single_exit only, 50 seeds,
escape-sector capture):

```bash
python experiments/run_ablation_dg_2d.py --smoke-test
python experiments/run_ablation_dg_2d.py
```

The DG ablation writes `ablation_dg_50seed_per_trial.csv`,
`ablation_dg_50seed_summary.csv`, `fig_ablation_layers.png`, and
`table_ablation_layers.md` in the run directory.

Run a suite:

```bash
python experiments/run_all_experiments.py --trials 1 --sections comparison --scenarios free
python experiments/run_all_experiments.py --trials 50
```

Each run writes to `results/<YYYYMMDD_HHMMSS>_<experiment>/` unless `--run-dir` or legacy output options are used.

## Analyze Results

```bash
python scripts/analyze_run.py --run-dir results/<YYYYMMDD_HHMMSS>_comparison
```

This aggregates trial JSON, writes summaries, compares methods, and generates the figures available for that experiment type.

Useful lower-level commands:

```bash
python scripts/aggregate_results.py --run-dir results/<YYYYMMDD_HHMMSS>_comparison
python scripts/summarize_experiments.py --run-dir results/<YYYYMMDD_HHMMSS>_comparison
python scripts/compare_methods.py --run-dir results/<YYYYMMDD_HHMMSS>_comparison --section comparison
python scripts/plot_comparison_capture_structure.py --run-dir results/<YYYYMMDD_HHMMSS>_comparison
python scripts/visualize_run.py results/<YYYYMMDD_HHMMSS>_comparison/fcem/free/fcem_free_t000.json
```

## Visualization

Backend:

```bash
pip install fastapi "uvicorn[standard]"
python scripts/run_viz_server.py --reload
```

Frontend:

```bash
cd viz
npm install
npm run dev
```

Open `http://localhost:5173`. The backend default port is `18888`; pass `--port 19090` if needed.

## Metrics

Core structural metrics live in [`metrics/structure.py`](metrics/structure.py):

| Metric | Meaning |
| --- | --- |
| `D_ang` | angular gap uniformity |
| `C_cov` | angular coverage score |
| `G_max` | largest full-circle escape gap, radians |
| `C_col` | centroid offset relative to mean radius |
| `C_sync` | synchronized arrival coverage |
| `pre_capture_*` | aggregate structural metrics over the final pre-capture window |

The active capture mode is configured in [`config/default.yaml`](config/default.yaml). The default is `escape_sector`, which combines distance and escape-sector closure instead of distance-only success.

## OPEN MARL

OPEN MARL checkpoints are not shipped with the repository. Train one locally:

```bash
python experiments/train_open_marl.py --stage 1 --total-steps 5000 --n-envs 8
python experiments/train_open_marl.py --stage 1 --device cuda --n-envs 64
python experiments/train_open_marl.py --stage 2 --resume checkpoints/open_marl/best.pt --device cuda
```

See [`checkpoints/open_marl/README.md`](checkpoints/open_marl/README.md) and [`config/baselines/open_marl.yaml`](config/baselines/open_marl.yaml).

## Project Layout

```text
common/          dynamics, capture checks, evader policies, obstacles
fcem/            manifold generation, contraction, controller, slot assignment
baselines/       active baselines and method registry
metrics/         structural, synchronization, comparison, and logging metrics
envs/            2D simulation plus PyFlyt/PX4 interfaces
experiments/     batch runners and config loading
scripts/         aggregation, plotting, analysis, visualization helpers
config/          default, scenario, baseline, and experiment YAML
paper/           TRO draft and outline
tests/           smoke and unit tests
viz/             Vue frontend
```

## Test

```bash
pytest -q
```

Current audit status: `78 passed` on 2026-06-30, with environment warnings from SciPy/PyTorch NumPy compatibility.
