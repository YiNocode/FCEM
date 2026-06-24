"""Batch comparison experiments in 2D."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.registry import METHODS, normalize_method
from envs.sim2d import Sim2D
from experiments.config_loader import (
    CONFIG_DIR,
    build_experiment_base_config,
    config_override_for_scenario,
    deep_merge,
    format_dynamics_line,
    load_config,
    load_experiment_config,
    obstacles_from_scenario,
)
from experiments.runner_args import add_dynamics_args, dynamics_overrides_from_args
from experiments.runner_common import run_fcem_trial, trial_summary
from metrics.pre_capture import pre_capture_k_from_config, pre_capture_structure_metrics


def run_trial(
    method: str,
    scenario_name: str,
    trial_id: int,
    base_config: dict[str, Any],
) -> dict[str, Any]:
    config_override = config_override_for_scenario(base_config, scenario_name)

    if method == "fcem":
        return run_fcem_trial(
            method=method,
            scenario_name=scenario_name,
            trial_id=trial_id,
            base_config=base_config,
            ablation_flags=None,
            output_parts=(method,),
            config_override=config_override,
        )

    cfg = load_config(scenario_name)
    cfg = deep_merge(cfg, config_override_for_scenario(base_config, scenario_name))
    cfg["seed"] = base_config.get("seed", 42) + trial_id
    obstacles = obstacles_from_scenario(cfg["scenario"])
    controller = METHODS[method](None)
    rng = np.random.default_rng(cfg["seed"])
    sim = Sim2D(cfg, obstacles, controller, rng)
    result = sim.run()

    from metrics.experiment_logger import ExperimentLogger

    out_dir = Path(base_config.get("output_dir", cfg["output_dir"])) / method / scenario_name
    logger = ExperimentLogger(out_dir, method, scenario_name, trial_id, cfg)
    for frame in result["frames"]:
        logger.log_step(frame)
    summary = trial_summary(result, cfg["dt"])
    k_pre = pre_capture_k_from_config(cfg)
    summary.update(
        pre_capture_structure_metrics(
            result["frames"],
            summary.get("capture_step"),
            summary.get("captured", False),
            k=k_pre,
        )
    )
    logger.finalize(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2D method comparison experiments")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_DIR / "experiments" / "comparison.yaml"),
        help="experiment config (use evader_faster_comparison.yaml for v_e > v_p)",
    )
    add_dynamics_args(parser)
    args = parser.parse_args()

    exp_cfg = load_experiment_config(Path(args.config))
    base = build_experiment_base_config(exp_cfg, dynamics_overrides_from_args(args))
    n_trials = args.trials if args.trials is not None else base["n_trials"]
    methods = args.methods or exp_cfg["methods"]
    scenarios = args.scenarios or exp_cfg["scenarios"]

    print(format_dynamics_line(base))
    print(f"Output: {base.get('output_dir')}")
    print(f"Trials={n_trials}, methods={methods}, scenarios={scenarios}\n")

    results: list[tuple[str, str, float, float | None, int]] = []
    for method in methods:
        method_key = normalize_method(method)
        if method_key not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        for scenario in scenarios:
            captures = 0
            times: list[float] = []
            for trial in range(n_trials):
                summary = run_trial(method_key, scenario, trial, base)
                captures += int(summary["captured"])
                if summary.get("time_to_capture_s") is not None:
                    times.append(float(summary["time_to_capture_s"]))
                status = "OK" if summary["captured"] else summary.get("failure_reason", "timeout")
                print(
                    f"{method}/{scenario} trial {trial}: "
                    f"captured={summary['captured']}, steps={summary['num_steps']}, {status}"
                )
            rate = captures / n_trials
            mean_t = sum(times) / len(times) if times else None
            results.append((method, scenario, rate, mean_t, captures))
            t_str = f", mean_capture={mean_t:.1f}s" if mean_t is not None else ""
            print(f"==> {method}/{scenario} capture rate: {rate:.2%} ({captures}/{n_trials}){t_str}\n")

    print("=== Summary ===")
    for method, scenario, rate, mean_t, captures in results:
        t_str = f"  mean_t={mean_t:.1f}s" if mean_t is not None else ""
        print(f"  {method:15s} {scenario:20s} {rate:.2%} ({captures}/{n_trials}){t_str}")


if __name__ == "__main__":
    main()
