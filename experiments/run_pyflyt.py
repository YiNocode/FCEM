"""PyFlyt experiment runner (2.5D). Falls back gracefully if PyFlyt unavailable."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.registry import METHODS, SLOT_METHODS, normalize_method
from envs.pyflyt_env import PyFlytEnv
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
from experiments.runner_common import trial_summary
from metrics.experiment_logger import ExperimentLogger
from metrics.pre_capture import pre_capture_k_from_config, pre_capture_structure_metrics


def run_trial(method: str, scenario_name: str, trial_id: int, base_config: dict) -> dict:
    cfg = load_config(scenario_name)
    cfg = deep_merge(cfg, config_override_for_scenario(base_config, scenario_name))
    cfg["seed"] = base_config.get("seed", 42) + trial_id
    if "max_steps" in base_config:
        cfg["max_steps"] = base_config["max_steps"]
    if "pyflyt" in base_config:
        cfg["pyflyt"] = deep_merge(cfg.get("pyflyt", {}), base_config["pyflyt"])
    obstacles = obstacles_from_scenario(cfg["scenario"])
    controller = METHODS[method](None)
    rng = np.random.default_rng(cfg["seed"])

    env = PyFlytEnv(
        cfg,
        obstacles,
        controller,
        rng,
        use_plan_state=method in SLOT_METHODS,
    )
    try:
        env.reset()
        result = env.run()
    finally:
        env.close()

    out_dir = Path(base_config.get("output_dir", cfg["output_dir"])) / "pyflyt" / method / scenario_name
    logger = ExperimentLogger(out_dir, method, scenario_name, trial_id, cfg)
    for frame in result["frames"]:
        logger.log_step(frame)
    summary = trial_summary(result, cfg["dt"])
    summary["backend"] = "pyflyt"
    summary["method"] = method
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
    parser = argparse.ArgumentParser(description="Run PyFlyt 2.5D comparison experiments")
    parser.add_argument("--scenario", type=str, default="free")
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--method", type=str, default=None, choices=list(METHODS))
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="override max_steps for quick viz")
    parser.add_argument("--render", action="store_true", help="force PyBullet GUI on")
    parser.add_argument("--no-render", action="store_true", help="disable GUI")
    parser.add_argument("--verbose", action="store_true", help="print per-step positions/distances")
    parser.add_argument("--log-every", type=int, default=None, help="print every N steps (default from config)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_DIR / "experiments" / "comparison.yaml"),
        help="experiment config for default methods/scenarios",
    )
    add_dynamics_args(parser)
    args = parser.parse_args()

    if not PyFlytEnv.check():
        print("PyFlyt not installed — skipping 3D experiment.")
        print("Install on WSL2/Linux: pip install pyflyt")
        sys.exit(0)

    exp_cfg = load_experiment_config(Path(args.config))
    base = build_experiment_base_config(exp_cfg, dynamics_overrides_from_args(args))
    if args.max_steps is not None:
        base["max_steps"] = args.max_steps
    if args.render:
        base["pyflyt"] = {**base.get("pyflyt", {}), "render": True}
    if args.no_render:
        base["pyflyt"] = {**base.get("pyflyt", {}), "render": False}
    pyflyt_overrides: dict = {}
    if args.verbose or args.render:
        pyflyt_overrides["verbose"] = True
    if args.log_every is not None:
        pyflyt_overrides["log_every"] = max(1, args.log_every)
    if pyflyt_overrides:
        base["pyflyt"] = {**base.get("pyflyt", {}), **pyflyt_overrides}

    print(format_dynamics_line(base))

    if args.methods:
        methods = args.methods
    elif args.method:
        methods = [args.method]
    else:
        methods = exp_cfg.get("methods", ["fcem"])

    scenarios = args.scenarios or [args.scenario]

    results = []
    for method in methods:
        method_key = normalize_method(method)
        if method_key not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        for scenario in scenarios:
            captures = 0
            for trial in range(args.trials):
                summary = run_trial(method_key, scenario, trial, base)
                captures += int(summary["captured"])
                print(
                    f"{method}/{scenario} trial {trial}: "
                    f"captured={summary['captured']}, steps={summary['num_steps']}"
                )
            rate = captures / args.trials
            results.append((method, scenario, rate))
            print(f"==> {method}/{scenario} capture rate: {rate:.2%} ({captures}/{args.trials})")

    if len(results) > 1:
        print("\n=== Summary ===")
        for method, scenario, rate in results:
            print(f"  {method:15s} {scenario:20s} {rate:.2%}")


if __name__ == "__main__":
    main()
