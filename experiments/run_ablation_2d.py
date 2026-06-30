"""Legacy component-level ablation experiments (see ablation_components.yaml)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.config_loader import (
    CONFIG_DIR,
    build_experiment_base_config,
    config_override_for_scenario,
    format_dynamics_line,
    load_experiment_config,
)
from experiments.runner_args import add_dynamics_args, add_run_output_args, dynamics_overrides_from_args, run_output_kwargs_from_args
from experiments.runner_common import run_fcem_trial


def main() -> None:
    parser = argparse.ArgumentParser(description="Run legacy component ablation experiments")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_DIR / "experiments" / "ablation_components.yaml"),
    )
    add_dynamics_args(parser)
    add_run_output_args(parser)
    args = parser.parse_args()

    exp_cfg = load_experiment_config(Path(args.config))
    base = build_experiment_base_config(
        exp_cfg,
        dynamics_overrides_from_args(args),
        **run_output_kwargs_from_args(args),
    )
    n_trials = args.trials if args.trials is not None else base["n_trials"]
    scenarios = args.scenarios or exp_cfg.get("scenarios", [])
    print(format_dynamics_line(base))
    print(f"Output: {base.get('output_dir')}")

    for variant in exp_cfg["variants"]:
        name = variant["name"]
        flags = variant.get("flags", {})
        method = f"fcem_{name}"

        for scenario in scenarios:
            captures = 0
            for trial in range(n_trials):
                summary = run_fcem_trial(
                    method=method,
                    scenario_name=scenario,
                    trial_id=trial,
                    base_config=base,
                    ablation_flags=flags,
                    output_parts=("ablation", "components", name),
                    extra_summary={"variant": name},
                    config_override=config_override_for_scenario(base, scenario),
                )
                captures += int(summary["captured"])
                print(f"{name}/{scenario} trial {trial}: captured={summary['captured']}")
            print(f"==> {name}/{scenario} success rate: {captures / n_trials:.2%}")


if __name__ == "__main__":
    main()
