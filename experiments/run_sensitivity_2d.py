"""Hyperparameter sensitivity sweeps (one-at-a-time)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.config_loader import (
    CONFIG_DIR,
    build_experiment_base_config,
    config_override_for_scenario,
    deep_merge,
    format_dynamics_line,
    load_experiment_config,
)
from experiments.runner_args import add_dynamics_args, dynamics_overrides_from_args
from experiments.runner_common import run_fcem_trial


def _set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _config_override(param: str, value: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if "." in param:
        _set_nested(cfg, param, value)
    else:
        cfg[param] = value
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hyperparameter sensitivity sweeps")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_DIR / "experiments" / "ablation_sensitivity.yaml"),
    )
    add_dynamics_args(parser)
    args = parser.parse_args()

    exp_cfg = load_experiment_config(Path(args.config))
    base = build_experiment_base_config(exp_cfg, dynamics_overrides_from_args(args))
    n_trials = args.trials if args.trials is not None else base["n_trials"]
    scenarios = args.scenarios or exp_cfg.get("scenarios", [])
    print(format_dynamics_line(base))

    for sweep in exp_cfg["sweeps"]:
        param = sweep["param"]
        for value in sweep["values"]:
            tag = str(value).replace(".", "p")
            method = f"fcem_sens_{param.replace('.', '_')}_{tag}"
            for scenario in scenarios:
                override = deep_merge(
                    config_override_for_scenario(base, scenario),
                    _config_override(param, value),
                )
                captures = 0
                for trial in range(n_trials):
                    summary = run_fcem_trial(
                        method=method,
                        scenario_name=scenario,
                        trial_id=trial,
                        base_config=base,
                        ablation_flags={},
                        output_parts=("ablation", "sensitivity", param.replace(".", "_"), tag),
                        extra_summary={"sweep_param": param, "sweep_value": value},
                        config_override=override,
                    )
                    captures += int(summary["captured"])
                    print(f"{param}={value}/{scenario} trial {trial}: captured={summary['captured']}")
                print(f"==> {param}={value}/{scenario} success rate: {captures / n_trials:.2%}")


if __name__ == "__main__":
    main()
