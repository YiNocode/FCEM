"""Speed-pressure sweep: capture rate vs evader speed ratio (v_e / v_p)."""

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
from experiments.runner_args import add_dynamics_args, add_run_output_args, dynamics_overrides_from_args, run_output_kwargs_from_args
from experiments.runner_common import run_fcem_trial, trial_summary
from metrics.pre_capture import pre_capture_k_from_config, pre_capture_structure_metrics


def _ratio_tag(ratio: float) -> str:
    return str(ratio).replace(".", "p")


def _run_baseline_trial(
    method: str,
    scenario_name: str,
    trial_id: int,
    base_config: dict[str, Any],
    config_override: dict[str, Any],
    output_parts: tuple[str, ...],
    extra_summary: dict[str, Any],
) -> dict[str, Any]:
    cfg = load_config(scenario_name)
    cfg = deep_merge(cfg, config_override)
    cfg["seed"] = base_config.get("seed", 42) + trial_id
    obstacles = obstacles_from_scenario(cfg["scenario"])
    controller = METHODS[method](None)
    rng = np.random.default_rng(cfg["seed"])
    sim = Sim2D(cfg, obstacles, controller, rng)
    result = sim.run()

    from metrics.experiment_logger import ExperimentLogger

    out_dir = Path(base_config.get("output_dir", cfg["output_dir"])).joinpath(*output_parts, scenario_name)
    logger = ExperimentLogger(out_dir, method, scenario_name, trial_id, cfg)
    for frame in result["frames"]:
        logger.log_step(frame)
    summary = trial_summary(result, cfg["dt"], cfg)
    k_pre = pre_capture_k_from_config(cfg)
    summary.update(
        pre_capture_structure_metrics(
            result["frames"],
            summary.get("capture_step"),
            summary.get("captured", False),
            k=k_pre,
        )
    )
    summary.update(extra_summary)
    logger.finalize(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evader speed-ratio pressure sweep")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=None,
        help="Override speed_ratios from config (v_e/v_p)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_DIR / "experiments" / "speed_pressure.yaml"),
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
    methods = args.methods or exp_cfg["methods"]
    scenarios = args.scenarios or exp_cfg["scenarios"]
    ratios = args.ratios or exp_cfg.get("speed_ratios", [])
    pursuer_vmax = float(exp_cfg.get("pursuer_vmax", base.get("pursuer_vmax", 4.0)))
    ref_evader_vmax = float(base.get("evader_vmax", 8.0))
    ref_evader_amax = float(base.get("evader_amax", 4.0))
    scale_amax = bool(exp_cfg.get("scale_evader_amax", True))

    print(format_dynamics_line(base))
    print(f"Output: {base.get('output_dir')}")
    print(f"Fixed pursuer_vmax={pursuer_vmax}, ratios={ratios}, scale_evader_amax={scale_amax}")
    print(f"Trials={n_trials}, methods={methods}, scenarios={scenarios}\n")

    results: list[tuple[float, str, str, float, int]] = []
    for ratio in ratios:
        evader_vmax = pursuer_vmax * ratio
        evader_amax = (
            ref_evader_amax * (evader_vmax / ref_evader_vmax) if scale_amax else ref_evader_amax
        )
        ratio_tag = _ratio_tag(ratio)
        sweep_meta = {"sweep_param": "speed_ratio", "sweep_value": ratio}
        print(f"\n--- speed_ratio={ratio:.2f} (v_e={evader_vmax:.1f}, a_e={evader_amax:.2f} m/s²) ---")

        for method in methods:
            method_key = normalize_method(method)
            if method_key not in METHODS and method_key != "fcem":
                raise ValueError(f"Unknown method: {method}")
            for scenario in scenarios:
                scenario_override = deep_merge(
                    config_override_for_scenario(base, scenario),
                    {
                        "pursuer_vmax": pursuer_vmax,
                        "evader_vmax": evader_vmax,
                        "evader_amax": evader_amax,
                    },
                )
                output_parts = ("speed_pressure", f"ratio_{ratio_tag}", method_key)
                captures = 0
                for trial in range(n_trials):
                    if method_key == "fcem":
                        summary = run_fcem_trial(
                            method=method_key,
                            scenario_name=scenario,
                            trial_id=trial,
                            base_config=base,
                            ablation_flags=None,
                            output_parts=output_parts,
                            extra_summary=sweep_meta,
                            config_override=scenario_override,
                        )
                    else:
                        summary = _run_baseline_trial(
                            method_key,
                            scenario,
                            trial,
                            base,
                            scenario_override,
                            output_parts,
                            sweep_meta,
                        )
                    captures += int(summary["captured"])
                    status = "OK" if summary["captured"] else summary.get("failure_reason", "timeout")
                    print(
                        f"ratio={ratio:.2f} {method_key}/{scenario} trial {trial}: "
                        f"captured={summary['captured']}, {status}"
                    )
                rate = captures / n_trials
                results.append((ratio, method_key, scenario, rate, captures))
                print(f"==> ratio={ratio:.2f} {method_key}/{scenario}: {rate:.2%} ({captures}/{n_trials})\n")

    print("=== Summary ===")
    print(f"  {'ratio':>6}  {'method':15s} {'scenario':20s} {'success':>8}")
    for ratio, method, scenario, rate, captures in results:
        print(f"  {ratio:6.2f}  {method:15s} {scenario:20s} {rate:7.1%}")


if __name__ == "__main__":
    main()
