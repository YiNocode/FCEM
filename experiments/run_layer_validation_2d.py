"""Layer-wise validation: remove one layer at a time (E1–E4)."""

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
    deep_merge,
    format_dynamics_line,
    load_experiment_config,
)
from experiments.layer_registry import resolve_variant
from experiments.runner_args import add_dynamics_args, add_run_output_args, dynamics_overrides_from_args, run_output_kwargs_from_args
from experiments.runner_common import run_fcem_trial


def _print_delta(
    variant: str,
    scenario: str,
    rate: float,
    mean_ttc: float | None,
    full_rate: float,
    full_ttc: float | None,
) -> None:
    delta_rate = rate - full_rate
    if full_ttc is not None and mean_ttc is not None:
        delta_ttc = mean_ttc - full_ttc
        print(
            f"  {variant}/{scenario}: success={rate:.2%} (Δ{delta_rate:+.2%}), "
            f"ttc={mean_ttc:.1f}s (Δ{delta_ttc:+.1f}s)"
        )
    else:
        print(f"  {variant}/{scenario}: success={rate:.2%} (Δ{delta_rate:+.2%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run layer-wise validation (E1–E4)")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_DIR / "experiments" / "layer_validation.yaml"),
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

    stats: dict[tuple[str, str], dict] = {}
    for variant in exp_cfg["variants"]:
        name = variant["name"]
        remove_layer = variant.get("remove_layer")
        flags = resolve_variant(remove_layer=remove_layer)
        method = f"fcem_{name}"

        for scenario in scenarios:
            captures = 0
            ttc_list: list[float] = []
            for trial in range(n_trials):
                summary = run_fcem_trial(
                    method=method,
                    scenario_name=scenario,
                    trial_id=trial,
                    base_config=base,
                    ablation_flags=flags,
                    output_parts=("layer_validation", name),
                    extra_summary={"variant": name, "remove_layer": remove_layer or ""},
                    config_override=config_override_for_scenario(base, scenario),
                )
                captures += int(summary["captured"])
                if summary.get("time_to_capture_s") is not None:
                    ttc_list.append(summary["time_to_capture_s"])
                print(f"{name}/{scenario} trial {trial}: captured={summary['captured']}")

            rate = captures / n_trials
            mean_ttc = sum(ttc_list) / len(ttc_list) if ttc_list else None
            stats[(name, scenario)] = {"rate": rate, "mean_ttc": mean_ttc}
            print(f"==> {name}/{scenario} success rate: {rate:.2%} ({captures}/{n_trials})")

    print("\n=== Layer drop vs full ===")
    for scenario in scenarios:
        full = stats.get(("full", scenario), {})
        full_rate = full.get("rate", 0.0)
        full_ttc = full.get("mean_ttc")
        for variant in exp_cfg["variants"]:
            name = variant["name"]
            if name == "full":
                continue
            s = stats.get((name, scenario), {})
            _print_delta(
                name, scenario, s.get("rate", 0.0), s.get("mean_ttc"),
                full_rate, full_ttc,
            )


if __name__ == "__main__":
    main()
