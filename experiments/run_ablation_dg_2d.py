"""FCEM layer ablation under differential-game evader.

Default execution is the requested 50-seed run over random_obstacles and
single_exit. Use --smoke-test for a tiny startup/logic check.
"""

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
from experiments.layer_registry import resolve_variant
from experiments.runner_args import (
    add_dynamics_args,
    add_run_output_args,
    dynamics_overrides_from_args,
    run_output_kwargs_from_args,
)
from experiments.runner_common import format_trial_progress_line, run_fcem_trial
from scripts.analyze_ablation_layers import (
    EXPECTED_SCENARIOS,
    EXPECTED_VARIANTS,
    analyze_ablation_run,
)


def _runtime_overrides(exp_cfg: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if "capture_mode" in exp_cfg:
        overrides["capture_mode"] = exp_cfg["capture_mode"]
    if isinstance(exp_cfg.get("runtime"), dict):
        overrides = deep_merge(overrides, exp_cfg["runtime"])
    return overrides


def _variant_flags(variant: dict[str, Any]) -> dict[str, bool]:
    if "flags" in variant:
        return dict(variant.get("flags") or {})
    remove_layer = variant.get("remove_layer")
    return resolve_variant(remove_layer=remove_layer)


def _validate_scope(
    exp_cfg: dict[str, Any],
    base: dict[str, Any],
    scenarios: list[str],
    variants: list[dict[str, Any]],
    planned_trials: int,
    *,
    smoke_test: bool,
) -> None:
    errors: list[str] = []
    variant_names = tuple(str(v.get("name", "")) for v in variants)
    if tuple(scenarios) != EXPECTED_SCENARIOS:
        errors.append(f"scenarios must be {list(EXPECTED_SCENARIOS)}, got {scenarios}")
    if variant_names != EXPECTED_VARIANTS:
        errors.append(f"variants must be {list(EXPECTED_VARIANTS)}, got {list(variant_names)}")
    policy = str(base.get("evader_policy", "")).lower()
    if policy not in {"differential_game", "differential-game", "game"}:
        errors.append(f"evader_policy must be differential_game/game, got {policy!r}")
    if str(base.get("capture_mode", "")).lower() != "escape_sector":
        errors.append(f"capture_mode must be escape_sector, got {base.get('capture_mode')!r}")
    configured_trials = int(exp_cfg.get("n_trials", base.get("n_trials", 0)))
    if configured_trials != 50:
        errors.append(f"configured n_trials must be 50, got {configured_trials}")
    if not smoke_test and planned_trials != 50:
        errors.append(f"planned trials must be 50 unless --smoke-test is used, got {planned_trials}")

    if errors:
        raise SystemExit("Invalid DG ablation scope:\n- " + "\n- ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FCEM DG layer ablation")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_DIR / "experiments" / "ablation_dg_50seed.yaml"),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run all variants/scenarios with one seed and a tiny horizon.",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=8,
        help="max_steps used by --smoke-test.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Do not write ablation_dg_50seed CSV/table/figure artifacts.",
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
    base = deep_merge(base, _runtime_overrides(exp_cfg))
    base["n_trials"] = int(exp_cfg.get("n_trials", base.get("n_trials", 50)))

    scenarios = list(args.scenarios or exp_cfg.get("scenarios", []))
    variants = list(exp_cfg.get("variants", []))
    n_trials = 1 if args.smoke_test else (args.trials if args.trials is not None else base["n_trials"])
    if args.smoke_test:
        base["max_steps"] = min(int(base.get("max_steps", args.smoke_steps)), args.smoke_steps)

    _validate_scope(
        exp_cfg,
        base,
        scenarios,
        variants,
        int(n_trials),
        smoke_test=args.smoke_test,
    )

    print(format_dynamics_line(base))
    print(f"Evader policy: {base.get('evader_policy')}")
    print(f"Capture mode: {base.get('capture_mode')}")
    print(f"Scenarios: {', '.join(scenarios)}")
    print(f"Variants: {', '.join(v['name'] for v in variants)}")
    print(f"Trials per scenario/variant: {n_trials}")
    if args.smoke_test:
        print(f"Smoke horizon: max_steps={base['max_steps']}")
    print(f"Output: {base.get('output_dir')}")

    for variant in variants:
        name = str(variant["name"])
        remove_layer = variant.get("remove_layer")
        flags = _variant_flags(variant)
        method = f"fcem_{name}"

        for scenario in scenarios:
            for trial in range(int(n_trials)):
                summary = run_fcem_trial(
                    method=method,
                    scenario_name=scenario,
                    trial_id=trial,
                    base_config=base,
                    ablation_flags=flags,
                    output_parts=("ablation_dg_50seed", name),
                    extra_summary={
                        "variant": name,
                        "remove_layer": remove_layer or "",
                    },
                    config_override=config_override_for_scenario(base, scenario),
                )
                prefix = f"{name}/{scenario} trial {trial}"
                print(format_trial_progress_line(summary, prefix=prefix))

    if not args.skip_analysis:
        outputs = analyze_ablation_run(Path(base["output_dir"]))
        print("\nArtifacts:")
        for label, path in outputs.items():
            print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
