"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from common.obstacles import Obstacle
from experiments.run_output import aggregate_section_name, make_run_dir, write_run_manifest

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_experiment_config(path: Path) -> dict[str, Any]:
    """Load experiment YAML, resolving optional ``_base`` chain."""
    cfg = load_yaml(path)
    base_name = cfg.pop("_base", None)
    if base_name:
        base_path = path.parent / base_name
        if not base_path.exists():
            base_path = CONFIG_DIR / "experiments" / base_name
        base_cfg = load_experiment_config(base_path) if base_path.exists() else {}
        cfg = deep_merge(base_cfg, cfg)
    return cfg


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(
    scenario_name: str | None = None,
    ablation_flags: dict[str, bool] | None = None,
) -> dict[str, Any]:
    cfg = load_yaml(CONFIG_DIR / "default.yaml")
    if scenario_name:
        scenario = load_yaml(CONFIG_DIR / "scenarios" / f"{scenario_name}.yaml")
        cfg["scenario"] = scenario
        scenario_dynamics = scenario.get("dynamics")
        if scenario_dynamics:
            cfg = deep_merge(cfg, scenario_dynamics)
        if "evader_policy" in scenario:
            cfg["evader_policy"] = scenario["evader_policy"]
        if "evader_game" in scenario:
            cfg["evader_game"] = deep_merge(cfg.get("evader_game", {}), scenario["evader_game"])
    if ablation_flags:
        cfg["ablation"] = deep_merge(cfg.get("ablation", {}), ablation_flags)
    return cfg


def obstacles_from_scenario(scenario: dict[str, Any]) -> list[Obstacle]:
    return [
        Obstacle(np.array(obs["center"], dtype=float), float(obs["radius"]))
        for obs in scenario.get("obstacles", [])
    ]


def dynamics_cli_overrides(
    pursuer_vmax: float | None = None,
    evader_vmax: float | None = None,
    pursuer_amax: float | None = None,
    evader_amax: float | None = None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if pursuer_vmax is not None:
        overrides["pursuer_vmax"] = pursuer_vmax
    if evader_vmax is not None:
        overrides["evader_vmax"] = evader_vmax
    if pursuer_amax is not None:
        overrides["pursuer_amax"] = pursuer_amax
    if evader_amax is not None:
        overrides["evader_amax"] = evader_amax
    return overrides


def build_experiment_base_config(
    exp_cfg: dict[str, Any],
    cli_overrides: dict[str, Any] | None = None,
    *,
    run_dir: str | Path | None = None,
    timestamped: bool = True,
    write_manifest: bool = True,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Load default config + experiment dynamics_file/dynamics + CLI overrides."""
    base = load_config()
    dynamics: dict[str, Any] = dict(exp_cfg.get("dynamics") or {})
    dyn_file = exp_cfg.get("dynamics_file")
    if dyn_file:
        dyn_path = CONFIG_DIR / dyn_file
        if not dyn_path.exists():
            dyn_path = Path(dyn_file)
        dynamics = deep_merge(dynamics, load_yaml(dyn_path))
    if cli_overrides:
        dynamics = deep_merge(dynamics, cli_overrides)
    base = deep_merge(base, dynamics)

    exp_name = str(exp_cfg.get("name") or exp_cfg.get("output_subdir") or "experiment")
    results_root = Path(base.get("output_dir", "results"))
    legacy = not timestamped and run_dir is None
    legacy_subdir = str(exp_cfg["output_subdir"]) if legacy and "output_subdir" in exp_cfg else None

    output_path = make_run_dir(
        results_root,
        exp_name,
        run_dir=Path(run_dir) if run_dir else None,
        timestamped=timestamped and run_dir is None,
        legacy_subdir=legacy_subdir,
    )
    base["output_dir"] = str(output_path)
    base["experiment_name"] = exp_name
    base["experiment_section"] = aggregate_section_name(exp_cfg)

    manifest_path = output_path / "run_manifest.json"
    if write_manifest and not manifest_path.exists():
        vp = float(base.get("pursuer_vmax", 0.0))
        ve = float(base.get("evader_vmax", 0.0))
        write_run_manifest(
            output_path,
            exp_cfg,
            config_path=config_path,
            extra={
                "pursuer_vmax": vp,
                "pursuer_amax": float(base.get("pursuer_amax", 0.0)),
                "evader_vmax": ve,
                "evader_amax": float(base.get("evader_amax", 0.0)),
                "evader_policy": base.get("evader_policy"),
                "speed_ratio": round(ve / vp, 4) if vp > 0 else None,
            },
        )

    return base


def config_override_for_scenario(
    base_config: dict[str, Any],
    scenario_name: str,
) -> dict[str, Any]:
    """Merge base experiment config without clobbering per-scenario dynamics."""
    skip = {"seed", "output_dir", "n_trials", "scenario"}
    override = {k: v for k, v in base_config.items() if k not in skip}
    scenario = load_yaml(CONFIG_DIR / "scenarios" / f"{scenario_name}.yaml")
    for key in scenario.get("dynamics") or {}:
        override.pop(key, None)
    return override


def format_dynamics_line(cfg: dict[str, Any]) -> str:
    vp = float(cfg["pursuer_vmax"])
    ve = float(cfg["evader_vmax"])
    return f"Dynamics: pursuer_vmax={vp}, evader_vmax={ve} (ratio v_e/v_p={ve / vp:.2f})"
