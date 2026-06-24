"""Shared helpers for 2D experiment runners."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from envs.sim2d import Sim2D, make_fcem_controller
from experiments.config_loader import load_config, obstacles_from_scenario
from metrics.experiment_logger import ExperimentLogger
from metrics.pre_capture import pre_capture_k_from_config, pre_capture_structure_metrics


def trial_summary(result: dict[str, Any], dt: float) -> dict[str, Any]:
    captured = bool(result["captured"])
    failed = bool(result.get("failed", False))
    capture_step = result.get("capture_step")
    failure_step = result.get("failure_step")
    time_to_capture_s = None
    if captured and capture_step is not None:
        time_to_capture_s = float(capture_step) * dt
    return {
        "captured": captured,
        "failed": failed,
        "failure_reason": result.get("failure_reason"),
        "failure_step": failure_step,
        "success": captured and not failed,
        "capture_step": capture_step,
        "num_steps": result["num_steps"],
        "time_to_capture_s": time_to_capture_s,
    }


def _safe_segment(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def run_fcem_trial(
    method: str,
    scenario_name: str,
    trial_id: int,
    base_config: dict[str, Any],
    ablation_flags: dict[str, bool] | None,
    output_parts: tuple[str, ...],
    extra_summary: dict[str, Any] | None = None,
    config_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = load_config(scenario_name, ablation_flags=ablation_flags)
    if config_override:
        from experiments.config_loader import deep_merge
        cfg = deep_merge(cfg, config_override)
    cfg["seed"] = base_config.get("seed", 42) + trial_id
    obstacles = obstacles_from_scenario(cfg["scenario"])
    controller = make_fcem_controller(ablation_flags or {})
    rng = np.random.default_rng(cfg["seed"])
    sim = Sim2D(cfg, obstacles, controller, rng)
    result = sim.run()

    safe_parts = tuple(_safe_segment(p) for p in output_parts)
    out_root = Path(base_config.get("output_dir", cfg["output_dir"]))
    out_dir = out_root.joinpath(*safe_parts, scenario_name)
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
    if extra_summary:
        summary.update(extra_summary)
    logger.finalize(summary)
    return summary


def mean_structure_from_frames(frames: list[dict[str, Any]]) -> dict[str, float]:
    if not frames:
        return {}
    d_ang, c_cov, g_max, c_sync = [], [], [], []
    for frame in frames:
        metrics = frame.get("metrics", {})
        d_ang.append(float(metrics.get("D_ang", 0.0)))
        c_cov.append(float(metrics.get("C_cov", 0.0)))
        g_max.append(math.degrees(float(metrics.get("G_max", 0.0))))
        c_sync.append(float(metrics.get("C_sync", frame.get("C_sync", 0.0))))
    return {
        "mean_D_ang": float(np.mean(d_ang)),
        "mean_C_cov": float(np.mean(c_cov)),
        "mean_G_max_deg": float(np.mean(g_max)),
        "mean_C_sync": float(np.mean(c_sync)),
    }
