"""Shared helpers for 2D experiment runners."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from envs.sim2d import Sim2D, make_fcem_controller
from experiments.config_loader import load_config, obstacles_from_scenario
from metrics.experiment_logger import ExperimentLogger
from metrics.comparison_metrics import adjusted_time_to_capture_s, episode_t_max_s
from metrics.escape_sector_pre_capture import escape_sector_window_metrics
from metrics.pre_capture import (
    canonical_metrics_from_step,
    capture_window_metrics,
    pre_capture_k_from_config,
    pre_capture_structure_metrics,
)


def trial_summary(result: dict[str, Any], dt: float, config: dict[str, Any] | None = None) -> dict[str, Any]:
    captured = bool(result["captured"])
    failed = bool(result.get("failed", False))
    capture_step = result.get("capture_step")
    failure_step = result.get("failure_step")
    time_to_capture_s = None
    if captured and capture_step is not None:
        time_to_capture_s = float(capture_step) * dt
    cfg = config or {"dt": dt, "max_steps": int(result.get("num_steps", 0) or 0)}
    t_max_s = episode_t_max_s(cfg)
    return {
        "captured": captured,
        "failed": failed,
        "failure_reason": result.get("failure_reason"),
        "failure_step": failure_step,
        "success": captured and not failed,
        "capture_step": capture_step,
        "num_steps": result["num_steps"],
        "t_max_s": t_max_s,
        "time_to_capture_s": time_to_capture_s,
        "time_to_capture_adj_s": adjusted_time_to_capture_s(captured, time_to_capture_s, t_max_s),
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

    summary = trial_summary(result, cfg["dt"], cfg)
    enrich_trial_structure_metrics(summary, result["frames"], cfg)
    if extra_summary:
        summary.update(extra_summary)
    logger.finalize(summary)
    return summary


def _fmt_metric_val(val: Any, precision: int = 3) -> str | None:
    if val in ("", None):
        return None
    try:
        f = float(val)
        if f != f:
            return None
        return f"{f:.{precision}f}"
    except (TypeError, ValueError):
        return None


def final_structure_metrics(frames: list[dict[str, Any]]) -> dict[str, float | str]:
    empty: dict[str, float | str] = {
        "final_D_ang": "",
        "final_C_cov": "",
        "final_G_max_deg": "",
    }
    if not frames:
        return empty
    canon = canonical_metrics_from_step(frames[-1])
    if canon is None:
        return empty
    return {
        "final_D_ang": canon["D_ang"],
        "final_C_cov": canon["C_cov"],
        "final_G_max_deg": math.degrees(canon["G_max"]),
    }


def enrich_trial_structure_metrics(
    summary: dict[str, Any],
    frames: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Add capture / pre-capture / final-step structural metrics to a trial summary."""
    k_pre = pre_capture_k_from_config(config)
    captured = bool(summary.get("captured", False))
    capture_step = summary.get("capture_step")
    summary.update(
        pre_capture_structure_metrics(frames, capture_step, captured, k=k_pre)
    )
    summary.update(capture_window_metrics(frames, capture_step, captured))
    summary.update(escape_sector_window_metrics(frames, capture_step, captured))
    if not captured:
        summary.update(final_structure_metrics(frames))

    summary["method"] = summary.get("method", "")
    summary["scenario"] = config.get("scenario", {}).get("name", summary.get("scenario", ""))
    summary["evader_policy"] = config.get("evader_policy", "")
    summary["seed"] = config.get("seed", "")
    summary["timeout"] = not captured and not summary.get("failed", False)
    summary["collision"] = summary.get("failure_reason") == "obstacle_collision"
    summary["boundary_violation"] = summary.get("failure_reason") == "boundary_violation"
    return summary


def format_trial_progress_line(summary: dict[str, Any], prefix: str = "") -> str:
    """One-line trial progress with the key structural metrics used in comparison tables."""
    captured = bool(summary.get("captured", False))
    status = "OK" if captured else (summary.get("failure_reason") or "timeout")
    parts: list[str] = []
    if prefix:
        parts.append(f"{prefix.rstrip(':')}:")
    parts.append(f"captured={summary['captured']}")
    parts.append(f"steps={summary['num_steps']}")
    if captured:
        metric_specs = (
            ("capD", "capture_D_ang", 3),
            ("capC", "capture_C_cov", 3),
            ("capG", "capture_G_max_deg", 1),
            ("p5D", "pre_capture_5_D_ang", 3),
            ("p10D", "pre_capture_10_D_ang", 3),
            ("free", "free_escape_angle_at_capture_deg", 1),
            ("blk", "blocked_escape_angle_at_capture_deg", 1),
            ("unblk", "unblocked_escape_angle_at_capture_deg", 1),
            ("escC", "C_esc_at_capture", 3),
            ("escG", "G_esc_at_capture_deg", 1),
        )
    else:
        metric_specs = (
            ("finD", "final_D_ang", 3),
            ("finC", "final_C_cov", 3),
            ("finG", "final_G_max_deg", 1),
        )
    for label, key, prec in metric_specs:
        formatted = _fmt_metric_val(summary.get(key), prec)
        if formatted is not None:
            parts.append(f"{label}={formatted}")
    parts.append(str(status))
    return ", ".join(parts)


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
