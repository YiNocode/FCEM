"""Per-step diagnostics recomputed from logged positions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from common.capture import capture_params_from_config, evaluate_capture_conditions
from metrics.capture_mechanism import distance_to_nearest_wall
from metrics.escape_sector_metrics import (
    compute_escape_sector_metrics,
    escape_metrics_config_from_config,
    escape_status_from_metrics,
)
from metrics.structure import angular_gaps_rad, structural_metrics_from_positions


def obstacles_from_config(config: dict[str, Any]) -> list[Any]:
    scenario = config.get("scenario", {})
    if isinstance(scenario, dict):
        return list(scenario.get("obstacles", []))
    return []


def metric_value(step: dict[str, Any], key: str) -> Any:
    if key in step and step[key] not in ("", None):
        return step[key]
    metrics = step.get("metrics", {})
    if isinstance(metrics, dict) and key in metrics and metrics[key] not in ("", None):
        return metrics[key]
    extra = step.get("extra", {})
    if isinstance(extra, dict) and key in extra and extra[key] not in ("", None):
        return extra[key]
    return None


def _positions_from_step(step: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    evader = step.get("evader")
    pursuers = step.get("pursuers")
    if evader is None or pursuers is None:
        return None
    return np.asarray(evader, dtype=float).reshape(2), np.asarray(pursuers, dtype=float).reshape(-1, 2)


def _jsonable_positions(arr: np.ndarray) -> list:
    return np.asarray(arr, dtype=float).tolist()


def _fallback_float(step: dict[str, Any], key: str, default: float = float("nan")) -> float:
    val = metric_value(step, key)
    if val in ("", None):
        return default
    return float(val)


def full_circle_diagnostics_from_step(step: dict[str, Any]) -> dict[str, Any]:
    pos = _positions_from_step(step)
    if pos is None:
        g_full = metric_value(step, "G_max_full_deg")
        if g_full is None:
            g = metric_value(step, "G_max")
            g_full = math.degrees(float(g)) if g is not None else float("nan")
        return {
            "D_ang_full": _fallback_float(step, "D_ang_full", _fallback_float(step, "D_ang")),
            "C_cov_full": _fallback_float(step, "C_cov_full", _fallback_float(step, "C_cov")),
            "G_max_full_deg": float(g_full),
            "distances_to_evader": "",
            "angles_deg": "",
            "full_gaps_deg": "",
        }

    evader, pursuers = pos
    metrics = structural_metrics_from_positions(evader, pursuers)
    rel = pursuers - evader[None, :]
    distances = np.linalg.norm(rel, axis=1)
    angles = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2.0 * math.pi)
    gaps = angular_gaps_rad(angles)

    return {
        "D_ang_full": float(metrics["D_ang"]),
        "C_cov_full": float(metrics["C_cov"]),
        "G_max_full_deg": math.degrees(float(metrics["G_max"])),
        "G_max_full_rad": float(metrics["G_max"]),
        "distances_to_evader": _jsonable_positions(distances),
        "angles_deg": _jsonable_positions(np.degrees(np.sort(angles))),
        "full_gaps_deg": _jsonable_positions(np.degrees(gaps)),
    }


def diagnostics_from_step_record(
    step: dict[str, Any],
    *,
    bounds: tuple[float, float, float, float],
    esc_cfg: dict[str, Any] | None = None,
    obstacles: list[Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute full-circle, escape-sector, and capture-validity diagnostics."""
    config = config or {}
    esc_cfg = esc_cfg or escape_metrics_config_from_config(config)
    obstacles = obstacles if obstacles is not None else obstacles_from_config(config)

    row: dict[str, Any] = {}
    pos = _positions_from_step(step)
    full = full_circle_diagnostics_from_step(step)
    row.update(full)
    row["capD"] = full["D_ang_full"]
    row["capC"] = full["C_cov_full"]
    row["capG_full_deg"] = full["G_max_full_deg"]

    if pos is None:
        for key in (
            "C_esc",
            "G_esc_deg",
            "free_escape_angle_deg",
            "blocked_escape_angle_deg",
            "unblocked_escape_angle_deg",
        ):
            row[key] = _fallback_float(step, key)
        row["escape_status"] = str(metric_value(step, "escape_status") or escape_status_from_metrics(row))
        row["evader_pos"] = ""
        row["pursuer_positions"] = ""
        row["distance_to_nearest_wall"] = ""
        row["ray_length"] = esc_cfg.get("ray_length", 6.0)
        row["pursuer_block_radius"] = esc_cfg.get("pursuer_block_radius", 1.0)
        return row

    evader, pursuers = pos
    row["evader_pos"] = _jsonable_positions(evader)
    row["pursuer_positions"] = _jsonable_positions(pursuers)
    row["distance_to_nearest_wall"] = distance_to_nearest_wall(evader, bounds)
    row["ray_length"] = esc_cfg["ray_length"]
    row["pursuer_block_radius"] = esc_cfg["pursuer_block_radius"]

    esc = compute_escape_sector_metrics(
        evader,
        pursuers,
        obstacles,
        bounds,
        ray_length=esc_cfg["ray_length"],
        num_angles=esc_cfg["num_angles"],
        num_ray_samples=esc_cfg["num_ray_samples"],
        pursuer_block_radius=esc_cfg["pursuer_block_radius"],
        obstacle_margin=esc_cfg["obstacle_margin"],
        boundary_margin=esc_cfg["boundary_margin"],
        exit_config=esc_cfg["exit_config"],
        min_forward_block_dist=esc_cfg["min_forward_block_dist"],
        G_esc_allow_deg=esc_cfg["G_esc_allow_deg"],
        C_esc_min=esc_cfg["C_esc_min"],
    )
    for key in (
        "C_esc",
        "G_esc_deg",
        "free_escape_angle_deg",
        "blocked_escape_angle_deg",
        "unblocked_escape_angle_deg",
        "exit_blockage",
        "escape_status",
    ):
        row[key] = esc[key]

    cap_params = capture_params_from_config(config)
    cap_flags = evaluate_capture_conditions(
        pursuers,
        evader,
        cap_params["capture_radius"],
        float(full.get("G_max_full_rad", 0.0)),
        cap_params["g_max_allowed"],
        esc,
        g_esc_allow_deg=cap_params["g_esc_allow_deg"],
        c_esc_min=cap_params["c_esc_min"],
        capture_mode="escape_sector",
    )
    row["valid_distance_capture"] = cap_flags["capture_condition_valid_distance"]
    row["valid_full_circle_capture"] = cap_flags["capture_condition_valid_full_circle"]
    row["valid_escape_sector_capture"] = cap_flags["capture_condition_valid_escape_sector"]
    row["capture_condition_valid_distance"] = row["valid_distance_capture"]
    row["capture_condition_valid_full_circle"] = row["valid_full_circle_capture"]
    row["capture_condition_valid_escape_sector"] = row["valid_escape_sector_capture"]
    row["captured"] = cap_flags["captured"]
    return row
