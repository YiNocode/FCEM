"""Cross-method structural metrics for fair comparison and plotting."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fcem.boundary_trap import detect_trap_mode, structural_metrics_free_cone
from metrics.pre_capture import pre_capture_step_window, step_metric_value
from metrics.structure import structural_metrics_from_positions


def trap_thresholds_from_config(config: dict[str, Any]) -> tuple[float, float]:
    trap_cfg = config.get("fcem", {}).get("trap", {})
    return (
        float(trap_cfg.get("boundary_trap_threshold", 2.5)),
        float(trap_cfg.get("corner_trap_threshold", 3.0)),
    )


def world_bounds_from_config(config: dict[str, Any]) -> tuple[float, float, float, float]:
    w = config["world"]
    return float(w["xmin"]), float(w["xmax"]), float(w["ymin"]), float(w["ymax"])


def step_trap_mode(
    step: dict[str, Any],
    bounds: tuple[float, float, float, float],
    config: dict[str, Any],
) -> str:
    mode = step.get("trap_mode")
    if mode:
        return str(mode)
    extra = step.get("extra") or {}
    mode = extra.get("trap_mode")
    if mode:
        return str(mode)
    evader = step.get("evader")
    if evader is None:
        return "open_space"
    b_th, c_th = trap_thresholds_from_config(config)
    trap = detect_trap_mode(np.asarray(evader, dtype=float), bounds, b_th, c_th)
    return trap.mode


def _positions_from_step(step: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    evader = step.get("evader")
    pursuers = step.get("pursuers")
    if evader is None or pursuers is None:
        return None
    return np.asarray(evader, dtype=float), np.asarray(pursuers, dtype=float)


def canonical_structure_from_step(step: dict[str, Any]) -> dict[str, float] | None:
    """Full-circle encirclement metrics recomputed from logged positions."""
    pos = _positions_from_step(step)
    if pos is None:
        return None
    evader, pursuers = pos
    return structural_metrics_from_positions(evader, pursuers)


def trap_aware_structure_from_step(
    step: dict[str, Any],
    bounds: tuple[float, float, float, float],
    config: dict[str, Any],
) -> dict[str, float] | None:
    """Trap / free-cone metrics recomputed from logged positions."""
    pos = _positions_from_step(step)
    if pos is None:
        return None
    evader, pursuers = pos
    b_th, c_th = trap_thresholds_from_config(config)
    trap = detect_trap_mode(evader, bounds, b_th, c_th)
    return structural_metrics_free_cone(evader, pursuers, trap)


def _mean_or_blank(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def episode_structure_stats(
    steps: list[dict[str, Any]],
    bounds: tuple[float, float, float, float],
    config: dict[str, Any],
) -> dict[str, float | str]:
    """Episode-level structural aggregates used in aggregated_comparison.csv."""
    can_d, can_c, can_g_deg = [], [], []
    free_d, free_c, free_g_deg = [], [], []
    open_d, open_c, open_g_deg = [], [], []
    trap_d, trap_c, trap_g_deg = [], [], []

    for step in steps:
        canonical = canonical_structure_from_step(step)
        if canonical is not None:
            can_d.append(canonical["D_ang"])
            can_c.append(canonical["C_cov"])
            can_g_deg.append(math.degrees(canonical["G_max"]))

        trap_metrics = trap_aware_structure_from_step(step, bounds, config)
        if trap_metrics is not None:
            free_d.append(float(trap_metrics.get("D_free", trap_metrics["D_ang"])))
            free_c.append(float(trap_metrics.get("C_free", trap_metrics["C_cov"])))
            free_g_deg.append(math.degrees(float(trap_metrics.get("G_free", trap_metrics["G_max"]))))

        mode = step_trap_mode(step, bounds, config)
        if mode == "open_space" and canonical is not None:
            open_d.append(canonical["D_ang"])
            open_c.append(canonical["C_cov"])
            open_g_deg.append(math.degrees(canonical["G_max"]))
        elif mode in ("boundary", "corner") and trap_metrics is not None:
            trap_d.append(float(trap_metrics.get("D_free", trap_metrics["D_ang"])))
            trap_c.append(float(trap_metrics.get("C_free", trap_metrics["C_cov"])))
            trap_g_deg.append(math.degrees(float(trap_metrics.get("G_free", trap_metrics["G_max"]))))

    return {
        "mean_canonical_D_ang": _mean_or_blank(can_d),
        "mean_canonical_C_cov": _mean_or_blank(can_c),
        "mean_canonical_G_max_deg": _mean_or_blank(can_g_deg),
        "mean_D_free": _mean_or_blank(free_d),
        "mean_C_free": _mean_or_blank(free_c),
        "mean_G_free_deg": _mean_or_blank(free_g_deg),
        "mean_open_D_ang": _mean_or_blank(open_d),
        "mean_open_C_cov": _mean_or_blank(open_c),
        "mean_open_G_max_deg": _mean_or_blank(open_g_deg),
        "mean_trap_D_free": _mean_or_blank(trap_d),
        "mean_trap_C_free": _mean_or_blank(trap_c),
        "mean_trap_G_free_deg": _mean_or_blank(trap_g_deg),
        "open_space_step_frac": (len(open_d) / len(steps)) if steps else "",
    }


def pre_capture_canonical_metrics(
    steps: list[dict[str, Any]],
    capture_step: int | None,
    captured: bool,
    k: int,
) -> dict[str, float | str]:
    """Last-k-step full-circle metrics from positions (successful trials only)."""
    empty: dict[str, float | str] = {
        "pre_capture_canonical_D_ang": "",
        "pre_capture_canonical_C_cov": "",
        "pre_capture_canonical_G_max_deg": "",
        "pre_capture_canonical_C_sync": "",
    }
    if not captured or capture_step is None or k <= 0:
        return empty

    window = pre_capture_step_window(steps, capture_step, k)
    if not window:
        return empty

    d_vals, c_vals, g_vals, sync_vals = [], [], [], []
    for step in window:
        canonical = canonical_structure_from_step(step)
        if canonical is None:
            continue
        d_vals.append(canonical["D_ang"])
        c_vals.append(canonical["C_cov"])
        g_vals.append(math.degrees(canonical["G_max"]))
        sync = step_metric_value(step, "C_sync")
        if sync is not None:
            sync_vals.append(sync)

    return {
        "pre_capture_canonical_D_ang": _mean_or_blank(d_vals),
        "pre_capture_canonical_C_cov": _mean_or_blank(c_vals),
        "pre_capture_canonical_G_max_deg": _mean_or_blank(g_vals),
        "pre_capture_canonical_C_sync": _mean_or_blank(sync_vals),
    }


def pre_capture_trap_metrics(
    steps: list[dict[str, Any]],
    capture_step: int | None,
    captured: bool,
    k: int,
    bounds: tuple[float, float, float, float],
    config: dict[str, Any],
) -> dict[str, float | str]:
    """Last-k-step trap / free-cone metrics from positions (successful trials only)."""
    empty: dict[str, float | str] = {
        "pre_capture_D_free": "",
        "pre_capture_C_free": "",
        "pre_capture_G_free_deg": "",
    }
    if not captured or capture_step is None or k <= 0:
        return empty

    window = pre_capture_step_window(steps, capture_step, k)
    if not window:
        return empty

    d_vals, c_vals, g_vals = [], [], []
    for step in window:
        trap_metrics = trap_aware_structure_from_step(step, bounds, config)
        if trap_metrics is None:
            continue
        d_vals.append(float(trap_metrics.get("D_free", trap_metrics["D_ang"])))
        c_vals.append(float(trap_metrics.get("C_free", trap_metrics["C_cov"])))
        g_vals.append(math.degrees(float(trap_metrics.get("G_free", trap_metrics["G_max"]))))

    return {
        "pre_capture_D_free": _mean_or_blank(d_vals),
        "pre_capture_C_free": _mean_or_blank(c_vals),
        "pre_capture_G_free_deg": _mean_or_blank(g_vals),
    }


# Fixed reference scales for radar charts (not per-scenario min-max).
RADAR_TTC_REF_S = 120.0
# For 3 pursuers: ideal max gap = 120°, worst = 360° (all agents on one ray).
RADAR_G_MAX_IDEAL_DEG = 120.0
RADAR_G_MAX_WORST_DEG = 360.0


def episode_t_max_s(config: dict[str, Any]) -> float:
    """Episode horizon in seconds (timeout penalty for adjusted TTC)."""
    dt = float(config.get("dt", 0.1))
    max_steps = int(config.get("max_steps", 1200))
    return max_steps * dt


def adjusted_time_to_capture_s(
    captured: bool,
    time_to_capture_s: float | None,
    t_max_s: float,
) -> float:
    """
    Timeout-adjusted TTC: actual capture time if successful, else T_max.

    Penalizes failed / timeout trials so low-success methods cannot look fast.
    """
    if captured and time_to_capture_s is not None:
        return float(time_to_capture_s)
    return float(t_max_s)


def radar_fixed_scale(metric: str, value: float) -> float:
    """Map a raw metric to [0, 1] with a fixed reference (higher is better)."""
    if metric == "success":
        return float(np.clip(value, 0.0, 1.0))
    if metric == "inv_ttc":
        if value <= 0.0:
            return 0.0
        return float(np.clip(RADAR_TTC_REF_S / value, 0.0, 1.0))
    if metric in ("d_ang", "c_cov", "c_sync"):
        return float(np.clip(value, 0.0, 1.0))
    if metric == "inv_g":
        span = RADAR_G_MAX_WORST_DEG - RADAR_G_MAX_IDEAL_DEG
        return float(np.clip((RADAR_G_MAX_WORST_DEG - value) / span, 0.0, 1.0))
    raise KeyError(metric)
