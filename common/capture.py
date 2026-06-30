"""Capture condition checking."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def check_capture(
    pursuers: np.ndarray,
    evader: np.ndarray,
    capture_radius: float,
    g_max: float,
    g_max_allowed: float,
    trap_mode: str = "open_space",
    g_free: float | None = None,
    g_free_allowed: float | None = None,
) -> bool:
    dists = np.linalg.norm(pursuers - evader[None, :], axis=1)
    all_inside = bool(np.all(dists <= capture_radius))
    if trap_mode in ("boundary", "corner") and g_free is not None and g_free_allowed is not None:
        structure_closed = g_free <= g_free_allowed
    else:
        structure_closed = g_max <= g_max_allowed
    return all_inside and structure_closed


def check_capture_with_dists(
    target: np.ndarray,
    pursuers: np.ndarray,
    capture_radius: float,
    g_max: float,
    g_max_allowed: float,
) -> tuple[bool, np.ndarray]:
    dists = np.linalg.norm(pursuers - target[None, :], axis=1)
    captured = bool(np.all(dists <= capture_radius)) and g_max <= g_max_allowed
    return captured, dists


def all_pursuers_within_capture_radius(
    pursuers: np.ndarray,
    evader: np.ndarray,
    capture_radius: float,
) -> bool:
    dists = np.linalg.norm(pursuers - evader[None, :], axis=1)
    return bool(np.all(dists <= capture_radius))


def evaluate_capture_conditions(
    pursuers: np.ndarray,
    evader: np.ndarray,
    capture_radius: float,
    g_max: float,
    g_max_allowed: float,
    escape_metrics: dict[str, float],
    *,
    g_esc_allow_deg: float = 30.0,
    c_esc_min: float | None = 0.85,
    capture_mode: str = "escape_sector",
) -> dict[str, bool]:
    """Evaluate distance, full-circle, and escape-sector capture validity flags."""
    distance_ok = all_pursuers_within_capture_radius(pursuers, evader, capture_radius)
    full_circle_ok = distance_ok and g_max <= g_max_allowed

    g_esc_deg = float(escape_metrics.get("G_esc_deg", 360.0))
    c_esc = float(escape_metrics.get("C_esc", 0.0))
    escape_ok = distance_ok and g_esc_deg <= g_esc_allow_deg
    if c_esc_min is not None:
        escape_ok = escape_ok and c_esc >= float(c_esc_min)

    mode = str(capture_mode).lower()
    if mode == "distance_only":
        captured = distance_ok
    elif mode == "full_circle":
        captured = full_circle_ok
    else:
        captured = escape_ok

    return {
        "capture_condition_valid_distance": distance_ok,
        "capture_condition_valid_full_circle": full_circle_ok,
        "capture_condition_valid_escape_sector": escape_ok,
        "valid_distance_capture": distance_ok,
        "valid_full_circle_capture": full_circle_ok,
        "valid_escape_sector_capture": escape_ok,
        "captured": captured,
    }


def capture_params_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract capture-mode parameters from merged experiment config."""
    metrics_cfg = config.get("metrics", {})
    if not isinstance(metrics_cfg, dict):
        metrics_cfg = {}
    fcem_cfg = config.get("fcem", {})

    c_esc_min = metrics_cfg.get("C_esc_min", 0.85)
    if c_esc_min is not None:
        c_esc_min = float(c_esc_min)

    return {
        "capture_mode": str(config.get("capture_mode", "escape_sector")),
        "capture_radius": float(fcem_cfg.get("capture_radius", 1.8)),
        "g_max_allowed": math.radians(float(config.get("G_max_allowed_deg", 140.0))),
        "g_esc_allow_deg": float(metrics_cfg.get("G_esc_allow_deg", 30.0)),
        "c_esc_min": c_esc_min,
    }
