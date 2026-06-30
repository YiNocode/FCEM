"""Pre-capture escape-sector and full-circle diagnostic aggregates."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from metrics.pre_capture import pre_capture_step_window, step_metric_value


def _nan() -> float:
    return float("nan")


def _escape_value(step: dict[str, Any], key: str) -> float | None:
    metrics = step.get("metrics", {})
    if key in metrics and metrics[key] not in ("", None):
        val = metrics[key]
        if isinstance(val, float) and math.isnan(val):
            return None
        return float(val)
    if key in step and step[key] not in ("", None):
        return float(step[key])
    return step_metric_value(step, key)


def _full_circle_value(step: dict[str, Any], key: str) -> float | None:
    metrics = step.get("metrics", {})
    full_key = {
        "D_ang": "D_ang_full",
        "C_cov": "C_cov_full",
        "G_max_deg": "G_max_full_deg",
    }.get(key, key)
    if full_key in metrics and metrics[full_key] not in ("", None):
        return float(metrics[full_key])
    if key == "G_max_deg":
        g = step_metric_value(step, "G_max")
        return math.degrees(g) if g is not None else None
    return step_metric_value(step, key)


def _mean_from_window(
    window: list[dict[str, Any]],
    extractor,
    keys: tuple[str, ...],
) -> dict[str, float | str]:
    buckets: dict[str, list[float]] = {k: [] for k in keys}
    for step in window:
        for key in keys:
            val = extractor(step, key)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                buckets[key].append(float(val))
    return {k: (sum(v) / len(v) if v else "") for k, v in buckets.items()}


def _escape_keys(prefix: str) -> dict[str, float | str]:
    return {
        f"C_esc_{prefix}": _nan(),
        f"G_esc_{prefix}_deg": _nan(),
        f"free_escape_angle_{prefix}_deg": _nan(),
        f"blocked_escape_angle_{prefix}_deg": _nan(),
        f"unblocked_escape_angle_{prefix}_deg": _nan(),
    }


def _full_keys(prefix: str) -> dict[str, float | str]:
    return {
        f"D_ang_full_{prefix}": _nan(),
        f"C_cov_full_{prefix}": _nan(),
        f"G_max_full_{prefix}_deg": _nan(),
    }


ESCAPE_STEP_KEYS = (
    "C_esc",
    "G_esc_deg",
    "free_escape_angle_deg",
    "blocked_escape_angle_deg",
    "unblocked_escape_angle_deg",
)

FULL_STEP_KEYS = ("D_ang", "C_cov", "G_max_deg")


def escape_at_capture_metrics(
    steps: list[dict[str, Any]],
    capture_step: int | None,
    captured: bool,
) -> dict[str, float | str]:
    out = _escape_keys("at_capture")
    if not captured or capture_step is None:
        return out

    window = pre_capture_step_window(steps, capture_step, 1)
    if not window:
        return out

    means = _mean_from_window(window, _escape_value, ESCAPE_STEP_KEYS)
    return {
        "C_esc_at_capture": means["C_esc"] if means["C_esc"] != "" else _nan(),
        "G_esc_at_capture_deg": means["G_esc_deg"] if means["G_esc_deg"] != "" else _nan(),
        "free_escape_angle_at_capture_deg": (
            means["free_escape_angle_deg"] if means["free_escape_angle_deg"] != "" else _nan()
        ),
        "blocked_escape_angle_at_capture_deg": (
            means["blocked_escape_angle_deg"] if means["blocked_escape_angle_deg"] != "" else _nan()
        ),
        "unblocked_escape_angle_at_capture_deg": (
            means["unblocked_escape_angle_deg"]
            if means["unblocked_escape_angle_deg"] != ""
            else _nan()
        ),
    }


def escape_window_mean_metrics(
    steps: list[dict[str, Any]],
    capture_step: int | None,
    captured: bool,
    k: int,
    label: str,
) -> dict[str, float | str]:
    out = _escape_keys(f"final{k}_mean")
    if not captured or capture_step is None or k <= 0:
        return out

    window = pre_capture_step_window(steps, capture_step, k)
    if not window:
        return out

    means = _mean_from_window(window, _escape_value, ESCAPE_STEP_KEYS)
    return {
        f"C_esc_final{k}_mean": means["C_esc"] if means["C_esc"] != "" else _nan(),
        f"G_esc_final{k}_mean_deg": means["G_esc_deg"] if means["G_esc_deg"] != "" else _nan(),
        f"free_escape_angle_final{k}_mean_deg": (
            means["free_escape_angle_deg"] if means["free_escape_angle_deg"] != "" else _nan()
        ),
        f"unblocked_escape_angle_final{k}_mean_deg": (
            means["unblocked_escape_angle_deg"]
            if means["unblocked_escape_angle_deg"] != ""
            else _nan()
        ),
    }


def full_circle_at_capture_metrics(
    steps: list[dict[str, Any]],
    capture_step: int | None,
    captured: bool,
) -> dict[str, float | str]:
    out = _full_keys("at_capture")
    if not captured or capture_step is None:
        return out

    window = pre_capture_step_window(steps, capture_step, 1)
    if not window:
        return out

    means = _mean_from_window(window, _full_circle_value, FULL_STEP_KEYS)
    return {
        "D_ang_full_at_capture": means["D_ang"] if means["D_ang"] != "" else _nan(),
        "C_cov_full_at_capture": means["C_cov"] if means["C_cov"] != "" else _nan(),
        "G_max_full_at_capture_deg": means["G_max_deg"] if means["G_max_deg"] != "" else _nan(),
    }


def full_circle_window_mean_metrics(
    steps: list[dict[str, Any]],
    capture_step: int | None,
    captured: bool,
    k: int,
    label: str,
) -> dict[str, float | str]:
    out = _full_keys(f"final{k}_mean")
    if not captured or capture_step is None or k <= 0:
        return out

    window = pre_capture_step_window(steps, capture_step, k)
    if not window:
        return out

    means = _mean_from_window(window, _full_circle_value, FULL_STEP_KEYS)
    return {
        f"D_ang_full_final{k}_mean": means["D_ang"] if means["D_ang"] != "" else _nan(),
        f"C_cov_full_final{k}_mean": means["C_cov"] if means["C_cov"] != "" else _nan(),
        f"G_max_full_final{k}_mean_deg": means["G_max_deg"] if means["G_max_deg"] != "" else _nan(),
    }


def escape_sector_window_metrics(
    steps: list[dict[str, Any]],
    capture_step: int | None,
    captured: bool,
) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    out.update(escape_at_capture_metrics(steps, capture_step, captured))
    out.update(escape_window_mean_metrics(steps, capture_step, captured, 5, "final5"))
    out.update(escape_window_mean_metrics(steps, capture_step, captured, 10, "final10"))
    out.update(full_circle_at_capture_metrics(steps, capture_step, captured))
    out.update(full_circle_window_mean_metrics(steps, capture_step, captured, 5, "final5"))
    out.update(full_circle_window_mean_metrics(steps, capture_step, captured, 10, "final10"))
    return out
