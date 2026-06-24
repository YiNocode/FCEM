"""Pre-capture structural metrics (last K steps before capture)."""

from __future__ import annotations

import math
from typing import Any


def step_metric_value(step: dict[str, Any], key: str) -> float | None:
    if key in step and step[key] not in ("", None):
        return float(step[key])
    metrics = step.get("metrics", {})
    if key in metrics and metrics[key] not in ("", None):
        return float(metrics[key])
    if key == "C_sync":
        extra = step.get("extra", {})
        if "C_sync" in extra and extra["C_sync"] not in ("", None):
            return float(extra["C_sync"])
    return None


def pre_capture_step_window(
    steps: list[dict[str, Any]],
    capture_step: int | None,
    k: int,
) -> list[dict[str, Any]]:
    """
    Return the last ``k`` steps up to and including ``capture_step``.

    Steps are matched by their ``step`` field when present; otherwise list index
    is used as the step id.
    """
    if not steps or capture_step is None or k <= 0:
        return []

    indexed: list[tuple[int, dict[str, Any]]] = []
    for i, s in enumerate(steps):
        step_id = s.get("step", i)
        indexed.append((int(step_id), s))

    eligible = [s for sid, s in indexed if sid <= capture_step]
    if not eligible:
        return []
    return eligible[-k:]


def pre_capture_structure_metrics(
    steps: list[dict[str, Any]],
    capture_step: int | None,
    captured: bool,
    k: int = 10,
) -> dict[str, float | str]:
    """
    Mean structural metrics over the last ``k`` steps before capture.

    Returns empty strings for failed / non-captured runs.
    """
    empty: dict[str, float | str] = {
        "pre_capture_D_ang": "",
        "pre_capture_C_cov": "",
        "pre_capture_G_max_deg": "",
        "pre_capture_C_sync": "",
        "pre_capture_window": k,
        "pre_capture_n_steps": "",
    }
    if not captured or capture_step is None:
        return empty

    window = pre_capture_step_window(steps, capture_step, k)
    if not window:
        return empty

    d_vals, c_vals, g_vals, sync_vals = [], [], [], []
    for s in window:
        d = step_metric_value(s, "D_ang")
        c = step_metric_value(s, "C_cov")
        g = step_metric_value(s, "G_max")
        sync = step_metric_value(s, "C_sync")
        if d is not None:
            d_vals.append(d)
        if c is not None:
            c_vals.append(c)
        if g is not None:
            g_vals.append(math.degrees(g))
        if sync is not None:
            sync_vals.append(sync)

    return {
        "pre_capture_D_ang": sum(d_vals) / len(d_vals) if d_vals else "",
        "pre_capture_C_cov": sum(c_vals) / len(c_vals) if c_vals else "",
        "pre_capture_G_max_deg": sum(g_vals) / len(g_vals) if g_vals else "",
        "pre_capture_C_sync": sum(sync_vals) / len(sync_vals) if sync_vals else "",
        "pre_capture_window": k,
        "pre_capture_n_steps": len(window),
    }


def pre_capture_k_from_config(config: dict[str, Any]) -> int:
    metrics_cfg = config.get("metrics", {})
    if isinstance(metrics_cfg, dict) and "pre_capture_window" in metrics_cfg:
        return int(metrics_cfg["pre_capture_window"])
    return int(config.get("pre_capture_window", 10))
