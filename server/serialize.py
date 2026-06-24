"""JSON serialization for simulation frames."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from metrics.experiment_logger import _to_jsonable


def _trap_to_dict(trap: Any) -> dict[str, Any] | None:
    if trap is None:
        return None
    if hasattr(trap, "__dict__"):
        out: dict[str, Any] = {}
        for key, val in trap.__dict__.items():
            if key.startswith("_"):
                continue
            out[key] = _to_jsonable(val)
        return out
    return _to_jsonable(trap)


def frame_to_client(frame: dict[str, Any]) -> dict[str, Any]:
    """Convert a Sim2D frame to a JSON-safe dict for the Vue client."""
    metrics = frame.get("metrics") or {}
    g_max = metrics.get("G_max", 0.0)
    g_max_deg = math.degrees(float(g_max)) if g_max is not None else 0.0

    out: dict[str, Any] = {
        "step": int(frame.get("step", 0)),
        "evader": _to_jsonable(frame.get("evader")),
        "evader_v": _to_jsonable(frame.get("evader_v")),
        "pursuers": _to_jsonable(frame.get("pursuers")),
        "pursuer_v": _to_jsonable(frame.get("pursuer_v")),
        "R": float(frame.get("R", 0.0)),
        "captured": bool(frame.get("captured", False)),
        "failed": bool(frame.get("failed", False)),
        "failure_reason": frame.get("failure_reason"),
        "metrics": {
            "D_ang": float(metrics.get("D_ang", 0.0)),
            "C_cov": float(metrics.get("C_cov", 0.0)),
            "G_max_deg": g_max_deg,
            "C_col": float(metrics.get("C_col", 0.0)),
            "C_sync": float(metrics.get("C_sync", frame.get("C_sync", 0.0))),
        },
        "q": float(frame.get("q", 1.0)),
        "phase": frame.get("phase", ""),
        "trap_mode": frame.get("trap_mode") or (
            frame.get("trap").mode if frame.get("trap") is not None and hasattr(frame.get("trap"), "mode") else "open_space"
        ),
    }

    for key in ("slots", "center", "curve", "assignment", "slot_vel"):
        if key in frame and frame[key] is not None:
            out[key] = _to_jsonable(frame[key])

    trap = frame.get("trap")
    if trap is not None:
        out["trap"] = _trap_to_dict(trap)

    return out


def session_meta(
    *,
    method: str,
    scenario: str,
    config: dict[str, Any],
    obstacles: list[dict[str, Any]],
) -> dict[str, Any]:
    world = config.get("world", {})
    fcem = config.get("fcem", {})
    return {
        "method": method,
        "scenario": scenario,
        "world": world,
        "obstacles": obstacles,
        "dt": float(config.get("dt", 0.1)),
        "max_steps": int(config.get("max_steps", 1200)),
        "capture_radius": float(fcem.get("capture_radius", 1.8)),
        "pursuer_vmax": float(config.get("pursuer_vmax", 0.0)),
        "evader_vmax": float(config.get("evader_vmax", 0.0)),
        "evader_policy": config.get("evader_policy", "game"),
        "ablation": dict(config.get("ablation", {})),
    }
