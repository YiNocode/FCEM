"""Per-step experiment logging."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np


class TimingBlock:
    """Context manager for timing code blocks in milliseconds."""

    def __enter__(self) -> TimingBlock:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self._t1 = time.perf_counter()

    @property
    def ms(self) -> float:
        return (self._t1 - self._t0) * 1000.0


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _metric_float(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    val = metrics.get(key, default)
    if val in ("", None):
        return default
    if isinstance(val, float) and math.isnan(val):
        return float("nan")
    return float(val)


@dataclass
class StepRecord:
    step: int
    D_ang: float
    C_cov: float
    G_max: float
    C_col: float
    timing_ms: float
    C_sync: float = 0.0
    R: float = 0.0
    q: float = 0.0
    slot_error: float = 0.0
    phase: str = ""
    evader: list[float] = field(default_factory=list)
    pursuers: list[list[float]] = field(default_factory=list)
    timing_detail: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    # Escape-sector metrics
    C_esc: float = 0.0
    G_esc_deg: float = 0.0
    free_escape_angle_deg: float = 0.0
    blocked_escape_angle_deg: float = 0.0
    unblocked_escape_angle_deg: float = 0.0
    exit_blockage: float = float("nan")
    escape_status: str = ""
    # Full-circle diagnostic aliases
    D_ang_full: float = 0.0
    C_cov_full: float = 0.0
    G_max_full_deg: float = 0.0
    # Capture condition validity flags
    capture_condition_valid_distance: bool = False
    capture_condition_valid_full_circle: bool = False
    capture_condition_valid_escape_sector: bool = False
    valid_distance_capture: bool = False
    valid_full_circle_capture: bool = False
    valid_escape_sector_capture: bool = False


@dataclass
class ExperimentLogger:
    out_dir: Path | str
    method: str
    scenario: str
    trial: int
    config: dict[str, Any]
    records: list[StepRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.run_id = f"{self.method}_{self.scenario}_t{self.trial:03d}"

    def log_step(self, frame: dict[str, Any]) -> None:
        """Log from a simulation frame dict."""
        metrics = frame["metrics"]
        timing_raw = frame.get("timing_ms", 0.0)
        if isinstance(timing_raw, dict):
            timing_ms = float(timing_raw.get("total_ms", 0.0))
        else:
            timing_ms = float(timing_raw)

        timing_detail = timing_raw if isinstance(timing_raw, dict) else {}
        extra = {
            k: v
            for k, v in frame.items()
            if k
            not in (
                "step",
                "metrics",
                "timing_ms",
                "evader",
                "evader_v",
                "pursuers",
                "pursuer_v",
                "captured",
                "curve",
                "slots",
            )
        }
        rec = StepRecord(
            step=int(frame["step"]),
            D_ang=float(metrics["D_ang"]),
            C_cov=float(metrics["C_cov"]),
            G_max=float(metrics["G_max"]),
            C_col=float(metrics["C_col"]),
            C_sync=float(metrics.get("C_sync", frame.get("C_sync", 0.0))),
            timing_ms=timing_ms,
            R=float(frame.get("R", 0.0)),
            q=float(frame.get("q", 0.0)),
            slot_error=float(frame.get("slot_error", 0.0)),
            phase=str(frame.get("phase", "")),
            evader=_to_jsonable(frame["evader"]),
            pursuers=_to_jsonable(frame["pursuers"]),
            timing_detail=_to_jsonable(timing_detail),
            extra=_to_jsonable(extra),
            C_esc=_metric_float(metrics, "C_esc"),
            G_esc_deg=_metric_float(metrics, "G_esc_deg"),
            free_escape_angle_deg=_metric_float(metrics, "free_escape_angle_deg"),
            blocked_escape_angle_deg=_metric_float(metrics, "blocked_escape_angle_deg"),
            unblocked_escape_angle_deg=_metric_float(metrics, "unblocked_escape_angle_deg"),
            exit_blockage=_metric_float(metrics, "exit_blockage", default=float("nan")),
            escape_status=str(metrics.get("escape_status", "")),
            D_ang_full=_metric_float(metrics, "D_ang_full", default=float(metrics["D_ang"])),
            C_cov_full=_metric_float(metrics, "C_cov_full", default=float(metrics["C_cov"])),
            G_max_full_deg=_metric_float(
                metrics,
                "G_max_full_deg",
                default=math.degrees(float(metrics["G_max"])),
            ),
            capture_condition_valid_distance=bool(
                metrics.get("capture_condition_valid_distance", False)
            ),
            capture_condition_valid_full_circle=bool(
                metrics.get("capture_condition_valid_full_circle", False)
            ),
            capture_condition_valid_escape_sector=bool(
                metrics.get("capture_condition_valid_escape_sector", False)
            ),
            valid_distance_capture=bool(
                metrics.get(
                    "valid_distance_capture",
                    metrics.get("capture_condition_valid_distance", False),
                )
            ),
            valid_full_circle_capture=bool(
                metrics.get(
                    "valid_full_circle_capture",
                    metrics.get("capture_condition_valid_full_circle", False),
                )
            ),
            valid_escape_sector_capture=bool(
                metrics.get(
                    "valid_escape_sector_capture",
                    metrics.get("capture_condition_valid_escape_sector", False),
                )
            ),
        )
        self.records.append(rec)

    def finalize(self, summary: dict[str, Any]) -> Path:
        self.metadata.update(summary)
        return self.save()

    def save(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.run_id}.json"
        payload = {
            "run_id": self.run_id,
            "method": self.method,
            "scenario": self.scenario,
            "trial": self.trial,
            "config": _to_jsonable(self.config),
            "metadata": _to_jsonable(self.metadata),
            "records": [asdict(r) for r in self.records],
        }
        path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
        return path

    def summary_metrics(self) -> dict[str, float]:
        if not self.records:
            return {}
        timings = [r.timing_ms for r in self.records]
        return {
            "mean_timing_ms": float(np.mean(timings)),
            "max_timing_ms": float(np.max(timings)),
            "final_D_ang": self.records[-1].D_ang,
            "final_C_cov": self.records[-1].C_cov,
            "final_G_max_deg": math.degrees(self.records[-1].G_max),
            "final_C_col": self.records[-1].C_col,
        }
