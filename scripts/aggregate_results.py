"""Aggregate experiment JSON results into summary CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.comparison_metrics import (
    adjusted_time_to_capture_s,
    episode_structure_stats,
    episode_t_max_s,
    pre_capture_canonical_metrics,
    pre_capture_trap_metrics,
    world_bounds_from_config,
)
from metrics.pre_capture import (
    capture_window_metrics,
    pre_capture_k_from_config,
    pre_capture_structure_metrics,
    step_metric_value,
)
from experiments.run_output import MANIFEST_FILENAME, experiment_section_for_json
from metrics.escape_sector_pre_capture import escape_sector_window_metrics
from metrics.step_diagnostics import diagnostics_from_step_record


def _get_steps(data: dict) -> list:
    return data.get("records") or data.get("steps") or []


def _get_summary(data: dict) -> dict:
    return data.get("metadata") or data.get("summary") or {}


def _metric_value(step: dict, key: str) -> float | None:
    return step_metric_value(step, key)


def load_run(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _escape_from_summary_or_steps(
    summary: dict,
    steps: list,
    capture_step: int | None,
    captured: bool,
    config: dict,
    bounds: tuple[float, float, float, float],
) -> dict:
    """Prefer metadata escape-sector keys; recompute from steps when absent."""
    escape_keys = (
        "C_esc_at_capture",
        "G_esc_at_capture_deg",
        "free_escape_angle_at_capture_deg",
        "blocked_escape_angle_at_capture_deg",
        "unblocked_escape_angle_at_capture_deg",
        "C_esc_final5_mean",
        "G_esc_final5_mean_deg",
        "free_escape_angle_final5_mean_deg",
        "unblocked_escape_angle_final5_mean_deg",
        "C_esc_final10_mean",
        "G_esc_final10_mean_deg",
        "free_escape_angle_final10_mean_deg",
        "unblocked_escape_angle_final10_mean_deg",
        "D_ang_full_at_capture",
        "C_cov_full_at_capture",
        "G_max_full_at_capture_deg",
        "D_ang_full_final5_mean",
        "C_cov_full_final5_mean",
        "G_max_full_final5_mean_deg",
        "D_ang_full_final10_mean",
        "C_cov_full_final10_mean",
        "G_max_full_final10_mean_deg",
    )
    if any(k in summary for k in escape_keys):
        return {k: summary.get(k, "") for k in escape_keys}

    return escape_sector_window_metrics(steps, capture_step, captured)


PER_STEP_COLUMNS = [
    "method",
    "scenario",
    "trial_id",
    "step",
    "captured",
    "evader_pos",
    "pursuer_positions",
    "distances_to_evader",
    "angles_deg",
    "full_gaps_deg",
    "D_ang_full",
    "C_cov_full",
    "G_max_full_deg",
    "capD",
    "capC",
    "capG_full_deg",
    "free_escape_angle_deg",
    "blocked_escape_angle_deg",
    "unblocked_escape_angle_deg",
    "C_esc",
    "G_esc_deg",
    "escape_status",
    "ray_length",
    "pursuer_block_radius",
    "distance_to_nearest_wall",
    "valid_distance_capture",
    "valid_full_circle_capture",
    "valid_escape_sector_capture",
]

_LIST_COLUMNS = {
    "evader_pos",
    "pursuer_positions",
    "distances_to_evader",
    "angles_deg",
    "full_gaps_deg",
}


def _csv_cell(key: str, value: object) -> str | bool | float | int:
    if value in ("", None):
        return ""
    if key in _LIST_COLUMNS:
        if isinstance(value, str):
            return value
        return json.dumps(value, separators=(",", ":"))
    if key == "captured":
        return bool(value)
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def _step_export_row(
    step: dict[str, Any],
    method: str,
    scenario: str,
    trial_id: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    bounds = world_bounds_from_config(config) if "world" in config else (0.0, 40.0, 0.0, 40.0)
    from metrics.escape_sector_metrics import escape_metrics_config_from_config
    from metrics.step_diagnostics import obstacles_from_config

    diag = diagnostics_from_step_record(
        step,
        bounds=bounds,
        esc_cfg=escape_metrics_config_from_config(config),
        obstacles=obstacles_from_config(config),
        config=config,
    )

    row: dict[str, Any] = {
        "method": method,
        "scenario": scenario,
        "trial_id": trial_id,
        "step": step.get("step", ""),
    }
    for col in PER_STEP_COLUMNS[4:]:
        if col in step and step[col] not in ("", None):
            row[col] = step[col]
        elif col in diag:
            row[col] = diag[col]
        elif col in ("capD", "capC", "capG_full_deg"):
            legacy = {
                "capD": "D_ang_full",
                "capC": "C_cov_full",
                "capG_full_deg": "G_max_full_deg",
            }[col]
            val = step.get(legacy)
            if val in ("", None) and "metrics" in step:
                val = step["metrics"].get(legacy)
            row[col] = val if val not in ("", None) else ""
        else:
            val = _metric_value(step, col)
            if val is None and "metrics" in step:
                val = step["metrics"].get(col)
            row[col] = val if val is not None else ""
    return row


def export_per_step_metrics(results_dir: Path, out_csv: Path) -> int:
    rows: list[dict] = []
    for json_path in sorted(results_dir.rglob("*.json")):
        if json_path.name == MANIFEST_FILENAME:
            continue
        data = load_run(json_path)
        steps = _get_steps(data)
        config = data.get("config", {})
        method = data.get("method", "")
        scenario = data.get("scenario", "")
        trial_id = data.get("trial", data.get("trial_id", -1))
        for s in steps:
            raw = _step_export_row(s, method, scenario, trial_id, config)
            rows.append({col: _csv_cell(col, raw.get(col, "")) for col in PER_STEP_COLUMNS})

    if not rows:
        print(f"No step records found under {results_dir}")
        return 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PER_STEP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} step rows to {out_csv}")
    return len(rows)


def aggregate_trial_row(data: dict, json_path: Path, results_dir: Path) -> dict:
    summary = _get_summary(data)
    steps = _get_steps(data)
    config = data.get("config", {})
    dt = float(config.get("dt", 0.1))

    timing_keys = ["prediction_ms", "manifold_gen_ms", "assignment_ms", "low_level_ms", "total_ms"]
    avg_timing = {k: 0.0 for k in timing_keys}
    n_timed = 0
    for s in steps:
        td = s.get("timing_detail") or s.get("timing_ms")
        if isinstance(td, dict):
            n_timed += 1
            for k in timing_keys:
                avg_timing[k] += float(td.get(k, 0.0))
    if n_timed:
        for k in timing_keys:
            avg_timing[k] /= n_timed

    mean_d_ang = mean_c_cov = mean_g_max_deg = mean_c_col = mean_c_sync = ""
    if steps:
        d_vals, c_vals, g_vals, col_vals, sync_vals = [], [], [], [], []
        for s in steps:
            canon = None
            evader = s.get("evader")
            pursuers = s.get("pursuers")
            if evader is not None and pursuers is not None:
                from metrics.structure import structural_metrics_from_positions

                canon = structural_metrics_from_positions(
                    np.asarray(evader, dtype=float),
                    np.asarray(pursuers, dtype=float),
                )
            if canon is not None:
                d_vals.append(canon["D_ang"])
                c_vals.append(canon["C_cov"])
                g_vals.append(math.degrees(canon["G_max"]))
                col_vals.append(canon["C_col"])
            else:
                d = _metric_value(s, "D_ang")
                c = _metric_value(s, "C_cov")
                g = _metric_value(s, "G_max")
                col = _metric_value(s, "C_col")
                if d is not None:
                    d_vals.append(d)
                if c is not None:
                    c_vals.append(c)
                if g is not None:
                    g_vals.append(math.degrees(g))
                if col is not None:
                    col_vals.append(col)
            sync = _metric_value(s, "C_sync")
            if sync is not None:
                sync_vals.append(sync)
        if d_vals:
            mean_d_ang = sum(d_vals) / len(d_vals)
        if c_vals:
            mean_c_cov = sum(c_vals) / len(c_vals)
        if g_vals:
            mean_g_max_deg = sum(g_vals) / len(g_vals)
        if col_vals:
            mean_c_col = sum(col_vals) / len(col_vals)
        if sync_vals:
            mean_c_sync = sum(sync_vals) / len(sync_vals)

        last = steps[-1]
        last_canon = None
        if last.get("evader") is not None and last.get("pursuers") is not None:
            from metrics.structure import structural_metrics_from_positions

            last_canon = structural_metrics_from_positions(
                np.asarray(last["evader"], dtype=float),
                np.asarray(last["pursuers"], dtype=float),
            )
        if last_canon is not None:
            final_d_ang = last_canon["D_ang"]
            final_c_cov = last_canon["C_cov"]
            final_g_max_deg = math.degrees(last_canon["G_max"])
            final_c_col = last_canon["C_col"]
        else:
            final_d_ang = _metric_value(last, "D_ang") or ""
            final_c_cov = _metric_value(last, "C_cov") or ""
            final_g_max = _metric_value(last, "G_max")
            final_g_max_deg = math.degrees(final_g_max) if final_g_max is not None else ""
            final_c_col = _metric_value(last, "C_col") or ""
        final_c_sync = _metric_value(last, "C_sync") or ""
    else:
        final_d_ang = final_c_cov = final_g_max_deg = final_c_col = final_c_sync = ""

    captured = bool(summary.get("captured", False))
    capture_step = summary.get("capture_step")
    failed = bool(summary.get("failed", False))
    t_max_s = episode_t_max_s(config)
    time_to_capture_s = summary.get("time_to_capture_s")
    if time_to_capture_s is None and captured and capture_step is not None:
        time_to_capture_s = float(capture_step) * dt
    time_to_capture_adj_s = adjusted_time_to_capture_s(captured, time_to_capture_s, t_max_s)

    k_pre = pre_capture_k_from_config(config)
    bounds = world_bounds_from_config(config) if "world" in config else (0.0, 40.0, 0.0, 40.0)
    pre_cap = pre_capture_structure_metrics(steps, capture_step, captured, k=k_pre)
    pre_cap.update(capture_window_metrics(steps, capture_step, captured))
    pre_cap.update(pre_capture_canonical_metrics(steps, capture_step, captured, k=k_pre))
    pre_cap.update(pre_capture_trap_metrics(steps, capture_step, captured, k=k_pre, bounds=bounds, config=config))
    pre_cap.update(_escape_from_summary_or_steps(summary, steps, capture_step, captured, config, bounds))
    episode_stats = episode_structure_stats(steps, bounds, config)

    rel = json_path.relative_to(results_dir)
    parts = rel.parts
    experiment_section = experiment_section_for_json(json_path, results_dir, parts)

    t_capture = time_to_capture_s if captured else ""
    return {
        "file": str(rel),
        "experiment_section": experiment_section,
        "method": data.get("method", ""),
        "scenario": data.get("scenario", ""),
        "trial_id": data.get("trial", data.get("trial_id", -1)),
        "evader_policy": summary.get("evader_policy", config.get("evader_policy", "")),
        "seed": summary.get("seed", config.get("seed", "")),
        "success": captured,
        "captured": captured,
        "capture_step": capture_step,
        "t_capture": t_capture,
        "timeout": summary.get("timeout", not captured and not failed),
        "collision": summary.get("collision", summary.get("failure_reason") == "obstacle_collision"),
        "boundary_violation": summary.get(
            "boundary_violation", summary.get("failure_reason") == "boundary_violation"
        ),
        "failed": failed,
        "failure_reason": summary.get("failure_reason", ""),
        "t_max_s": t_max_s,
        "time_to_capture_s": time_to_capture_s if captured else "",
        "time_to_capture_adj_s": time_to_capture_adj_s,
        "num_steps": summary.get("num_steps", len(steps)),
        **{f"avg_{k}": avg_timing[k] for k in timing_keys},
        "mean_D_ang": mean_d_ang,
        "mean_C_cov": mean_c_cov,
        "mean_G_max_deg": mean_g_max_deg,
        "mean_C_col": mean_c_col,
        "mean_C_sync": mean_c_sync,
        "final_D_ang": final_d_ang,
        "final_C_cov": final_c_cov,
        "final_G_max_deg": final_g_max_deg,
        "final_C_col": final_c_col,
        "final_C_sync": final_c_sync,
        **pre_cap,
        **episode_stats,
        "variant": summary.get("variant", ""),
        "stack": summary.get("stack", ""),
        "sweep_param": summary.get("sweep_param", ""),
        "sweep_value": summary.get("sweep_value", ""),
    }


def aggregate(results_dir: Path, out_csv: Path) -> list[dict]:
    rows = []
    for json_path in sorted(results_dir.rglob("*.json")):
        if json_path.name == MANIFEST_FILENAME:
            continue
        data = load_run(json_path)
        rows.append(aggregate_trial_row(data, json_path, results_dir))

    if not rows:
        print(f"No JSON files found under {results_dir}")
        return []

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    all_keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")
    return rows


def aggregate_with_exports(
    results_dir: Path,
    out_csv: Path | None = None,
    export_per_step: bool = True,
) -> list[dict]:
    out = out_csv or results_dir / "aggregated_comparison.csv"
    rows = aggregate(results_dir, out)
    if rows and export_per_step:
        export_per_step_metrics(results_dir, results_dir / "per_step_metrics.csv")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Experiment run directory; outputs aggregated_comparison.csv inside it",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Alias of --run-dir; if omitted, uses latest timestamped run or results/",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV (default: <run-dir>/aggregated_comparison.csv)",
    )
    parser.add_argument(
        "--skip-per-step",
        action="store_true",
        help="Only write trial-level aggregated CSV; skip expensive per-step diagnostics export",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or args.results_dir
    if run_dir:
        results_path = Path(run_dir)
    else:
        from experiments.run_output import latest_run_dir

        latest = latest_run_dir(Path("results"))
        if latest is not None:
            results_path = latest
            print(f"Using latest run directory: {results_path}")
        else:
            results_path = Path("results")

    out_csv = Path(args.out) if args.out else results_path / "aggregated_comparison.csv"
    aggregate_with_exports(results_path, out_csv, export_per_step=not args.skip_per_step)


if __name__ == "__main__":
    main()
