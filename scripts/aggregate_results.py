"""Aggregate experiment JSON results into summary CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.pre_capture import (
    pre_capture_k_from_config,
    pre_capture_structure_metrics,
    step_metric_value,
)


def _get_steps(data: dict) -> list:
    return data.get("records") or data.get("steps") or []


def _get_summary(data: dict) -> dict:
    return data.get("metadata") or data.get("summary") or {}


def _metric_value(step: dict, key: str) -> float | None:
    return step_metric_value(step, key)


def load_run(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
            d = _metric_value(s, "D_ang")
            c = _metric_value(s, "C_cov")
            g = _metric_value(s, "G_max")
            col = _metric_value(s, "C_col")
            sync = _metric_value(s, "C_sync")
            if d is not None:
                d_vals.append(d)
            if c is not None:
                c_vals.append(c)
            if g is not None:
                g_vals.append(math.degrees(g))
            if col is not None:
                col_vals.append(col)
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
    time_to_capture_s = summary.get("time_to_capture_s")
    if time_to_capture_s is None and captured and capture_step is not None:
        time_to_capture_s = float(capture_step) * dt

    k_pre = pre_capture_k_from_config(config)
    pre_cap = pre_capture_structure_metrics(steps, capture_step, captured, k=k_pre)

    rel = json_path.relative_to(results_dir)
    parts = rel.parts
    experiment_section = parts[0] if parts else ""

    return {
        "file": str(rel),
        "experiment_section": experiment_section,
        "method": data.get("method", ""),
        "scenario": data.get("scenario", ""),
        "trial_id": data.get("trial", data.get("trial_id", -1)),
        "success": captured,
        "captured": captured,
        "capture_step": capture_step,
        "time_to_capture_s": time_to_capture_s if captured else "",
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
        "variant": summary.get("variant", ""),
        "stack": summary.get("stack", ""),
        "sweep_param": summary.get("sweep_param", ""),
        "sweep_value": summary.get("sweep_value", ""),
    }


def aggregate(results_dir: Path, out_csv: Path) -> list[dict]:
    rows = []
    for json_path in sorted(results_dir.rglob("*.json")):
        data = load_run(json_path)
        rows.append(aggregate_trial_row(data, json_path, results_dir))

    if not rows:
        print(f"No JSON files found under {results_dir}")
        return []

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--out", type=str, default="results/aggregated.csv")
    args = parser.parse_args()
    aggregate(Path(args.results_dir), Path(args.out))


if __name__ == "__main__":
    main()
