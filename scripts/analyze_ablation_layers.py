"""Analyze FCEM DG layer ablation runs into the requested artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_output import MANIFEST_FILENAME, latest_run_dir
from metrics.comparison_metrics import adjusted_time_to_capture_s, episode_t_max_s
from metrics.escape_sector_pre_capture import escape_sector_window_metrics

OUTPUT_PREFIX = "ablation_dg_50seed"

EXPECTED_SCENARIOS = ("random_obstacles", "single_exit")
EXPECTED_VARIANTS = (
    "full_fcem",
    "no_l1_prediction",
    "no_l2_multi_candidate_manifold",
    "no_l3_executability_assignment",
    "no_l4_slot_velocity_feedforward",
)
VARIANT_LABELS = {
    "full_fcem": "Full",
    "no_l1_prediction": "No L1",
    "no_l2_multi_candidate_manifold": "No L2",
    "no_l3_executability_assignment": "No L3",
    "no_l4_slot_velocity_feedforward": "No L4",
}

METRIC_COLUMNS = [
    "success_rate",
    "adjusted_TTC_with_timeout",
    "conditional_TTC_success_only",
    "C_esc_at_capture",
    "G_esc_at_capture_deg",
    "C_esc_final5_mean",
    "G_esc_final5_mean_deg",
    "mean_slot_error",
    "final5_slot_error",
    "assignment_switch_count",
    "infeasible_slot_rate",
    "collision_rate",
    "boundary_violation_rate",
    "decision_time_p95",
    "decision_time_p99",
]

TRIAL_COLUMNS = [
    "variant",
    "scenario",
    "evader_policy",
    "method",
    "trial_id",
    "seed",
    "remove_layer",
    *METRIC_COLUMNS,
    "captured",
    "timeout",
    "failure_reason",
    "num_steps",
    "t_max_s",
]

SUMMARY_COLUMNS = [
    "variant",
    "scenario",
    "evader_policy",
    "n_trials",
    *METRIC_COLUMNS,
]

ESCAPE_COLUMNS = (
    "C_esc_at_capture",
    "G_esc_at_capture_deg",
    "C_esc_final5_mean",
    "G_esc_final5_mean_deg",
)


def _to_bool(val: object) -> bool:
    return str(val).lower() in ("true", "1", "yes")


def _to_float(val: object) -> float | None:
    if val in ("", None):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _csv_cell(val: object) -> object:
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    return val


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _result_jsons(run_dir: Path) -> list[Path]:
    return [
        p
        for p in sorted(run_dir.rglob("*.json"))
        if p.name != MANIFEST_FILENAME
    ]


def _summary(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("metadata") or data.get("summary") or {}


def _records(data: dict[str, Any]) -> list[dict[str, Any]]:
    return data.get("records") or data.get("steps") or []


def _extra(record: dict[str, Any]) -> dict[str, Any]:
    extra = record.get("extra")
    return extra if isinstance(extra, dict) else {}


def _assignment(record: dict[str, Any]) -> tuple[int, ...] | None:
    raw = _extra(record).get("assignment", record.get("assignment"))
    if raw in ("", None):
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    try:
        return tuple(int(x) for x in raw)
    except TypeError:
        return None


def _assignments(records: list[dict[str, Any]]) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for record in records:
        assignment = _assignment(record)
        if assignment is not None:
            out.append(assignment)
    return out


def _assignment_switch_count(records: list[dict[str, Any]]) -> int:
    count = 0
    prev: tuple[int, ...] | None = None
    for assignment in _assignments(records):
        if prev is not None and assignment != prev:
            count += 1
        prev = assignment
    return count


def _slot_errors(records: list[dict[str, Any]]) -> list[float]:
    vals: list[float] = []
    for record in records:
        val = _to_float(record.get("slot_error"))
        if val is not None:
            vals.append(val)
    return vals


def _infeasible_values(records: list[dict[str, Any]]) -> list[float]:
    vals: list[float] = []
    for record in records:
        extra = _extra(record)
        if "infeasible" in extra:
            vals.append(1.0 if _to_bool(extra.get("infeasible")) else 0.0)
    return vals


def _decision_times(records: list[dict[str, Any]]) -> list[float]:
    vals: list[float] = []
    for record in records:
        detail = record.get("timing_detail")
        if isinstance(detail, dict):
            val = _to_float(detail.get("total_ms"))
        else:
            val = None
        if val is None:
            val = _to_float(record.get("timing_ms"))
        if val is not None:
            vals.append(val)
    return vals


def _escape_values(
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    capture_step: int | None,
    success: bool,
) -> dict[str, float | None]:
    computed: dict[str, Any] | None = None
    out: dict[str, float | None] = {}
    for key in ESCAPE_COLUMNS:
        val = _to_float(summary.get(key))
        if val is None:
            if computed is None:
                computed = escape_sector_window_metrics(records, capture_step, success)
            val = _to_float(computed.get(key))
        out[key] = val
    return out


def _trial_row(data: dict[str, Any]) -> dict[str, Any]:
    summary = _summary(data)
    records = _records(data)
    config = data.get("config", {})
    captured = _to_bool(summary.get("captured", False))
    failed = _to_bool(summary.get("failed", False))
    success = _to_bool(summary.get("success", captured and not failed))
    capture_step_raw = summary.get("capture_step")
    capture_step = int(capture_step_raw) if capture_step_raw not in ("", None) else None

    t_max_s = _to_float(summary.get("t_max_s")) or episode_t_max_s(config)
    ttc = _to_float(summary.get("time_to_capture_s"))
    adjusted_ttc = _to_float(summary.get("time_to_capture_adj_s"))
    if adjusted_ttc is None:
        adjusted_ttc = adjusted_time_to_capture_s(success, ttc, t_max_s)

    slot_errors = _slot_errors(records)
    final5_slot = slot_errors[-5:]
    infeasible = _infeasible_values(records)
    timings = _decision_times(records)
    escape = _escape_values(summary, records, capture_step, success)

    failure_reason = str(summary.get("failure_reason") or "")
    collision = failure_reason == "obstacle_collision" or _to_bool(summary.get("collision", False))
    boundary = failure_reason == "boundary_violation" or _to_bool(
        summary.get("boundary_violation", False)
    )

    variant = str(summary.get("variant") or data.get("method", "").replace("fcem_", ""))
    row: dict[str, Any] = {
        "variant": variant,
        "scenario": data.get("scenario", summary.get("scenario", "")),
        "evader_policy": summary.get("evader_policy", config.get("evader_policy", "")),
        "method": data.get("method", ""),
        "trial_id": data.get("trial", data.get("trial_id", "")),
        "seed": summary.get("seed", config.get("seed", "")),
        "remove_layer": summary.get("remove_layer", ""),
        "success_rate": 1.0 if success else 0.0,
        "adjusted_TTC_with_timeout": adjusted_ttc,
        "conditional_TTC_success_only": ttc if success else None,
        "mean_slot_error": _mean(slot_errors),
        "final5_slot_error": _mean(final5_slot),
        "assignment_switch_count": _assignment_switch_count(records),
        "infeasible_slot_rate": _mean(infeasible),
        "collision_rate": 1.0 if collision else 0.0,
        "boundary_violation_rate": 1.0 if boundary else 0.0,
        "decision_time_p95": _percentile(timings, 95.0),
        "decision_time_p99": _percentile(timings, 99.0),
        "captured": captured,
        "timeout": _to_bool(summary.get("timeout", not captured and not failed)),
        "failure_reason": failure_reason,
        "num_steps": summary.get("num_steps", len(records)),
        "t_max_s": t_max_s,
        "_timings": timings,
    }
    row.update(escape)
    return row


def load_trial_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = [_trial_row(_load_json(path)) for path in _result_jsons(run_dir)]
    return sorted(rows, key=_row_sort_key)


def _row_sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    scenario = str(row.get("scenario", ""))
    variant = str(row.get("variant", ""))
    scenario_idx = EXPECTED_SCENARIOS.index(scenario) if scenario in EXPECTED_SCENARIOS else 99
    variant_idx = EXPECTED_VARIANTS.index(variant) if variant in EXPECTED_VARIANTS else 99
    try:
        trial_idx = int(row.get("trial_id", 0))
    except (TypeError, ValueError):
        trial_idx = 0
    return scenario_idx, variant_idx, trial_idx


def write_trial_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRIAL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _csv_cell(row.get(col)) for col in TRIAL_COLUMNS})
    return path


def summarize_trial_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("variant", "")),
            str(row.get("scenario", "")),
            str(row.get("evader_policy", "")),
        )
        groups[key].append(row)

    out: list[dict[str, Any]] = []
    for (variant, scenario, evader_policy), items in groups.items():
        row: dict[str, Any] = {
            "variant": variant,
            "scenario": scenario,
            "evader_policy": evader_policy,
            "n_trials": len(items),
        }
        for metric in METRIC_COLUMNS:
            if metric == "decision_time_p95":
                row[metric] = _percentile(
                    [v for item in items for v in item.get("_timings", [])],
                    95.0,
                )
            elif metric == "decision_time_p99":
                row[metric] = _percentile(
                    [v for item in items for v in item.get("_timings", [])],
                    99.0,
                )
            else:
                vals = [
                    v
                    for item in items
                    if (v := _to_float(item.get(metric))) is not None
                ]
                row[metric] = _mean(vals)
        out.append(row)
    return sorted(out, key=_row_sort_key)


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _csv_cell(row.get(col)) for col in SUMMARY_COLUMNS})
    return path


def _fmt(val: object, precision: int = 3) -> str:
    f = _to_float(val)
    if f is None:
        return ""
    if abs(f) >= 100:
        return f"{f:.1f}"
    return f"{f:.{precision}f}"


def write_markdown_table(rows: list[dict[str, Any]], path: Path) -> Path:
    cols = [
        ("scenario", "Scenario"),
        ("variant", "Variant"),
        ("success_rate", "Success"),
        ("adjusted_TTC_with_timeout", "Adj. TTC"),
        ("conditional_TTC_success_only", "TTC succ"),
        ("C_esc_at_capture", "Cesc@cap"),
        ("G_esc_at_capture_deg", "Gesc@cap"),
        ("C_esc_final5_mean", "Cesc final5"),
        ("G_esc_final5_mean_deg", "Gesc final5"),
        ("mean_slot_error", "Slot err"),
        ("final5_slot_error", "Slot err final5"),
        ("assignment_switch_count", "Assign switches"),
        ("infeasible_slot_rate", "Infeasible"),
        ("collision_rate", "Collision"),
        ("boundary_violation_rate", "Boundary"),
        ("decision_time_p95", "p95 ms"),
        ("decision_time_p99", "p99 ms"),
    ]
    lines = [
        "| " + " | ".join(title for _, title in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        cells: list[str] = []
        for key, _ in cols:
            if key == "variant":
                cells.append(VARIANT_LABELS.get(str(row.get(key, "")), str(row.get(key, ""))))
            elif key == "scenario":
                cells.append(str(row.get(key, "")))
            else:
                cells.append(_fmt(row.get(key), precision=3))
        lines.append("| " + " | ".join(cells) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _summary_metric(
    rows: list[dict[str, Any]],
    variant: str,
    scenario: str,
    metric: str,
) -> float | None:
    for row in rows:
        if row.get("variant") == variant and row.get("scenario") == scenario:
            return _to_float(row.get(metric))
    return None


def _plot_ablation_layers_pillow(rows: list[dict[str, Any]], path: Path) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    metrics = [
        ("success_rate", "Success rate"),
        ("adjusted_TTC_with_timeout", "Adjusted TTC (s)"),
        ("C_esc_at_capture", "C_esc at capture"),
        ("G_esc_at_capture_deg", "G_esc at capture (deg)"),
        ("final5_slot_error", "Final-5 slot error"),
        ("assignment_switch_count", "Assignment switches"),
    ]
    colors = {
        "random_obstacles": (47, 111, 166),
        "single_exit": (207, 111, 51),
    }
    width, height = 1500, 900
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.text((24, 18), "FCEM layer ablation under differential-game evader", fill=(20, 20, 20), font=font)
    legend_x = width - 300
    for i, scenario in enumerate(EXPECTED_SCENARIOS):
        y = 22 + i * 22
        draw.rectangle((legend_x, y, legend_x + 18, y + 12), fill=colors[scenario])
        draw.text((legend_x + 26, y - 2), scenario, fill=(20, 20, 20), font=font)

    panel_w = (width - 80) // 3
    panel_h = (height - 100) // 2
    for idx, (metric, title) in enumerate(metrics):
        row_idx = idx // 3
        col_idx = idx % 3
        left = 35 + col_idx * panel_w
        top = 75 + row_idx * panel_h
        plot_left = left + 58
        plot_top = top + 30
        plot_w = panel_w - 82
        plot_h = panel_h - 92
        axis_bottom = plot_top + plot_h

        vals = [
            _summary_metric(rows, variant, scenario, metric)
            for variant in EXPECTED_VARIANTS
            for scenario in EXPECTED_SCENARIOS
        ]
        finite = [v for v in vals if v is not None]
        y_max = 1.0 if metric == "success_rate" else (max(finite) * 1.15 if finite else 1.0)
        if y_max <= 0.0:
            y_max = 1.0

        draw.text((left + 6, top + 4), title, fill=(20, 20, 20), font=font)
        draw.line((plot_left, plot_top, plot_left, axis_bottom), fill=(40, 40, 40), width=2)
        draw.line((plot_left, axis_bottom, plot_left + plot_w, axis_bottom), fill=(40, 40, 40), width=2)
        draw.text((plot_left - 50, plot_top - 6), _fmt(y_max, 2), fill=(70, 70, 70), font=font)
        draw.text((plot_left - 28, axis_bottom - 8), "0", fill=(70, 70, 70), font=font)

        group_w = plot_w / len(EXPECTED_VARIANTS)
        bar_w = max(8, int(group_w * 0.28))
        for v_idx, variant in enumerate(EXPECTED_VARIANTS):
            group_center = plot_left + group_w * (v_idx + 0.5)
            for s_idx, scenario in enumerate(EXPECTED_SCENARIOS):
                val = _summary_metric(rows, variant, scenario, metric)
                if val is None:
                    continue
                bar_h = int((val / y_max) * (plot_h - 4))
                x0 = int(group_center + (s_idx - 0.5) * bar_w - bar_w / 2)
                x1 = x0 + bar_w
                y0 = axis_bottom - bar_h
                draw.rectangle((x0, y0, x1, axis_bottom - 1), fill=colors[scenario])
            label = VARIANT_LABELS[variant]
            draw.text((int(group_center - 18), axis_bottom + 8), label, fill=(35, 35, 35), font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def plot_ablation_layers(rows: list[dict[str, Any]], path: Path) -> Path:
    try:
        import contextlib
        import io

        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Matplotlib unavailable ({exc}); using Pillow fallback for {path}")
        return _plot_ablation_layers_pillow(rows, path)

    metrics = [
        ("success_rate", "Success rate"),
        ("adjusted_TTC_with_timeout", "Adjusted TTC (s)"),
        ("C_esc_at_capture", "C_esc at capture"),
        ("G_esc_at_capture_deg", "G_esc at capture (deg)"),
        ("final5_slot_error", "Final-5 slot error"),
        ("assignment_switch_count", "Assignment switches"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    x = np.arange(len(EXPECTED_VARIANTS))
    width = 0.36

    for ax, (metric, title) in zip(axes.flat, metrics):
        for i, scenario in enumerate(EXPECTED_SCENARIOS):
            vals = [
                _summary_metric(rows, variant, scenario, metric)
                for variant in EXPECTED_VARIANTS
            ]
            y = [float("nan") if v is None else v for v in vals]
            ax.bar(x + (i - 0.5) * width, y, width, label=scenario)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [VARIANT_LABELS[v] for v in EXPECTED_VARIANTS],
            rotation=25,
            ha="right",
        )

    axes[0, 0].legend(loc="best")
    fig.suptitle("FCEM layer ablation under differential-game evader")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def analyze_ablation_run(run_dir: Path, out_dir: Path | None = None) -> dict[str, Path]:
    out_dir = out_dir or run_dir
    trial_rows = load_trial_rows(run_dir)
    if not trial_rows:
        raise SystemExit(f"No trial JSON files found under {run_dir}")

    summary_rows = summarize_trial_rows(trial_rows)
    outputs = {
        "per_trial": write_trial_csv(
            trial_rows,
            out_dir / f"{OUTPUT_PREFIX}_per_trial.csv",
        ),
        "summary": write_summary_csv(
            summary_rows,
            out_dir / f"{OUTPUT_PREFIX}_summary.csv",
        ),
        "figure": plot_ablation_layers(
            summary_rows,
            out_dir / "fig_ablation_layers.png",
        ),
        "table": write_markdown_table(
            summary_rows,
            out_dir / "table_ablation_layers.md",
        ),
    }
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DG layer ablation outputs")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        latest = latest_run_dir(Path("results"), "ablation_dg_50seed")
        if latest is None:
            raise SystemExit("No --run-dir specified and no ablation_dg_50seed run found")
        run_dir = latest

    outputs = analyze_ablation_run(run_dir, Path(args.out_dir) if args.out_dir else None)
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
