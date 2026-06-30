"""Reprocess existing experiment logs with boundary-aware escape-sector metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_output import MANIFEST_FILENAME, latest_run_dir
from metrics.capture_mechanism import (
    CAPTURE_MECHANISMS,
    classify_capture_mechanism,
    distance_to_nearest_corner,
    distance_to_nearest_wall,
)
from metrics.escape_sector_metrics import escape_metrics_config_from_config
from metrics.step_diagnostics import diagnostics_from_step_record, obstacles_from_config

PER_STEP_REPROCESSED_COLUMNS = [
    "file",
    "method",
    "scenario",
    "trial_id",
    "step",
    "time_s",
    "captured_reprocessed",
    "D_ang_full",
    "C_cov_full",
    "G_max_full_deg",
    "free_escape_angle_deg",
    "blocked_escape_angle_deg",
    "unblocked_escape_angle_deg",
    "C_esc",
    "G_esc_deg",
    "escape_status",
    "valid_distance_capture",
    "valid_full_circle_capture",
    "valid_escape_sector_capture",
    "evader_pos",
    "pursuer_positions",
    "distances_to_evader",
    "angles_deg",
    "full_gaps_deg",
    "ray_length",
    "pursuer_block_radius",
    "distance_to_nearest_wall",
]

PER_TRIAL_REPROCESSED_COLUMNS = [
    "file",
    "method",
    "scenario",
    "trial_id",
    "logged_captured",
    "logged_capture_step",
    "captured_reprocessed",
    "capture_step_reprocessed",
    "time_to_capture_s_reprocessed",
    "num_steps",
    "capture_mechanism",
    "evader_pos_at_capture",
    "distance_to_nearest_wall_at_capture",
    "distance_to_nearest_corner_at_capture",
    "valid_distance_capture",
    "valid_full_circle_capture",
    "valid_escape_sector_capture",
    "D_ang_full_at_capture",
    "C_cov_full_at_capture",
    "G_max_full_deg_at_capture",
    "free_escape_angle_deg_at_capture",
    "blocked_escape_angle_deg_at_capture",
    "unblocked_escape_angle_deg_at_capture",
    "C_esc_at_capture",
    "G_esc_deg_at_capture",
    "escape_status_at_capture",
    "boundary_blocked_angle_deg_at_capture",
    "obstacle_blocked_angle_deg_at_capture",
]

LIST_COLUMNS = {
    "evader_pos",
    "pursuer_positions",
    "distances_to_evader",
    "angles_deg",
    "full_gaps_deg",
    "evader_pos_at_capture",
}


def _get_steps(data: dict[str, Any]) -> list[dict[str, Any]]:
    return data.get("records") or data.get("steps") or []


def _get_summary(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("metadata") or data.get("summary") or {}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")


def _to_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _csv_cell(key: str, value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if key in LIST_COLUMNS and value != "":
        return json.dumps(value, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _csv_cell(col, row.get(col, "")) for col in columns})


def _load_json_logs(run_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    logs: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(run_dir.rglob("*.json")):
        if path.name == MANIFEST_FILENAME:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if _get_steps(data):
            logs.append((path, data))
    return logs


def _bounds_from_config(config: dict[str, Any]) -> tuple[float, float, float, float]:
    world = config.get("world") or {}
    return (
        float(world.get("xmin", 0.0)),
        float(world.get("xmax", 40.0)),
        float(world.get("ymin", 0.0)),
        float(world.get("ymax", 40.0)),
    )


def _step_id(step: dict[str, Any], fallback: int) -> int:
    try:
        return int(step.get("step", fallback))
    except (TypeError, ValueError):
        return fallback


def _method_scenario_trial(data: dict[str, Any], summary: dict[str, Any]) -> tuple[str, str, Any]:
    method = data.get("method") or summary.get("method") or ""
    scenario = data.get("scenario") or summary.get("scenario") or ""
    trial = data.get("trial", data.get("trial_id", summary.get("trial_id", -1)))
    return str(method), str(scenario), trial


def reprocess_log(
    json_path: Path,
    data: dict[str, Any],
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    steps = _get_steps(data)
    summary = _get_summary(data)
    config = data.get("config", {})
    bounds = _bounds_from_config(config)
    obstacles = obstacles_from_config(config)
    esc_cfg = escape_metrics_config_from_config(config)
    dt = float(config.get("dt", 0.1))
    method, scenario, trial_id = _method_scenario_trial(data, summary)
    rel = str(json_path.relative_to(run_dir))

    per_step_rows: list[dict[str, Any]] = []
    first_capture: dict[str, Any] | None = None

    for idx, step in enumerate(steps):
        diagnostics = diagnostics_from_step_record(
            step,
            bounds=bounds,
            esc_cfg=esc_cfg,
            obstacles=obstacles,
            config=config,
        )
        step_num = _step_id(step, idx)
        row: dict[str, Any] = {
            "file": rel,
            "method": method,
            "scenario": scenario,
            "trial_id": trial_id,
            "step": step_num,
            "time_s": step_num * dt,
            **{col: diagnostics.get(col, "") for col in PER_STEP_REPROCESSED_COLUMNS},
        }
        row["file"] = rel
        row["method"] = method
        row["scenario"] = scenario
        row["trial_id"] = trial_id
        row["step"] = step_num
        row["time_s"] = step_num * dt
        row["captured_reprocessed"] = bool(diagnostics.get("valid_escape_sector_capture", False))
        per_step_rows.append(row)
        if first_capture is None and row["captured_reprocessed"]:
            first_capture = row

    logged_captured = _to_bool(summary.get("captured", False))
    logged_capture_step = summary.get("capture_step", "")
    captured_reprocessed = first_capture is not None
    capture_step = first_capture["step"] if first_capture else ""
    capture_time = first_capture["time_s"] if first_capture else ""

    mechanism = "invalid_or_ambiguous"
    mechanism_details = {
        "boundary_blocked_angle_deg": float("nan"),
        "obstacle_blocked_angle_deg": float("nan"),
    }
    evader_pos = ""
    wall_dist = ""
    corner_dist = ""

    if first_capture is not None:
        evader_pos = first_capture.get("evader_pos", "")
        if evader_pos not in ("", None):
            wall_dist = distance_to_nearest_wall(evader_pos, bounds)
            corner_dist = distance_to_nearest_corner(evader_pos, bounds)
        esc_at_capture = {
            "free_escape_angle_deg": first_capture.get("free_escape_angle_deg"),
            "blocked_escape_angle_deg": first_capture.get("blocked_escape_angle_deg"),
            "unblocked_escape_angle_deg": first_capture.get("unblocked_escape_angle_deg"),
            "C_esc": first_capture.get("C_esc"),
            "G_esc_deg": first_capture.get("G_esc_deg"),
        }
        mechanism, mechanism_details = classify_capture_mechanism(
            evader_pos,
            obstacles,
            bounds,
            esc_at_capture,
            valid_distance_capture=bool(first_capture.get("valid_distance_capture", False)),
            valid_escape_sector_capture=bool(first_capture.get("valid_escape_sector_capture", False)),
            ray_length=esc_cfg["ray_length"],
            num_angles=esc_cfg["num_angles"],
            num_ray_samples=esc_cfg["num_ray_samples"],
            obstacle_margin=esc_cfg["obstacle_margin"],
            boundary_margin=esc_cfg["boundary_margin"],
        )

    trial_row: dict[str, Any] = {
        "file": rel,
        "method": method,
        "scenario": scenario,
        "trial_id": trial_id,
        "logged_captured": logged_captured,
        "logged_capture_step": logged_capture_step,
        "captured_reprocessed": captured_reprocessed,
        "capture_step_reprocessed": capture_step,
        "time_to_capture_s_reprocessed": capture_time,
        "num_steps": len(steps),
        "capture_mechanism": mechanism,
        "evader_pos_at_capture": evader_pos,
        "distance_to_nearest_wall_at_capture": wall_dist,
        "distance_to_nearest_corner_at_capture": corner_dist,
        "valid_distance_capture": bool(first_capture and first_capture.get("valid_distance_capture", False)),
        "valid_full_circle_capture": bool(first_capture and first_capture.get("valid_full_circle_capture", False)),
        "valid_escape_sector_capture": bool(first_capture and first_capture.get("valid_escape_sector_capture", False)),
        "D_ang_full_at_capture": first_capture.get("D_ang_full", "") if first_capture else "",
        "C_cov_full_at_capture": first_capture.get("C_cov_full", "") if first_capture else "",
        "G_max_full_deg_at_capture": first_capture.get("G_max_full_deg", "") if first_capture else "",
        "free_escape_angle_deg_at_capture": first_capture.get("free_escape_angle_deg", "") if first_capture else "",
        "blocked_escape_angle_deg_at_capture": first_capture.get("blocked_escape_angle_deg", "") if first_capture else "",
        "unblocked_escape_angle_deg_at_capture": first_capture.get("unblocked_escape_angle_deg", "") if first_capture else "",
        "C_esc_at_capture": first_capture.get("C_esc", "") if first_capture else "",
        "G_esc_deg_at_capture": first_capture.get("G_esc_deg", "") if first_capture else "",
        "escape_status_at_capture": first_capture.get("escape_status", "") if first_capture else "",
        "boundary_blocked_angle_deg_at_capture": mechanism_details["boundary_blocked_angle_deg"],
        "obstacle_blocked_angle_deg_at_capture": mechanism_details["obstacle_blocked_angle_deg"],
    }
    return per_step_rows, trial_row


def build_capture_mechanism_summary(trial_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trial_rows:
        grouped[(str(row.get("method", "")), str(row.get("scenario", "")))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (method, scenario), rows in sorted(grouped.items()):
        counts = Counter(str(r.get("capture_mechanism", "invalid_or_ambiguous")) for r in rows)
        n_trials = len(rows)
        n_captures = sum(1 for r in rows if _to_bool(r.get("captured_reprocessed")))
        for mechanism in CAPTURE_MECHANISMS:
            n = counts.get(mechanism, 0)
            summary_rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "capture_mechanism": mechanism,
                    "n_trials": n_trials,
                    "n_captures": n,
                    "fraction_of_trials": n / n_trials if n_trials else 0.0,
                    "fraction_of_reprocessed_captures": (
                        n / n_captures if n_captures else 0.0
                    ),
                }
            )
    return summary_rows


def _group_labels(rows: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    groups = sorted({(str(r["method"]), str(r["scenario"])) for r in rows})
    return [(m, s, f"{m}\n{s}") for m, s in groups]


def _plot_placeholder(path: Path, title: str, message: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1200, 700), "white")
    draw = ImageDraw.Draw(img)
    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()
    draw.text((80, 260), title, fill=(15, 23, 42), font=title_font)
    draw.text((80, 320), message[:220], fill=(71, 85, 105), font=body_font)
    img.save(path)


def _draw_text(draw: Any, xy: tuple[int, int], text: str, fill: tuple[int, int, int]) -> None:
    from PIL import ImageFont

    draw.text(xy, text, fill=fill, font=ImageFont.load_default())


def _mean_for_group(
    rows: list[dict[str, Any]],
    method: str,
    scenario: str,
    field: str,
) -> float:
    vals = [
        _to_float(row.get(field))
        for row in rows
        if row.get("method") == method and row.get("scenario") == scenario
    ]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def _draw_bar_panel(
    draw: Any,
    box: tuple[int, int, int, int],
    title: str,
    labels: list[str],
    values: list[float],
    color: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    _draw_text(draw, (x0, y0), title, (15, 23, 42))
    plot_x0, plot_y0 = x0 + 45, y0 + 38
    plot_x1, plot_y1 = x1 - 12, y1 - 72
    draw.line((plot_x0, plot_y0, plot_x0, plot_y1, plot_x1, plot_y1), fill=(148, 163, 184), width=1)
    finite = [v for v in values if math.isfinite(v)]
    y_max = max(finite) if finite else 1.0
    if y_max <= 1.0:
        y_max = 1.0
    else:
        y_max *= 1.12
    n = max(len(values), 1)
    slot = max((plot_x1 - plot_x0) / n, 1.0)
    bar_w = max(int(slot * 0.62), 3)
    for i, value in enumerate(values):
        cx = int(plot_x0 + slot * (i + 0.5))
        if math.isfinite(value):
            h = int((value / y_max) * max(plot_y1 - plot_y0, 1))
            draw.rectangle((cx - bar_w // 2, plot_y1 - h, cx + bar_w // 2, plot_y1), fill=color)
            _draw_text(draw, (cx - bar_w // 2, max(plot_y1 - h - 16, plot_y0)), f"{value:.2g}", (51, 65, 85))
        label = labels[i].replace("\n", "/")
        if len(label) > 18:
            label = label[:17] + "."
        _draw_text(draw, (max(cx - 42, plot_x0), plot_y1 + 10), label, (71, 85, 105))
    _draw_text(draw, (x0 + 4, plot_y0 - 4), f"{y_max:.2g}", (100, 116, 139))


def _save_metric_panels(
    rows: list[dict[str, Any]],
    panels: list[tuple[str, str]],
    out_path: Path,
    title: str,
    color: tuple[int, int, int],
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    groups = _group_labels(rows)
    labels = [label for _, _, label in groups]
    width, height = 1500, 560
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 24), title, fill=(15, 23, 42), font=ImageFont.load_default())
    panel_w = (width - 80) // len(panels)
    for i, (field, panel_title) in enumerate(panels):
        values = [_mean_for_group(rows, method, scenario, field) for method, scenario, _ in groups]
        box = (40 + i * panel_w, 70, 40 + (i + 1) * panel_w - 20, height - 30)
        _draw_bar_panel(draw, box, panel_title, labels, values, color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _heat_color(frac: float) -> tuple[int, int, int]:
    frac = float(np.clip(frac, 0.0, 1.0))
    stops = [
        (255, 255, 255),
        (254, 224, 139),
        (251, 140, 60),
        (220, 38, 127),
        (63, 0, 125),
    ]
    idx = min(int(frac * (len(stops) - 1)), len(stops) - 2)
    local = frac * (len(stops) - 1) - idx
    a, b = stops[idx], stops[idx + 1]
    return tuple(int(a[j] + (b[j] - a[j]) * local) for j in range(3))


def plot_escape_sector_structure(trial_rows: list[dict[str, Any]], out_path: Path) -> None:
    rows = [r for r in trial_rows if _to_bool(r.get("captured_reprocessed"))]
    if not rows:
        _plot_placeholder(out_path, "Escape-Sector Structure", "No reprocessed captures found")
        return

    panels = [
        ("C_esc_at_capture", "C_esc"),
        ("G_esc_deg_at_capture", "G_esc (deg)"),
        ("unblocked_escape_angle_deg_at_capture", "Unblocked angle (deg)"),
    ]
    _save_metric_panels(
        rows,
        panels,
        out_path,
        "Boundary-aware escape-sector metrics at reprocessed capture",
        (37, 99, 235),
    )


def plot_full_circle_diagnostics(trial_rows: list[dict[str, Any]], out_path: Path) -> None:
    rows = [r for r in trial_rows if _to_bool(r.get("captured_reprocessed"))]
    if not rows:
        _plot_placeholder(out_path, "Full-Circle Diagnostics", "No reprocessed captures found")
        return

    panels = [
        ("D_ang_full_at_capture", "D_ang_full"),
        ("C_cov_full_at_capture", "C_cov_full"),
        ("G_max_full_deg_at_capture", "G_max_full_deg"),
    ]
    _save_metric_panels(
        rows,
        panels,
        out_path,
        "Full-circle diagnostics only, not rectangular-workspace capture criterion",
        (100, 116, 139),
    )


def plot_capture_position_heatmap(
    trial_rows: list[dict[str, Any]],
    out_path: Path,
    bounds: tuple[float, float, float, float],
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    points = []
    for row in trial_rows:
        if not _to_bool(row.get("captured_reprocessed")):
            continue
        pos = row.get("evader_pos_at_capture")
        if isinstance(pos, str):
            try:
                pos = json.loads(pos)
            except json.JSONDecodeError:
                continue
        if pos not in ("", None):
            points.append(np.asarray(pos, dtype=float))

    if not points:
        _plot_placeholder(out_path, "Capture Position Heatmap", "No reprocessed captures found")
        return

    pts = np.vstack(points)
    xmin, xmax, ymin, ymax = bounds
    width, height = 900, 820
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((50, 26), "Evader position at reprocessed capture", fill=(15, 23, 42), font=ImageFont.load_default())
    px0, py0, px1, py1 = 85, 80, 805, 720
    bins = 24
    counts = np.zeros((bins, bins), dtype=int)
    for x, y in pts:
        ix = min(max(int((x - xmin) / max(xmax - xmin, 1e-9) * bins), 0), bins - 1)
        iy = min(max(int((y - ymin) / max(ymax - ymin, 1e-9) * bins), 0), bins - 1)
        counts[ix, iy] += 1
    max_count = max(int(counts.max()), 1)
    cell_w = (px1 - px0) / bins
    cell_h = (py1 - py0) / bins
    for ix in range(bins):
        for iy in range(bins):
            c = counts[ix, iy]
            if c <= 0:
                continue
            x0 = int(px0 + ix * cell_w)
            x1 = int(px0 + (ix + 1) * cell_w)
            y1 = int(py1 - iy * cell_h)
            y0 = int(py1 - (iy + 1) * cell_h)
            draw.rectangle((x0, y0, x1, y1), fill=_heat_color(c / max_count))
    draw.rectangle((px0, py0, px1, py1), outline=(15, 23, 42), width=2)
    for x, y in pts:
        sx = int(px0 + (x - xmin) / max(xmax - xmin, 1e-9) * (px1 - px0))
        sy = int(py1 - (y - ymin) / max(ymax - ymin, 1e-9) * (py1 - py0))
        draw.ellipse((sx - 4, sy - 4, sx + 4, sy + 4), fill=(56, 189, 248), outline=(15, 23, 42))
    draw.text((px0, py1 + 18), f"x: {xmin:g} to {xmax:g}", fill=(71, 85, 105), font=ImageFont.load_default())
    draw.text((px0, py1 + 40), f"y: {ymin:g} to {ymax:g}", fill=(71, 85, 105), font=ImageFont.load_default())
    draw.text((px1 - 145, py1 + 18), f"n={len(pts)}", fill=(71, 85, 105), font=ImageFont.load_default())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def plot_capture_mechanism_distribution(
    trial_rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if not trial_rows:
        _plot_placeholder(out_path, "Capture Mechanism Distribution", "No trials found")
        return

    groups = _group_labels(trial_rows)
    colors = {
        "open_field_capture": (37, 99, 235),
        "boundary_assisted_capture": (22, 163, 74),
        "corner_assisted_capture": (245, 158, 11),
        "obstacle_assisted_capture": (220, 38, 38),
        "mixed_assisted_capture": (124, 58, 237),
        "invalid_or_ambiguous": (148, 163, 184),
    }
    width, height = 1300, 720
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((50, 28), "Capture mechanism distribution", fill=(15, 23, 42), font=ImageFont.load_default())
    px0, py0, px1, py1 = 80, 90, 980, 610
    draw.line((px0, py0, px0, py1, px1, py1), fill=(148, 163, 184), width=1)
    totals = []
    group_counts: list[dict[str, int]] = []
    for method, scenario, _ in groups:
        counts = {
            mechanism: sum(
                1
                for row in trial_rows
                if row.get("method") == method
                and row.get("scenario") == scenario
                and row.get("capture_mechanism") == mechanism
            )
            for mechanism in CAPTURE_MECHANISMS
        }
        group_counts.append(counts)
        totals.append(sum(counts.values()))
    y_max = max(totals) if totals else 1
    slot = max((px1 - px0) / max(len(groups), 1), 1)
    bar_w = max(int(slot * 0.55), 8)
    for i, (_, _, label) in enumerate(groups):
        cx = int(px0 + slot * (i + 0.5))
        bottom = py1
        for mechanism in CAPTURE_MECHANISMS:
            count = group_counts[i][mechanism]
            if count <= 0:
                continue
            h = int(count / max(y_max, 1) * (py1 - py0))
            draw.rectangle(
                (cx - bar_w // 2, bottom - h, cx + bar_w // 2, bottom),
                fill=colors[mechanism],
                outline=(255, 255, 255),
            )
            bottom -= h
        short_label = label.replace("\n", "/")
        if len(short_label) > 20:
            short_label = short_label[:19] + "."
        draw.text((max(cx - 48, px0), py1 + 12), short_label, fill=(71, 85, 105), font=ImageFont.load_default())
    draw.text((px0 + 4, py0 - 18), f"{y_max}", fill=(100, 116, 139), font=ImageFont.load_default())
    lx, ly = 1020, 105
    for mechanism in CAPTURE_MECHANISMS:
        draw.rectangle((lx, ly, lx + 16, ly + 16), fill=colors[mechanism])
        draw.text((lx + 24, ly + 2), mechanism, fill=(51, 65, 85), font=ImageFont.load_default())
        ly += 32
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def reprocess_run(
    run_dir: Path,
    out_dir: Path | None = None,
    *,
    write_per_step: bool = True,
    make_plots: bool = True,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else run_dir
    logs = _load_json_logs(run_dir)
    if not logs:
        print(f"No experiment JSON logs with step records found under {run_dir}")
        return {"logs": 0, "trials": 0, "steps": 0}

    all_step_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    first_bounds = (0.0, 40.0, 0.0, 40.0)

    for json_path, data in logs:
        config = data.get("config", {})
        if not trial_rows:
            first_bounds = _bounds_from_config(config)
        step_rows, trial_row = reprocess_log(json_path, data, run_dir)
        all_step_rows.extend(step_rows)
        trial_rows.append(trial_row)

    _write_csv(out_dir / "per_trial_reprocessed.csv", trial_rows, PER_TRIAL_REPROCESSED_COLUMNS)
    if write_per_step:
        _write_csv(out_dir / "per_step_reprocessed.csv", all_step_rows, PER_STEP_REPROCESSED_COLUMNS)

    summary_rows = build_capture_mechanism_summary(trial_rows)
    summary_columns = [
        "method",
        "scenario",
        "capture_mechanism",
        "n_trials",
        "n_captures",
        "fraction_of_trials",
        "fraction_of_reprocessed_captures",
    ]
    _write_csv(out_dir / "capture_mechanism_summary.csv", summary_rows, summary_columns)

    if make_plots:
        plot_escape_sector_structure(trial_rows, out_dir / "fig_escape_sector_structure.png")
        plot_full_circle_diagnostics(trial_rows, out_dir / "fig_full_circle_diagnostics.png")
        plot_capture_position_heatmap(trial_rows, out_dir / "fig_capture_position_heatmap.png", first_bounds)
        plot_capture_mechanism_distribution(trial_rows, out_dir / "fig_capture_mechanism_distribution.png")

    print(f"Reprocessed {len(trial_rows)} trial logs and {len(all_step_rows)} step records")
    print(f"Wrote {out_dir / 'per_trial_reprocessed.csv'}")
    if write_per_step:
        print(f"Wrote {out_dir / 'per_step_reprocessed.csv'}")
    print(f"Wrote {out_dir / 'capture_mechanism_summary.csv'}")
    return {"logs": len(logs), "trials": len(trial_rows), "steps": len(all_step_rows)}


def _default_run_dir() -> Path:
    latest = latest_run_dir(Path("results"))
    return latest if latest is not None else Path("results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=str, default=None, help="Directory containing experiment JSON logs")
    parser.add_argument("--results-dir", type=str, default=None, help="Alias of --run-dir")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory for reprocessed outputs")
    parser.add_argument("--no-per-step", action="store_true", help="Skip per_step_reprocessed.csv")
    parser.add_argument("--skip-plots", action="store_true", help="Skip PNG figure generation")
    args = parser.parse_args()

    run_dir = Path(args.run_dir or args.results_dir) if (args.run_dir or args.results_dir) else _default_run_dir()
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    reprocess_run(
        run_dir,
        out_dir,
        write_per_step=not args.no_per_step,
        make_plots=not args.skip_plots,
    )


if __name__ == "__main__":
    main()
