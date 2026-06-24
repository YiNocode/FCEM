"""Visualize experiment runs (trajectories) from JSON files or a directory."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _get_steps(data: dict) -> list:
    return data.get("records") or data.get("steps") or []


def _get_summary(data: dict) -> dict:
    return data.get("metadata") or data.get("summary") or {}


def _metric(step: dict, key: str, nested_key: str | None = None) -> float:
    if key in step and step[key] not in ("", None):
        return float(step[key])
    metrics = step.get("metrics", {})
    if nested_key and nested_key in metrics:
        return float(metrics[nested_key])
    if key in metrics:
        return float(metrics[key])
    return 0.0


def collect_run_jsons(root: Path, recursive: bool = True) -> list[Path]:
    """Collect experiment JSON files under ``root``."""
    if root.is_file():
        return [root] if root.suffix.lower() == ".json" else []

    pattern = "**/*.json" if recursive else "*.json"
    files = sorted(root.glob(pattern))
    runs: list[Path] = []
    for path in files:
        if path.name.endswith(".viz.json"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if _get_steps(data):
            runs.append(path)
    return runs


def _default_out_path(run_json: Path, out_dir: Path | None) -> Path:
    if out_dir is not None:
        return out_dir / f"{run_json.stem}.png"
    return run_json.with_name(f"{run_json.stem}.viz.png")


def visualize(run_json: Path, out_path: Path) -> bool:
    data = json.loads(run_json.read_text(encoding="utf-8"))
    steps = _get_steps(data)
    if not steps:
        print(f"Skip (no steps): {run_json}")
        return False

    cfg = data.get("config", {})
    world = cfg.get("world", {"xmin": 0, "xmax": 40, "ymin": 0, "ymax": 40})
    obstacles = cfg.get("scenario", {}).get("obstacles", [])

    method = data.get("method", run_json.stem)
    scenario = data.get("scenario", "")
    trial = data.get("trial", data.get("trial_id", ""))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(world["xmin"], world["xmax"])
    ax.set_ylim(world["ymin"], world["ymax"])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)

    for obs in obstacles:
        c = obs["center"]
        r = obs["radius"]
        ax.add_patch(plt.Circle(c, r, fill=False, linestyle="--"))

    evader_traj = np.array([np.array(s["evader"]) for s in steps])
    ax.plot(evader_traj[:, 0], evader_traj[:, 1], "k-", label="evader", linewidth=1.5)

    n_p = len(steps[0]["pursuers"])
    for i in range(n_p):
        traj = np.array([np.array(s["pursuers"])[i] for s in steps])
        ax.plot(traj[:, 0], traj[:, 1], linewidth=1.0, label=f"P{i}")

    last = steps[-1]
    evader = np.array(last["evader"])
    pursuers = np.array(last["pursuers"])
    ax.scatter([evader[0]], [evader[1]], marker="*", s=150, c="black")
    ax.scatter(pursuers[:, 0], pursuers[:, 1], marker="^", s=80)

    cap_r = cfg.get("fcem", {}).get("capture_radius", 1.8)
    ax.add_patch(plt.Circle(evader, cap_r, fill=False, linestyle="-.", color="green"))

    d_ang = _metric(last, "D_ang")
    c_cov = _metric(last, "C_cov")
    g_max = _metric(last, "G_max")
    summary = _get_summary(data)
    info = (
        f"{method} | {scenario} | trial={trial}\n"
        f"captured={summary.get('captured')} step={last['step']}\n"
        f"D_ang={d_ang:.2f} C_cov={c_cov:.2f} G_max={math.degrees(g_max):.0f}°"
    )
    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.8),
    )
    ax.set_title(f"{method} / {scenario} / t{trial}")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return True


def visualize_folder(
    folder: Path,
    out_dir: Path | None = None,
    recursive: bool = True,
) -> int:
    """Render trajectory plots for all run JSON files under ``folder``."""
    runs = collect_run_jsons(folder, recursive=recursive)
    if not runs:
        print(f"No run JSON files found under {folder}")
        return 0

    count = 0
    folder = folder.resolve()
    for run_json in runs:
        run_json = run_json.resolve()
        if out_dir is not None:
            try:
                rel = run_json.relative_to(folder)
                out_path = out_dir / rel.parent / f"{run_json.stem}.png"
            except ValueError:
                out_path = out_dir / f"{run_json.stem}.png"
        else:
            out_path = _default_out_path(run_json, None)
        if visualize(run_json, out_path):
            count += 1
    print(f"Done: {count}/{len(runs)} plots")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize trajectory from one run JSON or all runs in a folder"
    )
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        help="Path to a run .json file, or a directory containing run JSON files",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory of run JSON files (batch mode; alternative to path)",
    )
    parser.add_argument("--out", type=str, default=None, help="Output image (single-file mode)")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for batch mode (default: next to each JSON as *.viz.png)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan JSON files in the top level of the folder",
    )
    args = parser.parse_args()

    target = args.dir or args.path
    if not target:
        parser.error("Provide a run JSON path or --dir / path to a folder")

    root = Path(target)
    if not root.exists():
        raise SystemExit(f"Path not found: {root}")

    if root.is_dir():
        out_dir = Path(args.out_dir) if args.out_dir else None
        visualize_folder(root, out_dir=out_dir, recursive=not args.no_recursive)
        return

    out = Path(args.out) if args.out else _default_out_path(root, None)
    visualize(root, out)


if __name__ == "__main__":
    main()
