"""Compare methods from aggregated results (success rate, time-to-capture, structure metrics)."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_paths import add_run_dir_arg, resolve_run_paths


def _to_bool(val: object) -> bool:
    return str(val).lower() in ("true", "1", "yes")


def _to_float(val: object) -> float | None:
    if val in ("", None):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def compare(csv_path: Path, section: str | None = None) -> None:
    if not csv_path.exists():
        print(f"File not found: {csv_path}. Run aggregate_results.py first.")
        return

    stats: dict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "success": [],
            "ttc_cond": [],
            "ttc_adj": [],
            "can_d": [],
            "can_c": [],
            "can_g": [],
            "can_sync": [],
            "free_d": [],
            "free_c": [],
            "free_g": [],
            "open_d": [],
            "open_c": [],
            "open_g": [],
            "cap_d": [],
            "cap_c": [],
            "cap_g": [],
            "pre5_d": [],
            "pre5_c": [],
            "pre5_g": [],
            "pre10_d": [],
            "pre10_c": [],
            "pre10_g": [],
            "esc_c": [],
            "esc_g": [],
            "esc_u": [],
            "full_g": [],
        }
    )
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if section and row.get("experiment_section") != section:
                continue
            key = (row["method"], row["scenario"])
            captured = _to_bool(row.get("success", row.get("captured")))
            stats[key]["success"].append(captured)
            adj = _to_float(row.get("time_to_capture_adj_s"))
            if adj is not None:
                stats[key]["ttc_adj"].append(adj)
            if captured:
                ttc = _to_float(row.get("time_to_capture_s"))
                if ttc is not None:
                    stats[key]["ttc_cond"].append(ttc)
                for col, dest in (
                    ("pre_capture_canonical_D_ang", "can_d"),
                    ("pre_capture_canonical_C_cov", "can_c"),
                    ("pre_capture_canonical_G_max_deg", "can_g"),
                    ("pre_capture_canonical_C_sync", "can_sync"),
                    ("pre_capture_D_free", "free_d"),
                    ("pre_capture_C_free", "free_c"),
                    ("pre_capture_G_free_deg", "free_g"),
                    ("capture_D_ang", "cap_d"),
                    ("capture_C_cov", "cap_c"),
                    ("capture_G_max_deg", "cap_g"),
                    ("pre_capture_5_D_ang", "pre5_d"),
                    ("pre_capture_5_C_cov", "pre5_c"),
                    ("pre_capture_5_G_max_deg", "pre5_g"),
                    ("pre_capture_10_D_ang", "pre10_d"),
                    ("pre_capture_10_C_cov", "pre10_c"),
                    ("pre_capture_10_G_max_deg", "pre10_g"),
                    ("C_esc_at_capture", "esc_c"),
                    ("G_esc_at_capture_deg", "esc_g"),
                    ("unblocked_escape_angle_at_capture_deg", "esc_u"),
                    ("G_max_full_at_capture_deg", "full_g"),
                ):
                    v = _to_float(row.get(col))
                    if v is not None:
                        stats[key][dest].append(v)
            for col, dest in (
                ("mean_open_D_ang", "open_d"),
                ("mean_open_C_cov", "open_c"),
                ("mean_open_G_max_deg", "open_g"),
            ):
                v = _to_float(row.get(col))
                if v is not None:
                    stats[key][dest].append(v)

    print(
        f"{'Method':<22} {'Scenario':<18} {'Success':>8} "
        f"{'TTC|cap':>8} {'TTCadj':>8} {'medAdj':>8} "
        f"{'capD':>6} {'capC':>6} {'capG':>6} "
        f"{'p5D':>6} {'p10D':>6} {'N':>5}"
    )
    print("-" * 120)
    print(
        "TTC|cap = conditional mean (successful trials only); TTCadj/medAdj = timeout-adjusted (all trials); "
        "cap*=at capture; p5/p10=mean D_ang over final 5/10 steps before capture"
    )
    print("-" * 120)
    for (method, scenario), data in sorted(stats.items()):
        n = len(data["success"])
        rate = sum(data["success"]) / n if n else 0.0

        def _avg(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else float("nan")

        def _fmt(v: float, w: int, prec: int) -> str:
            return f"{v:{w}.{prec}f}" if v == v else " " * (w - 3) + "n/a"

        print(
            f"{method:<22} {scenario:<18} {rate:>7.1%} "
            f"{_fmt(_avg(data['ttc_cond']), 8, 1)} {_fmt(_avg(data['ttc_adj']), 8, 1)} "
            f"{_fmt(_median(data['ttc_adj']) if data['ttc_adj'] else float('nan'), 8, 1)} "
            f"{_fmt(_avg(data['cap_d']), 6, 3)} {_fmt(_avg(data['cap_c']), 6, 3)} "
            f"{_fmt(_avg(data['cap_g']), 6, 1)} "
            f"{_fmt(_avg(data['pre5_d']), 6, 3)} {_fmt(_avg(data['pre10_d']), 6, 3)} {n:>5}"
        )

    print("\n=== Boundary-aware escape-sector summary ===")
    print(
        f"{'method':<22} {'scenario':<18} {'success_rate':>12} {'adjusted_ttc':>12} "
        f"{'C_esc_at_capture':>16} {'G_esc_at_capture_deg':>20} "
        f"{'unblocked_at_capture_deg':>24} {'G_max_full_at_capture_deg':>24}"
    )
    print("-" * 150)
    for (method, scenario), data in sorted(stats.items()):
        n = len(data["success"])
        rate = sum(data["success"]) / n if n else 0.0

        def _avg(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else float("nan")

        def _fmt(v: float, w: int, prec: int) -> str:
            return f"{v:{w}.{prec}f}" if v == v else " " * (w - 3) + "n/a"

        print(
            f"{method:<22} {scenario:<18} {rate:>11.1%} "
            f"{_fmt(_avg(data['ttc_adj']), 12, 1)} "
            f"{_fmt(_avg(data['esc_c']), 16, 3)} "
            f"{_fmt(_avg(data['esc_g']), 20, 1)} "
            f"{_fmt(_avg(data['esc_u']), 24, 1)} "
            f"{_fmt(_avg(data['full_g']), 24, 1)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    add_run_dir_arg(parser)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--section", type=str, default="comparison", help="Filter by experiment_section")
    parser.add_argument("--all-sections", action="store_true", help="Include all sections")
    args = parser.parse_args()
    paths = resolve_run_paths(args, experiment_name="comparison")
    csv_path = Path(args.csv) if args.csv else paths.aggregated_csv
    section = None if args.all_sections else args.section
    compare(csv_path, section=section)


if __name__ == "__main__":
    main()
