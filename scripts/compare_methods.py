"""Compare methods from aggregated results (success rate, time-to-capture, structure metrics)."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _to_bool(val: object) -> bool:
    return str(val).lower() in ("true", "1", "yes")


def _to_float(val: object) -> float | None:
    if val in ("", None):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def compare(csv_path: Path, section: str | None = None) -> None:
    if not csv_path.exists():
        print(f"File not found: {csv_path}. Run aggregate_results.py first.")
        return

    stats: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"success": [], "ttc": [], "d_ang": [], "c_cov": [], "g_max": [], "c_sync": [],
                 "pre_d": [], "pre_c": [], "pre_g": [], "pre_sync": []}
    )
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if section and row.get("experiment_section") != section:
                continue
            key = (row["method"], row["scenario"])
            stats[key]["success"].append(_to_bool(row.get("success", row.get("captured"))))
            if _to_bool(row.get("success", row.get("captured"))):
                ttc = _to_float(row.get("time_to_capture_s"))
                if ttc is not None:
                    stats[key]["ttc"].append(ttc)
            d = _to_float(row.get("mean_D_ang"))
            c = _to_float(row.get("mean_C_cov"))
            g = _to_float(row.get("mean_G_max_deg"))
            sync = _to_float(row.get("mean_C_sync"))
            if d is not None:
                stats[key]["d_ang"].append(d)
            if c is not None:
                stats[key]["c_cov"].append(c)
            if g is not None:
                stats[key]["g_max"].append(g)
            if sync is not None:
                stats[key]["c_sync"].append(sync)
            if _to_bool(row.get("success", row.get("captured"))):
                for col, dest in (
                    ("pre_capture_D_ang", "pre_d"),
                    ("pre_capture_C_cov", "pre_c"),
                    ("pre_capture_G_max_deg", "pre_g"),
                    ("pre_capture_C_sync", "pre_sync"),
                ):
                    v = _to_float(row.get(col))
                    if v is not None:
                        stats[key][dest].append(v)

    print(
        f"{'Method':<22} {'Scenario':<18} {'Success':>8} {'TTC(s)':>8} "
        f"{'D_ang':>8} {'C_cov':>8} {'G_max':>8} {'C_sync':>8} "
        f"{'pD_ang':>8} {'pC_cov':>8} {'pGmax':>8} {'pSync':>8} {'N':>5}"
    )
    print("-" * 140)
    for (method, scenario), data in sorted(stats.items()):
        n = len(data["success"])
        rate = sum(data["success"]) / n if n else 0.0
        ttc = sum(data["ttc"]) / len(data["ttc"]) if data["ttc"] else float("nan")
        d_ang = sum(data["d_ang"]) / len(data["d_ang"]) if data["d_ang"] else float("nan")
        c_cov = sum(data["c_cov"]) / len(data["c_cov"]) if data["c_cov"] else float("nan")
        g_max = sum(data["g_max"]) / len(data["g_max"]) if data["g_max"] else float("nan")
        c_sync = sum(data["c_sync"]) / len(data["c_sync"]) if data["c_sync"] else float("nan")
        pre_d = sum(data["pre_d"]) / len(data["pre_d"]) if data["pre_d"] else float("nan")
        pre_c = sum(data["pre_c"]) / len(data["pre_c"]) if data["pre_c"] else float("nan")
        pre_g = sum(data["pre_g"]) / len(data["pre_g"]) if data["pre_g"] else float("nan")
        pre_sync = sum(data["pre_sync"]) / len(data["pre_sync"]) if data["pre_sync"] else float("nan")
        ttc_s = f"{ttc:7.1f}" if data["ttc"] else "    n/a"
        def _fmt(v: float, w: int, prec: int) -> str:
            return f"{v:{w}.{prec}f}" if not (v != v) else " " * (w - 3) + "n/a"

        print(
            f"{method:<22} {scenario:<18} {rate:>7.1%} {ttc_s} "
            f"{d_ang:>8.3f} {c_cov:>8.3f} {g_max:>8.1f} {c_sync:>8.3f} "
            f"{_fmt(pre_d, 8, 3)} {_fmt(pre_c, 8, 3)} {_fmt(pre_g, 8, 1)} {_fmt(pre_sync, 8, 3)} {n:>5}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/aggregated.csv")
    parser.add_argument("--section", type=str, default="comparison", help="Filter by experiment_section")
    parser.add_argument("--all-sections", action="store_true", help="Include all sections")
    args = parser.parse_args()
    section = None if args.all_sections else args.section
    compare(Path(args.csv), section=section)


if __name__ == "__main__":
    main()
