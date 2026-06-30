"""Summarize aggregated trial CSV into per-section summary statistics."""

from __future__ import annotations

import argparse
import csv
import math
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


def _mean_std(values: list[float]) -> tuple[float | str, float | str]:
    if not values:
        return "", ""
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def _median(values: list[float]) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def summarize_rows(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k, "") for k in group_keys)
        groups[key].append(row)

    out = []
    for key, items in sorted(groups.items()):
        n = len(items)
        successes = [_to_bool(r.get("success", r.get("captured"))) for r in items]
        success_rate = sum(successes) / n if n else 0.0

        ttc = [
            _to_float(r.get("time_to_capture_s"))
            for r in items
            if _to_bool(r.get("success", r.get("captured")))
        ]
        ttc = [v for v in ttc if v is not None]
        mean_ttc, std_ttc = _mean_std(ttc)

        ttc_adj = [_to_float(r.get("time_to_capture_adj_s")) for r in items]
        ttc_adj = [v for v in ttc_adj if v is not None]
        mean_ttc_adj, std_ttc_adj = _mean_std(ttc_adj)
        median_ttc_adj = _median(ttc_adj)

        d_ang = [_to_float(r.get("mean_D_ang")) for r in items]
        d_ang = [v for v in d_ang if v is not None]
        mean_d, std_d = _mean_std(d_ang)

        c_cov = [_to_float(r.get("mean_C_cov")) for r in items]
        c_cov = [v for v in c_cov if v is not None]
        mean_c, std_c = _mean_std(c_cov)

        g_max = [_to_float(r.get("mean_G_max_deg")) for r in items]
        g_max = [v for v in g_max if v is not None]
        mean_g, std_g = _mean_std(g_max)

        c_sync = [_to_float(r.get("mean_C_sync")) for r in items]
        c_sync = [v for v in c_sync if v is not None]
        mean_sync, std_sync = _mean_std(c_sync)

        def _pre_cap_col(key: str) -> tuple[float | str, float | str]:
            vals = [
                _to_float(r.get(key))
                for r in items
                if _to_bool(r.get("success", r.get("captured")))
            ]
            vals = [v for v in vals if v is not None]
            return _mean_std(vals)

        def _mean_std_col(key: str) -> tuple[float | str, float | str]:
            vals = [_to_float(r.get(key)) for r in items]
            vals = [v for v in vals if v is not None]
            return _mean_std(vals)

        pre_can_d, std_pre_can_d = _pre_cap_col("pre_capture_canonical_D_ang")
        pre_can_c, std_pre_can_c = _pre_cap_col("pre_capture_canonical_C_cov")
        pre_can_g, std_pre_can_g = _pre_cap_col("pre_capture_canonical_G_max_deg")
        pre_can_sync, std_pre_can_sync = _pre_cap_col("pre_capture_canonical_C_sync")

        pre_d, std_pre_d = _pre_cap_col("pre_capture_D_ang")
        pre_c, std_pre_c = _pre_cap_col("pre_capture_C_cov")
        pre_g, std_pre_g = _pre_cap_col("pre_capture_G_max_deg")
        pre_sync, std_pre_sync = _pre_cap_col("pre_capture_C_sync")

        pre_d_free, std_pre_d_free = _pre_cap_col("pre_capture_D_free")
        pre_c_free, std_pre_c_free = _pre_cap_col("pre_capture_C_free")
        pre_g_free, std_pre_g_free = _pre_cap_col("pre_capture_G_free_deg")

        cap_d, std_cap_d = _pre_cap_col("capture_D_ang")
        cap_c, std_cap_c = _pre_cap_col("capture_C_cov")
        cap_g, std_cap_g = _pre_cap_col("capture_G_max_deg")

        pre5_d, std_pre5_d = _pre_cap_col("pre_capture_5_D_ang")
        pre5_c, std_pre5_c = _pre_cap_col("pre_capture_5_C_cov")
        pre5_g, std_pre5_g = _pre_cap_col("pre_capture_5_G_max_deg")

        pre10_d, std_pre10_d = _pre_cap_col("pre_capture_10_D_ang")
        pre10_c, std_pre10_c = _pre_cap_col("pre_capture_10_C_cov")
        pre10_g, std_pre10_g = _pre_cap_col("pre_capture_10_G_max_deg")

        can_d, std_can_d = _mean_std_col("mean_canonical_D_ang")
        can_c, std_can_c = _mean_std_col("mean_canonical_C_cov")
        can_g, std_can_g = _mean_std_col("mean_canonical_G_max_deg")

        open_d, std_open_d = _mean_std_col("mean_open_D_ang")
        open_c, std_open_c = _mean_std_col("mean_open_C_cov")
        open_g, std_open_g = _mean_std_col("mean_open_G_max_deg")

        row = {
            "n_trials": n,
            "success_rate": success_rate,
            "mean_time_to_capture_s": mean_ttc,
            "std_time_to_capture_s": std_ttc,
            "mean_time_to_capture_adj_s": mean_ttc_adj,
            "std_time_to_capture_adj_s": std_ttc_adj,
            "median_time_to_capture_adj_s": median_ttc_adj,
            "mean_D_ang": mean_d,
            "std_D_ang": std_d,
            "mean_C_cov": mean_c,
            "std_C_cov": std_c,
            "mean_G_max_deg": mean_g,
            "std_G_max_deg": std_g,
            "mean_C_sync": mean_sync,
            "std_C_sync": std_sync,
            "mean_canonical_D_ang": can_d,
            "std_mean_canonical_D_ang": std_can_d,
            "mean_canonical_C_cov": can_c,
            "std_mean_canonical_C_cov": std_can_c,
            "mean_canonical_G_max_deg": can_g,
            "std_mean_canonical_G_max_deg": std_can_g,
            "mean_open_D_ang": open_d,
            "std_mean_open_D_ang": std_open_d,
            "mean_open_C_cov": open_c,
            "std_mean_open_C_cov": std_open_c,
            "mean_open_G_max_deg": open_g,
            "std_mean_open_G_max_deg": std_open_g,
            "pre_capture_canonical_D_ang": pre_can_d,
            "std_pre_capture_canonical_D_ang": std_pre_can_d,
            "pre_capture_canonical_C_cov": pre_can_c,
            "std_pre_capture_canonical_C_cov": std_pre_can_c,
            "pre_capture_canonical_G_max_deg": pre_can_g,
            "std_pre_capture_canonical_G_max_deg": std_pre_can_g,
            "pre_capture_canonical_C_sync": pre_can_sync,
            "std_pre_capture_canonical_C_sync": std_pre_can_sync,
            "pre_capture_D_ang": pre_d,
            "std_pre_capture_D_ang": std_pre_d,
            "pre_capture_C_cov": pre_c,
            "std_pre_capture_C_cov": std_pre_c,
            "pre_capture_G_max_deg": pre_g,
            "std_pre_capture_G_max_deg": std_pre_g,
            "pre_capture_C_sync": pre_sync,
            "std_pre_capture_C_sync": std_pre_sync,
            "pre_capture_D_free": pre_d_free,
            "std_pre_capture_D_free": std_pre_d_free,
            "pre_capture_C_free": pre_c_free,
            "std_pre_capture_C_free": std_pre_c_free,
            "pre_capture_G_free_deg": pre_g_free,
            "std_pre_capture_G_free_deg": std_pre_g_free,
            "capture_D_ang": cap_d,
            "std_capture_D_ang": std_cap_d,
            "capture_C_cov": cap_c,
            "std_capture_C_cov": std_cap_c,
            "capture_G_max_deg": cap_g,
            "std_capture_G_max_deg": std_cap_g,
            "pre_capture_5_D_ang": pre5_d,
            "std_pre_capture_5_D_ang": std_pre5_d,
            "pre_capture_5_C_cov": pre5_c,
            "std_pre_capture_5_C_cov": std_pre5_c,
            "pre_capture_5_G_max_deg": pre5_g,
            "std_pre_capture_5_G_max_deg": std_pre5_g,
            "pre_capture_10_D_ang": pre10_d,
            "std_pre_capture_10_D_ang": std_pre10_d,
            "pre_capture_10_C_cov": pre10_c,
            "std_pre_capture_10_C_cov": std_pre10_c,
            "pre_capture_10_G_max_deg": pre10_g,
            "std_pre_capture_10_G_max_deg": std_pre10_g,
        }
        for i, k in enumerate(group_keys):
            row[k] = key[i]
        out.append(row)
    return out


def summarize_csv(
    csv_path: Path,
    out_dir: Path,
    group_keys: tuple[str, ...] = ("experiment_section", "method", "scenario"),
) -> Path:
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"No rows in {csv_path}")
        return out_dir / "empty.csv"

    section_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        section = row.get("experiment_section") or "all"
        section_rows[section].append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for section, section_items in section_rows.items():
        summary = summarize_rows(section_items, group_keys)
        out_path = out_dir / f"{section}.csv"
        fieldnames = list(summary[0].keys()) if summary else []
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary)
        written.append(out_path)
        print(f"Wrote {len(summary)} summary rows to {out_path}")

    all_summary = summarize_rows(rows, group_keys)
    all_path = out_dir / "all.csv"
    if all_summary:
        with all_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_summary[0].keys()))
            writer.writeheader()
            writer.writerows(all_summary)
        print(f"Wrote {len(all_summary)} summary rows to {all_path}")
    return all_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize aggregated experiment CSV")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None, help="Experiment run directory")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        csv_path = Path(args.csv) if args.csv else run_dir / "aggregated_comparison.csv"
        out_dir = Path(args.out_dir) if args.out_dir else run_dir / "summary"
    else:
        csv_path = Path(args.csv or "results/aggregated_comparison.csv")
        out_dir = Path(args.out_dir or "results/summary")

    summarize_csv(csv_path, out_dir)


if __name__ == "__main__":
    main()
