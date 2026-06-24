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

        pre_d, std_pre_d = _pre_cap_col("pre_capture_D_ang")
        pre_c, std_pre_c = _pre_cap_col("pre_capture_C_cov")
        pre_g, std_pre_g = _pre_cap_col("pre_capture_G_max_deg")
        pre_sync, std_pre_sync = _pre_cap_col("pre_capture_C_sync")

        row = {
            "n_trials": n,
            "success_rate": success_rate,
            "mean_time_to_capture_s": mean_ttc,
            "std_time_to_capture_s": std_ttc,
            "mean_D_ang": mean_d,
            "std_D_ang": std_d,
            "mean_C_cov": mean_c,
            "std_C_cov": std_c,
            "mean_G_max_deg": mean_g,
            "std_G_max_deg": std_g,
            "mean_C_sync": mean_sync,
            "std_C_sync": std_sync,
            "pre_capture_D_ang": pre_d,
            "std_pre_capture_D_ang": std_pre_d,
            "pre_capture_C_cov": pre_c,
            "std_pre_capture_C_cov": std_pre_c,
            "pre_capture_G_max_deg": pre_g,
            "std_pre_capture_G_max_deg": std_pre_g,
            "pre_capture_C_sync": pre_sync,
            "std_pre_capture_C_sync": std_pre_sync,
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
    parser.add_argument("--csv", type=str, default="results/aggregated.csv")
    parser.add_argument("--out-dir", type=str, default="results/summary")
    args = parser.parse_args()
    summarize_csv(Path(args.csv), Path(args.out_dir))


if __name__ == "__main__":
    main()
