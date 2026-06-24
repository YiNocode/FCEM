"""Generate Layer → Experiment mapping table (Tab. X) for the paper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.layer_registry import layer_mapping_table


def to_markdown(rows: list[dict[str, str]]) -> str:
    lines = [
        "| Layer | Experiment | Module | Remove flags | Description |",
        "|-------|------------|--------|--------------|-------------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['layer']} | {row['experiment']} | `{row['module']}` | "
            f"`{row['remove_flags']}` | {row['description']} |"
        )
    return "\n".join(lines) + "\n"


def to_latex(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Layer & Experiment & Module & Remove flags \\",
        r"\midrule",
    ]
    for row in rows:
        flags = row["remove_flags"].replace("_", r"\_")
        module = row["module"].replace("_", r"\_")
        lines.append(f"{row['layer']} & {row['experiment']} & \\texttt{{{module}}} & \\texttt{{{flags}}} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate layer mapping table")
    parser.add_argument("--format", choices=("markdown", "latex", "both"), default="both")
    parser.add_argument("--out-dir", type=str, default="results/figures")
    args = parser.parse_args()

    rows = layer_mapping_table()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.format in ("markdown", "both"):
        md_path = out_dir / "tab_layer_mapping.md"
        md_path.write_text(to_markdown(rows), encoding="utf-8")
        print(f"Wrote {md_path}")

    if args.format in ("latex", "both"):
        tex_path = out_dir / "tab_layer_mapping.tex"
        tex_path.write_text(to_latex(rows), encoding="utf-8")
        print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
