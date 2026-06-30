"""Timestamped experiment run directories and manifests."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "run_manifest.json"
_RUN_DIR_PATTERN = re.compile(r"^\d{8}_\d{6}_")


def sanitize_experiment_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").strip()


def aggregate_section_name(exp_cfg: dict[str, Any]) -> str:
    """Section label used in aggregated CSV (matches plot script filters)."""
    name = str(exp_cfg.get("name") or exp_cfg.get("section") or "experiment")
    if name.startswith("ablation"):
        return "ablation"
    return name


def make_run_dir(
    results_root: Path,
    experiment_name: str,
    *,
    run_dir: Path | None = None,
    timestamped: bool = True,
    legacy_subdir: str | None = None,
) -> Path:
    if run_dir is not None:
        path = Path(run_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    if not timestamped and legacy_subdir:
        path = results_root / legacy_subdir
    elif timestamped:
        safe_name = sanitize_experiment_name(experiment_name)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = results_root / f"{stamp}_{safe_name}"
    else:
        path = results_root / sanitize_experiment_name(experiment_name)

    path.mkdir(parents=True, exist_ok=True)
    return path


def exp_cfg_output_subdir_fallback(experiment_name: str) -> str:
    """Legacy flat layout: results/<output_subdir>/ (name may use underscores)."""
    return experiment_name.replace("_", "/")


def write_run_manifest(
    run_dir: Path,
    exp_cfg: dict[str, Any],
    *,
    config_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "experiment_name": exp_cfg.get("name", "experiment"),
        "experiment_section": aggregate_section_name(exp_cfg),
        "description": exp_cfg.get("description", ""),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": config_path,
    }
    if extra:
        manifest.update(extra)
    path = run_dir / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def read_run_manifest(run_dir: Path) -> dict[str, Any] | None:
    path = Path(run_dir) / MANIFEST_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_run_root(json_path: Path, results_root: Path) -> Path | None:
    """Locate the run directory that owns a trial JSON file."""
    results_root = results_root.resolve()
    p = json_path.parent.resolve()
    while True:
        if (p / MANIFEST_FILENAME).exists():
            return p
        if p == results_root or p.parent == p:
            return None
        p = p.parent


def experiment_section_for_json(json_path: Path, results_root: Path, rel_parts: tuple[str, ...]) -> str:
    run_root = find_run_root(json_path, results_root)
    if run_root is not None:
        manifest = read_run_manifest(run_root)
        if manifest and manifest.get("experiment_section"):
            return str(manifest["experiment_section"])
    return rel_parts[0] if rel_parts else ""


def list_run_dirs(results_root: Path, experiment_name: str | None = None) -> list[Path]:
    if not results_root.is_dir():
        return []
    suffix = f"_{sanitize_experiment_name(experiment_name)}" if experiment_name else None
    runs: list[Path] = []
    for p in results_root.iterdir():
        if not p.is_dir() or not _RUN_DIR_PATTERN.match(p.name):
            continue
        if suffix and not p.name.endswith(suffix):
            continue
        runs.append(p)
    return sorted(runs, key=lambda x: x.name)


def latest_run_dir(
    results_root: Path = Path("results"),
    experiment_name: str | None = None,
) -> Path | None:
    runs = list_run_dirs(results_root, experiment_name)
    return runs[-1] if runs else None
