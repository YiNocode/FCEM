"""Layer registry: map L1–L4 stacks and single-layer removal to ablation flags."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.config_loader import CONFIG_DIR, load_yaml

LAYERS_PATH = CONFIG_DIR / "experiments" / "layers.yaml"

ALL_LAYER_IDS = ("L1", "L2", "L3", "L4")

# Flags toggled when a layer is disabled in a progressive stack.
_LAYER_DISABLE_FLAGS: dict[str, dict[str, bool]] = {
    "L1": {
        "ablate_no_esc_dir": True,
        "ablate_no_center_shift": True,
    },
    "L2": {
        "ablate_single_manifold": True,
    },
    "L3": {
        "ablate_no_executability": True,
        "ablate_nearest_assign": True,
    },
    "L4": {
        "ablate_no_slot_vel_ff": True,
    },
}


def load_layers_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or LAYERS_PATH)


def layer_definitions(path: Path | None = None) -> list[dict[str, Any]]:
    return list(load_layers_config(path).get("layers", []))


def layer_order(path: Path | None = None) -> list[str]:
    cfg = load_layers_config(path)
    order = cfg.get("layer_order")
    if order:
        return list(order)
    return [layer["id"] for layer in layer_definitions(path)]


def _merge_flags(*flag_dicts: dict[str, bool]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for flags in flag_dicts:
        out.update(flags)
    return out


def baseline_fallback_flags() -> dict[str, bool]:
    """All layers disabled — minimal stack for progressive ablation base."""
    flags: dict[str, bool] = {}
    for layer_id in ALL_LAYER_IDS:
        flags.update(_LAYER_DISABLE_FLAGS[layer_id])
    return flags


def flags_for_remove_layer(layer_id: str, path: Path | None = None) -> dict[str, bool]:
    """Remove one layer from full FCEM; other layers stay enabled."""
    layer_id = layer_id.upper()
    for layer in layer_definitions(path):
        if layer["id"] == layer_id:
            return dict(layer.get("remove_flags", _LAYER_DISABLE_FLAGS.get(layer_id, {})))
    raise ValueError(f"Unknown layer: {layer_id}")


def layers_for_stack(enabled: list[str], path: Path | None = None) -> dict[str, bool]:
    """Build ablation flags for a progressive layer stack (enabled layers only)."""
    enabled_set = {layer.upper() for layer in enabled}
    order = layer_order(path)
    flags: dict[str, bool] = {}
    for layer_id in order:
        if layer_id not in enabled_set:
            flags.update(_LAYER_DISABLE_FLAGS[layer_id])
    return flags


def resolve_variant(
    *,
    remove_layer: str | None = None,
    enabled_layers: list[str] | None = None,
    path: Path | None = None,
) -> dict[str, bool]:
    """Resolve ablation flags from either remove-one-layer or progressive stack."""
    if remove_layer is not None and enabled_layers is not None:
        raise ValueError("Specify either remove_layer or enabled_layers, not both")
    if remove_layer is not None:
        return flags_for_remove_layer(remove_layer, path)
    if enabled_layers is not None:
        return layers_for_stack(enabled_layers, path)
    return {}


def layer_mapping_table(path: Path | None = None) -> list[dict[str, str]]:
    """Rows for Tab. X (Layer → Experiment mapping)."""
    rows = []
    for layer in layer_definitions(path):
        flags = layer.get("remove_flags", {})
        flag_str = ", ".join(f"{k}={v}" for k, v in flags.items())
        rows.append({
            "layer": layer["id"],
            "experiment": layer["experiment"],
            "name": layer["name"],
            "module": layer["module"],
            "remove_flags": flag_str,
            "description": layer.get("description", ""),
        })
    return rows
