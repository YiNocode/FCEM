"""Startup and logic checks for the DG layer ablation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.config_loader import load_experiment_config
from experiments.layer_registry import resolve_variant
from fcem.slot_assignment import AssignmentWeights, assign_slots


def test_ablation_dg_config_scope_and_variants() -> None:
    cfg = load_experiment_config(Path("config/experiments/ablation_dg_50seed.yaml"))

    assert cfg["n_trials"] == 50
    assert cfg["scenarios"] == ["random_obstacles", "single_exit"]
    assert cfg["capture_mode"] == "escape_sector"
    assert cfg["dynamics_file"] is None
    assert cfg["dynamics"]["evader_policy"] == "differential_game"
    assert [v["name"] for v in cfg["variants"]] == [
        "full_fcem",
        "no_l1_prediction",
        "no_l2_multi_candidate_manifold",
        "no_l3_executability_assignment",
        "no_l4_slot_velocity_feedforward",
    ]


def test_ablation_dg_layer_flag_mapping() -> None:
    assert resolve_variant(remove_layer=None) == {}
    assert resolve_variant(remove_layer="L1") == {
        "ablate_no_esc_dir": True,
        "ablate_no_center_shift": True,
    }
    assert resolve_variant(remove_layer="L2") == {"ablate_single_manifold": True}
    assert resolve_variant(remove_layer="L3") == {
        "ablate_no_executability": True,
        "ablate_nearest_assign": True,
    }
    assert resolve_variant(remove_layer="L4") == {"ablate_no_slot_vel_ff": True}


def test_no_l3_assignment_uses_greedy_nearest_fallback() -> None:
    pursuers = np.array([[1.0, 0.0], [0.0, 1.1]], dtype=float)
    slots = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=float)
    weights = AssignmentWeights(
        w_reach=1.0,
        w_ang=0.0,
        w_cov=0.0,
        w_sector=0.0,
        w_sync=0.0,
        w_switch=0.0,
        w_safe=0.0,
    )

    optimal, _, _ = assign_slots(
        pursuers,
        slots,
        weights=weights,
        pursuer_v=np.zeros_like(pursuers),
        target=np.array([5.0, 0.0]),
    )
    nearest, _, _ = assign_slots(
        pursuers,
        slots,
        weights=weights,
        pursuer_v=np.zeros_like(pursuers),
        target=np.array([5.0, 0.0]),
        ablate_nearest_assign=True,
    )

    assert optimal == (1, 0)
    assert nearest == (0, 1)
