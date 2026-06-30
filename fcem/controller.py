"""Main FCEM high-level controller."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from common.dynamics import clip_norm, norm
from fcem.assignment import select_best_candidate
from fcem.contraction import contraction_gate, phase_label, update_radius
from fcem.manifold import generate_candidate_manifolds
from metrics.structure import contraction_allowed, structural_metrics_from_positions


@dataclass
class FCEMConfig:
    R_init: float = 6.0
    R_terminal: float = 1.2
    capture_radius: float = 1.8
    contraction_rate: float = 0.095
    expansion_rate: float = 0.015
    min_R_for_closure: float = 2.20
    D_min: float = 0.35
    C_min: float = 0.24
    C_expand_min: float = 0.35
    G_max_allowed: float = 2.443460952792025  # 140 deg
    G_contract_threshold_deg: float = 140.0
    slot_error_frac: float = 0.78
    slot_error_abs: float = 1.10
    lookahead_time: float = 0.80
    center_shift_frac: float = 0.28
    slot_v_ff_gain: float = 0.85
    n_candidates: int = 3
    assignment_inertia: float = 0.12
    # Ablation flags
    enable_escape_lock: bool = True
    enable_obstacle_deform: bool = True
    enable_guarded_contraction: bool = True
    enable_prediction_shift: bool = True
    enable_slot_feedforward: bool = True
    enable_executability_rollout: bool = True
    enable_assignment_inertia: bool = True
    fixed_shrink: bool = False


@dataclass
class FCEMState:
    R: float
    prev_slots: np.ndarray | None = None
    prev_assignment: tuple[int, ...] | None = None
    phase_offset: float = 0.0


@dataclass
class FCEMOutput:
    center: np.ndarray
    slots: np.ndarray
    slot_angles: np.ndarray
    curve: np.ndarray
    assignment: tuple[int, ...]
    slot_vel: np.ndarray
    R: float
    q: float
    q_parts: dict[str, float]
    metrics: dict[str, float]
    slot_error: float
    phase: str
    candidate_score: float = 0.0


class FCEMController:
    def __init__(self, config: FCEMConfig, world_bounds: tuple[float, float, float, float]):
        self.config = config
        self.xmin, self.xmax, self.ymin, self.ymax = world_bounds
        self.state = FCEMState(R=config.R_init)

    def reset(self) -> None:
        self.state = FCEMState(R=self.config.R_init)

    def compute_center(
        self,
        target: np.ndarray,
        target_v: np.ndarray,
        pursuer_centroid: np.ndarray,
    ) -> np.ndarray:
        R = self.state.R
        if self.config.enable_prediction_shift:
            center_shift = (
                self.config.lookahead_time * target_v
                + 0.10 * (target - pursuer_centroid)
            )
            center_shift = clip_norm(center_shift, self.config.center_shift_frac * R)
            center = target + center_shift
        else:
            center = target.copy()
        center[0] = np.clip(center[0], self.xmin + R * 0.15, self.xmax - R * 0.15)
        center[1] = np.clip(center[1], self.ymin + R * 0.15, self.ymax - R * 0.15)
        return center

    def step(
        self,
        target: np.ndarray,
        target_v: np.ndarray,
        pursuers: np.ndarray,
        pursuer_v: np.ndarray,
        pursuer_centroid: np.ndarray,
        escape_dir: np.ndarray,
        obstacles: list,
        dt: float,
        kp: float,
        kd: float,
        vmax: float,
        amax: float,
        captured: bool = False,
    ) -> FCEMOutput:
        cfg = self.config
        R = self.state.R
        center = self.compute_center(target, target_v, pursuer_centroid)

        candidates = generate_candidate_manifolds(
            center=center,
            target=target,
            obstacles=obstacles,
            R=R,
            escape_dir=escape_dir,
            n_slots=len(pursuers),
            xmin=self.xmin,
            xmax=self.xmax,
            ymin=self.ymin,
            ymax=self.ymax,
            n_candidates=cfg.n_candidates,
            enable_obstacle_deform=cfg.enable_obstacle_deform,
            enable_escape_lock=cfg.enable_escape_lock,
        )

        phase_off, slot_angles, slots, curve, assignment, cand_score = select_best_candidate(
            candidates,
            pursuers,
            pursuer_v,
            target,
            self.state.prev_assignment,
            dt,
            kp,
            kd,
            vmax,
            amax,
            cfg.assignment_inertia,
            cfg.enable_executability_rollout,
            cfg.enable_assignment_inertia,
        )
        self.state.phase_offset = phase_off

        if self.state.prev_slots is None or not cfg.enable_slot_feedforward:
            slot_vel = np.zeros_like(slots)
        else:
            slot_vel = (slots - self.state.prev_slots) / dt

        metrics = structural_metrics_from_positions(target, pursuers)
        slot_error = float(np.mean([
            norm(pursuers[i] - slots[assignment[i]]) for i in range(len(pursuers))
        ]))

        q, q_parts = contraction_gate(
            metrics,
            slot_error,
            R,
            cfg.D_min,
            cfg.C_min,
            cfg.G_max_allowed,
            cfg.slot_error_frac,
            cfg.slot_error_abs,
        )

        fcem_cfg = {
            "C_expand_min": cfg.C_expand_min,
            "min_R_for_closure": cfg.min_R_for_closure,
        }
        config = {
            "G_contract_threshold_deg": cfg.G_contract_threshold_deg,
            "G_max_allowed_deg": math.degrees(cfg.G_max_allowed),
        }
        can_contract, gate_parts, _ = contraction_allowed(
            metrics,
            fcem_cfg,
            config,
            R,
            formation_expanded_latched=False,
        )
        q_parts = {**q_parts, **gate_parts}

        new_R = update_radius(
            R,
            q,
            cfg.R_init,
            cfg.R_terminal,
            cfg.contraction_rate,
            cfg.expansion_rate,
            cfg.enable_guarded_contraction,
            cfg.fixed_shrink,
            can_contract=can_contract,
        )
        self.state.R = new_R

        phase = phase_label(
            new_R, cfg.R_init, cfg.R_terminal, cfg.min_R_for_closure, captured
        )

        self.state.prev_slots = slots.copy()
        self.state.prev_assignment = assignment

        return FCEMOutput(
            center=center,
            slots=slots,
            slot_angles=slot_angles,
            curve=curve,
            assignment=assignment,
            slot_vel=slot_vel,
            R=new_R,
            q=q,
            q_parts=q_parts,
            metrics=metrics,
            slot_error=slot_error,
            phase=phase,
            candidate_score=cand_score,
        )

    @property
    def R(self) -> float:
        return self.state.R
