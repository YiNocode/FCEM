"""Escape-aware manifold generation and executability rollout."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from common.dynamics import clip_norm, norm, wrap_angle
from common.obstacles import Obstacle, point_segment_distance
from fcem.boundary_trap import (
    TrapState,
    build_arc_slots,
    corner_clamp_slots,
    detect_trap_mode,
    structural_metrics_free_cone,
)


@dataclass
class CandidateManifold:
    name: str
    center: np.ndarray
    R: float
    escape_dir: np.ndarray
    slot_angles: np.ndarray
    slots: np.ndarray
    curve: np.ndarray
    structure_score: float
    blocker_score: float


def _radius_at(
    theta: float,
    center: np.ndarray,
    target: np.ndarray,
    obstacles: list[Obstacle],
    R: float,
    bounds: tuple[float, float, float, float],
) -> float:
    xmin, xmax, ymin, ymax = bounds
    r = R
    u = np.array([math.cos(theta), math.sin(theta)])
    nominal = target + R * u

    for obs in obstacles:
        rel = obs.center - target
        obs_angle = math.atan2(rel[1], rel[0])
        angle_err = wrap_angle(theta - obs_angle)
        ray_dist = point_segment_distance(obs.center, target, nominal)
        penetration = max(0.0, obs.radius + 0.45 - ray_dist)
        bump = 1.55 * penetration * math.exp(-0.5 * (angle_err / 0.42) ** 2)
        r += bump

    p = center + r * u
    p[0] = np.clip(p[0], xmin + 0.25, xmax - 0.25)
    p[1] = np.clip(p[1], ymin + 0.25, ymax - 0.25)
    return norm(p - center)


def build_manifold(
    name: str,
    center: np.ndarray,
    target: np.ndarray,
    obstacles: list[Obstacle],
    R: float,
    escape_dir: np.ndarray,
    n_slots: int,
    bounds: tuple[float, float, float, float],
    phase_offset: float = 0.0,
    radius_scale: float = 1.0,
) -> CandidateManifold:
    R_eff = R * radius_scale
    theta_esc = math.atan2(escape_dir[1], escape_dir[0]) + phase_offset
    slot_angles = theta_esc + np.linspace(0.0, 2.0 * np.pi, n_slots, endpoint=False)

    dense_angles = theta_esc + np.linspace(0.0, 2.0 * np.pi, 181, endpoint=False)
    dense_points = []
    for theta in dense_angles:
        r = _radius_at(theta, center, target, obstacles, R_eff, bounds)
        dense_points.append(center + r * np.array([math.cos(theta), math.sin(theta)]))

    slots = []
    for theta in slot_angles:
        r = _radius_at(theta, center, target, obstacles, R_eff, bounds)
        slots.append(center + r * np.array([math.cos(theta), math.sin(theta)]))

    slots_arr = np.array(slots)
    curve_arr = np.array(dense_points)

    # Structure score: uniformity of slot spacing around target.
    rel = slots_arr - target
    angles = np.sort(np.arctan2(rel[:, 1], rel[:, 0]) % (2.0 * np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])
    ideal = 2.0 * np.pi / n_slots
    structure_score = float(1.0 - np.mean(np.abs(gaps - ideal)) / ideal)

    # Blocker score: penalize slots inside obstacles.
    blocker_score = 0.0
    for s in slots_arr:
        for obs in obstacles:
            d = norm(s - obs.center) - obs.radius
            if d < 0.5:
                blocker_score -= (0.5 - d) * 2.0

    return CandidateManifold(
        name=name,
        center=center.copy(),
        R=R_eff,
        escape_dir=escape_dir.copy(),
        slot_angles=slot_angles,
        slots=slots_arr,
        curve=curve_arr,
        structure_score=structure_score,
        blocker_score=blocker_score,
    )


def build_trap_manifold(
    name: str,
    center: np.ndarray,
    target: np.ndarray,
    obstacles: list[Obstacle],
    R: float,
    escape_dir: np.ndarray,
    n_slots: int,
    bounds: tuple[float, float, float, float],
    trap: TrapState,
    use_corner_clamp: bool = False,
    R_terminal: float = 1.2,
) -> CandidateManifold:
    """Boundary/corner mode: arc manifold or corner-clamp terminal slots."""
    if use_corner_clamp and trap.mode == "corner" and trap.corner != "none":
        slots_arr = corner_clamp_slots(target, R_terminal, trap.corner)
        slot_angles = np.arctan2(slots_arr[:, 1] - target[1], slots_arr[:, 0] - target[0])
        curve_arr = slots_arr.copy()
    else:
        slot_angles, slots_arr, curve_arr = build_arc_slots(
            target, center, R, trap, n_slots, obstacles, bounds
        )

    rel = slots_arr - target
    angles = np.sort(np.arctan2(rel[:, 1], rel[:, 0]) % (2.0 * np.pi))
    if trap.mode == "open_space":
        gaps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])
        ideal = 2.0 * np.pi / n_slots
    else:
        gaps = np.diff(angles) if len(angles) > 1 else np.array([trap.phi_free])
        ideal = trap.phi_free / max(n_slots, 1)
    structure_score = float(1.0 - np.mean(np.abs(gaps - ideal)) / (ideal + 1e-9))

    blocker_score = 0.0
    for s in slots_arr:
        for obs in obstacles:
            d = norm(s - obs.center) - obs.radius
            if d < 0.5:
                blocker_score -= (0.5 - d) * 2.0

    return CandidateManifold(
        name=name,
        center=center.copy(),
        R=R,
        escape_dir=escape_dir.copy(),
        slot_angles=slot_angles,
        slots=slots_arr,
        curve=curve_arr,
        structure_score=structure_score,
        blocker_score=blocker_score,
    )


def generate_candidate_manifolds(
    center: np.ndarray,
    target: np.ndarray,
    obstacles: list[Obstacle],
    R: float,
    escape_dir: np.ndarray,
    n_slots: int,
    bounds: tuple[float, float, float, float],
    ablate_single_manifold: bool = False,
    trap: TrapState | None = None,
    use_corner_clamp: bool = False,
    R_terminal: float = 1.2,
    trap_cfg: dict | None = None,
) -> tuple[list[CandidateManifold], TrapState]:
    trap_cfg = trap_cfg or {}
    if trap is None:
        trap = detect_trap_mode(
            target,
            bounds,
            boundary_trap_threshold=trap_cfg.get("boundary_trap_threshold", 2.5),
            corner_trap_threshold=trap_cfg.get("corner_trap_threshold", 3.0),
        )

    if trap.mode in ("boundary", "corner"):
        names = ["arc_base"] if ablate_single_manifold else ["arc_base", "arc_tight", "arc_wide"]
        scales = [1.0, 0.92, 1.06]
        manifolds = [
            build_trap_manifold(
                name,
                center,
                target,
                obstacles,
                R * scale,
                escape_dir,
                n_slots,
                bounds,
                trap,
                use_corner_clamp=use_corner_clamp and name == "arc_base",
                R_terminal=R_terminal,
            )
            for name, scale in zip(names[: 1 if ablate_single_manifold else 3], scales)
        ]
        return manifolds, trap

    if ablate_single_manifold:
        variants = [("base", 0.0, 1.0)]
    else:
        variants = [
            ("phase_0", 0.0, 1.0),
            ("phase_pi3", math.pi / 3.0, 1.0),
            ("radius_up", 0.0, 1.08),
        ]
    manifolds = [
        build_manifold(
            name, center, target, obstacles, R, escape_dir, n_slots, bounds,
            phase_offset=phase, radius_scale=scale,
        )
        for name, phase, scale in variants
    ]
    return manifolds, trap


def evaluate_executability(
    pursuers: np.ndarray,
    pursuer_vel: np.ndarray,
    slots: np.ndarray,
    obstacles: list[Obstacle],
    bounds: tuple[float, float, float, float],
    assignment: tuple[int, ...],
    dt: float,
    horizon_steps: int,
    pursuer_vmax: float,
    pursuer_amax: float,
    kp: float,
    kd: float,
    obstacle_influence: float,
    pursuer_obs_gain: float,
    boundary_margin: float,
    boundary_gain: float,
) -> tuple[float, bool]:
    """Short-horizon rollout; return mean slot error and feasibility flag."""
    from fcem.low_level.hermite_planner import enforce_position_clearance, plan_velocity_command

    pos = pursuers.copy()
    vel = pursuer_vel.copy()
    total_err = 0.0
    planner_kw = {
        "clearance": 0.55,
        "horizon_time": 0.65,
        "body_radius": 0.25,
        "boundary_margin": boundary_margin,
        "boundary_gain": boundary_gain,
        "kp_fallback": kp,
        "kd_fallback": kd,
    }
    req_clear = planner_kw["clearance"] + planner_kw["body_radius"]

    for _ in range(horizon_steps):
        step_err = 0.0
        for i, j in enumerate(assignment):
            slot = slots[j]
            vel[i] = plan_velocity_command(
                pos[i],
                vel[i],
                slot,
                np.zeros(2),
                obstacles,
                bounds,
                dt,
                pursuer_vmax,
                pursuer_amax,
                **planner_kw,
            )
            pos[i] = pos[i] + vel[i] * dt
            pos[i] = enforce_position_clearance(
                pos[i], obstacles, planner_kw["body_radius"], planner_kw["clearance"]
            )
            step_err += norm(pos[i] - slot)
        total_err += step_err / len(pursuers)

        for obs in obstacles:
            for p in pos:
                if norm(p - obs.center) < obs.radius + req_clear * 0.5:
                    return total_err / max(horizon_steps, 1), False

    mean_err = total_err / max(horizon_steps, 1)
    feasible = mean_err < 3.5
    return mean_err, feasible
