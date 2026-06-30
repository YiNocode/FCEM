"""Boundary-Assisted Trap Mode: detection, free-cone metrics, arc slots, recovery."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from common.dynamics import norm, wrap_angle
from common.obstacles import Obstacle, point_segment_distance

TrapMode = Literal["open_space", "boundary", "corner"]
CornerId = Literal[
    "top_left", "top_right", "bottom_left", "bottom_right", "none"
]
RecoveryMode = Literal["normal", "structure_recovery"]


@dataclass
class TrapState:
    mode: TrapMode
    corner: CornerId
    theta_min: float
    theta_max: float
    phi_free: float
    d_left: float
    d_right: float
    d_bottom: float
    d_top: float
    near_boundary: bool
    near_corner: bool


def wall_distances(
    target: np.ndarray, bounds: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    xmin, xmax, ymin, ymax = bounds
    d_left = float(target[0] - xmin)
    d_right = float(xmax - target[0])
    d_bottom = float(target[1] - ymin)
    d_top = float(ymax - target[1])
    return d_left, d_right, d_bottom, d_top


def detect_trap_mode(
    target: np.ndarray,
    bounds: tuple[float, float, float, float],
    boundary_trap_threshold: float = 2.5,
    corner_trap_threshold: float = 3.0,
) -> TrapState:
    d_left, d_right, d_bottom, d_top = wall_distances(target, bounds)
    dists = [d_left, d_right, d_bottom, d_top]
    sorted_d = sorted(dists)
    near_boundary = sorted_d[0] < boundary_trap_threshold
    near_corner = sorted_d[1] < corner_trap_threshold

    corner: CornerId = "none"
    if near_corner:
        if d_left < corner_trap_threshold and d_top < corner_trap_threshold:
            corner = "top_left"
        elif d_right < corner_trap_threshold and d_top < corner_trap_threshold:
            corner = "top_right"
        elif d_left < corner_trap_threshold and d_bottom < corner_trap_threshold:
            corner = "bottom_left"
        elif d_right < corner_trap_threshold and d_bottom < corner_trap_threshold:
            corner = "bottom_right"

    if near_corner and corner != "none":
        mode: TrapMode = "corner"
    elif near_boundary:
        mode = "boundary"
    else:
        mode = "open_space"

    theta_min, theta_max = free_cone_angles(corner, mode, d_left, d_right, d_bottom, d_top)
    phi_free = theta_max - theta_min
    if phi_free <= 0:
        phi_free = 2.0 * math.pi
        theta_min, theta_max = 0.0, 2.0 * math.pi

    return TrapState(
        mode=mode,
        corner=corner,
        theta_min=theta_min,
        theta_max=theta_max,
        phi_free=phi_free,
        d_left=d_left,
        d_right=d_right,
        d_bottom=d_bottom,
        d_top=d_top,
        near_boundary=near_boundary,
        near_corner=near_corner,
    )


def free_cone_angles(
    corner: CornerId,
    mode: TrapMode,
    d_left: float,
    d_right: float,
    d_bottom: float,
    d_top: float,
) -> tuple[float, float]:
    """Return (theta_min, theta_max) in [0, 2pi) spanning the free escape arc."""
    if mode == "open_space":
        return 0.0, 2.0 * math.pi

    if mode == "corner":
        table: dict[CornerId, tuple[float, float]] = {
            "top_right": (math.pi, 1.5 * math.pi),
            "top_left": (1.5 * math.pi, 2.0 * math.pi),  # wraps; handled below
            "bottom_right": (0.5 * math.pi, math.pi),
            "bottom_left": (0.0, 0.5 * math.pi),
            "none": (0.0, 2.0 * math.pi),
        }
        t0, t1 = table.get(corner, (0.0, 2.0 * math.pi))
        if corner == "top_left":
            # free: down + right  -> [-pi/2, pi/2] around east
            return -0.5 * math.pi, 0.5 * math.pi
        return t0, t1

    # boundary mode: single wall closest
    wall_idx = int(np.argmin([d_left, d_right, d_bottom, d_top]))
    if wall_idx == 0:  # left wall -> free right hemisphere
        return -0.5 * math.pi, 0.5 * math.pi
    if wall_idx == 1:  # right wall
        return 0.5 * math.pi, 1.5 * math.pi
    if wall_idx == 2:  # bottom wall
        return 0.0, math.pi
    # top wall
    return math.pi, 2.0 * math.pi


def angle_in_free_cone(angle: float, theta_min: float, theta_max: float) -> bool:
    a = wrap_angle(angle)
    if theta_min <= theta_max:
        return theta_min - 1e-6 <= a <= theta_max + 1e-6
    # wrapped interval (e.g. top_left)
    return a >= theta_min - 1e-6 or a <= theta_max + 1e-6


def structural_metrics_free_cone(
    target: np.ndarray,
    pursuers: np.ndarray,
    trap: TrapState,
) -> dict[str, float]:
    """Trap-aware free-cone metrics plus full-circle canonical encirclement metrics."""
    from metrics.structure import structural_metrics_from_positions

    canonical = structural_metrics_from_positions(target, pursuers)

    if trap.mode == "open_space":
        return {
            **canonical,
            "D_free": canonical["D_ang"],
            "C_free": canonical["C_cov"],
            "G_free": canonical["G_max"],
            "phi_free": 2.0 * math.pi,
            "mode": "open_space",
        }

    rel = pursuers - target
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    radii = np.linalg.norm(rel, axis=1)

    in_cone = np.array(
        [angle_in_free_cone(a, trap.theta_min, trap.theta_max) for a in angles]
    )
    cone_angles = angles[in_cone]
    if len(cone_angles) == 0:
        return {
            **canonical,
            "D_free": 0.0,
            "C_free": 0.0,
            "G_free": trap.phi_free,
            "phi_free": trap.phi_free,
            "mode": trap.mode,
        }

    # Virtual walls at cone boundaries.
    sorted_angles = np.sort(cone_angles)
    t0, t1 = trap.theta_min, trap.theta_max
    if t0 <= t1:
        boundary_gaps = [sorted_angles[0] - t0, t1 - sorted_angles[-1]]
    else:
        # wrapped cone
        boundary_gaps = [
            wrap_angle(sorted_angles[0] - t0) if sorted_angles[0] >= t0 else sorted_angles[0] + 2 * math.pi - t0,
            wrap_angle(t1 - sorted_angles[-1]) if sorted_angles[-1] <= t1 else t1 + 2 * math.pi - sorted_angles[-1],
        ]
    internal_gaps = np.diff(sorted_angles) if len(sorted_angles) > 1 else np.array([])
    all_gaps = np.r_[boundary_gaps, internal_gaps]
    all_gaps = np.clip(all_gaps, 0.0, trap.phi_free)

    n_active = max(len(cone_angles), 1)
    ideal = trap.phi_free / n_active
    D_free = float(np.clip(1.0 - np.mean(np.abs(all_gaps - ideal)) / (ideal + 1e-9), 0.0, 1.0))
    G_free = float(np.max(all_gaps)) if len(all_gaps) else trap.phi_free
    C_free = float(np.clip(1.0 - G_free / (trap.phi_free + 1e-9), 0.0, 1.0))

    return {
        **canonical,
        "D_free": D_free,
        "C_free": C_free,
        "G_free": G_free,
        "phi_free": trap.phi_free,
        "mode": trap.mode,
    }


def g_free_allowed(trap: TrapState, n_pursuers: int, g_max_open_deg: float) -> float:
    if trap.mode == "open_space":
        return math.radians(g_max_open_deg)
    # scale open-space threshold to free cone width
    return trap.phi_free * (g_max_open_deg / 360.0) * max(n_pursuers, 1)


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


def build_arc_slots(
    target: np.ndarray,
    center: np.ndarray,
    R: float,
    trap: TrapState,
    n_slots: int,
    obstacles: list[Obstacle],
    bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate slots along free escape arc (boundary/corner mode)."""
    t0, t1 = trap.theta_min, trap.theta_max
    if trap.mode == "open_space":
        slot_angles = np.linspace(0.0, 2.0 * math.pi, n_slots, endpoint=False)
    elif t0 <= t1:
        slot_angles = np.linspace(t0, t1, n_slots + 2)[1:-1]
    else:
        # wrapped arc: sample in two segments
        n1 = n_slots // 2 + 1
        n2 = n_slots - n1 + 2
        a1 = np.linspace(t0, 2.0 * math.pi, n1, endpoint=False)
        a2 = np.linspace(0.0, t1, n2)[1:-1]
        slot_angles = np.r_[a1, a2][:n_slots]

    slots = []
    for theta in slot_angles:
        r = _radius_at(theta, center, target, obstacles, R, bounds)
        slots.append(center + r * np.array([math.cos(theta), math.sin(theta)]))
    slots_arr = np.array(slots)

    dense_n = max(60, n_slots * 20)
    if trap.mode == "open_space":
        dense_angles = np.linspace(0.0, 2.0 * math.pi, dense_n, endpoint=False)
    elif t0 <= t1:
        dense_angles = np.linspace(t0, t1, dense_n)
    else:
        dense_angles = np.r_[
            np.linspace(t0, 2.0 * math.pi, dense_n // 2, endpoint=False),
            np.linspace(0.0, t1, dense_n // 2),
        ]
    curve = []
    for theta in dense_angles:
        r = _radius_at(theta, center, target, obstacles, R, bounds)
        curve.append(center + r * np.array([math.cos(theta), math.sin(theta)]))

    return slot_angles, slots_arr, np.array(curve)


def corner_clamp_slots(
    target: np.ndarray,
    R_trap: float,
    corner: CornerId,
) -> np.ndarray:
    """Terminal corner clamp: wall-assisted capture geometry."""
    s = 1.0 / math.sqrt(2.0)
    layouts: dict[CornerId, list[np.ndarray]] = {
        "top_right": [
            target + np.array([-R_trap, 0.0]),
            target + np.array([0.0, -R_trap]),
            target + np.array([-R_trap * s, -R_trap * s]),
        ],
        "top_left": [
            target + np.array([R_trap, 0.0]),
            target + np.array([0.0, -R_trap]),
            target + np.array([R_trap * s, -R_trap * s]),
        ],
        "bottom_right": [
            target + np.array([-R_trap, 0.0]),
            target + np.array([0.0, R_trap]),
            target + np.array([-R_trap * s, R_trap * s]),
        ],
        "bottom_left": [
            target + np.array([R_trap, 0.0]),
            target + np.array([0.0, R_trap]),
            target + np.array([R_trap * s, R_trap * s]),
        ],
        "none": [],
    }
    return np.array(layouts.get(corner, layouts["top_right"]))


def select_active_blockers(
    target: np.ndarray,
    pursuers: np.ndarray,
    slots: np.ndarray,
    assignment: tuple[int, ...],
    trap: TrapState,
    slot_error_threshold: float = 2.5,
) -> tuple[list[int], list[int]]:
    """Active blockers cover free cone; recoverers are far behind."""
    active: list[int] = []
    recover: list[int] = []
    for i, j in enumerate(assignment):
        rel = pursuers[i] - target
        angle = math.atan2(rel[1], rel[0])
        err = norm(pursuers[i] - slots[j])
        in_cone = angle_in_free_cone(angle, trap.theta_min, trap.theta_max)
        if trap.mode == "open_space":
            active.append(i)
        elif in_cone and err < slot_error_threshold * 2.5:
            active.append(i)
        else:
            recover.append(i)
    if not active and recover:
        # keep at least two closest in cone or nearest two overall
        dists = [norm(pursuers[i] - target) for i in range(len(pursuers))]
        order = np.argsort(dists)
        active = list(order[:2])
        recover = [i for i in range(len(pursuers)) if i not in active]
    return active, recover


def slot_error_active_only(
    pursuers: np.ndarray,
    slots: np.ndarray,
    assignment: tuple[int, ...],
    active_set: list[int],
) -> float:
    if not active_set:
        return float(np.mean([norm(pursuers[i] - slots[assignment[i]]) for i in range(len(pursuers))]))
    return float(np.mean([norm(pursuers[i] - slots[assignment[i]]) for i in active_set]))


def largest_escape_gap_direction(
    target: np.ndarray,
    pursuers: np.ndarray,
    trap: TrapState,
) -> np.ndarray:
    rel = pursuers - target
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    in_cone = [angle_in_free_cone(a, trap.theta_min, trap.theta_max) for a in angles]
    cone_angles = sorted([a for a, ok in zip(angles, in_cone) if ok])

    t0, t1 = trap.theta_min, trap.theta_max
    if not cone_angles:
        mid = 0.5 * (t0 + t1) if t0 <= t1 else wrap_angle(t0 + 0.5 * trap.phi_free)
        return np.array([math.cos(mid), math.sin(mid)])

    if t0 <= t1:
        edges = [t0] + cone_angles + [t1]
    else:
        edges = cone_angles + [t0, t1]
    gaps = [(wrap_angle(edges[i + 1] - edges[i]), 0.5 * (edges[i] + edges[i + 1])) for i in range(len(edges) - 1)]
    _, mid_angle = max(gaps, key=lambda x: x[0])
    return np.array([math.cos(mid_angle), math.sin(mid_angle)])


def apply_recovery_slots(
    target: np.ndarray,
    pursuers: np.ndarray,
    pursuer_vel: np.ndarray,
    slots: np.ndarray,
    assignment: tuple[int, ...],
    trap: TrapState,
    R: float,
    gap_dir: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Reassign gap-closing slot to best recoverer."""
    slots = slots.copy()
    _, recover = select_active_blockers(target, pursuers, slots, assignment, trap)
    if not recover:
        recover = [int(np.argmax([norm(pursuers[i] - target) for i in range(len(pursuers))]))]

    best_i = recover[0]
    best_score = -1e9
    for i in recover:
        align = float(np.dot(pursuer_vel[i], gap_dir)) if norm(pursuer_vel[i]) > 1e-6 else 0.0
        dist = -norm(pursuers[i] - (target + R * gap_dir))
        score = align + dist
        if score > best_score:
            best_score = score
            best_i = i

    j = assignment[best_i]
    slots[j] = target + R * gap_dir
    return slots, best_i
