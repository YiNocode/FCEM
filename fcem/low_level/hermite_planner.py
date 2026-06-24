"""MIGHTY-inspired 2D Hermite spline local planner for obstacle-aware tracking.

Plans a short cubic Hermite segment from (pos, vel) to a clearance-adjusted goal,
then returns the velocity at the next control step. Used by FCEM low-level tracking,
baselines, and the Mighty ROS bridge stub.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from common.dynamics import clip_norm, norm, unit
from common.obstacles import Obstacle, boundary_repulsion, point_segment_distance


def _closest_point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return a.copy()
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    return a + t * ab


def _hermite_position(
    s: float,
    p0: np.ndarray,
    v0: np.ndarray,
    p1: np.ndarray,
    v1: np.ndarray,
    duration: float,
) -> np.ndarray:
    s = float(np.clip(s, 0.0, 1.0))
    h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
    h10 = s**3 - 2.0 * s**2 + s
    h01 = -2.0 * s**3 + 3.0 * s**2
    h11 = s**3 - s**2
    return h00 * p0 + h10 * duration * v0 + h01 * p1 + h11 * duration * v1


def _hermite_velocity(
    s: float,
    p0: np.ndarray,
    v0: np.ndarray,
    p1: np.ndarray,
    v1: np.ndarray,
    duration: float,
) -> np.ndarray:
    s = float(np.clip(s, 0.0, 1.0))
    dh00 = 6.0 * s**2 - 6.0 * s
    dh10 = 3.0 * s**2 - 4.0 * s + 1.0
    dh01 = -6.0 * s**2 + 6.0 * s
    dh11 = 3.0 * s**2 - 2.0 * s
    return (dh00 * p0 + dh10 * duration * v0 + dh01 * p1 + dh11 * duration * v1) / max(
        duration, 1e-6
    )


def _min_path_clearance(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: list[Obstacle],
    n_samples: int,
) -> tuple[float, np.ndarray | None, Obstacle | None]:
    """Return minimum surface clearance along segment and the tightest sample."""
    if not obstacles:
        return float("inf"), None, None

    best_clear = float("inf")
    worst_pt: np.ndarray | None = None
    worst_obs: Obstacle | None = None
    for k in range(n_samples + 1):
        s = k / max(n_samples, 1)
        pt = start + s * (goal - start)
        for obs in obstacles:
            clear = norm(pt - obs.center) - obs.radius
            if clear < best_clear:
                best_clear = clear
                worst_pt = pt
                worst_obs = obs
    return best_clear, worst_pt, worst_obs


def _push_point_from_obstacle(
    point: np.ndarray,
    obs: Obstacle,
    clearance: float,
) -> np.ndarray:
    delta = point - obs.center
    d = norm(delta)
    if d < 1e-9:
        return point + np.array([clearance, 0.0])
    surface_gap = d - obs.radius
    if surface_gap >= clearance:
        return point
    push = (clearance - surface_gap) + 0.08
    return point + unit(delta) * push


def enforce_position_clearance(
    pos: np.ndarray,
    obstacles: list[Obstacle],
    body_radius: float,
    clearance: float = 0.55,
) -> np.ndarray:
    """Project a point outside all obstacle collision envelopes."""
    out = pos.copy()
    req = clearance + body_radius
    for _ in range(4):
        changed = False
        for obs in obstacles:
            pushed = _push_point_from_obstacle(out, obs, req)
            if norm(pushed - out) > 1e-9:
                changed = True
                out = pushed
        if not changed:
            break
    return out


def adjust_goal_for_obstacles(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: list[Obstacle],
    clearance: float,
    n_samples: int = 20,
    max_iters: int = 10,
) -> np.ndarray:
    """Shift the terminal goal so the straight segment stays outside obstacles."""
    adjusted = goal.copy()
    if not obstacles:
        return adjusted

    for _ in range(max_iters):
        min_clear, worst_pt, worst_obs = _min_path_clearance(
            start, adjusted, obstacles, n_samples
        )
        if min_clear >= clearance or worst_pt is None or worst_obs is None:
            break

        closest = _closest_point_on_segment(worst_obs.center, start, adjusted)
        seg_delta = closest - worst_obs.center
        if norm(seg_delta) < 1e-9:
            normal = unit(adjusted - worst_obs.center)
            if norm(normal) < 1e-9:
                normal = unit(adjusted - start)
        else:
            normal = unit(seg_delta)

        deficit = clearance - min_clear
        adjusted = adjusted + normal * (deficit + 0.12)

        # Keep the pursuer-side endpoint from cutting through the same disc.
        adjusted = _push_point_from_obstacle(adjusted, worst_obs, clearance)

    return adjusted


def _segment_time(start: np.ndarray, goal: np.ndarray, vmax: float, horizon_time: float) -> float:
    dist = norm(goal - start)
    if dist < 1e-6:
        return 0.25
    return float(np.clip(dist / max(vmax, 0.5), 0.25, horizon_time))


def plan_velocity_command(
    pos: np.ndarray,
    vel: np.ndarray,
    goal: np.ndarray,
    goal_vel: np.ndarray,
    obstacles: list[Obstacle],
    bounds: tuple[float, float, float, float],
    dt: float,
    vmax: float,
    amax: float,
    clearance: float = 0.55,
    horizon_time: float = 0.65,
    body_radius: float = 0.25,
    boundary_margin: float = 5.0,
    boundary_gain: float = 2.20,
    n_path_samples: int = 20,
    kp_fallback: float = 2.0,
    kd_fallback: float = 1.0,
) -> np.ndarray:
    """Plan one-step XY velocity toward goal with Hermite spline obstacle clearance."""
    req_clear = clearance + body_radius

    # Hard escape if already penetrating the collision envelope.
    escape = np.zeros(2)
    for obs in obstacles:
        delta = pos - obs.center
        d = norm(delta)
        if d < 1e-9:
            escape += np.array([1.0, 0.0])
            continue
        surface_gap = d - obs.radius
        if surface_gap < body_radius * 0.85:
            strength = max(2.0, (body_radius - surface_gap) * 8.0)
            escape += unit(delta) * strength
    if norm(escape) > 1e-6:
        return clip_norm(escape * vmax, vmax)

    safe_goal = adjust_goal_for_obstacles(
        pos, goal, obstacles, req_clear, n_samples=n_path_samples
    )

    to_goal = safe_goal - pos
    dist = norm(to_goal)
    duration = _segment_time(pos, safe_goal, vmax, horizon_time)

    if dist < 0.08:
        v_des = goal_vel.copy()
    else:
        if norm(goal_vel) > 0.15:
            v1 = goal_vel
        else:
            v1 = unit(to_goal) * min(vmax, dist / max(duration, dt))
        s_next = min(1.0, dt / max(duration, dt))
        v_des = _hermite_velocity(s_next, pos, vel, safe_goal, v1, duration)

        # If the short Hermite segment still clips an obstacle, nudge velocity outward.
        probe = _hermite_position(s_next, pos, vel, safe_goal, v1, duration)
        for obs in obstacles:
            probe_clear = norm(probe - obs.center) - obs.radius
            if probe_clear < req_clear:
                v_des += unit(probe - obs.center) * (req_clear - probe_clear) * 2.5

    # Near-field repulsion + boundary barrier (safety margin around planner path).
    for obs in obstacles:
        v_des += _obstacle_escape_accel(pos, obs, req_clear) * dt
    v_des += boundary_repulsion(pos, bounds, boundary_margin, boundary_gain) * dt

    # Acceleration-limited tracking toward desired velocity.
    acc = kp_fallback * (v_des - vel) + kd_fallback * (v_des - vel)
    acc = clip_norm(acc, amax)
    return clip_norm(vel + acc * dt, vmax)


def _obstacle_escape_accel(pos: np.ndarray, obs: Obstacle, clearance: float) -> np.ndarray:
    delta = pos - obs.center
    d = norm(delta)
    if d < 1e-9:
        return np.array([1.0, 0.0])
    surface_gap = d - obs.radius
    influence = clearance + 1.20
    if surface_gap >= influence:
        return np.zeros(2)
    strength = 3.5 * (1.0 / max(surface_gap, 0.05) - 1.0 / influence)
    return max(0.0, strength) * unit(delta)


def planner_kwargs_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    lp = cfg.get("local_planner", {})
    return {
        "clearance": float(lp.get("clearance", 0.55)),
        "horizon_time": float(lp.get("horizon_time", 0.65)),
        "body_radius": float(cfg.get("pursuer_collision_radius", 0.25)),
        "boundary_margin": float(cfg.get("boundary_margin", 5.0)),
        "boundary_gain": float(cfg.get("boundary_gain", 2.20)),
        "n_path_samples": int(lp.get("n_path_samples", 20)),
        "kp_fallback": float(cfg.get("pursuer_kp", 2.0)),
        "kd_fallback": float(cfg.get("pursuer_kd", 1.0)),
    }


def segment_blocked(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: list[Obstacle],
    clearance: float,
    n_samples: int = 16,
) -> bool:
    min_clear, _, _ = _min_path_clearance(start, goal, obstacles, n_samples)
    return min_clear < clearance
