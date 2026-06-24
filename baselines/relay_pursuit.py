"""Relay pursuit with Voronoi + Apollonius switching (point-mass adaptation).

Reference: survey relay pursuit / Ramana & Kothari style single-active pursuer
with advantage-region handoff. Strictly one active interceptor at a time.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from baselines.common.geometry import (
    point_in_apollonius_region,
    predict_evader,
    ring_slots,
    shrink_radius,
    time_to_intercept,
    voronoi_nearest_point,
)
from baselines.common.low_level import baseline_cfg, empty_timing
from common.dynamics import clip_norm, unit
from common.obstacles import Obstacle
from fcem.low_level.hermite_planner import (
    enforce_position_clearance,
    plan_velocity_command,
    planner_kwargs_from_config,
)
from metrics.experiment_logger import TimingBlock


def make_relay_pursuit_controller() -> Any:
    state: dict[str, Any] = {"active_id": 0, "steps_since_switch": 0}

    def controller(
        step: int,
        evader: np.ndarray,
        evader_v: np.ndarray,
        pursuers: np.ndarray,
        pursuer_v: np.ndarray,
        obstacles: list[Obstacle],
        bounds: tuple[float, float, float, float],
        R: float,
        prev_slots: np.ndarray | None,
        prev_assignment: tuple[int, ...] | None,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        fcem_cfg = config["fcem"]
        bcfg = baseline_cfg(config, "relay_pursuit")
        switch_margin = float(bcfg.get("switch_dist_margin", 1.0))
        min_hold_steps = int(bcfg.get("min_hold_steps", 10))
        hold_gain = float(bcfg.get("hold_gain", 0.15))
        timing = empty_timing()
        n = len(pursuers)
        vmax = config["pursuer_vmax"]
        evader_speed = max(float(np.linalg.norm(evader_v)), 1e-3)
        speed_ratio = vmax / evader_speed
        pred_evader = predict_evader(evader, evader_v, 0.5, config["dt"])
        pkw = planner_kwargs_from_config(config)

        with TimingBlock() as tb:
            active = state["active_id"]
            best = active
            best_score = time_to_intercept(pursuers[active], pred_evader, vmax)
            for i in range(n):
                if i == active:
                    continue
                vor_pt = voronoi_nearest_point(pursuers[i], evader, pursuers)
                dist_vor = float(np.linalg.norm(vor_pt - evader))
                in_voronoi = dist_vor < switch_margin
                in_apo = point_in_apollonius_region(
                    pred_evader, pursuers[i], evader, speed_ratio
                )
                t_int = time_to_intercept(pursuers[i], pred_evader, vmax)
                if in_voronoi and in_apo and t_int + switch_margin < best_score:
                    best = i
                    best_score = t_int
            if state["steps_since_switch"] >= min_hold_steps and best != active:
                state["active_id"] = best
                state["steps_since_switch"] = 0
            else:
                state["steps_since_switch"] += 1
            active = state["active_id"]

            ring = ring_slots(evader, R, n)
            slots = ring.copy()
            slot_vel = np.zeros((n, 2))
            if prev_slots is not None:
                slot_vel = (ring - prev_slots) / config["dt"]
            assignment = tuple(range(n))
        timing["manifold_gen_ms"] = tb.ms

        with TimingBlock() as tb:
            pursuers = pursuers.copy()
            pursuer_v = pursuer_v.copy()
            for i in range(n):
                if i == active:
                    goal_vel = vmax * unit(pred_evader - pursuers[i])
                    cmd = plan_velocity_command(
                        pursuers[i],
                        pursuer_v[i],
                        pred_evader,
                        goal_vel,
                        obstacles,
                        bounds,
                        config["dt"],
                        vmax,
                        config["pursuer_amax"],
                        **pkw,
                    )
                    pursuer_v[i] = clip_norm(cmd, vmax)
                    pursuers[i] = pursuers[i] + pursuer_v[i] * config["dt"]
                else:
                    j = assignment[i]
                    target = slots[j]
                    sv = slot_vel[j]
                    from fcem.low_level.pd_tracker import pd_track_step

                    p, v = pd_track_step(
                        pursuers[i : i + 1],
                        pursuer_v[i : i + 1],
                        target[None, :],
                        sv[None, :],
                        (0,),
                        obstacles,
                        bounds,
                        config["dt"],
                        config["pursuer_kp"],
                        config["pursuer_kd"],
                        fcem_cfg.get("slot_v_ff_gain", 0.85),
                        vmax,
                        config["pursuer_amax"],
                        config.get("obstacle_influence", 2.20),
                        config.get("pursuer_obs_gain", 1.25),
                        config.get("boundary_margin", 5.0),
                        config.get("boundary_gain", 2.20),
                        planner_kwargs=pkw,
                    )
                    pursuers[i] = p[0]
                    pursuer_v[i] = v[0]
                pursuers[i] = enforce_position_clearance(
                    pursuers[i],
                    obstacles,
                    float(pkw.get("body_radius", 0.25)),
                    float(pkw.get("clearance", 0.55)),
                )
                xmin, xmax, ymin, ymax = bounds
                pursuers[i, 0] = np.clip(pursuers[i, 0], xmin + 0.15, xmax - 0.15)
                pursuers[i, 1] = np.clip(pursuers[i, 1], ymin + 0.15, ymax - 0.15)
        timing["low_level_ms"] = tb.ms
        R = shrink_radius(R, fcem_cfg)
        timing["total_ms"] = sum(timing.values())

        return {
            "pursuers": pursuers,
            "pursuer_v": pursuer_v,
            "R": R,
            "slots": slots,
            "assignment": assignment,
            "timing_ms": timing,
            "active_pursuer": active,
        }

    return controller
