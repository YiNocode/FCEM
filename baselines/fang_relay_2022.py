"""Fang et al. 2022 cooperative pursuit with relay switching (point-mass adaptation).

Reference: Fang et al., IEEE T-Cybernetics 2022 — multi-pursuer pursuit of a
faster free-moving evader with Apollonius-based relay switching.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from baselines.common.assignment import assign_slots
from baselines.common.geometry import (
    point_in_apollonius_region,
    predict_evader,
    ring_slots,
    shrink_radius,
    time_to_intercept,
)
from baselines.common.low_level import baseline_cfg, empty_timing, track_slots
from common.dynamics import clip_norm, unit
from common.obstacles import Obstacle
from fcem.low_level.hermite_planner import (
    enforce_position_clearance,
    plan_velocity_command,
    planner_kwargs_from_config,
)
from metrics.experiment_logger import TimingBlock


def make_fang_relay_2022_controller() -> Any:
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
        bcfg = baseline_cfg(config, "fang_relay_2022")
        hysteresis = float(bcfg.get("switch_hysteresis", 0.5))
        min_hold_steps = int(bcfg.get("min_hold_steps", 8))
        lookahead = float(bcfg.get("lookahead_time", 0.6))
        timing = empty_timing()
        n = len(pursuers)
        vmax = config["pursuer_vmax"]
        evader_speed = max(float(np.linalg.norm(evader_v)), 1e-3)
        speed_ratio = vmax / evader_speed
        pred_evader = predict_evader(evader, evader_v, lookahead, config["dt"])
        pkw = planner_kwargs_from_config(config)

        with TimingBlock() as tb:
            scores = []
            for i in range(n):
                t_int = time_to_intercept(pursuers[i], pred_evader, vmax)
                in_region = point_in_apollonius_region(
                    pred_evader, pursuers[i], evader, speed_ratio
                )
                scores.append(t_int if in_region else t_int + 1e6)
            best = int(np.argmin(scores))
            switched = False
            if state["steps_since_switch"] >= min_hold_steps:
                current = state["active_id"]
                if scores[best] + hysteresis < scores[current]:
                    state["active_id"] = best
                    switched = True
            if switched:
                state["steps_since_switch"] = 0
            else:
                state["steps_since_switch"] += 1
            active = state["active_id"]

            ring = ring_slots(evader, R, n)
            slots = ring.copy()
            slot_vel = np.zeros((n, 2))
            if prev_slots is not None:
                slot_vel = (ring - prev_slots) / config["dt"]
            assignment = assign_slots(pursuers, slots, method="nearest")
        timing["manifold_gen_ms"] = tb.ms
        timing["assignment_ms"] = tb.ms

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
