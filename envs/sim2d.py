"""Unified 2D simulation engine for FCEM and baselines."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from common.capture import check_capture
from common.evader_policy import evader_kwargs_from_config, evader_step
from common.obstacles import Obstacle, any_pursuer_obstacle_collision
from fcem.boundary_trap import (
    apply_recovery_slots,
    g_free_allowed as compute_g_free_allowed,
    largest_escape_gap_direction,
    select_active_blockers,
    slot_error_active_only,
    structural_metrics_free_cone,
)
from fcem.evader_prediction import predict_escape_direction, predict_manifold_center
from fcem.low_level.pd_tracker import pd_planner_kwargs_from_config, pd_track_step
from fcem.manifold_generation import evaluate_executability, generate_candidate_manifolds
from fcem.slot_assignment import assign_slots, assignment_weights_from_config, score_assignment
from metrics.experiment_logger import TimingBlock
from metrics.structure import contraction_gate, structural_metrics_from_positions
from metrics.sync import estimate_arrival_times, sync_coverage


def default_pursuer_init(bounds: tuple[float, float, float, float], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = bounds
    w, h = xmax - xmin, ymax - ymin
    pursuers = np.array(
        [
            [xmin + 0.125 * w, ymin + 0.15 * h],
            [xmax - 0.125 * w, ymin + 0.16 * h],
            [xmin + 0.20 * w, ymax - 0.13 * h],
        ],
        dtype=float,
    )
    pursuers += rng.uniform(-0.01 * w, 0.01 * w, pursuers.shape)
    pursuer_v = np.array([[0.8, 0.2], [-0.7, 0.2], [0.2, -0.7]], dtype=float)
    return pursuers, pursuer_v


def default_evader_init(bounds: tuple[float, float, float, float], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = bounds
    w, h = xmax - xmin, ymax - ymin
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    evader = np.array([cx, cy], dtype=float) + rng.uniform(-0.04 * w, 0.04 * w, 2)
    evader_v = np.array([0.18, 0.10], dtype=float) + rng.uniform(-0.1, 0.1, 2)
    return evader, evader_v


def init_from_scenario(
    config: dict[str, Any],
    bounds: tuple[float, float, float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use scenario ``init`` block when present; otherwise default placement."""
    init = config.get("scenario", {}).get("init")
    if not init:
        pursuers, pursuer_v = default_pursuer_init(bounds, rng)
        evader, evader_v = default_evader_init(bounds, rng)
        return pursuers, pursuer_v, evader, evader_v

    evader = np.array(init["evader"], dtype=float)
    evader_v = np.array(init["evader_v"], dtype=float)
    pursuers = np.array(init["pursuers"], dtype=float)
    pursuer_v = np.array(init["pursuer_v"], dtype=float)
    return pursuers, pursuer_v, evader, evader_v


class Sim2D:
    """2D encirclement simulation with pluggable pursuer controller."""

    def __init__(
        self,
        config: dict[str, Any],
        obstacles: list[Obstacle],
        controller: Callable[..., dict[str, Any]],
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config
        self.obstacles = obstacles
        self.controller = controller
        self.rng = rng or np.random.default_rng(config.get("seed", 0))

        w = config["world"]
        self.bounds = (w["xmin"], w["xmax"], w["ymin"], w["ymax"])
        self.dt = config["dt"]
        self.max_steps = config["max_steps"]

        self.pursuers, self.pursuer_v, self.evader, self.evader_v = init_from_scenario(
            config, self.bounds, self.rng
        )

        self.R = config["fcem"]["R_init"]
        self.prev_slots: np.ndarray | None = None
        self.prev_assignment: tuple[int, ...] | None = None
        self.captured = False
        self.capture_step: int | None = None
        self.failed = False
        self.failure_step: int | None = None
        self.failure_reason: str | None = None
        self.frames: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.pursuers, self.pursuer_v, self.evader, self.evader_v = init_from_scenario(
            self.config, self.bounds, self.rng
        )
        self.R = self.config["fcem"]["R_init"]
        self.prev_slots = None
        self.prev_assignment = None
        self.captured = False
        self.capture_step = None
        self.failed = False
        self.failure_step = None
        self.failure_reason = None
        self.frames = []

    def step_once(self, step: int) -> dict[str, Any]:
        cfg = self.config
        fcem_cfg = cfg["fcem"]
        centroid = np.mean(self.pursuers, axis=0)

        self.evader, self.evader_v = evader_step(
            self.evader,
            self.evader_v,
            centroid,
            self.obstacles,
            self.bounds,
            self.dt,
            cfg["evader_vmax"],
            cfg["evader_amax"],
            pursuers=self.pursuers,
            **evader_kwargs_from_config(cfg),
        )

        ctrl_out = self.controller(
            step=step,
            evader=self.evader,
            evader_v=self.evader_v,
            pursuers=self.pursuers,
            pursuer_v=self.pursuer_v,
            obstacles=self.obstacles,
            bounds=self.bounds,
            R=self.R,
            prev_slots=self.prev_slots,
            prev_assignment=self.prev_assignment,
            config=cfg,
        )

        self.pursuers = ctrl_out["pursuers"]
        self.pursuer_v = ctrl_out["pursuer_v"]
        self.R = ctrl_out.get("R", self.R)
        slots = ctrl_out.get("slots")
        if slots is not None:
            self.prev_slots = slots.copy()
        assignment = ctrl_out.get("assignment")
        if assignment is not None:
            self.prev_assignment = assignment

        metrics = ctrl_out.get("metrics")
        if metrics is None:
            trap = ctrl_out.get("trap")
            if trap is not None:
                metrics = structural_metrics_free_cone(self.evader, self.pursuers, trap)
            else:
                metrics = structural_metrics_from_positions(self.evader, self.pursuers)

        trap = ctrl_out.get("trap")
        g_max_allowed = math.radians(cfg["G_max_allowed_deg"])
        g_f_allow = None
        if trap is not None and trap.mode in ("boundary", "corner"):
            g_f_allow = compute_g_free_allowed(trap, len(self.pursuers), cfg["G_max_allowed_deg"])

        if not self.captured and check_capture(
            self.pursuers,
            self.evader,
            fcem_cfg["capture_radius"],
            metrics["G_max"],
            g_max_allowed,
            trap_mode=trap.mode if trap else "open_space",
            g_free=metrics.get("G_free"),
            g_free_allowed=g_f_allow,
        ):
            self.captured = True
            self.capture_step = step

        if not self.captured and not self.failed:
            body_r = float(cfg.get("pursuer_collision_radius", 0.25))
            hit, hit_ids = any_pursuer_obstacle_collision(
                self.pursuers, self.obstacles, body_r
            )
            if hit:
                self.failed = True
                self.failure_step = step
                self.failure_reason = "obstacle_collision"

        frame = {
            "step": step,
            "evader": self.evader.copy(),
            "evader_v": self.evader_v.copy(),
            "pursuers": self.pursuers.copy(),
            "pursuer_v": self.pursuer_v.copy(),
            "R": float(self.R),
            "metrics": metrics,
            "captured": self.captured,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            **{k: v for k, v in ctrl_out.items() if k not in ("pursuers", "pursuer_v")},
        }
        self.frames.append(frame)
        return frame

    def run(self) -> dict[str, Any]:
        for step in range(self.max_steps):
            self.step_once(step)
            if self.captured or self.failed:
                break
        return {
            "captured": self.captured,
            "capture_step": self.capture_step,
            "failed": self.failed,
            "failure_step": self.failure_step,
            "failure_reason": self.failure_reason,
            "num_steps": len(self.frames),
            "frames": self.frames,
        }


def make_fcem_controller(ablation: dict[str, bool] | None = None) -> Callable[..., dict[str, Any]]:
    ablation = ablation or {}
    ctrl_state = {
        "stall_counter": 0,
        "recovery_mode": False,
    }

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
        trap_cfg = fcem_cfg.get("trap", {})
        timing: dict[str, float] = {}
        centroid = np.mean(pursuers, axis=0)

        lookahead = fcem_cfg.get("lookahead_time", 0.80)
        assign_inertia = fcem_cfg.get("assignment_inertia", 0.12)
        assign_weights = assignment_weights_from_config(fcem_cfg, ablation)
        tau_T = float(fcem_cfg.get("tau_T", 2.5))
        v_min_frac = float(fcem_cfg.get("v_min_frac", 0.15))
        clearance = float(config.get("local_planner", {}).get("clearance", 0.55))
        assign_score_scale = float(fcem_cfg.get("assignment_score_scale", 0.15))
        if ctrl_state["recovery_mode"]:
            lookahead = fcem_cfg.get("recovery_lookahead_time", lookahead * 1.35)
            assign_inertia = fcem_cfg.get("recovery_assignment_inertia", assign_inertia * 0.25)

        with TimingBlock() as tb:
            escape_dir = predict_escape_direction(
                evader, evader_v, centroid, ablate_no_esc_dir=ablation.get("ablate_no_esc_dir", False)
            )
            center = predict_manifold_center(
                evader,
                evader_v,
                centroid,
                R,
                bounds,
                lookahead_time=lookahead,
                center_shift_frac=fcem_cfg.get("center_shift_frac", 0.28),
                ablate_no_center_shift=ablation.get("ablate_no_center_shift", False),
            )
        timing["prediction_ms"] = tb.ms

        use_corner_clamp = (
            R <= fcem_cfg["R_terminal"] + trap_cfg.get("corner_clamp_margin", 0.6)
        )

        with TimingBlock() as tb:
            candidates, trap = generate_candidate_manifolds(
                center,
                evader,
                obstacles,
                R,
                escape_dir,
                len(pursuers),
                bounds,
                ablate_single_manifold=ablation.get("ablate_single_manifold", False),
                use_corner_clamp=use_corner_clamp,
                R_terminal=fcem_cfg["R_terminal"],
                trap_cfg=trap_cfg,
            )
        timing["manifold_gen_ms"] = tb.ms

        best = None
        with TimingBlock() as tb:
            for cand in candidates:
                assignment, assign_cost, j_components = assign_slots(
                    pursuers,
                    cand.slots,
                    prev_assignment,
                    inertia=assign_inertia,
                    ablate_nearest_assign=ablation.get("ablate_nearest_assign", False),
                    pursuer_v=pursuer_v,
                    target=evader,
                    obstacles=obstacles,
                    weights=assign_weights,
                    tau_T=tau_T,
                    vmax=config["pursuer_vmax"],
                    v_min_frac=v_min_frac,
                    clearance=clearance,
                )
                infeasible = False
                exec_err = 0.0
                if not ablation.get("ablate_no_executability", False):
                    exec_err, feasible = evaluate_executability(
                        pursuers,
                        pursuer_v,
                        cand.slots,
                        obstacles,
                        bounds,
                        assignment,
                        config["dt"],
                        horizon_steps=fcem_cfg.get("exec_horizon_steps", 3),
                        pursuer_vmax=config["pursuer_vmax"],
                        pursuer_amax=config["pursuer_amax"],
                        kp=config["pursuer_kp"],
                        kd=config["pursuer_kd"],
                        obstacle_influence=config.get("obstacle_influence", 2.20),
                        pursuer_obs_gain=config.get("pursuer_obs_gain", 1.25),
                        boundary_margin=config.get("boundary_margin", 1.0),
                        boundary_gain=config.get("boundary_gain", 1.45),
                    )
                    infeasible = not feasible

                score = score_assignment(
                    cand.structure_score,
                    cand.blocker_score,
                    assign_cost,
                    infeasible,
                    infeasible_penalty=fcem_cfg.get("infeasible_penalty", 8.0),
                    assignment_score_scale=assign_score_scale,
                )
                entry = (score, cand, assignment, assign_cost, exec_err, infeasible, j_components)
                if best is None or score > best[0]:
                    best = entry
        timing["assignment_ms"] = tb.ms

        assert best is not None
        _, best_cand, assignment, assign_cost, exec_err, infeasible, j_components = best
        slots = best_cand.slots.copy()

        if ctrl_state["recovery_mode"]:
            gap_dir = largest_escape_gap_direction(evader, pursuers, trap)
            R_rec = max(fcem_cfg["R_terminal"], R * trap_cfg.get("recovery_radius_scale", 0.95))
            slots, recover_idx = apply_recovery_slots(
                evader, pursuers, pursuer_v, slots, assignment, trap, R_rec, gap_dir
            )

        if prev_slots is None:
            slot_vel = np.zeros_like(slots)
        else:
            slot_vel = (slots - prev_slots) / config["dt"]

        with TimingBlock() as tb:
            pursuers, pursuer_v = pd_track_step(
                pursuers.copy(),
                pursuer_v.copy(),
                slots,
                slot_vel,
                assignment,
                obstacles,
                bounds,
                config["dt"],
                config["pursuer_kp"],
                config["pursuer_kd"],
                fcem_cfg.get("slot_v_ff_gain", 0.85),
                config["pursuer_vmax"],
                config["pursuer_amax"],
                config.get("obstacle_influence", 2.20),
                config.get("pursuer_obs_gain", 1.25),
                config.get("boundary_margin", 1.0),
                config.get("boundary_gain", 1.45),
                ablate_no_slot_vel_ff=ablation.get("ablate_no_slot_vel_ff", False),
                planner_kwargs=pd_planner_kwargs_from_config(config),
            )
        timing["low_level_ms"] = tb.ms
        timing["total_ms"] = sum(timing.values())

        metrics = structural_metrics_free_cone(evader, pursuers, trap)
        T_hats = estimate_arrival_times(
            pursuers,
            pursuer_v,
            slots,
            assignment,
            config["pursuer_vmax"],
            v_min_frac,
        )
        C_sync = sync_coverage(T_hats, tau_T)
        metrics["C_sync"] = C_sync
        active_set, recover_set = select_active_blockers(
            evader, pursuers, slots, assignment, trap,
            slot_error_threshold=fcem_cfg.get("slot_error_abs", 1.10),
        )
        if trap.mode in ("boundary", "corner"):
            slot_error = slot_error_active_only(pursuers, slots, assignment, active_set)
        else:
            slot_error = float(np.mean([np.linalg.norm(pursuers[i] - slots[assignment[i]]) for i in range(len(pursuers))]))

        g_max_allowed = math.radians(config["G_max_allowed_deg"])
        g_max_gate = math.radians(
            config.get("G_max_gate_deg", config["G_max_allowed_deg"])
        )
        g_f_allow = compute_g_free_allowed(trap, len(pursuers), config["G_max_allowed_deg"])

        if ablation.get("ablate_no_guarded_contraction", False):
            R = max(fcem_cfg["R_terminal"], R - fcem_cfg["contraction_rate"])
            q, q_parts = 1.0, {}
        else:
            q, q_parts = contraction_gate(
                metrics,
                slot_error,
                R,
                fcem_cfg["D_min"],
                fcem_cfg["C_min"],
                g_max_allowed,
                fcem_cfg["slot_error_frac"],
                fcem_cfg["slot_error_abs"],
                trap_mode=trap.mode,
                g_free_allowed=g_f_allow,
                g_max_gate_allowed=g_max_gate,
                gate_slot_floor=fcem_cfg.get("gate_slot_floor", 0.0),
                T_min=float(fcem_cfg.get("T_min", 0.35)),
                C_sync=C_sync,
                ablate_no_sync_gate=ablation.get("ablate_no_sync_gate", False),
            )
            stall_thresh = trap_cfg.get("stall_q_threshold", 0.05)
            stall_steps = trap_cfg.get("stall_steps", 25)
            if q < stall_thresh:
                ctrl_state["stall_counter"] += 1
            else:
                ctrl_state["stall_counter"] = 0
                ctrl_state["recovery_mode"] = False

            if ctrl_state["stall_counter"] > stall_steps:
                ctrl_state["recovery_mode"] = True

            if q > stall_thresh and not ctrl_state["recovery_mode"]:
                R = max(fcem_cfg["R_terminal"], R - fcem_cfg["contraction_rate"] * q)
            elif not ctrl_state["recovery_mode"] and not fcem_cfg.get("hold_R_on_stall", True):
                R = min(fcem_cfg["R_init"], R + fcem_cfg["expansion_rate"])

        return {
            "pursuers": pursuers,
            "pursuer_v": pursuer_v,
            "R": R,
            "slots": slots,
            "curve": best_cand.curve,
            "center": best_cand.center,
            "escape_dir": escape_dir,
            "assignment": assignment,
            "slot_error": slot_error,
            "q": q,
            "q_parts": q_parts,
            "manifold_name": best_cand.name,
            "assign_cost": assign_cost,
            "exec_err": exec_err,
            "infeasible": infeasible,
            "timing_ms": timing,
            "trap": trap,
            "trap_mode": trap.mode,
            "trap_corner": trap.corner,
            "metrics": metrics,
            "C_sync": C_sync,
            "J_components": j_components,
            "active_set": active_set,
            "recover_set": recover_set,
            "recovery_mode": ctrl_state["recovery_mode"],
            "stall_counter": ctrl_state["stall_counter"],
        }

    return controller
