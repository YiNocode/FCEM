"""PyFlyt 2.5D environment wrapper with graceful fallback.

High-level FCEM planner outputs slot targets; low-level control sends onboard
velocity commands (PyFlyt mode 6: vx, vy, yaw-rate, vz) — not position setpoints.
"""

from __future__ import annotations

import math
import sys
from typing import Any, Callable, TextIO

import numpy as np

from common.capture import capture_params_from_config, evaluate_capture_conditions
from common.dynamics import clip_to_bounds, in_bounds, wall_clearances
from common.evader_policy import evader_kwargs_from_config, evader_step
from common.obstacles import Obstacle, any_pursuer_obstacle_collision
from envs.sim2d import init_from_scenario
from fcem.boundary_trap import (
    structural_metrics_free_cone,
)
from fcem.low_level.pyflyt_command import (
    direct_velocity_setpoints_mode6,
    slot_velocities,
    velocity_setpoints_mode6,
)
from metrics.escape_sector_metrics import (
    compute_escape_sector_metrics,
    escape_metrics_config_from_config,
)
from metrics.structure import structural_metrics_from_positions

_PYFLYT_AVAILABLE = False
_Aviary = None
_p = None
try:
    import pybullet as _p
    from PyFlyt.core import Aviary as _Aviary

    _PYFLYT_AVAILABLE = True
except ImportError:
    pass


class _VisualOverlay:
    """Scene markers and persistent boundary lines for the 40 m arena."""

    def __init__(self, client: Any, n_pursuers: int, marker_radius: float) -> None:
        self.client = client
        self.n_pursuers = n_pursuers
        self.marker_radius = marker_radius
        self.pursuer_ids: list[int] = []
        self.evader_id: int | None = None
        self.boundary_ids: list[int] = []
        self.obstacle_ids: list[int] = []

    def create(
        self,
        altitude: float,
        world_bounds: tuple[float, float, float, float],
        legal_bounds: tuple[float, float, float, float],
        obstacles: list[Obstacle] | None = None,
    ) -> None:
        if _p is None:
            return
        r = self.marker_radius
        pursuer_shape = self.client.createVisualShape(
            _p.GEOM_SPHERE,
            radius=r,
            rgbaColor=[0.25, 0.55, 0.95, 0.88],
        )
        evader_shape = self.client.createVisualShape(
            _p.GEOM_SPHERE,
            radius=r * 1.2,
            rgbaColor=[0.95, 0.25, 0.20, 0.92],
        )
        self.pursuer_ids = [
            self.client.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=pursuer_shape,
                basePosition=[0.0, 0.0, altitude],
            )
            for _ in range(self.n_pursuers)
        ]
        self.evader_id = self.client.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=evader_shape,
            basePosition=[0.0, 0.0, altitude],
        )
        self._draw_rect(world_bounds, z=0.15, color=[1.0, 0.85, 0.1], width=4.0)
        self._draw_rect(legal_bounds, z=0.25, color=[1.0, 1.0, 1.0], width=2.5)
        self._create_obstacles(obstacles or [], altitude)

    def _create_obstacles(self, obstacles: list[Obstacle], altitude: float) -> None:
        if _p is None:
            return
        pillar_h = max(0.35, altitude * 0.85)
        for obs in obstacles:
            vis = self.client.createVisualShape(
                _p.GEOM_CYLINDER,
                radius=float(obs.radius),
                length=pillar_h,
                rgbaColor=[0.35, 0.35, 0.38, 0.82],
            )
            body_id = self.client.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=vis,
                basePosition=[float(obs.center[0]), float(obs.center[1]), pillar_h * 0.5],
            )
            self.obstacle_ids.append(body_id)

    def _draw_rect(
        self,
        bounds: tuple[float, float, float, float],
        z: float,
        color: list[float],
        width: float,
    ) -> None:
        if _p is None:
            return
        xmin, xmax, ymin, ymax = bounds
        corners = [
            (xmin, ymin, z),
            (xmax, ymin, z),
            (xmax, ymax, z),
            (xmin, ymax, z),
        ]
        for i in range(4):
            line_id = self.client.addUserDebugLine(
                corners[i],
                corners[(i + 1) % 4],
                lineColorRGB=color,
                lineWidth=width,
            )
            self.boundary_ids.append(line_id)

    def update(self, pursuers_xy: np.ndarray, evader_xy: np.ndarray, altitude: float) -> None:
        if _p is None:
            return
        for body_id, xy in zip(self.pursuer_ids, pursuers_xy):
            self.client.resetBasePositionAndOrientation(
                body_id,
                [float(xy[0]), float(xy[1]), altitude],
                [0.0, 0.0, 0.0, 1.0],
            )
        if self.evader_id is not None:
            self.client.resetBasePositionAndOrientation(
                self.evader_id,
                [float(evader_xy[0]), float(evader_xy[1]), altitude],
                [0.0, 0.0, 0.0, 1.0],
            )

    def clear(self) -> None:
        if _p is None:
            return
        for body_id in self.pursuer_ids:
            self.client.removeBody(body_id)
        if self.evader_id is not None:
            self.client.removeBody(self.evader_id)
        for line_id in self.boundary_ids:
            self.client.removeUserDebugItem(line_id)
        for body_id in self.obstacle_ids:
            self.client.removeBody(body_id)
        self.pursuer_ids = []
        self.evader_id = None
        self.boundary_ids = []
        self.obstacle_ids = []


def _pursuers_from_aviary(aviary: Any) -> np.ndarray:
    return np.array([aviary.state(i)[3, :2] for i in range(aviary.num_drones)], dtype=float)


def _yaw_and_height_from_aviary(aviary: Any) -> tuple[np.ndarray, np.ndarray]:
    yaws, heights = [], []
    for i in range(aviary.num_drones):
        state = aviary.state(i)
        yaws.append(float(state[1, 2]))
        heights.append(float(state[3, 2]))
    return np.array(yaws, dtype=float), np.array(heights, dtype=float)


class PyFlytEnv:
    """2.5D multi-drone encirclement with velocity-level PyFlyt control."""

    def __init__(
        self,
        config: dict[str, Any],
        obstacles: list[Obstacle] | None = None,
        controller: Callable[..., dict[str, Any]] | None = None,
        rng: np.random.Generator | None = None,
        use_plan_state: bool | None = None,
    ) -> None:
        self.config = config
        self.obstacles = obstacles or []
        self.controller = controller
        self.rng = rng or np.random.default_rng(config.get("seed", 0))
        self.available = _PYFLYT_AVAILABLE

        pf_cfg = config.get("pyflyt", {})
        self.altitude = float(pf_cfg.get("altitude", 1.5))
        self.physics_hz = int(pf_cfg.get("physics_hz", 240))
        self.render = bool(pf_cfg.get("render", False))
        self.flight_mode = int(pf_cfg.get("flight_mode", 6))
        self.n_pursuers = int(pf_cfg.get("n_pursuers", 3))
        self.camera_follow = bool(pf_cfg.get("camera_follow", False))
        self.marker_radius = float(pf_cfg.get("marker_radius", 0.65))
        self.show_markers = bool(pf_cfg.get("show_markers", True))
        self.drone_model = str(pf_cfg.get("drone_model", "primitive_drone"))
        self.bounds_margin = float(pf_cfg.get("bounds_margin", 0.15))
        self.verbose = bool(pf_cfg.get("verbose", False))
        self.log_every = max(1, int(pf_cfg.get("log_every", 20)))
        self._log_stream: TextIO = sys.stdout
        self._overlay: _VisualOverlay | None = None
        self._last_cmd: np.ndarray | None = None
        if use_plan_state is None:
            use_plan_state = bool(pf_cfg.get("use_plan_state", True))
        self.use_plan_state = use_plan_state

        w = config["world"]
        self.bounds = (w["xmin"], w["xmax"], w["ymin"], w["ymax"])
        self.dt = config["dt"]
        self.max_steps = config["max_steps"]

        self._aviary: Any | None = None
        self.evader = np.zeros(2, dtype=float)
        self.evader_v = np.zeros(2, dtype=float)
        self.pursuers_physics = np.zeros((self.n_pursuers, 2), dtype=float)
        self.pursuer_v_physics = np.zeros((self.n_pursuers, 2), dtype=float)
        self.pursuers_plan = np.zeros((self.n_pursuers, 2), dtype=float)
        self.pursuer_v_plan = np.zeros((self.n_pursuers, 2), dtype=float)
        self._prev_pursuer_physics: np.ndarray | None = None

        self.R = config["fcem"]["R_init"]
        self.prev_slots: np.ndarray | None = None
        self.prev_assignment: tuple[int, ...] | None = None
        self.captured = False
        self.capture_step: int | None = None
        self.failed = False
        self.failure_step: int | None = None
        self.failure_reason: str | None = None
        self.frames: list[dict[str, Any]] = []

    def _require_aviary(self) -> Any:
        if not self.available or _Aviary is None:
            raise RuntimeError(
                "PyFlyt is not installed. Install in WSL2/Linux: pip install pyflyt"
            )
        if self._aviary is None:
            raise RuntimeError("Aviary not initialized — call reset() first")
        return self._aviary

    def _legal_bounds(self) -> tuple[float, float, float, float]:
        m = self.bounds_margin
        xmin, xmax, ymin, ymax = self.bounds
        return (xmin + m, xmax - m, ymin + m, ymax - m)

    def _clamp_point(self, xy: np.ndarray) -> np.ndarray:
        xmin, xmax, ymin, ymax = self.bounds
        return clip_to_bounds(xy, xmin, xmax, ymin, ymax, self.bounds_margin)

    def _arena_center(self) -> np.ndarray:
        xmin, xmax, ymin, ymax = self.bounds
        return np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax)], dtype=float)

    def _configure_camera(self, aviary: Any) -> None:
        if not self.render:
            return
        pf_cfg = self.config.get("pyflyt", {})
        cx, cy = self._arena_center()
        xmin, xmax, ymin, ymax = self.bounds
        span = max(xmax - xmin, ymax - ymin)
        distance = float(pf_cfg.get("camera_distance", span * 1.45))
        aviary.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=float(pf_cfg.get("camera_yaw", 0.0)),
            cameraPitch=float(pf_cfg.get("camera_pitch", -89.0)),
            cameraTargetPosition=[float(cx), float(cy), 0.0],
        )
        if _p is not None:
            aviary.configureDebugVisualizer(_p.COV_ENABLE_SHADOWS, 0)
            aviary.configureDebugVisualizer(_p.COV_ENABLE_GUI, 1)

    def _setup_visualization(self, aviary: Any) -> None:
        if not self.render:
            return
        self._configure_camera(aviary)
        if self.show_markers:
            self._overlay = _VisualOverlay(aviary, self.n_pursuers, self.marker_radius)
            self._overlay.create(self.altitude, self.bounds, self._legal_bounds(), self.obstacles)
            self._overlay.update(self.pursuers_physics, self.evader, self.altitude)

    def _update_visualization(self) -> None:
        if not self.render or self._aviary is None:
            return
        if self.camera_follow:
            pf_cfg = self.config.get("pyflyt", {})
            focus = np.mean(
                np.vstack([self.pursuers_physics, self.evader[None, :]]),
                axis=0,
            )
            span = max(self.bounds[1] - self.bounds[0], self.bounds[3] - self.bounds[2])
            self._aviary.resetDebugVisualizerCamera(
                cameraDistance=float(pf_cfg.get("camera_distance", span * 1.45)),
                cameraYaw=float(pf_cfg.get("camera_yaw", 0.0)),
                cameraPitch=float(pf_cfg.get("camera_pitch", -89.0)),
                cameraTargetPosition=[float(focus[0]), float(focus[1]), 0.0],
            )
        if self._overlay is not None:
            self._overlay.update(self.pursuers_physics, self.evader, self.altitude)

    def _create_aviary(self, pursuers_xy: np.ndarray) -> Any:
        assert _Aviary is not None
        start_pos = np.column_stack(
            [pursuers_xy[:, 0], pursuers_xy[:, 1], np.full(len(pursuers_xy), self.altitude)]
        )
        start_orn = np.zeros((len(pursuers_xy), 3), dtype=float)
        aviary = _Aviary(
            start_pos=start_pos,
            start_orn=start_orn,
            render=self.render,
            drone_type="quadx",
            drone_options={"drone_model": self.drone_model},
            physics_hz=self.physics_hz,
            seed=int(self.config.get("seed", 0)),
        )
        aviary.set_mode(self.flight_mode)
        zeros = np.zeros((len(pursuers_xy), 4), dtype=float)
        aviary.set_all_setpoints(zeros)
        return aviary

    def _physics_steps_per_control(self) -> int:
        return max(1, int(round(self.dt * self.physics_hz)))

    def _sync_physics_state(self) -> None:
        aviary = self._require_aviary()
        pursuers = _pursuers_from_aviary(aviary)
        if self._prev_pursuer_physics is not None:
            self.pursuer_v_physics = (pursuers - self._prev_pursuer_physics) / self.dt
        self.pursuers_physics = pursuers
        self._prev_pursuer_physics = pursuers.copy()

    def _apply_velocity_commands(
        self,
        slots: np.ndarray,
        assignment: tuple[int, ...],
    ) -> np.ndarray:
        aviary = self._require_aviary()
        sv = slot_velocities(slots, self.prev_slots, self.dt)
        yaws, heights = _yaw_and_height_from_aviary(aviary)
        setpoints = velocity_setpoints_mode6(
            self.pursuers_physics,
            self.pursuer_v_physics,
            slots,
            sv,
            assignment,
            yaws,
            self.altitude,
            heights,
            self.obstacles,
            self.bounds,
            self.config,
            ablation=self.config.get("ablation"),
        )
        aviary.set_all_setpoints(setpoints)
        self._last_cmd = setpoints.copy()
        return setpoints

    def _apply_direct_velocity_commands(
        self,
        pursuer_v_xy: np.ndarray,
        heading_targets: np.ndarray | None = None,
    ) -> np.ndarray:
        aviary = self._require_aviary()
        yaws, heights = _yaw_and_height_from_aviary(aviary)
        setpoints = direct_velocity_setpoints_mode6(
            pursuer_v_xy,
            yaws,
            heights,
            self.altitude,
            self.config,
            heading_targets=heading_targets,
        )
        aviary.set_all_setpoints(setpoints)
        self._last_cmd = setpoints.copy()
        return setpoints

    def _commands_from_controller(
        self,
        ctrl_out: dict[str, Any],
    ) -> np.ndarray | None:
        slots = ctrl_out.get("slots")
        assignment = ctrl_out.get("assignment")
        if slots is not None and assignment is not None:
            return self._apply_velocity_commands(slots, assignment)
        pursuer_v = ctrl_out.get("pursuer_v")
        if pursuer_v is not None:
            heading = self.evader[None, :] - self.pursuers_physics
            return self._apply_direct_velocity_commands(pursuer_v, heading_targets=heading)
        return None

    def _step_physics(self) -> None:
        aviary = self._require_aviary()
        for _ in range(self._physics_steps_per_control()):
            aviary.step()

    def _agent_telemetry(self, name: str, xy: np.ndarray, ref_xy: np.ndarray) -> dict[str, Any]:
        walls = wall_clearances(xy, self.bounds)
        return {
            "name": name,
            "x": float(xy[0]),
            "y": float(xy[1]),
            "dist_to_evader": float(np.linalg.norm(xy - ref_xy)),
            "walls": walls,
            "in_bounds": in_bounds(xy, self.bounds, self.bounds_margin),
        }

    def build_step_telemetry(self, step: int, ctrl_out: dict[str, Any] | None = None) -> dict[str, Any]:
        dists = np.linalg.norm(self.pursuers_physics - self.evader[None, :], axis=1)
        oob = [
            i
            for i in range(self.n_pursuers)
            if not in_bounds(self.pursuers_physics[i], self.bounds, self.bounds_margin)
        ]
        slot_err_phy = None
        if self.prev_slots is not None and self.prev_assignment is not None:
            slot_err_phy = float(
                np.mean(
                    [
                        np.linalg.norm(
                            self.pursuers_physics[i] - self.prev_slots[self.prev_assignment[i]]
                        )
                        for i in range(self.n_pursuers)
                    ]
                )
            )
        cmds = self._last_cmd
        cmd_rows = []
        if cmds is not None:
            for i in range(len(cmds)):
                cmd_rows.append(
                    {
                        "vx": float(cmds[i, 0]),
                        "vy": float(cmds[i, 1]),
                        "vr": float(cmds[i, 2]),
                        "vz": float(cmds[i, 3]),
                    }
                )
        return {
            "step": step,
            "world_bounds": self.bounds,
            "legal_bounds": self._legal_bounds(),
            "bounds_margin": self.bounds_margin,
            "evader": self._agent_telemetry("evader", self.evader, self.evader),
            "pursuers_physics": [
                self._agent_telemetry(f"P{i}", self.pursuers_physics[i], self.evader)
                for i in range(self.n_pursuers)
            ],
            "pursuers_plan": [
                self._agent_telemetry(f"P{i}_plan", self.pursuers_plan[i], self.evader)
                for i in range(self.n_pursuers)
            ],
            "velocity_cmds": cmd_rows,
            "pairwise_physics": dists.tolist(),
            "mean_dist_physics": float(np.mean(dists)),
            "oob_physics": oob,
            "R": float(self.R),
            "q": float(ctrl_out.get("q", 0.0)) if ctrl_out else None,
            "slot_error": ctrl_out.get("slot_error") if ctrl_out else None,
            "slot_error_physics": slot_err_phy,
            "captured": self.captured,
        }

    def format_step_telemetry(self, telemetry: dict[str, Any]) -> str:
        xmin, xmax, ymin, ymax = telemetry["world_bounds"]
        lx0, lx1, ly0, ly1 = telemetry["legal_bounds"]
        m = telemetry["bounds_margin"]
        lines = [
            (
                f"[step {telemetry['step']}] "
                f"world=[{xmin:.1f},{xmax:.1f}]x[{ymin:.1f},{ymax:.1f}] "
                f"legal=[{lx0:.2f},{lx1:.2f}]x[{ly0:.2f},{ly1:.2f}] margin={m:.2f} "
                f"R={telemetry['R']:.2f} q={telemetry.get('q')} "
                f"slot_err={telemetry.get('slot_error')} slot_err_phy={telemetry.get('slot_error_physics')} "
                f"captured={telemetry['captured']}"
            ),
        ]
        ev = telemetry["evader"]
        flag = "OK" if ev["in_bounds"] else "OUT"
        w = ev["walls"]
        lines.append(
            f"  evader  ({ev['x']:6.2f}, {ev['y']:6.2f}) [{flag}] "
            f"walls L/R/B/T={w['left']:5.2f}/{w['right']:5.2f}/"
            f"{w['bottom']:5.2f}/{w['top']:5.2f}"
        )
        for i, agent in enumerate(telemetry["pursuers_physics"]):
            flag = "OK" if agent["in_bounds"] else "OUT"
            w = agent["walls"]
            plan = telemetry["pursuers_plan"][i]
            cmd_s = ""
            if i < len(telemetry.get("velocity_cmds", [])):
                c = telemetry["velocity_cmds"][i]
                cmd_s = f" cmd=({c['vx']:+.2f},{c['vy']:+.2f},{c['vr']:+.2f},{c['vz']:+.2f})"
            lines.append(
                f"  P{i} phy ({agent['x']:6.2f}, {agent['y']:6.2f}) [{flag}] "
                f"d_ev={agent['dist_to_evader']:5.2f} "
                f"walls L/R/B/T={w['left']:5.2f}/{w['right']:5.2f}/"
                f"{w['bottom']:5.2f}/{w['top']:5.2f}{cmd_s} "
                f"plan=({plan['x']:6.2f},{plan['y']:6.2f})"
            )
        if telemetry["oob_physics"]:
            lines.append(f"  ** boundary violation: physics={telemetry['oob_physics']} **")
        return "\n".join(lines)

    def _log_step(self, step: int, telemetry: dict[str, Any]) -> None:
        if not self.verbose:
            return
        if step % self.log_every != 0 and not telemetry["oob_physics"]:
            return
        print(self.format_step_telemetry(telemetry), file=self._log_stream, flush=True)

    def reset(self) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError(
                "PyFlyt is not installed. Install in WSL2/Linux: pip install pyflyt"
            )

        self.close()
        self.pursuers_plan, self.pursuer_v_plan, self.evader, self.evader_v = init_from_scenario(
            self.config, self.bounds, self.rng
        )
        init_pursuers = self.pursuers_plan.copy()
        init_vel = self.pursuer_v_plan.copy()
        self._prev_pursuer_physics = None
        self.pursuer_v_physics = np.zeros((self.n_pursuers, 2), dtype=float)

        self._aviary = self._create_aviary(init_pursuers)
        self._sync_physics_state()
        self.pursuers_physics = init_pursuers.copy()
        self._setup_visualization(self._aviary)

        self.R = self.config["fcem"]["R_init"]
        self.prev_slots = None
        self.prev_assignment = None
        self.captured = False
        self.capture_step = None
        self.failed = False
        self.failure_step = None
        self.failure_reason = None
        self.frames = []
        self._last_cmd = None

        return {
            "ok": True,
            "mode": "pyflyt_velocity",
            "flight_mode": self.flight_mode,
            "n_pursuers": self.n_pursuers,
            "altitude": self.altitude,
            "evader": self.evader.copy(),
            "pursuers_physics": self.pursuers_physics.copy(),
        }

    def step(
        self,
        velocity_cmds: np.ndarray,
    ) -> dict[str, Any]:
        """Low-level step: send (n,4) velocity setpoints and advance physics one control tick."""
        aviary = self._require_aviary()
        aviary.set_all_setpoints(velocity_cmds)
        self._last_cmd = velocity_cmds.copy()
        self._step_physics()
        self._sync_physics_state()
        self._update_visualization()
        return {
            "ok": True,
            "pursuers_physics": self.pursuers_physics.copy(),
            "pursuer_v_physics": self.pursuer_v_physics.copy(),
        }

    def step_once(self, step: int) -> dict[str, Any]:
        if self.controller is None:
            raise RuntimeError("PyFlytEnv.step_once requires a controller")

        cfg = self.config
        fcem_cfg = cfg["fcem"]
        if self.use_plan_state:
            centroid = np.mean(self.pursuers_plan, axis=0)
            pursuers_for_evader = self.pursuers_plan
        else:
            centroid = np.mean(self.pursuers_physics, axis=0)
            pursuers_for_evader = self.pursuers_physics

        self.evader, self.evader_v = evader_step(
            self.evader,
            self.evader_v,
            centroid,
            self.obstacles,
            self.bounds,
            self.dt,
            cfg["evader_vmax"],
            cfg["evader_amax"],
            pursuers=pursuers_for_evader,
            **evader_kwargs_from_config(cfg),
        )
        self.evader = self._clamp_point(self.evader)

        self._sync_physics_state()

        if self.use_plan_state:
            pursuers_in = self.pursuers_plan
            pursuer_v_in = self.pursuer_v_plan
        else:
            pursuers_in = self.pursuers_physics
            pursuer_v_in = self.pursuer_v_physics

        ctrl_out = self.controller(
            step=step,
            evader=self.evader,
            evader_v=self.evader_v,
            pursuers=pursuers_in,
            pursuer_v=pursuer_v_in,
            obstacles=self.obstacles,
            bounds=self.bounds,
            R=self.R,
            prev_slots=self.prev_slots,
            prev_assignment=self.prev_assignment,
            config=cfg,
        )

        if self.use_plan_state:
            self.pursuers_plan = ctrl_out["pursuers"]
            self.pursuer_v_plan = ctrl_out["pursuer_v"]

        self.R = ctrl_out.get("R", self.R)
        slots = ctrl_out.get("slots")
        assignment = ctrl_out.get("assignment")
        if slots is not None:
            self.prev_slots = slots.copy()
        if assignment is not None:
            self.prev_assignment = assignment
        cmd_setpoints = self._commands_from_controller(ctrl_out)

        self._step_physics()
        self._sync_physics_state()
        self._update_visualization()

        trap = ctrl_out.get("trap")
        if trap is not None:
            metrics_physics = structural_metrics_free_cone(
                self.evader, self.pursuers_physics, trap
            )
        else:
            metrics_physics = structural_metrics_from_positions(
                self.evader, self.pursuers_physics
            )

        metrics_physics["D_ang_full"] = float(metrics_physics["D_ang"])
        metrics_physics["C_cov_full"] = float(metrics_physics["C_cov"])
        metrics_physics["G_max_full_deg"] = math.degrees(float(metrics_physics["G_max"]))

        esc_cfg = escape_metrics_config_from_config(cfg)
        esc = compute_escape_sector_metrics(
            self.evader,
            self.pursuers_physics,
            self.obstacles,
            self.bounds,
            ray_length=esc_cfg["ray_length"],
            num_angles=esc_cfg["num_angles"],
            num_ray_samples=esc_cfg["num_ray_samples"],
            pursuer_block_radius=esc_cfg["pursuer_block_radius"],
            obstacle_margin=esc_cfg["obstacle_margin"],
            boundary_margin=esc_cfg["boundary_margin"],
            exit_config=esc_cfg["exit_config"],
            min_forward_block_dist=esc_cfg["min_forward_block_dist"],
        )
        for key in (
            "C_esc",
            "G_esc_deg",
            "free_escape_angle_deg",
            "blocked_escape_angle_deg",
            "unblocked_escape_angle_deg",
            "exit_blockage",
        ):
            metrics_physics[key] = esc[key]

        cap_params = capture_params_from_config(cfg)
        cap_flags = evaluate_capture_conditions(
            self.pursuers_physics,
            self.evader,
            cap_params["capture_radius"],
            metrics_physics["G_max"],
            cap_params["g_max_allowed"],
            esc,
            g_esc_allow_deg=cap_params["g_esc_allow_deg"],
            c_esc_min=cap_params["c_esc_min"],
            capture_mode=cap_params["capture_mode"],
        )
        metrics_physics.update(
            {
                "capture_condition_valid_distance": cap_flags[
                    "capture_condition_valid_distance"
                ],
                "capture_condition_valid_full_circle": cap_flags[
                    "capture_condition_valid_full_circle"
                ],
                "capture_condition_valid_escape_sector": cap_flags[
                    "capture_condition_valid_escape_sector"
                ],
            }
        )

        if not self.captured and cap_flags["captured"]:
            self.captured = True
            self.capture_step = step

        if not self.captured and not self.failed:
            body_r = float(cfg.get("pursuer_collision_radius", 0.25))
            hit_phy, _ = any_pursuer_obstacle_collision(
                self.pursuers_physics, self.obstacles, body_r
            )
            hit_plan = False
            if self.use_plan_state:
                hit_plan, _ = any_pursuer_obstacle_collision(
                    self.pursuers_plan, self.obstacles, body_r
                )
            if hit_phy or hit_plan:
                self.failed = True
                self.failure_step = step
                self.failure_reason = "obstacle_collision"

        frame = {
            "step": step,
            "evader": self.evader.copy(),
            "evader_v": self.evader_v.copy(),
            "pursuers": self.pursuers_plan.copy(),
            "pursuer_v": self.pursuer_v_plan.copy(),
            "pursuers_plan": self.pursuers_plan.copy(),
            "pursuer_v_plan": self.pursuer_v_plan.copy(),
            "pursuers_physics": self.pursuers_physics.copy(),
            "pursuer_v_physics": self.pursuer_v_physics.copy(),
            "velocity_cmds": None if cmd_setpoints is None else cmd_setpoints.copy(),
            "R": float(self.R),
            "metrics": metrics_physics,
            "metrics_physics": metrics_physics,
            "metrics_plan": ctrl_out.get("metrics"),
            "captured": self.captured,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "backend": "pyflyt",
            **{
                k: v
                for k, v in ctrl_out.items()
                if k not in ("pursuers", "pursuer_v", "metrics")
            },
        }
        telemetry = self.build_step_telemetry(step, ctrl_out)
        frame["telemetry"] = telemetry
        self._log_step(step, telemetry)
        self.frames.append(frame)
        return frame

    def run(self) -> dict[str, Any]:
        if self._aviary is None:
            self.reset()
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
            "backend": "pyflyt",
        }

    def close(self) -> None:
        if self._overlay is not None:
            self._overlay.clear()
            self._overlay = None
        if self._aviary is not None:
            try:
                self._aviary.disconnect()
            except Exception:
                pass
            self._aviary = None

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def check() -> bool:
        return _PYFLYT_AVAILABLE
