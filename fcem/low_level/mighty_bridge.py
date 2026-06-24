"""ROS2 / Mighty bridge stub for PX4 low-level control.

Uses the same Hermite local planner as PyFlyt / sim2d for slot tracking.
When ROS2 is available, planned velocities can be forwarded to Mighty topics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from common.obstacles import Obstacle
from fcem.low_level.hermite_planner import plan_velocity_command, planner_kwargs_from_config

_ROS2_AVAILABLE = False
try:
    import rclpy  # noqa: F401
    _ROS2_AVAILABLE = True
except ImportError:
    pass


@dataclass
class MightyBridgeConfig:
    namespace: str = "/mighty"
    control_rate_hz: float = 50.0


class MightyBridge:
    """Stub ROS2 bridge for sending slot targets to PX4 via Mighty."""

    def __init__(self, config: MightyBridgeConfig | None = None) -> None:
        self.config = config or MightyBridgeConfig()
        self._connected = False

    @property
    def ros2_available(self) -> bool:
        return _ROS2_AVAILABLE

    def connect(self) -> bool:
        if not _ROS2_AVAILABLE:
            return False
        # Future: initialize rclpy node and publishers.
        self._connected = True
        return True

    def plan_slot_velocities(
        self,
        pursuers: np.ndarray,
        pursuer_vel: np.ndarray,
        slots: np.ndarray,
        slot_vel: np.ndarray,
        assignment: tuple[int, ...],
        obstacles: list[Obstacle],
        bounds: tuple[float, float, float, float],
        sim_config: dict[str, Any],
        slot_v_ff_gain: float = 0.85,
    ) -> np.ndarray:
        """Hermite planner velocity commands for each pursuer (MIGHTY-compatible layer)."""
        pkw = planner_kwargs_from_config(sim_config)
        dt = float(sim_config["dt"])
        vmax = float(sim_config["pursuer_vmax"])
        amax = float(sim_config["pursuer_amax"])
        cmds = np.zeros((len(pursuers), 2), dtype=float)
        for i, slot_idx in enumerate(assignment):
            cmds[i] = plan_velocity_command(
                pursuers[i],
                pursuer_vel[i],
                slots[slot_idx],
                slot_v_ff_gain * slot_vel[slot_idx],
                obstacles,
                bounds,
                dt,
                vmax,
                amax,
                **pkw,
            )
        return cmds

    def send_slot_targets(
        self,
        slot_positions: np.ndarray,
        slot_velocities: np.ndarray,
        assignment: tuple[int, ...],
        pursuers: np.ndarray | None = None,
        pursuer_vel: np.ndarray | None = None,
        obstacles: list[Obstacle] | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        sim_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._connected:
            return {"ok": False, "reason": "not_connected"}

        planned_v = None
        if (
            pursuers is not None
            and pursuer_vel is not None
            and obstacles is not None
            and bounds is not None
            and sim_config is not None
        ):
            ff = float(sim_config.get("fcem", {}).get("slot_v_ff_gain", 0.85))
            planned_v = self.plan_slot_velocities(
                pursuers,
                pursuer_vel,
                slot_positions,
                slot_velocities,
                assignment,
                obstacles,
                bounds,
                sim_config,
                slot_v_ff_gain=ff,
            )

        return {
            "ok": True,
            "namespace": self.config.namespace,
            "n_targets": len(slot_positions),
            "assignment": assignment,
            "planned_velocities": None if planned_v is None else planned_v.tolist(),
        }

    def disconnect(self) -> None:
        self._connected = False
