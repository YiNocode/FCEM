"""PX4 / Gazebo environment stub for Mighty integration on WSL2."""

from __future__ import annotations

from typing import Any

import numpy as np

from fcem.low_level.mighty_bridge import MightyBridge, MightyBridgeConfig


class PX4Env:
    """Stub PX4 environment; connects via MightyBridge when ROS2 is available."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.bridge = MightyBridge(MightyBridgeConfig())
        self.connected = False

    def connect(self) -> bool:
        self.connected = self.bridge.connect()
        return self.connected

    def reset(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "ros2": self.bridge.ros2_available,
            "mode": "stub",
        }

    def send_targets(self, slots: np.ndarray, slot_vel: np.ndarray, assignment: tuple[int, ...]) -> dict[str, Any]:
        return self.bridge.send_slot_targets(slots, slot_vel, assignment)

    def disconnect(self) -> None:
        self.bridge.disconnect()
        self.connected = False
