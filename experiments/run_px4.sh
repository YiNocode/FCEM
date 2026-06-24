#!/usr/bin/env bash
# PX4 + Mighty experiment launcher (run inside WSL2)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

echo "=== FCEM PX4 / Mighty experiment (stub) ==="
echo "Ensure ROS2, PX4, Gazebo, and Mighty are running in WSL2."

python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from envs.px4_env import PX4Env
from experiments.config_loader import load_config

cfg = load_config("free")
env = PX4Env(cfg)
print("ROS2 available via bridge:", env.bridge.ros2_available)
print("connect:", env.connect())
print("reset:", env.reset())
env.disconnect()
print("PX4 stub complete. Wire MightyBridge to actual topics when hardware is ready.")
PY
