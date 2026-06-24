"""Central registry for all pursuit methods (FCEM + baselines)."""

from __future__ import annotations

from typing import Any, Callable

from baselines.open_marl import make_open_marl_controller
from baselines.deghat_circumnavigation import make_deghat_circumnavigation_controller
from baselines.fang_relay_2022 import make_fang_relay_2022_controller
from baselines.fixed_ring_apf import make_fixed_ring_controller
from baselines.kou_xiang_fencing import make_kou_xiang_fencing_controller
from baselines.liao_mpc import make_liao_mpc_controller
from baselines.pure_pursuit_apf import make_pure_pursuit_controller
from baselines.relay_pursuit import make_relay_pursuit_controller
from baselines.yu_consensus import make_yu_consensus_controller
from envs.sim2d import make_fcem_controller

ControllerFactory = Callable[[dict[str, bool] | None], Any]

METHODS: dict[str, ControllerFactory] = {
    "fcem": lambda ablation=None: make_fcem_controller(ablation),
    "pure_pursuit": lambda ablation=None: make_pure_pursuit_controller(),
    "fixed_ring": lambda ablation=None: make_fixed_ring_controller(),
    "pure_pursuit_apf": lambda ablation=None: make_pure_pursuit_controller(),
    "fixed_ring_apf": lambda ablation=None: make_fixed_ring_controller(),
    "deghat_circumnavigation": lambda ablation=None: make_deghat_circumnavigation_controller(),
    "kou_xiang_fencing": lambda ablation=None: make_kou_xiang_fencing_controller(),
    "fang_relay_2022": lambda ablation=None: make_fang_relay_2022_controller(),
    "relay_pursuit": lambda ablation=None: make_relay_pursuit_controller(),
    "liao_mpc": lambda ablation=None: make_liao_mpc_controller(),
    "yu_consensus": lambda ablation=None: make_yu_consensus_controller(),
    "open_marl": lambda ablation=None: make_open_marl_controller(),
}

METHOD_ALIASES: dict[str, str] = {
    "pure_pursuit_apf": "pure_pursuit",
    "fixed_ring_apf": "fixed_ring",
}

# Slot-based high-level planners (PyFlyt use_plan_state).
SLOT_METHODS = frozenset({
    "fcem",
    "fixed_ring",
    "deghat_circumnavigation",
    "kou_xiang_fencing",
    "fang_relay_2022",
    "relay_pursuit",
    "liao_mpc",
    "yu_consensus",
})


def normalize_method(name: str) -> str:
    return METHOD_ALIASES.get(name, name)
