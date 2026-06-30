"""Central registry for all pursuit methods (FCEM + baselines)."""

from __future__ import annotations

from typing import Any, Callable

from baselines.ac_baseline import make_ac_baseline_controller
from baselines.pure_pursuit_apf import make_pure_pursuit_controller
from envs.sim2d import make_fcem_controller

ControllerFactory = Callable[[dict[str, bool] | None], Any]


def _make_liao_mpc_controller(ablation: dict[str, bool] | None = None) -> Any:
    from baselines.liao_mpc import make_liao_mpc_controller

    return make_liao_mpc_controller()


def _make_open_marl_controller(ablation: dict[str, bool] | None = None) -> Any:
    from baselines.open_marl import make_open_marl_controller

    return make_open_marl_controller()


METHODS: dict[str, ControllerFactory] = {
    "fcem": lambda ablation=None: make_fcem_controller(ablation),
    "pure_pursuit": lambda ablation=None: make_pure_pursuit_controller(),
    "pure_pursuit_apf": lambda ablation=None: make_pure_pursuit_controller(),
    "liao_mpc": _make_liao_mpc_controller,
    "open_marl": _make_open_marl_controller,
    "ac_baseline": lambda ablation=None: make_ac_baseline_controller(),
}

METHOD_ALIASES: dict[str, str] = {
    "pure_pursuit_apf": "pure_pursuit",
}

# Slot-based high-level planners (PyFlyt use_plan_state).
SLOT_METHODS = frozenset({
    "fcem",
    "liao_mpc",
})


def normalize_method(name: str) -> str:
    return METHOD_ALIASES.get(name, name)
