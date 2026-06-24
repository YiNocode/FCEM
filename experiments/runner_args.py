"""Shared CLI helpers for experiment runners."""

from __future__ import annotations

import argparse

from experiments.config_loader import dynamics_cli_overrides


def add_dynamics_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pursuer-vmax", type=float, default=None)
    parser.add_argument("--evader-vmax", type=float, default=None)
    parser.add_argument("--pursuer-amax", type=float, default=None)
    parser.add_argument("--evader-amax", type=float, default=None)


def dynamics_overrides_from_args(args: argparse.Namespace) -> dict:
    return dynamics_cli_overrides(
        pursuer_vmax=args.pursuer_vmax,
        evader_vmax=args.evader_vmax,
        pursuer_amax=args.pursuer_amax,
        evader_amax=args.evader_amax,
    )
