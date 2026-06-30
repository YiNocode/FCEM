"""Shared CLI helpers for experiment runners."""

from __future__ import annotations

import argparse

from experiments.config_loader import dynamics_cli_overrides


def add_dynamics_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pursuer-vmax", type=float, default=None)
    parser.add_argument("--evader-vmax", type=float, default=None)
    parser.add_argument("--pursuer-amax", type=float, default=None)
    parser.add_argument("--evader-amax", type=float, default=None)


def add_run_output_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("output")
    group.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Existing run directory (resume); default creates results/<timestamp>_<experiment>",
    )
    group.add_argument(
        "--legacy-output",
        action="store_true",
        help="Write to results/<output_subdir> without timestamp (deprecated)",
    )


def run_output_kwargs_from_args(args: argparse.Namespace) -> dict:
    return {
        "run_dir": args.run_dir,
        "timestamped": not getattr(args, "legacy_output", False),
        "config_path": getattr(args, "config", None),
    }


def dynamics_overrides_from_args(args: argparse.Namespace) -> dict:
    return dynamics_cli_overrides(
        pursuer_vmax=args.pursuer_vmax,
        evader_vmax=args.evader_vmax,
        pursuer_amax=args.pursuer_amax,
        evader_amax=args.evader_amax,
    )
