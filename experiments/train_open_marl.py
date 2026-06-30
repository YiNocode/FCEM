"""Train OPEN MARL baseline (EPN + MAPPO) on FCEM 2D environments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.open_marl.aeg import AEGConfig, CurriculumConfig
from baselines.open_marl.mappo import MAPPOConfig, train_open_marl
from baselines.open_marl.observation import OpenMarlConfig
from baselines.open_marl.reward import RewardConfig
from experiments.config_loader import build_experiment_base_config, deep_merge, load_experiment_config, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OPEN MARL baseline")
    parser.add_argument("--config", type=str, default="config/experiments/setup.yaml")
    parser.add_argument("--open-config", type=str, default="config/baselines/open_marl.yaml")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--save", type=str, default="checkpoints/open_marl/default.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    open_cfg_raw = load_yaml(ROOT / args.open_config)
    exp_cfg = load_experiment_config(ROOT / args.config)
    base_config = build_experiment_base_config(exp_cfg, timestamped=False, write_manifest=False)

    open_cfg = OpenMarlConfig.from_dict(open_cfg_raw)
    reward_raw = dict(open_cfg_raw.get("reward", {}))
    train_raw = open_cfg_raw.get("training", {})
    stage_cfg = train_raw.get("stages", {})

    if args.stage == 1:
        s1 = stage_cfg.get("stage1", {})
        reward_raw["use_smoothness"] = False
        total_steps_default = int(s1.get("total_steps", train_raw.get("total_steps", 10_500_000)))
        global_start = 0
        resume_path = args.resume
    else:
        s2 = stage_cfg.get("stage2", {})
        reward_raw["use_smoothness"] = True
        total_steps_default = int(s2.get("total_steps", 4_500_000))
        global_start = int(s2.get("global_step_start", 10_500_000))
        resume_path = args.resume or str(ROOT / "checkpoints/open_marl/best.pt")

    reward_cfg = RewardConfig.from_dict(reward_raw)
    mappo_cfg = MAPPOConfig.from_dict(open_cfg_raw.get("mappo", {}))
    aeg_cfg = AEGConfig.from_dict(open_cfg_raw.get("aeg", {}))
    curriculum_cfg = CurriculumConfig.from_dict(open_cfg_raw.get("curriculum", {}))

    total_steps = args.total_steps or total_steps_default
    n_envs = args.n_envs or int(train_raw.get("n_envs", 64))
    scenarios = list(train_raw.get("scenarios", ["free", "random_obstacles", "single_exit"]))
    seed = args.seed if args.seed is not None else int(train_raw.get("seed", 42))
    log_every = int(train_raw.get("log_every", 1000))
    episode_max_steps = int(train_raw.get("episode_max_steps", 1200))
    base_config = deep_merge(base_config, {"max_steps": episode_max_steps})
    default_eval_every = episode_max_steps * n_envs
    eval_every_raw = train_raw.get("eval_every")
    eval_every = (
        args.eval_every
        if args.eval_every is not None
        else int(eval_every_raw) if eval_every_raw is not None else default_eval_every
    )
    eval_episodes = args.eval_episodes if args.eval_episodes is not None else int(train_raw.get("eval_episodes", 32))

    save_path = Path(args.save)
    if not save_path.is_absolute():
        save_path = ROOT / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training OPEN MARL stage {args.stage}: total_steps={total_steps}, n_envs={n_envs}")
    print(f"scenarios={scenarios}, smoothness={reward_cfg.use_smoothness}")
    print(f"episode_max_steps={episode_max_steps}, eval_every={eval_every}")
    print(f"Save checkpoint to: {save_path}")
    if resume_path:
        print(f"Resume from: {resume_path}")
    print(f"Best checkpoint (strict capture): {save_path.parent / 'best.pt'}")

    train_open_marl(
        total_steps=global_start + total_steps if args.stage == 2 else total_steps,
        n_envs=n_envs,
        base_config=base_config,
        open_cfg=open_cfg,
        reward_cfg=reward_cfg,
        mappo_cfg=mappo_cfg,
        scenario_names=scenarios,
        save_path=str(save_path),
        device=args.device,
        seed=seed,
        log_every=log_every,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        episode_max_steps=episode_max_steps,
        aeg_cfg=aeg_cfg,
        curriculum_cfg=curriculum_cfg,
        resume_path=resume_path,
        global_step_start=global_start if args.stage == 2 else 0,
    )
    print(f"Done. Checkpoint saved to {save_path}")


if __name__ == "__main__":
    main()
