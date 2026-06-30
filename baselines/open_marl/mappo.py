"""MAPPO trainer for OPEN MARL (PPO + EPN supervised loss)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    from tqdm import tqdm
except ImportError as e:
    raise ImportError(
        "OPEN MARL requires PyTorch and tqdm. Install with: pip install -r requirements-rl.txt"
    ) from e

from baselines.open_marl.aeg import AEGConfig, CurriculumConfig, TaskSeed
from baselines.open_marl.env import OpenMarlVecEnv
from baselines.open_marl.networks import OpenMARLPolicy
from baselines.open_marl.observation import OpenMarlConfig
from baselines.open_marl.reward import RewardConfig


def _array_to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert numpy array to torch without relying on broken numpy C-API in some envs."""
    return torch.tensor(np.asarray(arr, dtype=np.float64).tolist(), dtype=torch.float32, device=device)


@dataclass
class MAPPOConfig:
    train_every: int = 2048
    num_minibatches: int = 16
    ppo_epochs: int = 4
    clip_param: float = 0.1
    value_clip_param: float = 0.2
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 0.995
    max_grad_norm: float = 10.0
    normalize_advantages: bool = True
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    epn_loss_coef: float = 1.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MAPPOConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        batch_count = batch.size
        if batch_count == 0:
            return
        batch_mean = float(batch.mean())
        batch_var = float(batch.var())
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        std = max(float(np.sqrt(self.var)), 1e-8)
        normed = (x - self.mean) / std
        return np.clip(normed, -10.0, 10.0)


def fill_epn_targets(
    buffer: RolloutBuffer,
    open_cfg: OpenMarlConfig,
) -> None:
    """Label EPN with true future evader positions (paper Sec IV-B)."""
    size = buffer.capacity if buffer.full else buffer.ptr
    k = open_cfg.future_prediction_step
    n_envs = buffer.n_envs

    for t in range(size):
        scale = max(float(buffer.arena_scales[t]), 1e-6)
        centroid = buffer.centroids[t]
        targets = np.zeros(open_cfg.epn_target_dim, dtype=np.float32)
        for ahead in range(1, k + 1):
            t_future = t + ahead * n_envs
            if t_future >= size:
                break
            rel = (buffer.evader_positions[t_future] - centroid) / scale
            targets[2 * (ahead - 1) : 2 * ahead] = rel.astype(np.float32)
        buffer.epn_targets[t] = targets


class RolloutBuffer:
    def __init__(
        self,
        capacity: int,
        n_envs: int,
        n_agents: int,
        obs_dim: int,
        hist_len: int,
        epn_input_dim: int,
        epn_target_dim: int,
        action_dim: int,
    ) -> None:
        self.capacity = capacity
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.ptr = 0
        self.full = False

        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.history = np.zeros((capacity, hist_len, epn_input_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((capacity, n_agents), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.epn_targets = np.zeros((capacity, epn_target_dim), dtype=np.float32)
        self.centroids = np.zeros((capacity, 2), dtype=np.float32)
        self.evader_positions = np.zeros((capacity, 2), dtype=np.float32)
        self.arena_scales = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        history: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        centroid: np.ndarray,
        evader_pos: np.ndarray,
        arena_scale: float,
    ) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.history[i] = history
        self.actions[i] = actions
        self.log_probs[i] = log_probs
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.values[i] = value
        self.centroids[i] = centroid
        self.evader_positions[i] = evader_pos
        self.arena_scales[i] = arena_scale
        self.ptr += 1
        if self.ptr >= self.capacity:
            self.full = True
            self.ptr = 0

    def compute_gae(self, last_value: float, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
        size = self.capacity if self.full else self.ptr
        advantages = np.zeros(size, dtype=np.float32)
        returns = np.zeros(size, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(size)):
            next_non_terminal = 1.0 - self.dones[t]
            next_value = last_value if t == size - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]
        return advantages, returns

    def iterate_minibatches(self, n_minibatches: int):
        size = self.capacity if self.full else self.ptr
        indices = np.arange(size)
        np.random.shuffle(indices)
        batch_size = max(1, size // n_minibatches)
        for start in range(0, size, batch_size):
            mb = indices[start : start + batch_size]
            yield mb


class MAPPOTrainer:
    def __init__(
        self,
        policy: OpenMARLPolicy,
        mappo_cfg: MAPPOConfig,
        device: str = "cpu",
    ) -> None:
        self.policy = policy
        self.cfg = mappo_cfg
        self.device = torch.device(device)
        self.reward_norm = RunningMeanStd()
        self.policy.to(self.device)
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": list(policy.epn.parameters())
                    + list(policy.encoder.parameters())
                    + list(policy.actor.parameters()),
                    "lr": mappo_cfg.actor_lr,
                },
                {
                    "params": list(policy.critic_body.parameters()) + list(policy.critic_head.parameters()),
                    "lr": mappo_cfg.critic_lr,
                },
            ]
        )

    def collect_step(
        self,
        obs: np.ndarray,
        history: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """obs (n_envs, n_agents, obs_dim) -> actions, log_probs, values."""
        n_envs, n_agents, _ = obs.shape
        actions = np.zeros((n_envs, n_agents, self.policy.action_dim), dtype=np.float32)
        log_probs = np.zeros((n_envs, n_agents), dtype=np.float32)
        values = np.zeros(n_envs, dtype=np.float32)

        with torch.no_grad():
            for e in range(n_envs):
                obs_t = _array_to_tensor(obs[e], self.device).unsqueeze(0)
                hist_t = _array_to_tensor(history[e], self.device).unsqueeze(0)
                act, lp, _, _ = self.policy.actor_forward_batch(obs_t, hist_t, deterministic=deterministic)
                v = self.policy.critic_forward(obs_t, hist_t)
                actions[e] = np.array(act.cpu().tolist()[0], dtype=np.float32)
                log_probs[e] = np.array(lp.cpu().tolist()[0], dtype=np.float32)
                values[e] = v.cpu().item()
        return actions, log_probs, values

    def update(self, buffer: RolloutBuffer, last_value: float) -> dict[str, float]:
        fill_epn_targets(buffer, self.policy.cfg)
        raw_rewards = buffer.rewards[: buffer.ptr if not buffer.full else buffer.capacity].copy()
        self.reward_norm.update(raw_rewards)
        buffer.rewards[: buffer.ptr if not buffer.full else buffer.capacity] = self.reward_norm.normalize(
            raw_rewards
        )

        advantages, returns = buffer.compute_gae(last_value, self.cfg.gamma, self.cfg.gae_lambda)
        if self.cfg.normalize_advantages:
            adv_mean = float(advantages.mean())
            adv_std = float(advantages.std())
            if np.isfinite(adv_std) and adv_std > 1e-8:
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            else:
                advantages = advantages - adv_mean
        advantages = np.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        size = buffer.capacity if buffer.full else buffer.ptr
        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "epn_loss": 0.0,
            "entropy": 0.0,
            "skipped_updates": 0.0,
        }
        n_updates = 0

        for _ in range(self.cfg.ppo_epochs):
            for mb in buffer.iterate_minibatches(self.cfg.num_minibatches):
                obs_mb = _array_to_tensor(buffer.obs[mb], self.device)
                hist_mb = _array_to_tensor(buffer.history[mb], self.device)
                act_mb = _array_to_tensor(buffer.actions[mb], self.device)
                old_lp_mb = _array_to_tensor(buffer.log_probs[mb], self.device)
                ret_mb = torch.tensor(returns[mb].tolist(), dtype=torch.float32, device=self.device)
                adv_mb = torch.tensor(advantages[mb].tolist(), dtype=torch.float32, device=self.device)
                epn_tgt_mb = _array_to_tensor(buffer.epn_targets[mb], self.device)

                feats, pred = self.policy.encode_agents(obs_mb, hist_mb)
                if not torch.isfinite(feats).all() or not torch.isfinite(pred).all():
                    stats["skipped_updates"] += 1.0
                    continue

                _, n_agents, _ = obs_mb.shape
                policy_losses = []
                entropies = []
                for a in range(n_agents):
                    act_a = act_mb[:, a]
                    old_lp_a = old_lp_mb[:, a]
                    lp, ent = self.policy.actor.evaluate(feats[:, a], act_a)
                    if not torch.isfinite(lp).all():
                        stats["skipped_updates"] += 1.0
                        policy_losses = []
                        break
                    log_ratio = torch.clamp(lp - old_lp_a, -20.0, 20.0)
                    ratio = torch.exp(log_ratio)
                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param) * adv_mb
                    policy_losses.append(-torch.min(surr1, surr2).mean())
                    entropies.append(ent.mean())
                if not policy_losses:
                    continue

                policy_loss = torch.stack(policy_losses).mean()
                value = self.policy.critic_from_encoded(feats, pred)
                old_value_mb = _array_to_tensor(buffer.values[mb], self.device)
                value_clipped = old_value_mb + torch.clamp(
                    value - old_value_mb,
                    -self.cfg.value_clip_param,
                    self.cfg.value_clip_param,
                )
                value_loss = torch.max(
                    (value - ret_mb).pow(2),
                    (value_clipped - ret_mb).pow(2),
                ).mean()
                epn_loss = nn.functional.mse_loss(pred, epn_tgt_mb)

                loss = (
                    policy_loss
                    + 0.5 * value_loss
                    - self.cfg.entropy_coef * torch.stack(entropies).mean()
                    + self.cfg.epn_loss_coef * epn_loss
                )
                if not torch.isfinite(loss):
                    stats["skipped_updates"] += 1.0
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    stats["skipped_updates"] += 1.0
                    self.optimizer.zero_grad()
                    continue
                self.optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["epn_loss"] += epn_loss.item()
                stats["entropy"] += torch.stack(entropies).mean().item()
                n_updates += 1

        for k in stats:
            stats[k] /= max(n_updates, 1)
        return stats


def evaluate_capture_rate(
    trainer: MAPPOTrainer,
    base_config: dict[str, Any],
    open_cfg: OpenMarlConfig,
    reward_cfg: RewardConfig,
    scenario_names: list[str],
    n_episodes: int = 32,
    n_envs: int = 8,
    seed: int = 12345,
    max_steps: int | None = None,
    aeg_cfg: AEGConfig | None = None,
    curriculum_cfg: CurriculumConfig | None = None,
    capture_mode: str = "strict",
) -> tuple[float, int, int]:
    """Deterministic rollouts; capture_mode in {'strict','loose'}."""
    eval_base = dict(base_config)
    if max_steps is not None:
        eval_base["max_steps"] = max_steps
    eval_reward = reward_cfg.with_overrides({"capture_mode": capture_mode})
    eval_env = OpenMarlVecEnv(
        min(n_envs, n_episodes),
        eval_base,
        open_cfg,
        eval_reward,
        scenario_names,
        seed=seed,
        aeg_cfg=aeg_cfg,
        curriculum_cfg=curriculum_cfg,
        global_step=10**12,
        total_steps=10**12,
    )
    obs, history = eval_env.reset()
    captures = 0
    completed = 0
    max_eval_steps = int(eval_base.get("max_steps", 1200)) * n_episodes * 4

    trainer.policy.eval()
    try:
        eval_steps = 0
        while completed < n_episodes and eval_steps < max_eval_steps:
            actions, _, _ = trainer.collect_step(obs, history, deterministic=True)
            obs, history, _, _dones, info = eval_env.step(actions)
            eval_steps += 1
            for e in range(eval_env.n_envs):
                if info["episode_done"][e]:
                    completed += 1
                    if info["captured"][e]:
                        captures += 1
                    if completed >= n_episodes:
                        break
    finally:
        trainer.policy.train()

    return captures / max(completed, 1), completed, captures


def train_open_marl(
    total_steps: int,
    n_envs: int,
    base_config: dict[str, Any],
    open_cfg: OpenMarlConfig,
    reward_cfg: RewardConfig,
    mappo_cfg: MAPPOConfig,
    scenario_names: list[str],
    save_path: str,
    device: str = "cpu",
    seed: int = 42,
    log_every: int = 1000,
    eval_every: int = 19200,
    eval_episodes: int = 32,
    episode_max_steps: int | None = None,
    eval_scenarios: list[str] | None = None,
    aeg_cfg: AEGConfig | None = None,
    curriculum_cfg: CurriculumConfig | None = None,
    resume_path: str | None = None,
    global_step_start: int = 0,
) -> OpenMARLPolicy:
    train_config = dict(base_config)
    if episode_max_steps is not None:
        train_config["max_steps"] = episode_max_steps

    aeg_cfg = aeg_cfg or AEGConfig()
    curriculum_cfg = curriculum_cfg or CurriculumConfig.from_dict({})

    env = OpenMarlVecEnv(
        n_envs,
        train_config,
        open_cfg,
        reward_cfg,
        scenario_names,
        seed=seed,
        aeg_cfg=aeg_cfg,
        curriculum_cfg=curriculum_cfg,
        global_step=global_step_start,
        total_steps=total_steps,
    )

    if resume_path and Path(resume_path).exists():
        policy, meta = OpenMARLPolicy.load_checkpoint(resume_path, device=device)
        policy.train()
        ckpt_step = int(meta.get("global_step", 0))
        if ckpt_step > global_step_start:
            global_step_start = ckpt_step
    else:
        action_scale = float(train_config.get("pursuer_amax", 3.2))
        policy = OpenMARLPolicy(open_cfg, action_dim=2, action_scale=action_scale)

    trainer = MAPPOTrainer(policy, mappo_cfg, device=device)

    obs, history = env.reset()
    buffer = RolloutBuffer(
        mappo_cfg.train_every,
        n_envs,
        open_cfg.n_pursuers,
        open_cfg.obs_dim,
        open_cfg.history_step,
        open_cfg.epn_input_dim,
        open_cfg.epn_target_dim,
        2,
    )

    global_step = global_step_start
    ep_returns: list[float] = []
    ep_return_accum = np.zeros(n_envs, dtype=np.float64)
    window_captures = 0
    window_episodes = 0
    window_reward_sum = 0.0
    window_reward_count = 0
    window_min_dist_sum = 0.0
    window_min_dist_count = 0
    window_g_max_sum = 0.0
    window_g_max_count = 0
    last_eval_step = -1
    last_capture_rate = 0.0
    best_capture_rate = -1.0
    best_path = str(Path(save_path).with_name("best.pt"))
    latest_path = str(Path(save_path).with_name("latest.pt"))
    eval_scenarios = eval_scenarios or scenario_names
    ep_max = int(train_config.get("max_steps", 1200))
    episode_seeds: list[TaskSeed | None] = [None] * n_envs

    pbar = tqdm(total=total_steps, initial=global_step_start, desc="OPEN MARL", unit="step", dynamic_ncols=True)

    while global_step < total_steps:
        env.set_training_progress(global_step, total_steps)
        actions, log_probs, values = trainer.collect_step(obs, history)
        next_obs, next_history, rewards, _dones, info = env.step(actions)

        for e in range(n_envs):
            if episode_seeds[e] is None:
                episode_seeds[e] = env.task_seed(e)

            buffer.add(
                obs[e],
                history[e],
                actions[e],
                log_probs[e],
                float(rewards[e]),
                bool(info["episode_done"][e]),
                float(values[e]),
                info["centroids"][e],
                info["evader_positions"][e],
                float(info["arena_scales"][e]),
            )
            ep_return_accum[e] += float(rewards[e])
            window_reward_sum += float(rewards[e])
            window_reward_count += 1
            dists = np.linalg.norm(env.pursuers[e] - env.evader[e][None, :], axis=1)
            window_min_dist_sum += float(np.min(dists))
            window_min_dist_count += 1
            window_g_max_sum += float(info["g_max"][e])
            window_g_max_count += 1

            if info["episode_done"][e]:
                window_episodes += 1
                if info["captured"][e]:
                    window_captures += 1
                ep_returns.append(float(ep_return_accum[e]))
                ep_return_accum[e] = 0.0

                if episode_seeds[e] is not None:
                    phase = env._current_phase()
                    if phase.capture_mode == "loose":
                        rate = 1.0 if info.get("loose_captured", info["captured"])[e] else 0.0
                    else:
                        rate = 1.0 if info.get("strict_captured", info["captured"])[e] else 0.0
                    env.aeg_sampler.update_archive(episode_seeds[e], rate)
                episode_seeds[e] = None

        obs, history = next_obs, next_history
        global_step += n_envs
        pbar.update(n_envs)

        stats: dict[str, float] = {}
        if buffer.ptr >= mappo_cfg.train_every or buffer.full:
            with torch.no_grad():
                obs_t = _array_to_tensor(obs[0:1], trainer.device)
                hist_t = _array_to_tensor(history[0:1], trainer.device)
                last_v = trainer.policy.critic_forward(obs_t, hist_t).cpu().item()
            stats = trainer.update(buffer, last_v)
            buffer = RolloutBuffer(
                mappo_cfg.train_every,
                n_envs,
                open_cfg.n_pursuers,
                open_cfg.obs_dim,
                open_cfg.history_step,
                open_cfg.epn_input_dim,
                open_cfg.epn_target_dim,
                2,
            )

        if global_step >= eval_every and global_step // eval_every > last_eval_step // eval_every:
            last_eval_step = global_step
            env_steps_per_iter = global_step // max(n_envs, 1)
            last_capture_rate, eval_n, eval_cap = evaluate_capture_rate(
                trainer,
                train_config,
                open_cfg,
                reward_cfg,
                eval_scenarios,
                n_episodes=eval_episodes,
                n_envs=min(16, eval_episodes),
                seed=seed + global_step,
                max_steps=ep_max,
                aeg_cfg=aeg_cfg,
                curriculum_cfg=curriculum_cfg,
                capture_mode="strict",
            )
            loose_rate, _, loose_cap = evaluate_capture_rate(
                trainer,
                train_config,
                open_cfg,
                reward_cfg,
                ["free"],
                n_episodes=min(16, eval_episodes),
                n_envs=min(8, eval_episodes),
                seed=seed + global_step + 1,
                max_steps=ep_max,
                capture_mode="loose",
            )
            train_rate = window_captures / max(window_episodes, 1)
            mean_ret = float(np.mean(ep_returns[-50:])) if ep_returns else 0.0
            mean_step_r = window_reward_sum / max(window_reward_count, 1)
            mean_min_dist = window_min_dist_sum / max(window_min_dist_count, 1)
            mean_g_max = window_g_max_sum / max(window_g_max_count, 1)
            phase = env._current_phase().name
            postfix: dict[str, Any] = {
                "capture": f"{last_capture_rate:.1%}",
                "train_cap": f"{train_rate:.1%}",
                "min_dist": f"{mean_min_dist:.1f}",
                "phase": phase,
            }
            if stats:
                postfix["epn"] = f"{stats['epn_loss']:.3f}"
            pbar.set_postfix(postfix)
            tqdm.write(
                f"[eval @ global_step={global_step} (~{env_steps_per_iter} steps/env)] "
                f"strict_capture={last_capture_rate:.1%} ({eval_cap}/{eval_n}) "
                f"loose_capture={loose_rate:.1%} ({loose_cap}/{min(16, eval_episodes)}) "
                f"train_rollout={train_rate:.1%} (n={window_episodes}) "
                f"mean_ep_return={mean_ret:.2f} step_reward={mean_step_r:.3f} "
                f"min_dist={mean_min_dist:.2f}m G_max={np.degrees(mean_g_max):.1f}deg "
                f"phase={phase} episode_max={ep_max}"
                + (
                    f" policy_loss={stats['policy_loss']:.4f} value_loss={stats['value_loss']:.4f} "
                    f"epn_loss={stats['epn_loss']:.4f}"
                    if stats
                    else ""
                )
            )
            policy.save_checkpoint(
                latest_path,
                extra={"capture_rate": last_capture_rate, "global_step": global_step},
            )
            if last_capture_rate > best_capture_rate:
                best_capture_rate = last_capture_rate
                policy.save_checkpoint(
                    best_path,
                    extra={"capture_rate": best_capture_rate, "global_step": global_step},
                )
                tqdm.write(f"[best] strict_capture={best_capture_rate:.1%} -> {best_path}")
            window_captures = 0
            window_episodes = 0
            window_reward_sum = 0.0
            window_reward_count = 0
            window_min_dist_sum = 0.0
            window_min_dist_count = 0
            window_g_max_sum = 0.0
            window_g_max_count = 0
        elif global_step % log_every < n_envs and stats:
            mean_ret = float(np.mean(ep_returns[-20:])) if ep_returns else 0.0
            pbar.set_postfix(
                ret=f"{mean_ret:.2f}",
                capture=f"{last_capture_rate:.1%}",
                pi_loss=f"{stats['policy_loss']:.3f}",
            )

    pbar.close()
    policy.save_checkpoint(save_path, extra={"global_step": global_step})
    if best_capture_rate >= 0:
        tqdm.write(f"Best strict capture_rate={best_capture_rate:.1%} saved to {best_path}")
    return policy
