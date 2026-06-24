"""MAPPO trainer for OPEN MARL (PPO + EPN supervised loss)."""

from __future__ import annotations

from dataclasses import dataclass
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

from baselines.open_marl.env import OpenMarlVecEnv
from baselines.open_marl.networks import OpenMARLPolicy
from baselines.open_marl.observation import OpenMarlConfig
from baselines.open_marl.reward import RewardConfig


@dataclass
class MAPPOConfig:
    train_every: int = 64
    num_minibatches: int = 16
    ppo_epochs: int = 4
    tp_epochs: int = 1
    clip_param: float = 0.1
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 0.995
    max_grad_norm: float = 10.0
    normalize_advantages: bool = True
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    tp_lr: float = 5e-4
    epn_loss_coef: float = 1.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MAPPOConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RolloutBuffer:
    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        hist_len: int,
        epn_input_dim: int,
        epn_target_dim: int,
        action_dim: int,
    ) -> None:
        self.capacity = capacity
        self.n_agents = n_agents
        self.ptr = 0
        self.full = False

        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.obs_all = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.history = np.zeros((capacity, hist_len, epn_input_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((capacity, n_agents), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.epn_targets = np.zeros((capacity, epn_target_dim), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        history: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        epn_targets: np.ndarray,
    ) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.obs_all[i] = obs
        self.history[i] = history
        self.actions[i] = actions
        self.log_probs[i] = log_probs
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.values[i] = value
        self.epn_targets[i] = epn_targets
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
        self.policy.to(self.device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": list(policy.epn.parameters()) + list(policy.encoder.parameters()) + list(policy.actor_body.parameters()) + list(policy.actor_mean.parameters()) + [policy.log_std], "lr": mappo_cfg.actor_lr},
                {"params": list(policy.critic_body.parameters()) + list(policy.critic_head.parameters()), "lr": mappo_cfg.critic_lr},
            ]
        )

    def collect_step(
        self,
        obs: np.ndarray,
        history: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """obs (n_envs, n_agents, obs_dim) -> actions, log_probs, value, epn unused."""
        n_envs, n_agents, _ = obs.shape
        actions = np.zeros((n_envs, n_agents, self.policy.action_dim), dtype=np.float32)
        log_probs = np.zeros((n_envs, n_agents), dtype=np.float32)

        values: list[float] = []
        with torch.no_grad():
            for e in range(n_envs):
                obs_t = torch.tensor(obs[e], device=self.device)
                hist_t = torch.tensor(history[e], device=self.device).unsqueeze(0)
                hist_exp = hist_t.expand(n_agents, -1, -1)
                for a in range(n_agents):
                    act, lp, _, _ = self.policy.actor_forward(
                        obs_t[a : a + 1],
                        hist_exp[a : a + 1],
                        deterministic=deterministic,
                    )
                    actions[e, a] = act.cpu().numpy()[0]
                    log_probs[e, a] = lp.cpu().numpy()[0]
                v = self.policy.critic_forward(obs_t.unsqueeze(0), hist_t)
                values.append(v.cpu().item())
        return actions, log_probs, float(np.mean(values)), np.array(values, dtype=np.float32)

    def update(self, buffer: RolloutBuffer, last_value: float) -> dict[str, float]:
        advantages, returns = buffer.compute_gae(last_value, self.cfg.gamma, self.cfg.gae_lambda)
        if self.cfg.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        size = buffer.capacity if buffer.full else buffer.ptr
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "epn_loss": 0.0, "entropy": 0.0}
        n_updates = 0

        for _ in range(self.cfg.ppo_epochs):
            for mb in buffer.iterate_minibatches(self.cfg.num_minibatches):
                obs_mb = torch.tensor(buffer.obs[mb], device=self.device)
                hist_mb = torch.tensor(buffer.history[mb], device=self.device)
                act_mb = torch.tensor(buffer.actions[mb], device=self.device)
                old_lp_mb = torch.tensor(buffer.log_probs[mb], device=self.device)
                ret_mb = torch.tensor(returns[mb], device=self.device)
                adv_mb = torch.tensor(advantages[mb], device=self.device)
                epn_tgt_mb = torch.tensor(buffer.epn_targets[mb], device=self.device)

                bsz, n_agents, obs_dim = obs_mb.shape
                policy_losses = []
                entropies = []
                for a in range(n_agents):
                    obs_a = obs_mb[:, a]
                    act_a = act_mb[:, a]
                    old_lp_a = old_lp_mb[:, a]
                    lp, ent, val, pred = self.policy.evaluate_actions(
                        obs_a, obs_mb, hist_mb, act_a
                    )
                    ratio = torch.exp(lp - old_lp_a)
                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param) * adv_mb
                    policy_losses.append(-torch.min(surr1, surr2).mean())
                    entropies.append(ent.mean())

                policy_loss = torch.stack(policy_losses).mean()
                value = self.policy.critic_forward(obs_mb, hist_mb)
                value_loss = nn.functional.mse_loss(value, ret_mb)
                _, _, _, pred_all = self.policy.evaluate_actions(
                    obs_mb[:, 0], obs_mb, hist_mb, act_mb[:, 0]
                )
                epn_loss = nn.functional.mse_loss(pred_all, epn_tgt_mb)

                loss = (
                    policy_loss
                    + 0.5 * value_loss
                    - self.cfg.entropy_coef * torch.stack(entropies).mean()
                    + self.cfg.epn_loss_coef * epn_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
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
) -> tuple[float, int, int]:
    """Run deterministic rollouts; return (capture_rate, n_episodes, n_captures)."""
    eval_base = dict(base_config)
    if max_steps is not None:
        eval_base["max_steps"] = max_steps
    eval_env = OpenMarlVecEnv(
        min(n_envs, n_episodes),
        eval_base,
        open_cfg,
        reward_cfg,
        scenario_names,
        seed=seed,
    )
    obs, history = eval_env.reset()
    captures = 0
    completed = 0
    max_eval_steps = int(eval_base.get("max_steps", 1200)) * n_episodes * 4

    trainer.policy.eval()
    try:
        eval_steps = 0
        while completed < n_episodes and eval_steps < max_eval_steps:
            actions, _, _, _ = trainer.collect_step(obs, history, deterministic=True)
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
    eval_every: int = 1000,
    eval_episodes: int = 32,
    episode_max_steps: int | None = None,
    eval_scenarios: list[str] | None = None,
) -> OpenMARLPolicy:
    train_config = dict(base_config)
    if episode_max_steps is not None:
        train_config["max_steps"] = episode_max_steps

    env = OpenMarlVecEnv(n_envs, train_config, open_cfg, reward_cfg, scenario_names, seed=seed)
    policy = OpenMARLPolicy(open_cfg, action_dim=2)
    trainer = MAPPOTrainer(policy, mappo_cfg, device=device)

    obs, history = env.reset()
    buffer = RolloutBuffer(
        mappo_cfg.train_every,
        open_cfg.n_pursuers,
        open_cfg.obs_dim,
        open_cfg.history_step,
        open_cfg.epn_input_dim,
        open_cfg.epn_target_dim,
        2,
    )

    global_step = 0
    ep_returns: list[float] = []
    ep_return_accum = np.zeros(n_envs, dtype=np.float64)
    train_captures = 0
    train_episodes = 0
    window_captures = 0
    window_episodes = 0
    window_reward_sum = 0.0
    window_reward_count = 0
    window_min_dist_sum = 0.0
    window_min_dist_count = 0
    last_eval_step = -1
    last_capture_rate = 0.0
    eval_scenarios = eval_scenarios or scenario_names
    ep_max = int(train_config.get("max_steps", 1200))

    pbar = tqdm(total=total_steps, desc="OPEN MARL", unit="step", dynamic_ncols=True)

    while global_step < total_steps:
        actions, log_probs, _, values = trainer.collect_step(obs, history)
        next_obs, next_history, rewards, _dones, info = env.step(actions)

        for e in range(n_envs):
            buffer.add(
                obs[e],
                history[e],
                actions[e],
                log_probs[e],
                float(rewards[e]),
                bool(info["episode_done"][e]),
                float(values[e]) if e < len(values) else 0.0,
                info["epn_targets"][e],
            )
            ep_return_accum[e] += float(rewards[e])
            window_reward_sum += float(rewards[e])
            window_reward_count += 1
            dists = np.linalg.norm(env.pursuers[e] - env.evader[e][None, :], axis=1)
            window_min_dist_sum += float(np.min(dists))
            window_min_dist_count += 1

            if info["episode_done"][e]:
                train_episodes += 1
                window_episodes += 1
                if info["captured"][e]:
                    train_captures += 1
                    window_captures += 1
                ep_returns.append(float(ep_return_accum[e]))
                ep_return_accum[e] = 0.0

        obs, history = next_obs, next_history
        global_step += n_envs
        pbar.update(n_envs)

        stats: dict[str, float] = {}
        if buffer.ptr >= mappo_cfg.train_every or buffer.full:
            with torch.no_grad():
                obs_t = torch.tensor(obs[0:1], device=trainer.device)
                hist_t = torch.tensor(history[0:1], device=trainer.device)
                last_v = trainer.policy.critic_forward(obs_t, hist_t).cpu().item()
            stats = trainer.update(buffer, last_v)
            buffer = RolloutBuffer(
                mappo_cfg.train_every,
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
            )
            train_rate = window_captures / max(window_episodes, 1)
            mean_ret = float(np.mean(ep_returns[-50:])) if ep_returns else 0.0
            mean_step_r = window_reward_sum / max(window_reward_count, 1)
            mean_min_dist = window_min_dist_sum / max(window_min_dist_count, 1)
            postfix: dict[str, Any] = {
                "capture": f"{last_capture_rate:.1%}",
                "train_cap": f"{train_rate:.1%}",
                "min_dist": f"{mean_min_dist:.1f}",
            }
            if stats:
                postfix["pi_loss"] = f"{stats['policy_loss']:.3f}"
            pbar.set_postfix(postfix)
            tqdm.write(
                f"[eval @ global_step={global_step} (~{env_steps_per_iter} steps/env)] "
                f"capture_rate={last_capture_rate:.1%} ({eval_cap}/{eval_n}) "
                f"train_rollout={train_rate:.1%} (n={window_episodes}) "
                f"mean_ep_return={mean_ret:.2f} step_reward={mean_step_r:.3f} "
                f"min_dist={mean_min_dist:.2f}m episode_max={ep_max}"
                + (
                    f" policy_loss={stats['policy_loss']:.4f} value_loss={stats['value_loss']:.4f}"
                    if stats
                    else ""
                )
            )
            window_captures = 0
            window_episodes = 0
            window_reward_sum = 0.0
            window_reward_count = 0
            window_min_dist_sum = 0.0
            window_min_dist_count = 0
        elif global_step % log_every < n_envs and stats:
            mean_ret = float(np.mean(ep_returns[-20:])) if ep_returns else 0.0
            pbar.set_postfix(
                ret=f"{mean_ret:.2f}",
                capture=f"{last_capture_rate:.1%}",
                pi_loss=f"{stats['policy_loss']:.3f}",
            )

    pbar.close()
    policy.save_checkpoint(save_path)
    return policy
