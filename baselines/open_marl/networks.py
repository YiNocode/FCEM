"""EPN + attention actor-critic for OPEN MARL (paper Fig.3 adaptation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise ImportError(
        "OPEN MARL requires PyTorch. Install with: pip install -r requirements-rl.txt"
    ) from e

from baselines.open_marl.observation import OpenMarlConfig, split_observation


def _mlp(sizes: list[int], layer_norm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if layer_norm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class EvaderPredictor(nn.Module):
    """LSTM predicts future K-step evader relative displacements."""

    def __init__(self, cfg: OpenMarlConfig, hidden_size: int = 128) -> None:
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(cfg.epn_input_dim, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, cfg.epn_target_dim)

    def forward(
        self,
        history: torch.Tensor,
        h: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        out, h_new = self.lstm(history, h)
        pred = self.head(out[:, -1])
        return pred, h_new


class ComponentObservationEncoder(nn.Module):
    """Paper Fig.3: separate MLPs per component, attention, self residual."""

    def __init__(
        self,
        cfg: OpenMarlConfig,
        embed_dim: int = 128,
        n_heads: int = 4,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim
        n_other = cfg.n_pursuers - 1
        n_ob = cfg.obs_max_obstacles

        self.self_encoder = _mlp([cfg.self_encoder_dim, embed_dim], layer_norm=True)
        self.other_encoder = _mlp([2, embed_dim], layer_norm=True)
        self.obstacle_encoder = _mlp([2, embed_dim], layer_norm=True)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.out_mlp = _mlp([embed_dim, hidden, hidden], layer_norm=True)
        self.n_other = n_other
        self.n_ob = n_ob

    def forward(self, obs: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, obs_dim) base observation without pred in vector
        pred: (B, K*2) EPN prediction
        returns: (B, hidden) feature h
        """
        o_self, o_other, o_ob = split_observation(obs, self.cfg)
        o_self_full = torch.cat([o_self, pred], dim=-1)
        h_self = self.self_encoder(o_self_full)

        tokens = [h_self.unsqueeze(1)]
        if self.n_other > 0:
            other_chunks = o_other.reshape(o_other.shape[0], self.n_other, 2)
            for j in range(self.n_other):
                tokens.append(self.other_encoder(other_chunks[:, j]).unsqueeze(1))
        if self.n_ob > 0:
            ob_chunks = o_ob.reshape(o_ob.shape[0], self.n_ob, 2)
            for j in range(self.n_ob):
                tokens.append(self.obstacle_encoder(ob_chunks[:, j]).unsqueeze(1))

        x = torch.cat(tokens, dim=1)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = x + self.ff(x)
        h_attn = x.mean(dim=1)
        h = self.out_mlp(h_attn + h_self)
        return h


class TanhGaussianActor(nn.Module):
    """Gaussian in pre-tanh space; actions scaled to [-action_scale, action_scale]."""

    def __init__(self, feature_dim: int, action_dim: int, hidden_units: list[int], action_scale: float) -> None:
        super().__init__()
        self.action_scale = action_scale
        self.body = _mlp([feature_dim, *hidden_units], layer_norm=True)
        self.mean = nn.Linear(hidden_units[-1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def _distribution(self, features: torch.Tensor) -> torch.distributions.Normal:
        x = self.body(features)
        mean = self.mean(x)
        # Clamp log_std to avoid vanishing/exploding scales during long PPO runs.
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def sample(
        self,
        features: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._distribution(features)
        if deterministic:
            pre_tanh = dist.mean
            log_prob = torch.zeros(features.shape[0], device=features.device)
            entropy = torch.zeros_like(log_prob)
        else:
            pre_tanh = dist.rsample()
            log_prob = dist.log_prob(pre_tanh).sum(-1)
            log_prob = log_prob - torch.log(
                self.action_scale * (1.0 - torch.tanh(pre_tanh).pow(2)).clamp(min=1e-6)
            ).sum(-1)
            entropy = dist.entropy().sum(-1)

        action = torch.tanh(pre_tanh) * self.action_scale
        return action, log_prob, entropy

    def evaluate(self, features: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._distribution(features)
        pre_tanh = torch.atanh(torch.clamp(actions / self.action_scale, -0.999, 0.999))
        log_prob = dist.log_prob(pre_tanh).sum(-1)
        log_prob = log_prob - torch.log(
            self.action_scale * (1.0 - torch.tanh(pre_tanh).pow(2)).clamp(min=1e-6)
        ).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class OpenMARLPolicy(nn.Module):
    """Combined EPN + component encoder + actor-critic with parameter sharing."""

    def __init__(
        self,
        cfg: OpenMarlConfig,
        action_dim: int = 2,
        hidden_units: list[int] | None = None,
        lstm_hidden: int = 128,
        n_attn_heads: int = 4,
        action_scale: float = 3.2,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_dim = action_dim
        self.action_scale = action_scale
        hidden_units = hidden_units or [256, 256, 256]

        self.epn = EvaderPredictor(cfg, hidden_size=lstm_hidden)
        self.encoder = ComponentObservationEncoder(cfg, embed_dim=embed_dim, n_heads=n_attn_heads, hidden=hidden_units[0])
        self.actor = TanhGaussianActor(hidden_units[0], action_dim, hidden_units, action_scale)

        critic_in = hidden_units[0] * cfg.n_pursuers + cfg.epn_target_dim
        self.critic_body = _mlp([critic_in, *hidden_units], layer_norm=True)
        self.critic_head = nn.Linear(hidden_units[-1], 1)

    def encode_agents(self, obs: torch.Tensor, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """obs (B, n_agents, obs_dim) -> features (B, n_agents, hidden), pred (B, K*2)."""
        pred, _ = self.epn(history)
        b, n, _ = obs.shape
        feats = []
        for a in range(n):
            feats.append(self.encoder(obs[:, a], pred))
        return torch.stack(feats, dim=1), pred

    def actor_forward(
        self,
        obs: torch.Tensor,
        history: torch.Tensor,
        h: tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """obs: (B, obs_dim) single agent; history: (B, T, epn_input_dim)."""
        pred, h_new = self.epn(history, h)
        feat = self.encoder(obs, pred)
        action, log_prob, entropy = self.actor.sample(feat, deterministic=deterministic)
        return action, log_prob, entropy, h_new

    def actor_forward_batch(
        self,
        obs_all: torch.Tensor,
        history: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """obs_all: (B, n_agents, obs_dim) — single EPN forward per env."""
        feats, pred = self.encode_agents(obs_all, history)
        b, n, _ = feats.shape
        actions = []
        log_probs = []
        entropies = []
        for a in range(n):
            act, lp, ent = self.actor.sample(feats[:, a], deterministic=deterministic)
            actions.append(act)
            log_probs.append(lp)
            entropies.append(ent)
        return (
            torch.stack(actions, dim=1),
            torch.stack(log_probs, dim=1),
            torch.stack(entropies, dim=1),
            pred,
        )

    def critic_from_encoded(self, feats: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        flat = feats.reshape(feats.shape[0], -1)
        x = torch.cat([flat, pred], dim=-1)
        return self.critic_head(self.critic_body(x)).squeeze(-1)

    def critic_forward(
        self,
        obs_all: torch.Tensor,
        history: torch.Tensor,
        h: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        feats, pred = self.encode_agents(obs_all, history)
        return self.critic_from_encoded(feats, pred)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        obs_all: torch.Tensor,
        history: torch.Tensor,
        actions: torch.Tensor,
        h: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feats, pred = self.encode_agents(obs_all, history)
        agent_idx = 0
        if obs_all.shape[1] > 1 and obs.shape[0] == obs_all.shape[0]:
            for a in range(obs_all.shape[1]):
                if torch.allclose(obs, obs_all[:, a]):
                    agent_idx = a
                    break
        lp, ent = self.actor.evaluate(feats[:, agent_idx], actions)
        value = self.critic_forward(obs_all, history, h)
        return lp, ent, value, pred

    def save_checkpoint(self, path: str, extra: dict[str, Any] | None = None) -> None:
        meta = {
            "version": 2,
            "cfg": {
                "n_pursuers": self.cfg.n_pursuers,
                "history_step": self.cfg.history_step,
                "future_prediction_step": self.cfg.future_prediction_step,
                "obs_max_obstacles": self.cfg.obs_max_obstacles,
                "mask_value": self.cfg.mask_value,
                "arena_scale": self.cfg.arena_scale,
            },
            "obs_dim": self.cfg.obs_dim,
            "action_dim": self.action_dim,
            "epn_input_dim": self.cfg.epn_input_dim,
            "epn_target_dim": self.cfg.epn_target_dim,
            "hidden_units": [256, 256, 256],
            "action_scale": self.action_scale,
            **(extra or {}),
        }
        torch.save({"state_dict": self.state_dict(), "meta": meta}, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str | torch.device = "cpu") -> tuple[OpenMARLPolicy, dict[str, Any]]:
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=device)
        meta = ckpt["meta"]
        if meta.get("version", 1) < 2:
            raise ValueError(
                f"Checkpoint {path} uses legacy format (v{meta.get('version', 1)}). "
                "Retrain with: python experiments/train_open_marl.py --stage 1"
            )
        cfg = OpenMarlConfig.from_dict(meta.get("cfg", meta))
        hidden = meta.get("hidden_units", [256, 256, 256])
        policy = cls(
            cfg,
            action_dim=int(meta.get("action_dim", 2)),
            hidden_units=hidden,
            action_scale=float(meta.get("action_scale", 3.2)),
        )
        policy.load_state_dict(ckpt["state_dict"])
        policy.to(device)
        policy.eval()
        return policy, meta
