"""EPN + attention actor-critic for OPEN MARL (2D adaptation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    raise ImportError(
        "OPEN MARL requires PyTorch. Install with: pip install -r requirements-rl.txt"
    ) from e

from baselines.open_marl.observation import OpenMarlConfig


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
        """
        history: (B, T, epn_input_dim)
        returns prediction (B, K*2), new hidden
        """
        out, h_new = self.lstm(history, h)
        pred = self.head(out[:, -1])
        return pred, h_new


class PartialAttentionEncoder(nn.Module):
    """Entity-token attention over observation components."""

    def __init__(self, token_dim: int, n_heads: int = 4, hidden: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(token_dim, hidden)
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, n_entities, token_dim) -> (B, hidden)"""
        x = self.proj(tokens)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = x + self.ff(x)
        return x.mean(dim=1)


class OpenMARLPolicy(nn.Module):
    """Combined EPN + actor-critic with parameter sharing."""

    def __init__(
        self,
        cfg: OpenMarlConfig,
        action_dim: int = 2,
        hidden_units: list[int] | None = None,
        lstm_hidden: int = 128,
        n_attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_dim = action_dim
        hidden_units = hidden_units or [256, 256, 256]

        self.epn = EvaderPredictor(cfg, hidden_size=lstm_hidden)

        # Tokenize: self(4), each other(2), each obstacle(2), pred chunks(2)
        self.token_dim = 4
        self.encoder = PartialAttentionEncoder(self.token_dim, n_attn_heads, hidden_units[0])

        actor_in = hidden_units[0] + cfg.obs_dim + cfg.epn_target_dim
        self.actor_body = _mlp([actor_in, *hidden_units], layer_norm=True)
        self.actor_mean = nn.Linear(hidden_units[-1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        critic_in = cfg.obs_dim * cfg.n_pursuers + cfg.epn_target_dim
        self.critic_body = _mlp([critic_in, *hidden_units], layer_norm=True)
        self.critic_head = nn.Linear(hidden_units[-1], 1)

    def _obs_to_tokens(self, obs: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """obs (B, obs_dim), pred (B, K*2) -> (B, n_tokens, token_dim)"""
        b = obs.shape[0]
        cfg = self.cfg
        tokens = []
        idx = 0
        self_vel = obs[:, idx : idx + 2]
        idx += 2
        rel_ev = obs[:, idx : idx + 2]
        idx += 2
        tokens.append(torch.cat([self_vel, rel_ev], dim=-1))

        for _ in range(cfg.n_pursuers - 1):
            rel = obs[:, idx : idx + 2]
            idx += 2
            pad = torch.zeros(b, 2, device=obs.device, dtype=obs.dtype)
            tokens.append(torch.cat([rel, pad], dim=-1))

        for _ in range(cfg.obs_max_obstacles):
            rel = obs[:, idx : idx + 2]
            idx += 2
            pad = torch.zeros(b, 2, device=obs.device, dtype=obs.dtype)
            tokens.append(torch.cat([rel, pad], dim=-1))

        k = cfg.future_prediction_step
        for t in range(k):
            chunk = pred[:, 2 * t : 2 * t + 2]
            pad = torch.zeros(b, 2, device=obs.device, dtype=obs.dtype)
            tokens.append(torch.cat([chunk, pad], dim=-1))

        return torch.stack(tokens, dim=1)

    def forward_epn(
        self,
        history: torch.Tensor,
        h: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.epn(history, h)

    def actor_forward(
        self,
        obs: torch.Tensor,
        history: torch.Tensor,
        h: tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        obs: (B, obs_dim)
        history: (B, T, epn_input_dim)
        returns action, log_prob, entropy, h_new
        """
        pred, h_new = self.epn(history, h)
        enc = self.encoder(self._obs_to_tokens(obs, pred))
        x = torch.cat([enc, obs, pred], dim=-1)
        x = self.actor_body(x)
        mean = self.actor_mean(x)
        std = self.log_std.exp().expand_as(mean)

        if deterministic:
            action = mean
            log_prob = torch.zeros(mean.shape[0], device=mean.device)
            entropy = torch.zeros_like(log_prob)
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)

        return action, log_prob, entropy, h_new

    def critic_forward(
        self,
        obs_all: torch.Tensor,
        history: torch.Tensor,
        h: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """obs_all: (B, n_agents, obs_dim) -> value (B,)"""
        b, n, d = obs_all.shape
        pred, _ = self.epn(history, h)
        flat_obs = obs_all.reshape(b, n * d)
        x = torch.cat([flat_obs, pred], dim=-1)
        return self.critic_head(self.critic_body(x)).squeeze(-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        obs_all: torch.Tensor,
        history: torch.Tensor,
        actions: torch.Tensor,
        h: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred, h_new = self.epn(history, h)
        enc = self.encoder(self._obs_to_tokens(obs, pred))
        x = torch.cat([enc, obs, pred], dim=-1)
        x = self.actor_body(x)
        mean = self.actor_mean(x)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic_forward(obs_all, history, h)
        return log_prob, entropy, value, pred

    @dataclass
    class CheckpointMeta:
        cfg: dict[str, Any]
        obs_dim: int
        action_dim: int
        epn_input_dim: int
        epn_target_dim: int

    def save_checkpoint(self, path: str, extra: dict[str, Any] | None = None) -> None:
        meta = {
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
            **(extra or {}),
        }
        torch.save({"state_dict": self.state_dict(), "meta": meta}, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str | torch.device = "cpu") -> tuple[OpenMARLPolicy, dict[str, Any]]:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        meta = ckpt["meta"]
        cfg = OpenMarlConfig.from_dict(meta.get("cfg", meta))
        hidden = meta.get("hidden_units", [256, 256, 256])
        policy = cls(cfg, action_dim=int(meta.get("action_dim", 2)), hidden_units=hidden)
        policy.load_state_dict(ckpt["state_dict"])
        policy.to(device)
        policy.eval()
        return policy, meta
