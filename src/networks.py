"""Actor-Critic networks for SAC, written from scratch in PyTorch."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def mlp(sizes, activation=nn.ReLU, out_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        layers.append(activation() if i < len(sizes) - 2 else out_activation())
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    """Tanh-squashed diagonal Gaussian policy.

    Outputs actions already scaled to the environment's action bounds.
    log_prob includes both the tanh Jacobian and the linear rescale (act_scale)
    correction, so it is the density w.r.t. env actions.
    """

    def __init__(self, obs_dim: int, act_dim: int, act_low: torch.Tensor,
                 act_high: torch.Tensor, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden_sizes], out_activation=nn.ReLU)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.register_buffer("act_low", act_low)
        self.register_buffer("act_high", act_high)
        # inactive [-1, 1] range of tanh maps linearly to [low, high]
        self.register_buffer("act_scale", (self.act_high - self.act_low) / 2.0)
        self.register_buffer("act_bias", (self.act_high + self.act_low) / 2.0)

    def forward(self, obs, deterministic=False, with_logprob=True):
        h = self.net(obs)
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        if deterministic:
            pre_tanh = mu
        else:
            pre_tanh = mu + std * torch.randn_like(std)

        action = torch.tanh(pre_tanh)
        if with_logprob:
            logp = self._logprob(pre_tanh, mu, std)
        else:
            logp = None

        action = self.act_scale * action + self.act_bias
        return action, logp

    def _logprob(self, pre_tanh, mu, std):
        # Diagonal Gaussian log prob centered on mu.
        gauss = -0.5 * (
            ((pre_tanh - mu) / std) ** 2
            + 2.0 * torch.log(std)
            + torch.log(torch.tensor(2 * torch.pi, device=pre_tanh.device))
        )
        logp = gauss.sum(dim=-1, keepdim=True)
        # Tanh change-of-variables: log(1 - tanh(x)^2), with epsilon for stability.
        logp -= torch.sum(
            torch.log(1.0 - torch.tanh(pre_tanh) ** 2 + 1e-6), dim=-1, keepdim=True
        )
        # Linear action-rescale Jacobian: a = scale*tanh(u)+bias, so dP/da =
        # dP/du / |scale|. For unit-bound envs (scale=1) this term is exactly 0.
        logp -= torch.sum(torch.log(self.act_scale), dim=-1, keepdim=True)
        return logp


class TwinCritic(nn.Module):
    """Two independent Q-networks (twin clipped critics; min over Q1, Q2)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, *hidden_sizes, 1])
        self.q2 = mlp([obs_dim + act_dim, *hidden_sizes, 1])

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)
