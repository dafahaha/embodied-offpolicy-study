"""Soft Actor-Critic (SAC) implemented from scratch.

Reference objective: Haarnoja et al. 2018, "Soft Actor-Critic: Off-Policy
Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor".
We use twin critics, soft target updates, and automatic entropy temperature.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn.functional as F

from .networks import SquashedGaussianActor, TwinCritic
from .replay_buffer import ReplayBuffer


class SAC:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_low: np.ndarray,
        act_high: np.ndarray,
        device: str = "cpu",
        hidden_sizes=(256, 256),
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        reward_scale: float = 1.0,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale

        act_low_t = torch.as_tensor(act_low, dtype=torch.float32, device=self.device)
        act_high_t = torch.as_tensor(act_high, dtype=torch.float32, device=self.device)

        self.actor = SquashedGaussianActor(
            obs_dim, act_dim, act_low_t, act_high_t, hidden_sizes
        ).to(self.device)
        self.critic = TwinCritic(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Entropy temperature
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if automatic_entropy_tuning:
            self.target_entropy = -float(act_dim)
            self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True,
                                           device=self.device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.log_alpha = torch.tensor(np.log(alpha), device=self.device)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.actor(obs_t, deterministic=deterministic, with_logprob=False)
        return a.squeeze(0).cpu().numpy()

    def update(self, batch):
        obs, act, rew, next_obs, done = batch

        # ---- Critic update ----
        with torch.no_grad():
            next_a, next_logp = self.actor(next_obs)
            tq1, tq2 = self.critic_target(next_obs, next_a)
            tq = torch.min(tq1, tq2) - self.alpha * next_logp
            target = self.reward_scale * rew + (1.0 - done) * self.gamma * tq
        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ---- Actor update ----
        a, logp = self.actor(obs)
        q1_pi, q2_pi = self.critic(obs, a)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * logp - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ---- Alpha update ----
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        # ---- Soft target update ----
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.mul_(1.0 - self.tau)
                pt.data.add_(self.tau * p.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "q1_mean": float(q1.mean().item()),
            "q2_mean": float(q2.mean().item()),
            "logp_mean": float(logp.mean().item()),
        }
