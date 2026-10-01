"""Uniform experience replay buffer (numpy-backed, circular).

Kept deliberately simple and tested so the SAC loop has no hidden magic.
"""
from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Circular buffer storing (s, a, r, s', done) transitions.

    Observations are stored as float32 numpy arrays. `sample` returns
    torch tensors on the requested device.
    """

    def __init__(self, obs_dim: int, act_dim: int, capacity: int, device: str = "cpu"):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return (
            to_t(self.obs[idx]),
            to_t(self.actions[idx]),
            to_t(self.rewards[idx]),
            to_t(self.next_obs[idx]),
            to_t(self.dones[idx]),
        )

    def __len__(self) -> int:
        return self.size
