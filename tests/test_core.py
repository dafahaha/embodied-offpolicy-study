import numpy as np
import torch

from src.replay_buffer import ReplayBuffer
from src.networks import SquashedGaussianActor, TwinCritic


def test_replay_buffer_shapes_and_circular():
    buf = ReplayBuffer(obs_dim=3, act_dim=1, capacity=10, device="cpu")
    for i in range(5):
        buf.add(np.zeros(3), np.zeros(1), 1.0, np.ones(3), False)
    assert len(buf) == 5
    # overflow -> circular overwrite
    for i in range(10):
        buf.add(np.full(3, i), np.full(1, i), float(i), np.full(3, i + 1), True)
    assert len(buf) == 10
    obs, act, rew, next_obs, done = buf.sample(4)
    assert obs.shape == (4, 3)
    assert act.shape == (4, 1)
    assert rew.shape == (4, 1)
    assert next_obs.shape == (4, 3)
    assert done.shape == (4, 1)
    assert obs.dtype == torch.float32


def test_actor_output_in_bounds_and_logprob():
    low = np.array([-1.0])
    high = np.array([1.0])
    actor = SquashedGaussianActor(3, 1, torch.tensor(low), torch.tensor(high), (16, 16))
    obs = torch.randn(8, 3)
    a, logp = actor(obs)
    assert a.shape == (8, 1)
    assert logp.shape == (8, 1)
    # squashed action must lie strictly within bounds
    assert (a >= -1.0 - 1e-5).all() and (a <= 1.0 + 1e-5).all()
    # deterministic pass returns the mean mapped through tanh
    a_det, _ = actor(obs, deterministic=True, with_logprob=False)
    assert a_det.shape == (8, 1)


def test_critic_twin_shapes():
    critic = TwinCritic(3, 1, (16, 16))
    q1, q2 = critic(torch.randn(5, 3), torch.randn(5, 1))
    assert q1.shape == (5, 1)
    assert q2.shape == (5, 1)
