import numpy as np
import pandas as pd
import torch
import pytest
from pytest import approx

from src.replay_buffer import ReplayBuffer
from src.networks import SquashedGaussianActor, TwinCritic
from src.sac import SAC
from ci_check import learn_gate_pass
from plot import asymptotic_stats, validate_grid_and_counts, load_runs
import train as train_mod


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


# ---------------------------------------------------------------------------
# G1: analytic log_prob must include the linear action-rescale Jacobian.
# For bounds [-2, 2], act_scale = 2. At pre_tanh=0, mu=0, std=1:
#   logp = logN(0|0,1) - log(1-tanh(0)^2) - log(2)
#        = -0.91894    - 0               - 0.69315 = -1.6121.
# (The review's "-0.919" is the *pre-fix* value that omits -log(act_scale);
# asserting it here would lock in the missing-Jacobian bug.)
# ---------------------------------------------------------------------------
def test_logprob_analytic_rescale_jacobian():
    low = np.array([-2.0])
    high = np.array([2.0])
    actor = SquashedGaussianActor(3, 1, torch.tensor(low), torch.tensor(high), (16, 16))
    pre_tanh = torch.zeros(1, 1)
    mu = torch.zeros(1, 1)
    std = torch.ones(1, 1)
    logp = actor._logprob(pre_tanh, mu, std).item()
    hand = (-0.5 * np.log(2 * np.pi)            # logN(0 | mu=0, std=1)
            - np.log(1.0 - np.tanh(0.0) ** 2)    # tanh change of variables
            - np.log(2.0))                        # linear rescale Jacobian
    assert logp == approx(hand, abs=1e-3)
    assert logp == approx(-1.6121, abs=1e-3)
    # unit-bound env (act_scale=1) -> rescale term is exactly 0
    actor_unit = SquashedGaussianActor(3, 1, torch.tensor([-1.0]),
                                       torch.tensor([1.0]), (16, 16))
    logp_unit = actor_unit._logprob(pre_tanh, mu, std).item()
    assert logp_unit == approx(-0.91894, abs=1e-3)


# ---------------------------------------------------------------------------
# G2: SAC update smoke test — one real update step, losses finite, target moved.
# ---------------------------------------------------------------------------
def _toy_sac(seed=0, reward_scale=1.0, zero_init_critic=False):
    torch.manual_seed(seed)
    agent = SAC(obs_dim=3, act_dim=1,
                act_low=np.array([-1.0]), act_high=np.array([1.0]),
                hidden_sizes=(16, 16), reward_scale=reward_scale,
                automatic_entropy_tuning=True)
    if zero_init_critic:
        for p in agent.critic.parameters():
            torch.nn.init.zeros_(p)
        # critic_target is a deep copy of the critic, so it is zero too
    buf = ReplayBuffer(3, 1, capacity=64, device="cpu")
    for i in range(32):
        buf.add(np.random.randn(3), np.zeros(1), 1.0,
                np.random.randn(3), i % 2 == 0)
    return agent, buf


def test_sac_update_smoke_finite_and_target_moves():
    agent, buf = _toy_sac()
    with torch.no_grad():
        t_before = next(iter(agent.critic_target.parameters())).clone()
    m = agent.update(buf.sample(16))
    for k in ("critic_loss", "actor_loss", "q1_mean"):
        assert np.isfinite(m[k]), f"{k} not finite: {m[k]}"
    t_after = next(iter(agent.critic_target.parameters()))
    # soft target (tau=0.005) must track the updated critic
    assert not torch.allclose(t_before, t_after)


def test_reward_scale_scales_bellman_target_linear():
    # Zero-initialised critic: q=0 everywhere. With done=1 for every row the
    # target collapses to reward_scale * r, so critic_loss = 2*mean(target^2)
    # scales as reward_scale^2 -> 100x smaller when reward_scale=0.1.
    torch.manual_seed(0)
    agent = SAC(obs_dim=3, act_dim=1,
                act_low=np.array([-1.0]), act_high=np.array([1.0]),
                hidden_sizes=(16, 16), reward_scale=0.1)
    for p in agent.critic.parameters():
        torch.nn.init.zeros_(p)
    torch.manual_seed(1)
    buf = ReplayBuffer(3, 1, capacity=16, device="cpu")
    for i in range(16):
        buf.add(np.zeros(3), np.zeros(1), 2.0, np.zeros(3), True)
    m_small = agent.update(buf.sample(16))

    torch.manual_seed(0)
    agent1 = SAC(obs_dim=3, act_dim=1,
                 act_low=np.array([-1.0]), act_high=np.array([1.0]),
                 hidden_sizes=(16, 16), reward_scale=1.0)
    for p in agent1.critic.parameters():
        torch.nn.init.zeros_(p)
    torch.manual_seed(1)
    buf1 = ReplayBuffer(3, 1, capacity=16, device="cpu")
    for i in range(16):
        buf1.add(np.zeros(3), np.zeros(1), 2.0, np.zeros(3), True)
    m_big = agent1.update(buf1.sample(16))

    # target = scale * r = 0.1*2.0=0.2 vs 1.0*2.0=2.0; q=0 -> loss = 2*target^2
    assert m_small["critic_loss"] == approx(2 * 0.2 ** 2, abs=1e-4)
    assert m_big["critic_loss"] == approx(2 * 2.0 ** 2, abs=1e-4)


def test_plot_stats_match_scipy_on_toy_arrays():
    from scipy import stats as s
    per_seed = {"A": np.array([100.0, 200.0, 300.0]),
                "B": np.array([150.0, 250.0, 350.0])}
    res = asymptotic_stats(per_seed)
    assert res["n"] == 3
    assert res["tcrit"] == approx(s.t.ppf(0.975, df=2))
    assert res["means"]["A"] == approx(200.0)
    assert res["cis"]["A"] == approx(s.t.ppf(0.975, df=2) * 100.0 / np.sqrt(3))
    t_exp = s.ttest_ind(per_seed["A"], per_seed["B"], equal_var=False)
    pair = res["welch"][0]
    assert pair[2] == approx(t_exp.statistic)
    assert pair[3] == approx(t_exp.pvalue)


# ---------------------------------------------------------------------------
# S3: CI learn-gate decision as a pure function.
# ---------------------------------------------------------------------------
def test_learn_gate_delta_200_passes():
    # first=-1200, mean(last 2)=-1000 -> Δ=+200 >= 150
    passed, delta, first, tail = learn_gate_pass(
        [-1200.0, -1100.0, -1000.0, -1000.0])
    assert first == approx(-1200.0)
    assert tail == approx(-1000.0)
    assert delta == approx(200.0)
    assert passed is True


def test_learn_gate_delta_50_fails():
    # first=-1200, mean(last 2)=-1150 -> Δ=+50 < 150
    passed, delta, _, _ = learn_gate_pass(
        [-1200.0, -1180.0, -1150.0, -1150.0])
    assert delta == approx(50.0)
    assert passed is False


# ---------------------------------------------------------------------------
# N-GATE: empty eval series must be a clean FAIL, not a bare IndexError.
# ---------------------------------------------------------------------------
def test_learn_gate_empty_series_is_clean_fail():
    passed, delta, first, tail = learn_gate_pass([])
    assert passed is False
    assert delta == 0.0
    assert np.isnan(first) and np.isnan(tail)


# ---------------------------------------------------------------------------
# G4: train.py divergence guard. Non-finite logp must abort with exit code 3;
# a finite-but-huge q1_mean must warn (not abort).
# ---------------------------------------------------------------------------
def test_guard_aborts_on_nonfinite_logp():
    bad = {"critic_loss": 1.0, "actor_loss": 1.0,
           "q1_mean": 1.0, "logp_mean": float("nan")}
    with pytest.raises(SystemExit) as ei:
        train_mod._guard_metrics(bad, step=1001)
    assert ei.value.code == 3


def test_guard_warns_on_large_q1_but_does_not_abort(capsys):
    ok = {"critic_loss": 1.0, "actor_loss": 1.0,
          "q1_mean": 2e4, "logp_mean": -1.0}
    train_mod._guard_metrics(ok, step=1001)  # no SystemExit
    err = capsys.readouterr().err
    assert "WARNING" in err and "q1_mean" in err


# ---------------------------------------------------------------------------
# T1: plot.py grid/count/non-finite guards as pure functions / synthetic CSV.
# ---------------------------------------------------------------------------
def _ev_df(steps):
    return pd.DataFrame({"env_step": list(steps),
                         "eval_return": [100.0] * len(steps)})


def test_validate_grid_aligned_passes():
    runs = {
        "A": [(0, _ev_df([5000, 10000, 120000])),
              (1, _ev_df([5000, 10000, 120000]))],
        "B": [(0, _ev_df([5000, 10000, 120000])),
              (1, _ev_df([5000, 10000, 120000]))],
    }
    grid = validate_grid_and_counts(runs)
    assert list(grid) == [5000, 10000, 120000]


def test_validate_grid_missing_end_point_raises():
    # equal seed counts (2 each) so we reach the grid comparison; one arm's
    # runs were killed before the final eval at 120000.
    runs = {
        "A": [(0, _ev_df([5000, 10000, 120000])),
              (1, _ev_df([5000, 10000, 120000]))],
        "B": [(0, _ev_df([5000, 10000])),
              (1, _ev_df([5000, 10000]))],
    }
    with pytest.raises(SystemExit, match="missing"):
        validate_grid_and_counts(runs)


def test_validate_grid_unequal_seed_counts_raises():
    runs = {
        "A": [(0, _ev_df([5000])), (1, _ev_df([5000]))],
        "B": [(0, _ev_df([5000]))],
    }
    with pytest.raises(SystemExit, match="unequal seed counts"):
        validate_grid_and_counts(runs)


def test_load_runs_rejects_nonfinite_eval_return(tmp_path, monkeypatch):
    run = tmp_path / "hopper_baseline_s0"
    run.mkdir()
    pd.DataFrame({
        "env_step": [5000, 10000],
        "phase": ["eval", "eval"],
        "eval_return": [100.0, float("nan")],
    }).to_csv(run / "progress.csv", index=False)
    monkeypatch.setattr("plot.LOG_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="non-finite eval_return"):
        load_runs()
