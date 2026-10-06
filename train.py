"""Training entry point.

Usage:
    python train.py --config configs/sac_hopper_baseline.yaml --seed 0 \
        --out logs/hopper_baseline_s0

Logs a per-episode + per-eval CSV at <out>/progress.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch

from src.sac import SAC
from src.replay_buffer import ReplayBuffer
from src.utils import set_seed, load_config


def evaluate(agent: SAC, env_id: str, seed: int, episodes: int = 5) -> float:
    env = gym.make(env_id)
    try:
        returns = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + 1000 + ep)
            done = False
            ep_ret = 0.0
            while not done:
                with torch.no_grad():
                    a = agent.act(obs, deterministic=True)
                obs, r, terminated, truncated, _ = env.step(a)
                ep_ret += r
                done = terminated or truncated
            returns.append(ep_ret)
    finally:
        env.close()
    return float(np.mean(returns))


# Soft magnitude guards: we only WARN, never abort on these — MuJoCo envs
# legitimately reach large asymptotic returns, and a hard abort here would
# false-positive on them. Both are sentinels that fire only when something
# has actually blown up.
#
# Healthy Hopper critics sit at |q1_mean| in the low hundreds (observed max
# ~207 on the baseline arm), so 1e4 is ~50x above healthy: it trips only on
# a genuine critic blow-up, not on ordinary training dynamics.
Q1_MEAN_WARN_ABS = 1e4
# Healthy logp_mean is about -2..-6 (tanh-squashed Gaussian). |logp_mean| > 1e3
# means the policy has gone to absurd pre-activations (the recorded 1e15
# incident was a finite float that np.isfinite() would not catch).
LOGP_MEAN_WARN_ABS = 1e3


def _guard_metrics(metrics: dict, step: int) -> None:
    """Hard-abort (exit 3) on any non-finite loss/logp; soft-warn on runaway Q/logp.

    Split out of the training loop so it can be unit-tested directly.
    """
    for k in ("critic_loss", "actor_loss", "q1_mean", "logp_mean"):
        v = float(metrics.get(k, float("nan")))
        if not np.isfinite(v):
            print(
                f"FATAL: training diverged — {k}={v} at env_step={step} "
                f"(critic_loss={metrics.get('critic_loss')}, "
                f"actor_loss={metrics.get('actor_loss')}, "
                f"q1_mean={metrics.get('q1_mean')}, "
                f"logp_mean={metrics.get('logp_mean')}).",
                file=sys.stderr,
            )
            sys.exit(3)
    q1 = float(metrics["q1_mean"])
    if abs(q1) > Q1_MEAN_WARN_ABS:
        print(
            f"WARNING: |q1_mean|={q1:.1f} > {Q1_MEAN_WARN_ABS:.0e} at "
            f"env_step={step} — critics may be blowing up (soft guard: "
            f"continuing, see train.py Q1_MEAN_WARN_ABS).",
            file=sys.stderr,
        )
    logp = float(metrics["logp_mean"])
    if abs(logp) > LOGP_MEAN_WARN_ABS:
        print(
            f"WARNING: |logp_mean|={logp:.1f} > {LOGP_MEAN_WARN_ABS:.0e} at "
            f"env_step={step} — policy may be blowing up (soft guard: "
            f"continuing, see train.py LOGP_MEAN_WARN_ABS).",
            file=sys.stderr,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    torch.set_num_threads(4)  # cap CPU threads so the host shell stays responsive

    cfg = load_config(args.config)
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    env = gym.make(cfg["env_id"])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    agent = SAC(
        obs_dim=obs_dim, act_dim=act_dim,
        act_low=act_low, act_high=act_high,
        device=args.device,
        hidden_sizes=tuple(cfg.get("hidden_sizes", [256, 256])),
        gamma=cfg.get("gamma", 0.99),
        tau=cfg.get("tau", 0.005),
        lr=cfg.get("lr", 3e-4),
        alpha=cfg.get("alpha", 0.2),
        automatic_entropy_tuning=cfg.get("automatic_entropy_tuning", True),
        reward_scale=cfg.get("reward_scale", 1.0),
    )
    buf = ReplayBuffer(obs_dim, act_dim, cfg.get("replay_size", 1_000_000), args.device)

    total_steps = int(cfg["total_steps"])
    start_steps = int(cfg.get("start_steps", 1000))
    batch_size = int(cfg.get("batch_size", 256))
    eval_every = int(cfg.get("eval_every", 5000))
    eval_episodes = int(cfg.get("eval_episodes", 5))

    csv_path = os.path.join(args.out, "progress.csv")
    fields = ["env_step", "episode", "phase", "episode_return", "episode_length",
              "eval_return", "time_sec", "alpha", "q1_mean"]
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()
    csv_file.flush()

    t0 = time.time()
    obs, _ = env.reset(seed=args.seed)
    ep_ret, ep_len, ep = 0.0, 0, 0
    next_eval_at = eval_every
    last_metrics = {}

    for step in range(1, total_steps + 1):
        if step < start_steps:
            a = env.action_space.sample()
        else:
            a = agent.act(obs, deterministic=False)

        next_obs, r, terminated, truncated, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        # Bootstrap on time-limit truncation, NOT on real termination.
        buf.add(obs, a, r, next_obs, terminated)
        obs = next_obs

        if step >= start_steps:
            last_metrics = agent.update(buf.sample(batch_size))
            # Divergence guard: abort loudly instead of writing NaN/Inf into
            # progress.csv (plot.py would otherwise spread it over the stats).
            _guard_metrics(last_metrics, step)

        if terminated or truncated:
            ep += 1
            writer.writerow({
                "env_step": step, "episode": ep, "phase": "train",
                "episode_return": ep_ret, "episode_length": ep_len,
                "eval_return": "", "time_sec": round(time.time() - t0, 2),
                "alpha": round(last_metrics.get("alpha", float("nan")), 4),
                "q1_mean": round(last_metrics.get("q1_mean", float("nan")), 3),
            })
            csv_file.flush()
            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0

        if step >= next_eval_at:
            ev = evaluate(agent, cfg["env_id"], args.seed, eval_episodes)
            writer.writerow({
                "env_step": step, "episode": ep, "phase": "eval",
                "episode_return": "", "episode_length": "",
                "eval_return": round(ev, 2),
                "time_sec": round(time.time() - t0, 2),
                "alpha": round(last_metrics.get("alpha", float("nan")), 4),
                "q1_mean": round(last_metrics.get("q1_mean", float("nan")), 3),
            })
            csv_file.flush()
            next_eval_at += eval_every
            if step % (eval_every * 4) == 0:
                print(f"[{args.out}] step={step}/{total_steps} eval={ev:.1f} "
                      f"t={time.time()-t0:.0f}s", flush=True)

    csv_file.close()
    env.close()
    print(f"DONE {args.out} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
