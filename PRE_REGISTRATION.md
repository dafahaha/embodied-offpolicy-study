# Pre-registration (before any Hopper result was inspected)

Written 2026-10-01, immediately after Pendulum validation and before launching any
Hopper run. No Hopper return numbers below were known at write time.

## Research question
Does scaling the rewards that enter the critic's Bellman backup by a fixed factor
change sample efficiency and asymptotic performance of SAC on Hopper-v4?

## Independent variable
`reward_scale` multiplies `r` in the Bellman target:
`y = reward_scale * r + gamma * (1 - done) * (min Q_target - alpha * log pi)`.
Two arms, everything else identical:

- **Baseline**: `reward_scale = 1.0` (rewards as emitted by the env).
- **RewardScaled**: `reward_scale = 0.1`.

The logged episode return is always the *unscaled* env return, so the two arms are
compared on the true objective.

## Pre-registered hypothesis
Scaling rewards by 0.1 shrinks the magnitude of the Q-targets. Because SAC's
max-entropy objective couples Q-magnitude to the temperature term
(`alpha * log pi`), smaller Q targets should reduce critic over-estimation bias
and stabilise learning, yielding:

1. Higher asymptotic (final-10k-step) deterministic eval return on Hopper-v4.
2. Faster early learning (higher eval return at, say, step 50k).

Direction: `RewardScaled >= Baseline`. We explicitly allow the result to come
out the other way; the point is a controlled, multi-seed measurement.

## Primary outcome
Mean deterministic eval return (5 episodes per eval point) over the last 20% of
training steps, averaged across 3 seeds, reported as mean +/- 95% CI.

## Design
- Env: Hopper-v4 (gymnasium 1.3.0, mujoco 3.14.0).
- Steps per run: **120,000** env steps (1 gradient step per step after 1000 warmup).
- Seeds: 0, 1, 2 per arm (6 runs total).
- Hardware: CPU only, PyTorch, 4 intra-op threads per run.
- No hyperparameter tuning on Hopper; both arms share all other settings
  (256x256 MLPs, lr 3e-4, gamma 0.99, tau 0.005, auto entropy tuning).

## What we will NOT claim
- This is not a benchmark SOTA run; 120k steps is a small, CPU-feasible budget.
- We will not extrapolate to HalfCheetah / 1M-step regimes.
- A single-seed winner would not be called a result; we report seed spread.

