# embodied-offpolicy-study

A small, honest, from-scratch reinforcement-learning study built to demonstrate
real research ability for a PhD (Embodied AI / RL) application.

- **Algorithm**: Soft Actor-Critic (SAC) implemented **from scratch in PyTorch**
  (no Stable-Baselines3). Twin critics, soft target updates, tanh-squashed Gaussian
  actor, automatic entropy temperature.
- **Envs**: Pendulum-v1 (validation) + Hopper-v4 (the study).
- **Study**: does critic-side **reward scaling** (0.1 vs 1.0) change sample
  efficiency / asymptotic return? 3 seeds per arm. Pre-registered in
  `PRE_REGISTRATION.md`.
- **Hardware honesty**: all runs are **CPU only**, **60k env steps per run**
  (planned 120k, cut because a 6-wide parallel batch got OOM-killed on a shared
  machine). This is *not* a 1M-step benchmark; numbers are reported as measured.

## Layout

```
src/            from-scratch SAC (replay buffer, actor, twin critic, agent)
tests/          pytest: buffer sampling, actor bounds/logprob, critic shapes
configs/        YAML per setting (pendulum_ci, pendulum, hopper baseline/rewardscale)
train.py        training loop -> logs/<run>/progress.csv (per-episode + eval rows)
ci_check.py     fast Pendulum learn-gate used by CI
plot.py         reads raw CSVs -> learning curve + asymptotic bar chart
logs/<run>/progress.csv   RAW step logs (committed, not fabricated)
figures/        plots produced by plot.py
PRE_REGISTRATION.md       hypothesis registered before results
RESEARCH_RETRO.md         short human-readable write-up
```

## Quick start

```bash
pip install -r requirements.txt

# unit tests
pytest tests/ -q

# validation: prove the agent learns on Pendulum
python train.py --config configs/sac_pendulum.yaml --seed 0 --out logs/pendulum_val_s0

# one arm of the study (repeat for each seed / config)
python train.py --config configs/sac_hopper_baseline.yaml    --seed 0 --out logs/hopper_baseline_s0
python train.py --config configs/sac_hopper_rewardscale.yaml --seed 0 --out logs/hopper_rewardscale_s0

# plot from the raw CSVs
python plot.py
```

## Results

Deterministic eval return (5 episodes), Hopper-v4, CPU, 60k steps/run, 3 seeds:

| Arm | asymptotic return (last 20%, mean) | 95% CI across seeds |
|---|---|---|
| Baseline (reward_scale=1.0) | **363** | +/- 210 |
| RewardScaled (reward_scale=0.1) | **1087** | +/- 1029 |

Per-seed final eval points: baseline = 250, 613, 336; reward-scaled = 591, 606,
1375.

**Interpretation (honest):** the mean points the way I pre-registered (scaling
rewards down helps), and the learning curve trends upward earlier for the scaled
arm. But the reward-scaled CI is enormous because one seed ran away to ~1375 while
the other two sat at ~600, and the two confidence intervals overlap heavily. With
n=3 and 60k steps this is a **directional signal, not a confirmed effect**. I am
not claiming a win.

See `figures/learning_curve.png` and `figures/asymptotic.png`. The summary table
is reproduced by running `python plot.py` on the committed CSVs.

## CI

`.github/workflows/ci.yml` on every push:
1. byte-compile (lint),
2. `pytest`,
3. train on Pendulum for a few thousand steps and **assert the deterministic eval
   return improves** (`ci_check.py`) — this proves the code actually learns, not
   just runs.

## Honest limitations

- Single environment (Hopper-v4), **60k steps**, CPU. Not SOTA, not 1M steps.
- Reward scaling is a known sensitivity; this is a controlled ablation, not a new
  algorithm. Effect is large but within seed noise; n=3.
- We bootstrap on time-limit truncation (correct), but do not normalise rewards.
- MuJoCo env warned that Hopper-v4 is superseded by v5; we stayed on v4 to match
  classic SAC benchmarks.
