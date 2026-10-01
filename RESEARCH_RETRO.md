# Research retrospective: SAC reward scaling on Hopper

This is a short, honest write-up of what I built and what the numbers actually
said. Not a polished paper — notes I'd want a reviewer (or future me) to read.

## Why this project exists

I came in with engineering projects but zero RL/embodied research on my CV. So
the goal was not to invent a new algorithm — it was to show I can take an
off-policy method, implement it correctly from scratch, instrument it, run a
controlled multi-seed experiment, and report whatever the data says.

## What I built

SAC from scratch in PyTorch: uniform replay buffer, tanh-squashed Gaussian
actor, twin critics, soft target updates, automatic entropy temperature.
Validated end-to-end on Pendulum first (eval return went from ~-1600 to ~-85
in 6k steps — sanity check that the loop learns). Then the actual study on
Hopper-v4.

The one research knob I varied was **reward scaling**: whether rewards entering
the critic's Bellman backup get multiplied by 0.1 vs left at 1.0. Pre-registered
the hypothesis in `PRE_REGISTRATION.md` before looking at any Hopper curve.

## The bug I hit (worth recording)

First Pendulum run looked like it learned nothing: Q-values went positive and
the deterministic eval returned the *identical* bad number at every checkpoint.
I almost wrote it off as "Pendulum just needs more steps", but Q being positive
with all-negative rewards was too wrong. I added a 20-line diagnostic and found
`log_prob` was exploding to 1e15: I had computed the Gaussian log-prob centered
at zero instead of at the predicted mean (`(x)^2` instead of `(x-mu)^2`). That
spurious gradient collapsed the policy. One-line fix, but the lesson is the
workflow: don't trust "more data will fix it" — check the quantities that *must*
have the right sign/magnitude (Q sign here) before burning GPU/CPU hours.

## What the data said

First pass at 60k steps (after an OOM cut) looked like reward scaling won:
scaled mean 1087 vs baseline 363. I honestly flagged it as "directional, not
confirmed" and stopped there.

Then I restored the pre-registered 120k budget and added a **fixed-alpha control**
arm (scale=0.1, alpha frozen at 0.2, no auto-tune) to separate reward scaling from
the auto-tuned alpha trajectory. At 120k, 3 seeds each:

- Baseline (scale=1.0, auto alpha): 1085 +/- 790.
- RewardScaled (scale=0.1, auto alpha): 1118 +/- 705.
- FixedAlpha (scale=0.1, alpha=0.2 fixed): 1404 +/- 715.

Pairwise Welch p in [0.55, 0.95]; one-way ANOVA F=0.22, p~0.90.

**The 60k "effect" did not survive.** At full budget all three arms are
indistinguishable, and the fixed-alpha arm is if anything slightly higher than the
auto-tuned scaled arm — so the earlier gap was early-training seed noise, not
reward scaling doing anything. That's the real result: a null. I'm not going to
pretend the first 60k numbers meant something.

The honest takeaway is the process: a pre-registered hypothesis, a control arm,
and enough budget to actually test it overturned a tempting-looking result. With
n=3 and this variance we have essentially zero power, so "no detectable effect" is
what I can claim — not "reward scaling has no effect".

## Budget honesty

- CPU only. **120k env steps per run** (restored). First attempt at 6-wide was
  OOM-killed on a shared machine; the runner now gates on >=8GB free RAM and
  caps at 3 concurrent, 4 torch threads each.
- 3 seeds per arm. Large seed-to-seed variance (one seed per arm tends to run
  away to ~1800-2100, others sit ~600-1000); I report it, don't smooth it away.
- Only Hopper. No HalfCheetah, no 1M steps.

## What I'd do next

Many more seeds (10+) before claiming anything; checkpointing so a killed run
resumes; normalised rewards as another arm; and treat any "looks better" early
curve as provisional until a control arm and full budget are in.
