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

<!-- Filled after runs; see figures/ and the plot.py summary. -->

## Budget honesty

- CPU only. **60k env steps per run** (I planned 120k, but the first 6-wide
  parallel batch got OOM-killed on a shared machine with only ~8GB free, so I
  cut to 60k and ran 3-wide). This is a small-budget ablation, not a benchmark.
- 3 seeds per arm. Effect sizes are within seed noise; I report them as such.
- Only Hopper. No HalfCheetah, no 1M steps.

## What I'd do next

Resume/checkpointing so a killed run continues instead of restarting; normalised
rewards as a third arm; more seeds once I have a box that isn't shared.
