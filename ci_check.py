"""Fast CI gate: train SAC on Pendulum for a few thousand steps and assert the
deterministic eval return actually improved. This proves the implementation
learns end-to-end (not just that the code runs).

Exits non-zero if learning is absent.
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)

# Margin (in eval-return points): 150 is a conservative floor far below the
# seed-0 measured healthy Δ (~+1600 at 4k steps, i.e. ~10x this threshold). A
# collapsed policy keeps eval ~constant (Δ≈0) and still fails; the loose floor
# just keeps CI from flaking on ordinary run-to-run noise.
LEARN_DELTA = 150.0


def learn_gate_pass(eval_returns):
    """Pure pass/fail decision on a sorted-by-step eval_return series.

    Pendulum rewards are negative, so "improved" means Δ > 0. We compare the
    mean of the last 2 eval points against the *first* eval point (step 1000,
    right after learning starts); averaging the tail dampens single-point noise
    versus using only the final point.
    Returns (passed, delta, first, tail_mean).
    """
    arr = np.asarray(eval_returns, dtype=float)
    if arr.size == 0:
        # train.py crashed before the first eval: report a clean FAIL rather
        # than letting arr[0] raise a bare IndexError traceback.
        return False, 0.0, float("nan"), float("nan")
    first = float(arr[0])
    tail_mean = float(arr[-2:].mean())
    delta = tail_mean - first
    return delta >= LEARN_DELTA, delta, first, tail_mean


def main():
    out = os.path.join(HERE, "logs", "ci_pendulum")
    subprocess.run([
        sys.executable, "train.py",
        "--config", os.path.join("configs", "pendulum_ci.yaml"),
        "--seed", "0", "--out", out,
    ], check=True, cwd=HERE)

    df = pd.read_csv(os.path.join(out, "progress.csv"))
    ev = df[df["phase"] == "eval"].sort_values("env_step")
    passed, delta, first, tail_mean = learn_gate_pass(ev["eval_return"].values)
    if np.isnan(first):
        print("FAIL: no eval points recorded — train.py must have crashed "
              "before the first eval", file=sys.stderr)
        sys.exit(1)
    print(f"CI Pendulum eval: first={first:.1f} mean(last 2)={tail_mean:.1f} Δ={delta:+.1f}")
    if not passed:
        print(f"FAIL: reward did not improve enough (Δ={delta:+.1f} < {LEARN_DELTA:.0f}); "
              f"code likely broken", file=sys.stderr)
        sys.exit(1)
    print("PASS: SAC learned on Pendulum")


if __name__ == "__main__":
    main()
