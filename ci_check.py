"""Fast CI gate: train SAC on Pendulum for a few thousand steps and assert the
deterministic eval return actually improved. This proves the implementation
learns end-to-end (not just that the code runs).

Exits non-zero if learning is absent.
"""
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(__file__)


def main():
    out = os.path.join(HERE, "logs", "ci_pendulum")
    subprocess.run([
        sys.executable, "train.py",
        "--config", os.path.join("configs", "pendulum_ci.yaml"),
        "--seed", "0", "--out", out,
    ], check=True, cwd=HERE)

    df = pd.read_csv(os.path.join(out, "progress.csv"))
    ev = df[df["phase"] == "eval"].sort_values("env_step")
    first = ev["eval_return"].iloc[0]
    last = ev["eval_return"].iloc[-1]
    print(f"CI Pendulum eval: first={first:.1f} last={last:.1f}")
    # Pendulum rewards are negative; "improved" means last is substantially higher.
    if last < first + 150:
        print("FAIL: reward did not improve enough (code likely broken)", file=sys.stderr)
        sys.exit(1)
    print("PASS: SAC learned on Pendulum")


if __name__ == "__main__":
    main()
