"""Plot learning curves and summary stats from raw per-run progress.csv files.

Usage:
    python plot.py

Reads every logs/hopper_*_s*/progress.csv, groups by arm (baseline vs
rewardscale), and writes figures/learning_curve.png + figures/asymptotic.png
plus a printed summary table. Everything plotted comes straight from the CSVs.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_ROOT = os.path.join(os.path.dirname(__file__), "logs")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")


def arm_of(run_dir: str) -> str:
    name = os.path.basename(run_dir)
    if name.startswith("hopper_baseline"):
        return "Baseline (reward_scale=1.0)"
    if name.startswith("hopper_rewardscale"):
        return "RewardScaled (reward_scale=0.1)"
    return None


def load_runs():
    runs = {}  # arm -> list of (seed, eval_df)
    for csv_path in glob.glob(os.path.join(LOG_ROOT, "hopper_*_s*", "progress.csv")):
        run_dir = os.path.dirname(csv_path)
        arm = arm_of(run_dir)
        if arm is None:
            continue
        df = pd.read_csv(csv_path)
        ev = df[df["phase"] == "eval"].copy()
        ev["env_step"] = ev["env_step"].astype(int)
        ev["eval_return"] = ev["eval_return"].astype(float)
        seed = int(re.search(r"_s(\d+)$", os.path.basename(run_dir)).group(1))
        runs.setdefault(arm, []).append((seed, ev))
    return runs


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    runs = load_runs()
    if not runs:
        raise SystemExit("No hopper eval logs found under logs/")

    # Common x grid from 0 to max step
    all_max = 0
    for arm, rs in runs.items():
        for _, ev in rs:
            all_max = max(all_max, int(ev["env_step"].max()))
    grid = np.arange(5000, all_max + 1, 5000)

    plt.figure(figsize=(7, 4.5))
    summary = {}
    colors = {"Baseline (reward_scale=1.0)": "#1f77b4",
              "RewardScaled (reward_scale=0.1)": "#d62728"}

    for arm, rs in sorted(runs.items()):
        interp = []
        for seed, ev in sorted(rs):
            s = ev.sort_values("env_step")
            y = np.interp(grid, s["env_step"].values, s["eval_return"].values)
            interp.append(y)
        arr = np.vstack(interp)  # (n_seeds, n_grid)
        mean = arr.mean(axis=0)
        sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        ci = 1.96 * sem
        c = colors.get(arm, "#333333")
        plt.plot(grid, mean, color=c, label=f"{arm} (n={arr.shape[0]})")
        plt.fill_between(grid, mean - ci, mean + ci, color=c, alpha=0.2)

        # asymptotic: mean over last 20% of grid, per seed then aggregate
        tail = arr[:, -max(1, int(0.2 * arr.shape[1])):]
        tail_per_seed = tail.mean(axis=1)
        summary[arm] = (tail_per_seed.mean(), tail_per_seed.std(ddof=1) /
                        np.sqrt(len(tail_per_seed)), len(tail_per_seed))

    plt.xlabel("environment steps")
    plt.ylabel("deterministic eval return (5 episodes)")
    plt.title("SAC on Hopper-v4: reward scaling (CPU, 60k steps/run)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out1 = os.path.join(FIG_DIR, "learning_curve.png")
    plt.savefig(out1, dpi=130)
    print("wrote", out1)

    # Asymptotic bar chart
    plt.figure(figsize=(5, 4))
    arms = list(summary.keys())
    means = [summary[a][0] for a in arms]
    cis = [summary[a][1] for a in arms]
    plt.bar(range(len(arms)), means, yerr=cis,
            color=[colors.get(a, "#333") for a in arms], capsize=8, alpha=0.8)
    plt.xticks(range(len(arms)), ["Baseline", "RewardScaled"], rotation=0)
    plt.ylabel("asymptotic eval return (last 20%)")
    plt.title("Mean +/- 95% CI across seeds")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out2 = os.path.join(FIG_DIR, "asymptotic.png")
    plt.savefig(out2, dpi=130)
    print("wrote", out2)

    print("\n=== Asymptotic (last 20% eval return) ===")
    for a, (m, sem, n) in summary.items():
        print(f"{a:35s} mean={m:8.1f}  95%CI~+/-{1.96*sem:6.1f}  seeds={n}")


if __name__ == "__main__":
    main()

