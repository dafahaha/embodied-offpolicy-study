"""Plot learning curves and summary stats from raw per-run progress.csv files.

Usage:
    python plot.py

Reads every logs/hopper_*_s*/progress.csv, groups by arm, and writes
figures/learning_curve.png + figures/asymptotic.png plus a printed summary table
and Welch / ANOVA statistics. Everything plotted comes straight from the CSVs.
"""
from __future__ import annotations

import glob
import math
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
    if name.startswith("hopper_fixedalpha"):
        return "FixedAlpha (auto_tune=off)"
    return None


def load_runs():
    runs = {}
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


def welch(a, b):
    """Welch's t-test (two-sided), normal approx. Returns (t, p). No scipy."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, 1.0
    t = (ma - mb) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p


def oneway_anova(groups):
    """One-way ANOVA (F, rough p). groups: list of arrays. Indicative at n=3."""
    k = len(groups)
    allv = np.concatenate(groups)
    grand = allv.mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    dfb, dfw = k - 1, len(allv) - k
    if ssw == 0:
        return 0.0, 1.0
    F = (ssb / dfb) / (ssw / dfw)
    p = math.exp(-0.5 * abs(F))  # rough; indicative only at n=3
    return F, p


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    runs = load_runs()
    if not runs:
        raise SystemExit("No hopper eval logs found under logs/")

    all_max = 0
    for arm, rs in runs.items():
        for _, ev in rs:
            all_max = max(all_max, int(ev["env_step"].max()))
    grid = np.arange(5000, all_max + 1, 5000)

    plt.figure(figsize=(7, 4.5))
    summary = {}
    per_seed = {}
    colors = {"Baseline (reward_scale=1.0)": "#1f77b4",
              "RewardScaled (reward_scale=0.1)": "#d62728",
              "FixedAlpha (auto_tune=off)": "#2ca02c"}
    short = {"Baseline (reward_scale=1.0)": "Baseline",
             "RewardScaled (reward_scale=0.1)": "RewardScaled",
             "FixedAlpha (auto_tune=off)": "FixedAlpha"}

    for arm, rs in sorted(runs.items()):
        interp = []
        for seed, ev in sorted(rs):
            s = ev.sort_values("env_step")
            y = np.interp(grid, s["env_step"].values, s["eval_return"].values)
            interp.append(y)
        arr = np.vstack(interp)
        mean = arr.mean(axis=0)
        sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        ci = 1.96 * sem
        c = colors.get(arm, "#333333")
        plt.plot(grid, mean, color=c, label=f"{arm} (n={arr.shape[0]})")
        plt.fill_between(grid, mean - ci, mean + ci, color=c, alpha=0.2)

        tail = arr[:, -max(1, int(0.2 * arr.shape[1])):]
        tail_per_seed = tail.mean(axis=1)
        summary[arm] = (tail_per_seed.mean(), tail_per_seed.std(ddof=1) /
                        np.sqrt(len(tail_per_seed)), len(tail_per_seed))
        per_seed[arm] = tail_per_seed

    plt.xlabel("environment steps")
    plt.ylabel("deterministic eval return (5 episodes)")
    plt.title(f"SAC on Hopper-v4: reward scaling (CPU, {all_max//1000}k steps/run)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out1 = os.path.join(FIG_DIR, "learning_curve.png")
    plt.savefig(out1, dpi=130)
    print("wrote", out1)

    plt.figure(figsize=(6, 4.5))
    arms = list(summary.keys())
    means = [summary[a][0] for a in arms]
    cis = [1.96 * summary[a][1] for a in arms]
    plt.bar(range(len(arms)), means, yerr=cis,
            color=[colors.get(a, "#333") for a in arms], capsize=8, alpha=0.8)
    plt.xticks(range(len(arms)), [short.get(a, a) for a in arms], rotation=0)
    plt.ylabel("asymptotic eval return (last 20%)")
    plt.title("Mean +/- 95% CI (1.96 x SEM) across seeds")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out2 = os.path.join(FIG_DIR, "asymptotic.png")
    plt.savefig(out2, dpi=130)
    print("wrote", out2)

    print("\n=== Asymptotic (last 20% eval return) ===")
    for a, (m, sem, n) in summary.items():
        print(f"{a:35s} mean={m:8.1f}  95%CI~+/-{1.96*sem:6.1f}  seeds={n}")

    print("\n=== Pairwise Welch t (indicative; n=3 each, low power) ===")
    keys = list(summary.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            t, p = welch(per_seed[keys[i]], per_seed[keys[j]])
            print(f"{short.get(keys[i],keys[i])} vs {short.get(keys[j],keys[j])}: "
                  f"t={t:+.2f}  p~{p:.3f}")

    if len(keys) >= 2:
        F, p = oneway_anova([per_seed[k] for k in keys])
        print(f"\nOne-way ANOVA across {len(keys)} arms: F={F:.2f}  p~{p:.3f} "
              "(indicative; do not over-interpret at n=3)")


if __name__ == "__main__":
    main()
