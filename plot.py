"""Plot learning curves and summary stats from raw per-run progress.csv files.

Usage:
    python plot.py

Reads every logs/hopper_*_s*/progress.csv, groups by arm, and writes
figures/learning_curve.png + figures/asymptotic.png plus a printed summary table
and exact Welch / ANOVA statistics (scipy). Everything plotted comes straight
from the CSVs.

Guards: every run must eval on the exact same env_step grid (up to the common
max) and every arm must have the same number of seeds; otherwise we raise
loudly instead of letting np.interp silently hold a killed run's last value.
Non-finite eval_return also aborts rather than poisoning the means/SDs.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_ROOT = os.path.join(os.path.dirname(__file__), "logs")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
# "last 20%" asymptotic window = last 4 eval points (105k-120k here).
TAIL_EVAL_POINTS = 4


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
        if not np.isfinite(ev["eval_return"].values).all():
            raise ValueError(
                f"non-finite eval_return in {csv_path}; refusing to plot "
                f"(a diverged run would silently poison the mean/SD)"
            )
        seed = int(re.search(r"_s(\d+)$", os.path.basename(run_dir)).group(1))
        runs.setdefault(arm, []).append((seed, ev))
    return runs


def validate_grid_and_counts(runs: dict):
    """Enforce equal seed counts and an identical eval grid across all runs.

    Returns the common env_step grid (int array, ascending). Raises SystemExit
    with a diagnostic listing what is missing/extra.
    """
    counts = {arm: len(rs) for arm, rs in runs.items()}
    if len(set(counts.values())) != 1:
        raise SystemExit(
            f"arms have unequal seed counts {counts}; cannot pool / compare"
        )
    ref_grid = None
    for arm, rs in runs.items():
        for seed, ev in rs:
            steps = tuple(sorted(int(s) for s in ev["env_step"].tolist()))
            if ref_grid is None:
                ref_grid = steps
                continue
            if steps != ref_grid:
                missing = sorted(set(ref_grid) - set(steps))
                extra = sorted(set(steps) - set(ref_grid))
                raise SystemExit(
                    f"run {arm}/s{seed} eval grid differs from the reference grid "
                    f"(missing={missing}, extra={extra}); refusing to silently "
                    f"hold its last value via np.interp"
                )
    return np.asarray(ref_grid, dtype=int)


def asymptotic_stats(per_seed: dict) -> dict:
    """Pure Welch / t-CI / ANOVA recomputation from per-seed tail means.

    Separated from plotting so it can be unit-tested against scipy directly.
    `per_seed` maps arm name -> 1-D array of per-seed tail means.
    """
    arms = list(per_seed.keys())
    n = len(per_seed[arms[0]])
    tcrit = stats.t.ppf(0.975, df=n - 1)  # 4.303 for n=3
    means = {a: float(per_seed[a].mean()) for a in arms}
    sds = {a: float(per_seed[a].std(ddof=1)) for a in arms}
    cis = {a: float(tcrit * per_seed[a].std(ddof=1) / np.sqrt(n)) for a in arms}
    welch = []
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            res = stats.ttest_ind(per_seed[arms[i]], per_seed[arms[j]],
                                  equal_var=False)
            welch.append((arms[i], arms[j], float(res.statistic),
                          float(res.pvalue)))
    F, p = stats.f_oneway(*[per_seed[a] for a in arms])
    return {"n": n, "tcrit": float(tcrit), "means": means, "sds": sds,
            "cis": cis, "welch": welch, "F": float(F), "p": float(p)}


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    runs = load_runs()
    if not runs:
        raise SystemExit("No hopper eval logs found under logs/")

    grid = validate_grid_and_counts(runs)
    all_max = int(grid.max())

    colors = {"Baseline (reward_scale=1.0)": "#1f77b4",
              "RewardScaled (reward_scale=0.1)": "#d62728",
              "FixedAlpha (auto_tune=off)": "#2ca02c"}
    short = {"Baseline (reward_scale=1.0)": "Baseline",
             "RewardScaled (reward_scale=0.1)": "RewardScaled",
             "FixedAlpha (auto_tune=off)": "FixedAlpha"}

    # ---- Learning curves: mean line +- 1 SD across seeds ----
    plt.figure(figsize=(7, 4.5))
    summary = {}
    per_seed = {}
    for arm, rs in sorted(runs.items()):
        cols = []
        for seed, ev in sorted(rs):
            s = ev.sort_values("env_step")
            # grids already validated identical -> direct alignment, no interp
            # (np.interp would silently hold a killed run's tail value).
            assert (s["env_step"].values == grid).all()
            cols.append(s["eval_return"].values.astype(float))
        arr = np.vstack(cols)
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1)
        c = colors.get(arm, "#333333")
        plt.plot(grid, mean, color=c, label=f"{short.get(arm,arm)} (n={arr.shape[0]})")
        plt.fill_between(grid, mean - sd, mean + sd, color=c, alpha=0.2)

        tail = arr[:, -TAIL_EVAL_POINTS:]
        tail_per_seed = tail.mean(axis=1)
        summary[arm] = tail_per_seed
        per_seed[arm] = tail_per_seed

    plt.xlabel("environment steps")
    plt.ylabel("deterministic eval return (5 episodes)")
    plt.title(f"SAC on Hopper-v4 (CPU, {all_max//1000}k steps/run); "
              f"shade = +/- 1 SD across seeds")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out1 = os.path.join(FIG_DIR, "learning_curve.png")
    plt.savefig(out1, dpi=130)
    print("wrote", out1)

    # ---- Asymptotic: raw seed points + mean + t-based 95% CI (df=n-1) ----
    stats_res = asymptotic_stats(per_seed)
    arms = list(summary.keys())
    n = stats_res["n"]
    tcrit = stats_res["tcrit"]

    plt.figure(figsize=(6.5, 4.5))
    x = np.arange(len(arms))
    plt.bar(x, [stats_res["means"][a] for a in arms],
            yerr=[stats_res["cis"][a] for a in arms],
            color=[colors.get(a, "#333") for a in arms],
            capsize=8, alpha=0.55)
    # overlay raw seed points, jittered
    rng = np.random.default_rng(0)
    for i, a in enumerate(arms):
        jitter = rng.uniform(-0.08, 0.08, size=len(per_seed[a]))
        plt.scatter(x[i] + jitter, per_seed[a], color=colors.get(a, "#333"),
                    zorder=3, s=28, edgecolor="black", linewidth=0.5)
    plt.xticks(x, [short.get(a, a) for a in arms])
    plt.ylabel("asymptotic eval return (last 4 eval points)")
    plt.title(f"mean + t-based 95% CI (df={n-1}); dots = individual seeds")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out2 = os.path.join(FIG_DIR, "asymptotic.png")
    plt.savefig(out2, dpi=130)
    print("wrote", out2)

    print(f"\n=== Asymptotic (last {TAIL_EVAL_POINTS} eval points, i.e. 105k-120k) ===")
    for a in arms:
        v = summary[a]
        print(f"{short.get(a,a):14s} mean={v.mean():8.1f}  SD={v.std(ddof=1):7.1f}  "
              f"t95%CI=+/-{tcrit*v.std(ddof=1)/np.sqrt(n):6.1f}  seeds={list(np.round(v).astype(int))}")

    print("\n=== Pairwise Welch t-test (scipy, equal_var=False) ===")
    for a_i, a_j, t, p in stats_res["welch"]:
        print(f"{short.get(a_i,a_i)} vs {short.get(a_j,a_j)}: "
              f"t={t:+.2f}  p={p:.3f}")

    print(f"\nOne-way ANOVA (scipy.f_oneway): F={stats_res['F']:.2f}  p={stats_res['p']:.3f}")


if __name__ == "__main__":
    main()
