"""Independent statistical recomputation for the SAC reward-scaling study.

This script re-derives every number shown on the study page directly from the
committed raw per-run CSVs under ``logs/hopper_*_s*/progress.csv``. It does
not import the project's ``plot.py``; it reads the CSVs itself so the page's
statistics can be audited independently.

Run from the repository root:

    python docs/assets/recompute_stats.py

It prints a summary table and writes ``docs/assets/results.json`` next to
this file. No training is performed; only the committed eval rows are read.

Caliber (identical to ``plot.py`` / PRE_REGISTRATION.md):
  * deterministic eval return, 5 episodes per eval point;
  * asymptotic window = last 4 eval points (105k-120k steps, the last 20%);
  * one per-seed tail mean per run; n = 3 seeds per arm;
  * sample SD uses ddof=1; t-based 95% CI uses t_{0.975, df=n-1=2} = 4.303.
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

# docs/assets/recompute_stats.py -> repo root is two levels up.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LOG_ROOT = os.path.join(ROOT, "logs")
TAIL_EVAL_POINTS = 4
# Pre-registered secondary endpoint (hypothesis #2): eval return at 50k steps,
# reported descriptively only (no inferential test at n = 3).
EARLY_STEP = 50000

ARM_LABELS = {
    "baseline": "Baseline (reward_scale=1.0, auto alpha)",
    "rewardscale": "RewardScaled (reward_scale=0.1, auto alpha)",
    "fixedalpha": "FixedAlpha (reward_scale=0.1, alpha=0.2 frozen)",
}


def arm_of(run_dir: str) -> str | None:
    name = os.path.basename(run_dir)
    if name.startswith("hopper_baseline"):
        return "baseline"
    if name.startswith("hopper_rewardscale"):
        return "rewardscale"
    if name.startswith("hopper_fixedalpha"):
        return "fixedalpha"
    return None


def load_tail_means() -> dict[str, list[float]]:
    """Return {arm: [per-seed tail means in seed order s0, s1, s2]}."""
    per_seed: dict[str, dict[int, np.ndarray]] = {}
    dynamics: dict[str, dict[int, dict[str, float]]] = {}
    for csv_path in glob.glob(os.path.join(LOG_ROOT, "hopper_*_s*", "progress.csv")):
        run_dir = os.path.dirname(csv_path)
        arm = arm_of(run_dir)
        if arm is None:
            continue
        df = pd.read_csv(csv_path)
        ev = df[df["phase"] == "eval"].copy()
        ev["env_step"] = ev["env_step"].astype(int)
        ev["eval_return"] = ev["eval_return"].astype(float)
        ev = ev.sort_values("env_step").reset_index(drop=True)
        if not np.isfinite(ev["eval_return"].values).all():
            raise ValueError(f"non-finite eval_return in {csv_path}")
        seed = int(re.search(r"_s(\d+)$", os.path.basename(run_dir)).group(1))
        per_seed.setdefault(arm, {})[seed] = ev["eval_return"].values

        # Descriptive training dynamics: last logged alpha and q1_mean per seed.
        tr = df[df["phase"] == "train"].dropna(subset=["alpha"])
        if len(tr):
            dynamics.setdefault(arm, {})[seed] = {
                "final_alpha": float(tr["alpha"].iloc[-1]),
                "final_q1_mean": float(tr["q1_mean"].iloc[-1]),
            }

    out: dict[str, list[float]] = {}
    dyn_out: dict[str, dict] = {}
    for arm, seeds in per_seed.items():
        tail = [float(arr[-TAIL_EVAL_POINTS:].mean()) for _, arr in sorted(seeds.items())]
        out[arm] = tail
        ds = [dynamics[arm][s] for s in sorted(dynamics[arm])]
        dyn_out[arm] = {
            "final_alpha_per_seed": [round(d["final_alpha"], 4) for d in ds],
            "final_alpha_mean": round(float(np.mean([d["final_alpha"] for d in ds])), 4),
            "final_q1_mean_per_seed": [round(d["final_q1_mean"], 1) for d in ds],
            "final_q1_mean_mean": round(float(np.mean([d["final_q1_mean"] for d in ds])), 1),
        }
    return out, dyn_out


def load_early_point(step: int = EARLY_STEP) -> dict:
    """Descriptive pre-registered secondary endpoint: single eval point at `step`.

    Returns {arm: {per_seed: [...s0,s1,s2...], mean, sd_ddof1}}. No inferential
    test is performed; n = 3 has no power and this endpoint was not pre-registered
    for hypothesis testing.
    """
    per_seed: dict[str, dict[int, float]] = {}
    for csv_path in glob.glob(os.path.join(LOG_ROOT, "hopper_*_s*", "progress.csv")):
        run_dir = os.path.dirname(csv_path)
        arm = arm_of(run_dir)
        if arm is None:
            continue
        df = pd.read_csv(csv_path)
        ev = df[df["phase"] == "eval"].copy()
        ev["env_step"] = ev["env_step"].astype(int)
        ev["eval_return"] = ev["eval_return"].astype(float)
        row = ev[ev["env_step"] == step]
        if len(row) != 1:
            raise ValueError(f"expected one eval row at {step} in {csv_path}, got {len(row)}")
        seed = int(re.search(r"_s(\d+)$", os.path.basename(run_dir)).group(1))
        per_seed.setdefault(arm, {})[seed] = float(row["eval_return"].iloc[0])

    out: dict[str, dict] = {}
    for arm in ["baseline", "rewardscale", "fixedalpha"]:
        vals = np.asarray([per_seed[arm][s] for s in sorted(per_seed[arm])])
        out[arm] = {
            "per_seed": [round(float(x), 2) for x in vals],
            "mean": round(float(vals.mean()), 2),
            "sd_ddof1": round(float(vals.std(ddof=1)), 2),
        }
    return out


def welch(a: np.ndarray, b: np.ndarray) -> dict:
    res = stats.ttest_ind(a, b, equal_var=False)
    # Welch-Satterthwaite degrees of freedom for the mean-difference CI.
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    se = np.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    tcrit = stats.t.ppf(0.975, df)
    diff = a.mean() - b.mean()
    return {
        "t": float(res.statistic),
        "p": float(res.pvalue),
        "welch_df": float(df),
        "mean_diff": float(diff),
        "se_diff": float(se),
        "ci95_low": float(diff - tcrit * se),
        "ci95_high": float(diff + tcrit * se),
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> dict:
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    d = (a.mean() - b.mean()) / sp
    # Hedges small-sample correction J(df=na+nb-2=4)
    J = 1.0 - 3.0 / (4.0 * (na + nb - 2) - 1.0)
    return {"cohens_d": float(d), "hedges_g": float(J * d), "pooled_sd": float(sp)}


def main() -> None:
    tails, dynamics = load_tail_means()
    n = 3
    tcrit = float(stats.t.ppf(0.975, df=n - 1))  # 4.303
    arms = ["baseline", "rewardscale", "fixedalpha"]

    summary = {}
    for arm in arms:
        v = np.asarray(tails[arm])
        mean = float(v.mean())
        sd = float(v.std(ddof=1))
        half = float(tcrit * sd / np.sqrt(n))
        summary[arm] = {
            "label": ARM_LABELS[arm],
            "per_seed_tail_means": [round(float(x), 2) for x in v],
            "mean": round(mean, 1),
            "sd": round(sd, 1),
            "ci95_halfwidth": round(half, 1),
            "ci95_low": round(mean - half, 1),
            "ci95_high": round(mean + half, 1),
        }

    pairwise = {}
    for x, y in [
        ("baseline", "rewardscale"),
        ("baseline", "fixedalpha"),
        ("fixedalpha", "rewardscale"),
    ]:
        pairwise[f"{x}_vs_{y}"] = welch(np.asarray(tails[x]), np.asarray(tails[y]))

    F, p_anova = stats.f_oneway(*[np.asarray(tails[a]) for a in arms])

    primary = welch(np.asarray(tails["baseline"]), np.asarray(tails["rewardscale"]))
    primary.update(cohens_d(np.asarray(tails["baseline"]), np.asarray(tails["rewardscale"])))

    # Pre-registered secondary endpoint (hypothesis #2: faster early learning at
    # 50k). Descriptive only: single eval point per seed, n=3, no inferential test.
    early = load_early_point(EARLY_STEP)

    results = {
        "caliber": {
            "env": "Hopper-v4",
            "steps_per_run": 120000,
            "seeds_per_arm": n,
            "asymptotic_window": "last 4 eval points (105k-120k)",
            "sd_ddof": 1,
            "tcrit_975_df2": round(tcrit, 4),
            "note": "t-based 95% CI; n=3, intervals are wide and overlap.",
        },
        "arms": summary,
        "pairwise_welch": pairwise,
        "anova": {"F": round(float(F), 3), "p": round(float(p_anova), 4)},
        "primary_contrast_baseline_vs_rewardscale": primary,
        "training_dynamics_final_step": dynamics,
        "early_learning_50k_descriptive": {
            "env_step": EARLY_STEP,
            "note": "Pre-registered secondary endpoint (hypothesis #2, faster early "
            "learning); single eval point per seed, n=3, descriptive only, no "
            "inferential test. The pre-registered RewardScaled arm shows no "
            "advantage over Baseline here.",
            "arms": early,
        },
    }

    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=== Asymptotic (last 4 eval points, 105k-120k) ===")
    for arm in arms:
        s = summary[arm]
        print(
            f"{arm:12s} mean={s['mean']:8.1f} SD={s['sd']:7.1f} "
            f"t95%CI=+/-{s['ci95_halfwidth']:7.1f} seeds={s['per_seed_tail_means']}"
        )
    print("\n=== Pairwise Welch (equal_var=False) ===")
    for k, v in pairwise.items():
        print(
            f"{k}: t={v['t']:+.3f} p={v['p']:.4f} "
            f"mean_diff={v['mean_diff']:+.1f} "
            f"95%CI=[{v['ci95_low']:.1f}, {v['ci95_high']:.1f}]"
        )
    print(f"\nANOVA F={F:.3f} p={p_anova:.4f}")
    print(f"Primary d={primary['cohens_d']:.3f}, g={primary['hedges_g']:.3f}")
    print("\n=== Final-step training dynamics (last train row per seed) ===")
    for arm in arms:
        d = dynamics[arm]
        print(
            f"{arm:12s} alpha={d['final_alpha_per_seed']} "
            f"(mean {d['final_alpha_mean']}), q1_mean={d['final_q1_mean_per_seed']}"
        )
    print(f"\n=== Early learning at {EARLY_STEP} (descriptive, no test) ===")
    for arm in arms:
        e = early[arm]
        print(
            f"{arm:12s} mean={e['mean']:8.2f} SD={e['sd_ddof1']:8.2f} " f"per_seed={e['per_seed']}"
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
