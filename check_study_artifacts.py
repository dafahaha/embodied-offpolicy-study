"""CI artifact-freshness gate for the committed Hopper figures and statistics.

P2-1 guard: the committed PNGs once lagged behind ``plot.py`` (rendered before
the fixed ``ARM_ORDER``), so the bars/legend order disagreed with the README
table. PNG bytes are NOT reproducible across renders (matplotlib metadata and
antialiasing differ; recorded as P3-8), so this gate cannot hash the images.
Instead it:

  1. recomputes the asymptotic statistics straight from the committed raw CSVs
     and asserts they equal the committed/expected key numbers — drift in the
     data or in ``plot.py``'s statistics fails the gate;
  2. asserts the fixed ``ARM_ORDER`` places Baseline, RewardScaled, FixedAlpha in
     that order (the old glob-insertion-order bug would swap the last two);
  3. asserts both committed copies (``figures/`` and ``docs/assets/``) exist and
     are byte-identical (a hand-edited single copy fails).

Run from the repository root:

    python check_study_artifacts.py

Exits non-zero on any mismatch. No training is performed.
"""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

from plot import ARM_ORDER, asymptotic_stats, load_runs, validate_grid_and_counts

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
DOC_ASSETS_DIR = os.path.join(HERE, "docs", "assets")

# Committed expected key numbers (independently recomputed from the raw CSVs;
# mirrors docs/assets/results.json and the README table). Tolerances are loose
# enough to absorb float noise but tight enough to catch real drift.
EXPECTED = {
    "Baseline (reward_scale=1.0)": {"mean": 1084.56, "sd": 697.72, "ci": 1733.22},
    "RewardScaled (reward_scale=0.1)": {"mean": 1118.16, "sd": 622.88, "ci": 1547.31},
    "FixedAlpha (auto_tune=off)": {"mean": 1403.64, "sd": 631.69, "ci": 1569.20},
}
# Welch p-values in ARM_ORDER pairwise order: B vs RS, B vs FA, FA vs RS.
EXPECTED_WELCH_P = [0.953421, 0.588913, 0.606984]
EXPECTED_ANOVA = {"F": 0.2172, "p": 0.810835}

TOL = {"mean": 1.0, "sd": 0.5, "ci": 1.0, "p": 5e-3, "F": 0.02, "anova_p": 5e-3}


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_arm_order() -> list[str]:
    """Assert ARM_ORDER fixes Baseline, RewardScaled, FixedAlpha in that order."""
    problems = []
    expected_order = [
        "Baseline (reward_scale=1.0)",
        "RewardScaled (reward_scale=0.1)",
        "FixedAlpha (auto_tune=off)",
    ]
    if ARM_ORDER != expected_order:
        problems.append(f"ARM_ORDER drifted: {ARM_ORDER} != {expected_order}")
    return problems


def check_stats(per_seed: dict) -> list[str]:
    """Recompute stats from per-seed tail means and compare to EXPECTED."""
    problems = []
    res = asymptotic_stats(per_seed)
    for arm, exp in EXPECTED.items():
        if arm not in res["means"]:
            problems.append(f"missing arm {arm} in recomputed stats")
            continue
        pairs = [
            ("mean", res["means"][arm], exp["mean"], TOL["mean"]),
            ("sd", res["sds"][arm], exp["sd"], TOL["sd"]),
            ("ci", res["cis"][arm], exp["ci"], TOL["ci"]),
        ]
        for label, got, want, tol in pairs:
            if abs(got - want) > tol:
                problems.append(
                    f"{arm} {label}: recomputed {got:.2f} != committed {want:.2f} " f"(tol {tol})"
                )
    for (a_i, a_j, _t, p), want_p in zip(res["welch"], EXPECTED_WELCH_P):
        if abs(p - want_p) > TOL["p"]:
            problems.append(f"Welch {a_i} vs {a_j}: p={p:.4f} != committed {want_p:.4f}")
    if abs(res["F"] - EXPECTED_ANOVA["F"]) > TOL["F"]:
        problems.append(f"ANOVA F={res['F']:.4f} != committed {EXPECTED_ANOVA['F']}")
    if abs(res["p"] - EXPECTED_ANOVA["p"]) > TOL["anova_p"]:
        problems.append(f"ANOVA p={res['p']:.4f} != committed {EXPECTED_ANOVA['p']}")
    return problems


def check_png_copies() -> list[str]:
    """Both committed copies must exist and be byte-identical to each other."""
    problems = []
    for name in ("learning_curve.png", "asymptotic.png"):
        a = os.path.join(FIG_DIR, name)
        b = os.path.join(DOC_ASSETS_DIR, name)
        if not os.path.isfile(a):
            problems.append(f"missing committed figure {a}")
            continue
        if not os.path.isfile(b):
            problems.append(f"missing committed copy {b}")
            continue
        if _md5(a) != _md5(b):
            problems.append(f"{name}: figures/ and docs/assets/ copies differ (drift)")
    return problems


def main() -> int:
    problems = []
    problems += check_arm_order()

    runs = load_runs()
    validate_grid_and_counts(runs)
    # Rebuild per_seed in fixed ARM_ORDER, exactly as plot.py's main() does.
    per_seed = {}
    ordered = [a for a in ARM_ORDER if a in runs] + [a for a in runs if a not in ARM_ORDER]
    for arm in ordered:
        cols = []
        for _seed, ev in sorted(runs[arm]):
            cols.append(ev.sort_values("env_step")["eval_return"].values.astype(float))
        arr = np.vstack(cols)
        per_seed[arm] = arr[:, -4:].mean(axis=1)

    problems += check_stats(per_seed)
    problems += check_png_copies()

    if problems:
        print("FAIL: study artifact freshness gate:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("PASS: committed figures/statistics match recomputed values; arm order fixed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
