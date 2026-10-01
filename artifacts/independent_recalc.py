"""Independent statistical recomputation from raw progress.csv files.
Written by the reviewer, NOT by the project author.
Re-derives everything from scratch without using plot.py.
"""
import os, glob, re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")

# --- 1. Load all 6 hopper runs independently ---
runs = {}  # arm -> {seed: eval_df}
for csv_path in glob.glob(os.path.join(LOGS, "hopper_*_s*", "progress.csv")):
    run_dir = os.path.dirname(csv_path)
    name = os.path.basename(run_dir)
    if name.startswith("hopper_baseline"):
        arm = "baseline"
    elif name.startswith("hopper_rewardscale"):
        arm = "rewardscale"
    else:
        continue
    seed = int(re.search(r"_s(\d+)$", name).group(1))
    df = pd.read_csv(csv_path)
    ev = df[df["phase"] == "eval"].copy()
    ev["env_step"] = ev["env_step"].astype(int)
    ev["eval_return"] = ev["eval_return"].astype(float)
    ev = ev.sort_values("env_step").reset_index(drop=True)
    runs.setdefault(arm, {})[seed] = ev
    print(f"Loaded {name}: {len(ev)} eval points, "
          f"step range [{ev['env_step'].min()}, {ev['env_step'].max()}]")

print()

# --- 2. For each arm, compute asymptotic = mean over last 20% of eval grid ---
# We follow the pre-reg: "last 20% of training steps".
# With eval_every=5000 and 60k steps, eval points at 5k,10k,...,60k = 12 points.
# Last 20% of 60k = last 12k = eval points at 50k,55k,60k (3 points).
# But plot.py uses last 20% of the *grid columns* after interpolation.
# Let's compute BOTH ways and compare.

for arm in ["baseline", "rewardscale"]:
    print(f"=== {arm} ===")
    seeds = sorted(runs[arm].keys())
    print(f"  seeds: {seeds}")

    # Method A: last 20% of env_steps (>= 48k for 60k total)
    # Pre-reg says "last 20% of training steps"
    tail_threshold = 0.8 * 60000  # = 48000
    tail_per_seed_prereg = []
    final_eval_per_seed = []
    for s in seeds:
        ev = runs[arm][s]
        tail = ev[ev["env_step"] >= tail_threshold]
        tail_mean = tail["eval_return"].mean()
        tail_per_seed_prereg.append(tail_mean)
        final_eval_per_seed.append(ev["eval_return"].iloc[-1])
        print(f"  seed {s}: eval points={len(ev)}, "
              f"tail points (>=48k)={len(tail)}, "
              f"tail mean={tail_mean:.1f}, final={ev['eval_return'].iloc[-1]:.1f}, "
              f"all eval={list(ev['eval_return'].round(1))}")

    tail_per_seed_prereg = np.array(tail_per_seed_prereg)
    mean_prereg = tail_per_seed_prereg.mean()
    sem_prereg = tail_per_seed_prereg.std(ddof=1) / np.sqrt(len(tail_per_seed_prereg))
    ci95_prereg = 1.96 * sem_prereg
    print(f"  PRE-REG method (last 20% steps >=48k): "
          f"mean={mean_prereg:.1f}, SEM={sem_prereg:.1f}, 95%CI=+/-{ci95_prereg:.1f}")
    print(f"  per-seed tail means: {list(tail_per_seed_prereg.round(1))}")
    print(f"  final eval points: {[round(x,1) for x in final_eval_per_seed]}")

    # Method B: plot.py method (last 20% of interpolated grid columns)
    # Build common grid
    all_steps = []
    for s in seeds:
        all_steps.extend(runs[arm][s]["env_step"].values)
    grid_max = max(all_steps)
    grid = np.arange(5000, grid_max + 1, 5000)
    interp = []
    for s in seeds:
        ev = runs[arm][s].sort_values("env_step")
        y = np.interp(grid, ev["env_step"].values, ev["eval_return"].values)
        interp.append(y)
    arr = np.vstack(interp)
    n_grid = arr.shape[1]
    n_tail = max(1, int(0.2 * n_grid))
    tail_plot = arr[:, -n_tail:]
    tail_per_seed_plot = tail_plot.mean(axis=1)
    mean_plot = tail_per_seed_plot.mean()
    sem_plot = tail_per_seed_plot.std(ddof=1) / np.sqrt(len(tail_per_seed_plot))
    ci95_plot = 1.96 * sem_plot
    print(f"  PLOT.PY method (last 20% of {n_grid} grid cols = last {n_tail}): "
          f"mean={mean_plot:.1f}, SEM={sem_plot:.1f}, 95%CI=+/-{ci95_plot:.1f}")
    print(f"  per-seed tail means: {list(tail_per_seed_plot.round(1))}")
    print()

# --- 3. Check training dynamics: alpha, Q-values ---
print("=== Training dynamics (last row of each run) ===")
for arm in ["baseline", "rewardscale"]:
    for s in sorted(runs[arm].keys()):
        csv_path = os.path.join(LOGS, f"hopper_{arm}_s{s}", "progress.csv")
        df = pd.read_csv(csv_path)
        train = df[df["phase"] == "train"]
        last_train = train.iloc[-1]
        ev = runs[arm][s]
        print(f"  {arm}_s{s}: final alpha={last_train['alpha']}, "
              f"final q1_mean={last_train['q1_mean']}, "
              f"last train return={last_train['episode_return']:.1f}, "
              f"final eval={ev['eval_return'].iloc[-1]:.1f}")

# --- 4. Check episode lengths (truncation vs termination) ---
print()
print("=== Episode length distribution (Hopper max is 1000) ===")
for arm in ["baseline", "rewardscale"]:
    for s in sorted(runs[arm].keys()):
        csv_path = os.path.join(LOGS, f"hopper_{arm}_s{s}", "progress.csv")
        df = pd.read_csv(csv_path)
        train = df[df["phase"] == "train"]
        lengths = train["episode_length"].dropna().values
        print(f"  {arm}_s{s}: n_episodes={len(lengths)}, "
              f"mean_len={lengths.mean():.0f}, max_len={lengths.max():.0f}, "
              f"frac_at_1000={np.mean(lengths>=1000):.2%}")

# --- 5. Welch's t-test between arms (descriptive, not powered) ---
print()
print("=== Welch's t-test (descriptive, n=3 per arm - NOT powered) ===")
from scipy import stats
for method_name, use_prereg in [("pre-reg(>=48k)", True), ("plot.py grid", False)]:
    means = {}
    for arm in ["baseline", "rewardscale"]:
        seeds = sorted(runs[arm].keys())
        if use_prereg:
            tail_thr = 0.8 * 60000
            tps = [runs[arm][s][runs[arm][s]["env_step"] >= tail_thr]["eval_return"].mean()
                   for s in seeds]
        else:
            all_steps = []
            for s in seeds:
                all_steps.extend(runs[arm][s]["env_step"].values)
            grid = np.arange(5000, max(all_steps)+1, 5000)
            interp = [np.interp(grid,
                                runs[arm][s].sort_values("env_step")["env_step"].values,
                                runs[arm][s].sort_values("env_step")["eval_return"].values)
                      for s in seeds]
            arr = np.vstack(interp)
            n_tail = max(1, int(0.2 * arr.shape[1]))
            tps = list(arr[:, -n_tail:].mean(axis=1))
        means[arm] = np.array(tps)
    t_stat, p_val = stats.ttest_ind(means["rewardscale"], means["baseline"], equal_var=False)
    print(f"  {method_name}: t={t_stat:.3f}, p={p_val:.4f} "
          f"(baseline={list(means['baseline'].round(1))}, "
          f"scaled={list(means['rewardscale'].round(1))})")
