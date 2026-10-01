"""Check alpha trajectory and other training dynamics over time."""
import os, glob, re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")

for arm in ["baseline", "rewardscale"]:
    for s in range(3):
        csv = os.path.join(LOGS, f"hopper_{arm}_s{s}", "progress.csv")
        df = pd.read_csv(csv)
        train = df[df["phase"] == "train"].copy()
        train["env_step"] = train["env_step"].astype(int)
        train = train.sort_values("env_step")
        # Sample alpha at key steps
        print(f"\n{arm}_s{s}:")
        for step_mark in [5000, 10000, 20000, 30000, 40000, 50000, 60000]:
            row = train[train["env_step"] >= step_mark]
            if len(row) > 0:
                r = row.iloc[0]
                print(f"  step~{int(r['env_step']):6d}: alpha={r['alpha']:.4f}, "
                      f"q1_mean={r['q1_mean']:.1f}, last_train_ret={r['episode_return']:.1f}")

# Also check: does rewardscale_s2 show instability?
print("\n\n=== rewardscale_s2 eval trajectory (the runaway seed) ===")
df = pd.read_csv(os.path.join(LOGS, "hopper_rewardscale_s2", "progress.csv"))
ev = df[df["phase"] == "eval"].sort_values("env_step")
for _, r in ev.iterrows():
    print(f"  step={int(r['env_step']):6d}: eval_return={r['eval_return']:.1f}, alpha={r['alpha']:.4f}")

print("\n=== baseline_s0 eval trajectory (has a 1043 spike at 45k) ===")
df = pd.read_csv(os.path.join(LOGS, "hopper_baseline_s0", "progress.csv"))
ev = df[df["phase"] == "eval"].sort_values("env_step")
for _, r in ev.iterrows():
    print(f"  step={int(r['env_step']):6d}: eval_return={r['eval_return']:.1f}, alpha={r['alpha']:.4f}")
