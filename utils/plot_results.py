#!/usr/bin/env python3
"""Plot benchmark results across runs."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BENCH_ROOT = Path(__file__).parent.parent.parent / "logs" / "benchmark_sparse_ase_vggt"
METRICS = ["pose_auc_5", "pose_ate_rmse", "ray_dirs_err_deg"]
METRIC_LABELS = {"pose_auc_5": "Pose AUC@5° (↑)", "pose_ate_rmse": "Pose ATE RMSE (↓)", "ray_dirs_err_deg": "Ray Dir Error deg (↓)"}
OUT = Path(__file__).parent / "results.png"

# Load all runs
runs = {}
for run_dir in sorted(BENCH_ROOT.iterdir()):
    result_file = run_dir / "per_dataset_results.json"
    if not result_file.exists():
        continue
    data = json.loads(result_file.read_text())
    runs[run_dir.name] = data.get("Average", {})

if not runs:
    print("No results found.")
    exit(1)

# Split into BA / no-BA groups keyed by threshold
no_ba, ba = {}, {}
for name, avg in runs.items():
    m = re.match(r"thres_([\d.]+)(_bundle_adjustment)?$", name)
    if not m:
        continue
    thres = float(m.group(1))
    (ba if m.group(2) else no_ba)[thres] = avg

thresholds = sorted(set(no_ba) | set(ba))
x = np.arange(len(thresholds))
labels = [str(t) for t in thresholds]

fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 4))
fig.suptitle("VGGT sparse benchmark — ASE", fontsize=13)

for ax, metric in zip(axes, METRICS):
    no_ba_vals = [no_ba.get(t, {}).get(metric, np.nan) for t in thresholds]
    ba_vals    = [ba.get(t, {}).get(metric, np.nan)    for t in thresholds]

    w = 0.35
    ax.bar(x - w/2, no_ba_vals, w, label="No BA",           color="#4C72B0")
    ax.bar(x + w/2, ba_vals,    w, label="BA (GT pts)",      color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([f"thres={l}" for l in labels])
    ax.set_title(METRIC_LABELS[metric])
    ax.legend()

plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"Saved to {OUT}")
