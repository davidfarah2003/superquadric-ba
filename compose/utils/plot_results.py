#!/usr/bin/env python3
"""Plot benchmark results: superbundle vs superbundle_surface."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG_ROOT = Path(__file__).parent.parent.parent / "logs"
RUNS = {
    "Mast3r BA": LOG_ROOT / "benchmark_ase_sparse_superbundle",
    "Mast3r BA + SUPERDEC": LOG_ROOT / "benchmark_ase_sparse_surface",
}
METRICS = ["pose_auc_5", "pose_ate_rmse", "ray_dirs_err_deg"]
METRIC_LABELS = {
    "pose_auc_5":       "Pose AUC@5° (↑)",
    "pose_ate_rmse":    "Pose ATE RMSE (↓)",
    "ray_dirs_err_deg": "Ray Dir Error deg (↓)",
}
OUT = Path(__file__).parent / "results.png"

results = {}
for label, run_dir in RUNS.items():
    result_file = run_dir / "per_dataset_results.json"
    if not result_file.exists():
        print(f"Missing: {result_file}")
        continue
    data = json.loads(result_file.read_text())
    results[label] = data.get("Average", {})

if not results:
    print("No results found.")
    exit(1)

labels = list(results.keys())
x = np.arange(len(METRICS))
w = 0.35
colors = ["#4C72B0", "#DD8452"]

fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 4))
fig.suptitle("VGGT sparse benchmark — ASE", fontsize=13)

for ax, metric in zip(axes, METRICS):
    for i, (label, avg) in enumerate(results.items()):
        offset = (i - (len(labels) - 1) / 2) * w
        val = avg.get(metric, np.nan)
        ax.bar(offset, val, w, label=label, color=colors[i % len(colors)])
    ax.set_xticks([])
    ax.set_title(METRIC_LABELS[metric])
    ax.legend()

plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"Saved to {OUT}")
