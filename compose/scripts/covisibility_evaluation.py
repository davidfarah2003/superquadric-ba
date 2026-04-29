#!/usr/bin/env python3
"""
Plots covisibility distributions for image pairs that are i steps apart (i=1..30).
For each scene in data/wai, loads the pairwise covisibility matrix and accumulates
covisibility values for pairs (a, b) where |a - b| == i.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

WAI_ROOT = Path(__file__).parent / "data" / "wai"
MAX_STEPS = 30
N_BINS = 100


def find_covisibility_file(scene_root: Path) -> Path | None:
    cov_dir = scene_root / "covisibility" / "v0"
    for f in cov_dir.glob("pairwise_covisibility--*.npy"):
        return f
    return None


def load_covisibility(scene_root: Path) -> np.ndarray | None:
    f = find_covisibility_file(scene_root)
    if f is None:
        print(f"  No covisibility file found in {scene_root}")
        return None
    return np.load(f)


def collect_values_by_step(matrix: np.ndarray) -> dict[int, np.ndarray]:
    n = matrix.shape[0]
    result = {}
    for step in range(1, MAX_STEPS + 1):
        indices_a = np.arange(n - step)
        indices_b = indices_a + step
        # Collect both (a,b) and (b,a) to be symmetric
        vals_ab = matrix[indices_a, indices_b]
        vals_ba = matrix[indices_b, indices_a]
        result[step] = np.concatenate([vals_ab, vals_ba])
    return result


def main():
    scenes = sorted(
        d for d in WAI_ROOT.iterdir()
        if d.is_dir() and (d / "scene_meta.json").exists()
    )
    if not scenes:
        print(f"No scenes found in {WAI_ROOT}")
        return

    # Accumulate values per step across all scenes
    all_values: dict[int, list[np.ndarray]] = {s: [] for s in range(1, MAX_STEPS + 1)}

    for scene_root in scenes:
        print(f"Loading scene {scene_root.name}...")
        matrix = load_covisibility(scene_root)
        if matrix is None:
            continue
        by_step = collect_values_by_step(matrix)
        for step, vals in by_step.items():
            all_values[step].append(vals)

    # Merge per-step arrays
    merged: dict[int, np.ndarray] = {}
    for step in range(1, MAX_STEPS + 1):
        if all_values[step]:
            merged[step] = np.concatenate(all_values[step])

    if not merged:
        print("No covisibility data found.")
        return

    # Plot — derive grid size from MAX_STEPS
    ncols = min(5, MAX_STEPS)
    nrows = (MAX_STEPS + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten()
    for ax in axes_flat[MAX_STEPS:]:
        ax.set_visible(False)
    # Bins over (0, 1] — exclude exact zeros
    bins = np.linspace(0, 1, N_BINS + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for idx, step in enumerate(range(1, MAX_STEPS + 1)):
        ax = axes_flat[idx]
        vals = merged.get(step, np.array([]))
        vals = vals[vals > 0.005]
        if len(vals) == 0:
            ax.set_title(f"step={step} (no data)")
            continue
        counts, _ = np.histogram(vals, bins=bins)
        argmax_val = bin_centers[np.argmax(counts)]
        print(f"step={step:2d}  argmax={argmax_val:.4f}")
        ax.bar(bin_centers, counts, width=bins[1] - bins[0], color="steelblue", align="center")
        ax.axvline(argmax_val, color="crimson", linewidth=1, linestyle="--", label=f"argmax={argmax_val:.3f}")
        ax.legend(fontsize=7)
        ax.set_title(f"step = {step}  (n={len(vals)})")
        ax.set_xlabel("Covisibility")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)

    fig.suptitle("Covisibility distribution by frame-step distance", fontsize=14)
    plt.tight_layout()

    out_path = Path(__file__).parent / "covisibility_by_step.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
