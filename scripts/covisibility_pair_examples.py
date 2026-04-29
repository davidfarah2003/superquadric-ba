#!/usr/bin/env python3
"""
For each step size 1..15, finds an image pair whose covisibility is within
epsilon of the argmax of that step's distribution (excluding values <= 0.005),
then saves a side-by-side PNG to compose/data/covisibility_analysis/.

Argmax values are recomputed from the covisibility matrices (same logic as
covisibility_evaluation.py) so this script is self-contained.
"""

from pathlib import Path

import json
import numpy as np
from PIL import Image

WAI_ROOT = Path(__file__).parent / "data" / "wai"
OUT_DIR = Path(__file__).parent / "data" / "covisibility_analysis"
MAX_STEPS = 30
N_BINS = 100
EPSILON = 0.01  # search window around argmax


def find_covisibility_file(scene_root: Path) -> Path | None:
    for f in (scene_root / "covisibility" / "v0").glob("pairwise_covisibility--*.npy"):
        return f
    return None


def load_scene(scene_root: Path) -> tuple[np.ndarray, list[dict]] | None:
    cov_file = find_covisibility_file(scene_root)
    if cov_file is None:
        return None
    matrix = np.load(cov_file)
    with open(scene_root / "scene_meta.json") as f:
        meta = json.load(f)
    return matrix, meta["frames"]


def compute_argmaxes(scenes: list[tuple[np.ndarray, list, Path]]) -> dict[int, float]:
    bins = np.linspace(0, 1, N_BINS + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    accumulated: dict[int, list[np.ndarray]] = {s: [] for s in range(1, MAX_STEPS + 1)}

    for matrix, _, _ in scenes:
        n = matrix.shape[0]
        for step in range(1, MAX_STEPS + 1):
            a = np.arange(n - step)
            b = a + step
            vals = np.concatenate([matrix[a, b], matrix[b, a]])
            accumulated[step].append(vals[vals > 0.005])

    argmaxes = {}
    for step in range(1, MAX_STEPS + 1):
        vals = np.concatenate(accumulated[step]) if accumulated[step] else np.array([])
        if len(vals) == 0:
            continue
        counts, _ = np.histogram(vals, bins=bins)
        argmaxes[step] = float(bin_centers[np.argmax(counts)])
        print(f"step={step:2d}  argmax={argmaxes[step]:.4f}")
    return argmaxes


def find_pair(
    step: int,
    target: float,
    scenes: list[tuple[np.ndarray, list, Path]],
) -> tuple[Path, Path, int, int, str, float] | None:
    """Returns (img_a, img_b, idx_a, idx_b, scene_name, covis) or None."""
    best = None
    best_dist = float("inf")

    for matrix, frames, scene_root in scenes:
        n = matrix.shape[0]
        a_indices = np.arange(n - step)
        b_indices = a_indices + step
        vals = matrix[a_indices, b_indices]
        dists = np.abs(vals - target)
        within = np.where(dists < EPSILON)[0]
        if len(within) == 0:
            continue
        closest = within[np.argmin(dists[within])]
        dist = dists[closest]
        if dist < best_dist:
            best_dist = dist
            a, b = int(a_indices[closest]), int(b_indices[closest])
            img_a = scene_root / frames[a]["image"]
            img_b = scene_root / frames[b]["image"]
            best = (img_a, img_b, a, b, scene_root.name, float(vals[closest]))

    return best


def make_side_by_side(img_a: Path, img_b: Path, title: str, out_path: Path) -> None:
    a = Image.open(img_a).convert("RGB")
    b = Image.open(img_b).convert("RGB")

    # Resize b to match a's height if needed
    if a.height != b.height:
        b = b.resize((int(b.width * a.height / b.height), a.height), Image.LANCZOS)

    gap = 8
    total_w = a.width + gap + b.width
    canvas = Image.new("RGB", (total_w, a.height + 30), (240, 240, 240))
    canvas.paste(a, (0, 30))
    canvas.paste(b, (a.width + gap, 30))

    # Draw title text via PIL (no matplotlib dependency)
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), title, fill=(30, 30, 30))
    except Exception:
        pass

    canvas.save(out_path)


def main():
    scenes_data = []
    for scene_root in sorted(WAI_ROOT.iterdir()):
        if not (scene_root / "scene_meta.json").exists():
            continue
        result = load_scene(scene_root)
        if result is None:
            print(f"Skipping {scene_root.name}: no covisibility file")
            continue
        matrix, frames = result
        scenes_data.append((matrix, frames, scene_root))

    if not scenes_data:
        print(f"No scenes found in {WAI_ROOT}")
        return

    print("Computing argmaxes...")
    argmaxes = compute_argmaxes(scenes_data)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for step in range(1, MAX_STEPS + 1):
        if step not in argmaxes:
            print(f"step={step}: no argmax, skipping")
            continue
        target = argmaxes[step]
        pair = find_pair(step, target, scenes_data)
        if pair is None:
            print(f"step={step}: no pair within epsilon={EPSILON} of argmax={target:.4f}")
            continue
        img_a, img_b, idx_a, idx_b, scene_name, covis = pair
        title = (
            f"step={step}  scene={scene_name}  "
            f"frames={idx_a}&{idx_b}  covis={covis:.4f}  argmax={target:.4f}"
        )
        out_path = OUT_DIR / f"step_{step:02d}.png"
        make_side_by_side(img_a, img_b, title, out_path)
        print(f"step={step:2d}  saved {out_path.name}  ({title})")

    print(f"\nAll outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
