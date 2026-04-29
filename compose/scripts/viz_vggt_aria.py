#!/usr/bin/env python3
"""
Run VGGT inference on Aria Synthetic Environments (ASE) WAI data and produce
paper-quality visualizations: a depth-comparison figure and a GLB point cloud.

Usage (on cluster — needs a GPU node):
    srun --account=3dv --gpus=5060ti:1 --mem=32G --time=01:00:00 --pty bash
    source ~/envs/3dv/bin/activate
    cd /work/courses/3dv/team39

    # Basic run — scene 0, 4 evenly-spaced frames
    python compose/scripts/run_vggt_on_aria.py

    # Custom
    python compose/scripts/run_vggt_on_aria.py \\
        --scene 0 \\
        --num_frames 4 \\
        --frame_start 0 \\
        --out_dir compose/data/compare/vggt

Outputs (written to --out_dir):
    depth_comparison.png   — paper figure: RGB | GT depth | VGGT depth (per view)
    pointcloud.glb         — 3D textured mesh / point cloud (viewable in VS Code)
    poses.png              — bird's-eye view of predicted camera poses
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Enable OpenEXR support in OpenCV (must be set before importing cv2)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch

# Make sure map-anything is importable
_repo = Path(__file__).resolve().parents[2] / "map-anything"
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from mapanything.models import model_factory
from mapanything.utils.device import get_device
from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.image import load_images
from mapanything.utils.viz import predictions_to_glb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VGGT_RESOLUTION = 518  # must match configs/dataset/benchmark_518_ase_wai.yaml


def load_gt_depth_exr(path: Path) -> np.ndarray:
    """Read a float32 EXR depth map (meters) and return as (H, W) array."""
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read EXR: {path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]  # EXR is typically stored as 3-channel; take first
    return depth.astype(np.float32)


def pick_frame_indices(total: int, num_frames: int, start: int = 0) -> list[int]:
    """Return `num_frames` evenly spaced frame indices starting from `start`."""
    available = total - start
    if num_frames >= available:
        return list(range(start, total))
    step = available / num_frames
    return [start + int(i * step) for i in range(num_frames)]


def colorize_depth(depth: np.ndarray, vmin=None, vmax=None, cmap="plasma") -> np.ndarray:
    """Map a (H, W) float32 depth to an (H, W, 3) uint8 image."""
    valid = depth > 0
    if vmin is None:
        vmin = depth[valid].min() if valid.any() else 0.0
    if vmax is None:
        vmax = np.percentile(depth[valid], 95) if valid.any() else 1.0
    depth_norm = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    depth_norm[~valid] = 0.0
    colored = (cm.get_cmap(cmap)(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    colored[~valid] = 0  # black for invalid
    return colored


# ---------------------------------------------------------------------------
# Figure: RGB | GT depth | VGGT depth
# ---------------------------------------------------------------------------

def save_depth_comparison_figure(
    rgb_list: list[np.ndarray],          # [(H,W,3) uint8]
    gt_depth_list: list[np.ndarray],     # [(H,W) float32]
    pred_depth_list: list[np.ndarray],   # [(H,W) float32]
    out_path: Path,
    cmap: str = "plasma",
    dpi: int = 300,
):
    n = len(rgb_list)
    nrows = 3   # RGB, GT depth, VGGT depth
    ncols = n   # one column per frame

    # Compute a shared depth scale across all views for a fair comparison
    all_gt = np.concatenate([d[d > 0].ravel() for d in gt_depth_list if (d > 0).any()])
    all_pr = np.concatenate([d[d > 0].ravel() for d in pred_depth_list if (d > 0).any()])
    vmin = min(all_gt.min(), all_pr.min())
    vmax = max(np.percentile(all_gt, 95), np.percentile(all_pr, 95))

    # 4 frames × 3 rows → 16:9 at cell=3
    cell = 3.0
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * cell * (16 / 12), nrows * cell))
    if ncols == 1:
        axes = axes[:, np.newaxis]

    row_labels = ["Input RGB", "GT Depth", "VGGT Depth"]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9, fontweight="bold", labelpad=6)

    for col_idx in range(n):
        axes[0, col_idx].set_title(f"Frame {col_idx}", fontsize=9, pad=4)

        axes[0, col_idx].imshow(rgb_list[col_idx])
        axes[0, col_idx].axis("off")

        axes[1, col_idx].imshow(colorize_depth(gt_depth_list[col_idx], vmin, vmax, cmap))
        axes[1, col_idx].axis("off")

        axes[2, col_idx].imshow(colorize_depth(pred_depth_list[col_idx], vmin, vmax, cmap))
        axes[2, col_idx].axis("off")

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.01)
    cbar.set_label("Depth (m)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved depth comparison figure → {out_path}")


# ---------------------------------------------------------------------------
# Figure: bird's-eye view of predicted poses
# ---------------------------------------------------------------------------

def save_pose_figure(
    pred_translations: list[np.ndarray],  # [(3,)]
    out_path: Path,
    dpi: int = 300,
):
    xs = [t[0] for t in pred_translations]
    zs = [t[2] for t in pred_translations]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(xs, zs, c=range(len(xs)), cmap="viridis", s=60, zorder=3)
    for i, (x, z) in enumerate(zip(xs, zs)):
        ax.annotate(str(i), (x, z), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Predicted Camera Positions (top-down)", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved pose figure → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="VGGT inference on Aria WAI data")
    p.add_argument("--scene", type=str, default="0",
                   help="Scene ID (default: 0)")
    p.add_argument("--num_frames", type=int, default=4,
                   help="Number of frames to use (default: 4)")
    p.add_argument("--frame_start", type=int, default=0,
                   help="First frame index (default: 0)")
    p.add_argument("--wai_root", type=str,
                   default="/work/courses/3dv/team39/compose/data/wai",
                   help="Root of the WAI dataset")
    p.add_argument("--out_dir", type=str,
                   default="/work/courses/3dv/team39/compose/data/compare/vggt",
                   help="Output directory for visualizations")
    p.add_argument("--cmap", type=str, default="plasma",
                   help="Matplotlib colormap for depth (default: plasma)")
    p.add_argument("--save_glb", action="store_true", default=True,
                   help="Export 3D point cloud as GLB (default: True)")
    p.add_argument("--as_mesh", action="store_true", default=False,
                   help="Export GLB as textured mesh instead of point cloud")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wai_path = Path(args.wai_root) / args.scene

    # ---- Load scene metadata ------------------------------------------------
    with open(wai_path / "scene_meta.json") as f:
        meta = json.load(f)

    total_frames = len(meta["frames"])
    frame_indices = pick_frame_indices(total_frames, args.num_frames, args.frame_start)
    print(f"Scene {args.scene}: {total_frames} frames total, using indices {frame_indices}")

    # ---- Load RGB images for VGGT -------------------------------------------
    # VGGT uses identity normalization (no DINOv2 mean/std) at 518x518
    image_paths = [str(wai_path / meta["frames"][i]["image"]) for i in frame_indices]
    views = load_images(
        image_paths,
        resize_mode="square",
        size=VGGT_RESOLUTION,
        norm_type="identity",
    )
    print(f"Loaded {len(views)} views at {VGGT_RESOLUTION}x{VGGT_RESOLUTION}")

    # ---- Load GT depth maps (for comparison figure) -------------------------
    gt_depths_orig = []
    for i in frame_indices:
        depth_path = wai_path / meta["frames"][i]["depth"]
        gt_depths_orig.append(load_gt_depth_exr(depth_path))

    # Also load RGB at original resolution for the figure
    rgb_orig = []
    for i in frame_indices:
        img_path = wai_path / meta["frames"][i]["image"]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_orig.append(img)

    # ---- Load VGGT model ----------------------------------------------------
    device = get_device()
    print(f"Using device: {device}")
    print("Loading VGGT-1B (from HuggingFace cache) ...")
    model = model_factory("vggt", name="vggt", torch_hub_force_reload=False,
                          load_pretrained_weights=True)
    model = model.to(device)
    model.eval()
    print("VGGT loaded.")

    # Move views to the correct device
    views_on_device = []
    for v in views:
        views_on_device.append({k: (val.to(device) if isinstance(val, torch.Tensor) else val)
                                 for k, val in v.items()})

    # ---- Run VGGT inference -------------------------------------------------
    print("Running VGGT forward pass ...")
    with torch.no_grad():
        preds = model(views_on_device)
    print("Inference complete.")

    # ---- Extract outputs ----------------------------------------------------
    rgb_viz = []       # (H, W, 3) uint8
    gt_depths = []     # (H, W) float32
    pred_depths = []   # (H, W) float32
    pred_translations = []
    world_points_list = []
    images_list = []
    masks_list = []

    for idx, pred in enumerate(preds):
        # Get the input image at VGGT resolution (already in [0,1])
        img_norm = views_on_device[idx]["img"][0]  # (3, H, W)
        img_np = img_norm.permute(1, 2, 0).cpu().float().numpy()  # (H, W, 3)
        img_np = np.clip(img_np, 0, 1)

        # Predicted Z-depth: (B, H, W, 1) — B=1 here
        # The VGGT wrapper returns `depth_along_ray`, not `depth_z` directly.
        # We reconstruct Z depth from the depth along ray and intrinsics.
        depth_along_ray = pred["depth_along_ray"][0, ..., 0].cpu().float().numpy()  # (H, W)

        # Predicted camera translation (world frame)
        cam_trans = pred["cam_trans"][0].cpu().float().numpy()  # (3,)
        pred_translations.append(cam_trans)

        # Confidence mask
        conf = pred["conf"][0].cpu().float().numpy()  # (H, W)
        conf_mask = conf > np.percentile(conf, 20)  # keep top 80% confident pixels

        # World-frame 3D points
        pts3d = pred["pts3d"][0].cpu().float().numpy()  # (H, W, 3)

        # Resize GT depth to VGGT resolution for comparison
        gt_d_resized = cv2.resize(gt_depths_orig[idx], (VGGT_RESOLUTION, VGGT_RESOLUTION),
                                   interpolation=cv2.INTER_NEAREST)

        # Collect for figure
        rgb_viz.append((img_np * 255).astype(np.uint8))
        gt_depths.append(gt_d_resized)
        pred_depths.append(depth_along_ray)

        # Collect for GLB
        world_points_list.append(pts3d)
        images_list.append(img_np)
        masks_list.append(conf_mask)

    # ---- Paper figure: depth comparison ------------------------------------
    save_depth_comparison_figure(
        rgb_viz, gt_depths, pred_depths,
        out_dir / "depth_comparison.png",
        cmap=args.cmap,
    )

    # ---- Paper figure: predicted camera poses (top-down) -------------------
    save_pose_figure(pred_translations, out_dir / "poses.png")

    # ---- Export GLB --------------------------------------------------------
    if args.save_glb:
        world_points = np.stack(world_points_list, axis=0)  # (V, H, W, 3)
        images = np.stack(images_list, axis=0)              # (V, H, W, 3)
        final_masks = np.stack(masks_list, axis=0)          # (V, H, W)

        predictions_dict = {
            "world_points": world_points,
            "images": images,
            "final_masks": final_masks,
        }

        glb_path = out_dir / "pointcloud.glb"
        scene_3d = predictions_to_glb(predictions_dict, as_mesh=args.as_mesh)
        scene_3d.export(str(glb_path))
        print(f"Saved 3D scene → {glb_path}")

    print(f"\nAll outputs written to: {out_dir}")
    print("  depth_comparison.png  — RGB | GT depth | VGGT depth")
    print("  poses.png             — top-down camera pose layout")
    if args.save_glb:
        print("  pointcloud.glb        — 3D scene (open in VS Code or browser)")


if __name__ == "__main__":
    main()
