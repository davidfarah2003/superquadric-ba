#!/usr/bin/env python3
"""
Standalone covisibility computation for WAI-formatted scenes.

Computes pairwise covisibility matrices needed by the map-anything benchmark.
For each pair of frames, reprojects the depth map from one view into the other
and checks how many points land at consistent depths.

Outputs: <scene_root>/covisibility/v0/pairwise_covisibility.npy

Usage:
    python scripts/compute_covisibility.py --wai_root data/wai --scenes 0 1 2
    python scripts/compute_covisibility.py --wai_root data/wai  # all scenes
"""

import argparse
import json
import os
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# --- Config ---
TARGET_DEPTH_SIZE = 224  # Long-side resize for depth (matches upstream default)
DEPTH_ASSOC_ERROR_THRES = 0.1
DEPTH_ASSOC_REL_ERROR_THRES = 0.005
DEPTH_ASSOC_ERROR_TEMP = 0.1
DENOMINATOR_MODE = "full"  # "full" or "valid_target_depth"


def load_scene_meta(scene_root: Path) -> dict:
    with open(scene_root / "scene_meta.json") as f:
        meta = json.load(f)
    # Build frame_names index (same as mapanything's _load_scene_meta)
    meta["frame_names"] = {
        frame["frame_name"]: idx for idx, frame in enumerate(meta["frames"])
    }
    return meta


def load_depth(scene_root: Path, frame: dict) -> np.ndarray:
    """Load depth map from EXR or PNG."""
    depth_path = scene_root / frame["depth"]
    depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise FileNotFoundError(f"Cannot load depth: {depth_path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth.astype(np.float32)


def get_intrinsics(frame: dict, scene_meta: dict) -> np.ndarray:
    """Get 3x3 intrinsics matrix for a frame."""
    # Per-frame intrinsics override scene-level
    fx = frame.get("fl_x", scene_meta.get("fl_x"))
    fy = frame.get("fl_y", scene_meta.get("fl_y"))
    cx = frame.get("cx", scene_meta.get("cx"))
    cy = frame.get("cy", scene_meta.get("cy"))
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def get_extrinsics(frame: dict) -> np.ndarray:
    """Get 4x4 camera-to-world matrix."""
    return np.array(frame["transform_matrix"], dtype=np.float32)


def resize_depth(depth: np.ndarray, target_size: int):
    """Resize depth so long side equals target_size. Returns (depth, scale_h, scale_w)."""
    h, w = depth.shape
    if h >= w:
        new_h = target_size
        new_w = int(round(w * target_size / h))
    else:
        new_w = target_size
        new_h = int(round(h * target_size / w))
    scale_h = new_h / h
    scale_w = new_w / w
    resized = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return resized, scale_h, scale_w


def unproject_depth(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """
    Unproject depth map to 3D points in camera frame.
    depth: (H, W), intrinsics: (3, 3)
    Returns: (H, W, 3) points in camera coordinates
    """
    h, w = depth.shape
    device = depth.device

    v, u = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    z = depth

    return torch.stack([x, y, z], dim=-1)


def compute_covisibility_for_scene(
    scene_root: Path,
    device: torch.device,
    overwrite: bool = False,
):
    """Compute and save pairwise covisibility for one WAI scene."""
    scene_name = scene_root.name
    out_dir = scene_root / "covisibility" / "v0"

    if (out_dir / "pairwise_covisibility.npy").exists() and not overwrite:
        print(f"  Scene {scene_name}: covisibility already exists, skipping")
        return

    scene_meta = load_scene_meta(scene_root)
    frames = scene_meta["frames"]
    num_frames = len(frames)
    print(f"  Scene {scene_name}: {num_frames} frames")

    # Load all depths, intrinsics, extrinsics
    depths = []
    valid_masks = []
    intrinsics_list = []
    cam2worlds = []
    world_pts_list = []

    depth_h, depth_w = 0, 0

    for frame in tqdm(frames, desc=f"  Loading {scene_name}", leave=False):
        depth_np = load_depth(scene_root, frame)
        K_np = get_intrinsics(frame, scene_meta)
        c2w_np = get_extrinsics(frame)

        # Resize depth
        depth_resized, scale_h, scale_w = resize_depth(depth_np, TARGET_DEPTH_SIZE)
        if depth_h == 0:
            depth_h, depth_w = depth_resized.shape

        # Scale intrinsics
        K_scaled = K_np.copy()
        K_scaled[0, :] *= scale_w
        K_scaled[1, :] *= scale_h

        depth_t = torch.from_numpy(depth_resized).to(device)
        K_t = torch.from_numpy(K_scaled).to(device)
        c2w_t = torch.from_numpy(c2w_np).to(device)

        valid = depth_t > 0

        # Unproject to camera-frame 3D, then transform to world
        pts_cam = unproject_depth(depth_t, K_t)  # (H, W, 3)
        pts_cam_h = torch.cat(
            [pts_cam, torch.ones(*pts_cam.shape[:-1], 1, device=device)], dim=-1
        )  # (H, W, 4)
        pts_world = (c2w_t @ pts_cam_h.reshape(-1, 4).T).T[:, :3].reshape(
            depth_h, depth_w, 3
        )

        depths.append(depth_t)
        valid_masks.append(valid)
        intrinsics_list.append(K_t)
        cam2worlds.append(c2w_t)
        world_pts_list.append(pts_world.cpu())  # Keep on CPU to save VRAM

    depths = torch.stack(depths)  # (N, H, W)
    valid_masks = torch.stack(valid_masks)  # (N, H, W)
    intrinsics_all = torch.stack(intrinsics_list)  # (N, 3, 3)
    cam2worlds_all = torch.stack(cam2worlds)  # (N, 4, 4)

    # Compute pairwise covisibility
    pairwise_covisibility = torch.zeros((num_frames, num_frames), dtype=torch.float32)

    for idx in tqdm(range(num_frames), desc=f"  Covisibility {scene_name}", leave=False):
        if not valid_masks[idx].any():
            continue

        # Get world points for this frame
        pts_w = world_pts_list[idx].to(device)  # (H, W, 3)
        pts_w_h = torch.cat(
            [pts_w, torch.ones(*pts_w.shape[:-1], 1, device=device)], dim=-1
        )  # (H, W, 4)
        pts_flat = pts_w_h.reshape(-1, 4)  # (H*W, 4)

        # Process in chunks of target views to avoid OOM
        chunk_size = 64
        for start in range(0, num_frames, chunk_size):
            end = min(start + chunk_size, num_frames)
            n_targets = end - start

            # Project into each target view
            w2c = torch.inverse(cam2worlds_all[start:end])  # (chunk, 4, 4)
            pts_in_target = (w2c @ pts_flat.T).permute(0, 2, 1)[
                :, :, :3
            ]  # (chunk, H*W, 3)
            pts_in_target = pts_in_target.reshape(n_targets, depth_h, depth_w, 3)

            # Project to 2D: u = fx * X/Z + cx, v = fy * Y/Z + cy
            K = intrinsics_all[start:end]  # (chunk, 3, 3)
            Z = pts_in_target[..., 2]  # (chunk, H, W)
            X = pts_in_target[..., 0]
            Y = pts_in_target[..., 1]

            fx = K[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
            fy = K[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
            cx = K[:, 0, 2].unsqueeze(-1).unsqueeze(-1)
            cy = K[:, 1, 2].unsqueeze(-1).unsqueeze(-1)

            u = fx * X / Z.clamp(min=1e-6) + cx
            v = fy * Y / Z.clamp(min=1e-6) + cy

            # Check in-bounds and positive depth
            in_bounds = (
                (u >= 0)
                & (u < depth_w)
                & (v >= 0)
                & (v < depth_h)
                & (Z > 0.04)
                & valid_masks[idx].unsqueeze(0)
            )

            # Sample target depth at projected locations
            # Normalize to [-1, 1] for grid_sample
            u_norm = 2 * u / (depth_w - 1) - 1
            v_norm = 2 * v / (depth_h - 1) - 1
            grid = torch.stack([u_norm, v_norm], dim=-1).clamp(-1, 1)

            target_depths = depths[start:end].unsqueeze(1)  # (chunk, 1, H, W)
            sampled_depth = F.grid_sample(
                target_depths, grid, mode="nearest", align_corners=True
            )[
                :, 0
            ]  # (chunk, H, W)

            # Check depth consistency
            reproj_error = torch.abs(Z - sampled_depth)
            depth_threshold = (
                DEPTH_ASSOC_ERROR_THRES
                + DEPTH_ASSOC_REL_ERROR_THRES * Z
                + (-np.log(0.5)) * DEPTH_ASSOC_ERROR_TEMP
            )
            consistent = in_bounds & (reproj_error < depth_threshold)

            # Compute scores
            if DENOMINATOR_MODE == "full":
                scores = consistent.sum(dim=[1, 2]).float() / (depth_h * depth_w)
            else:
                target_valid = valid_masks[start:end].sum(dim=[1, 2]).clamp(min=1).float()
                scores = consistent.sum(dim=[1, 2]).float() / target_valid

            pairwise_covisibility[idx, start:end] = scores.cpu()

        torch.cuda.empty_cache()

    # Save using mmap naming convention: name--NxN.npy
    out_dir.mkdir(parents=True, exist_ok=True)
    data = pairwise_covisibility.numpy().astype(np.float32)
    shape_str = f"{num_frames}x{num_frames}"
    mmap_name = f"pairwise_covisibility--{shape_str}.npy"
    with open(out_dir / mmap_name, "wb") as fid:
        np.save(fid, data)
    # Remove old non-mmap file if it exists
    old_file = out_dir / "pairwise_covisibility.npy"
    if old_file.exists():
        old_file.unlink()
    print(f"  Saved: {out_dir / mmap_name}")


def main():
    parser = argparse.ArgumentParser(description="Compute covisibility for WAI scenes")
    parser.add_argument(
        "--wai_root",
        type=str,
        default="/work/courses/3dv/team39/compose/data/wai",
        help="Root of WAI-formatted scenes",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Scene names to process (default: all subdirs with scene_meta.json)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing covisibility files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    args = parser.parse_args()

    wai_root = Path(args.wai_root)

    if args.scenes:
        scene_names = args.scenes
    else:
        scene_names = sorted(
            d.name
            for d in wai_root.iterdir()
            if d.is_dir() and (d / "scene_meta.json").exists()
        )

    print(f"Processing {len(scene_names)} scenes from {wai_root}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for scene_name in scene_names:
        scene_root = wai_root / scene_name
        if not (scene_root / "scene_meta.json").exists():
            print(f"  Skipping {scene_name}: no scene_meta.json")
            continue
        compute_covisibility_for_scene(scene_root, device, overwrite=args.overwrite)

    print("Done!")


if __name__ == "__main__":
    main()
