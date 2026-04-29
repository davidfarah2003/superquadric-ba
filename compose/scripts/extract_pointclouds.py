#!/usr/bin/env python3
"""
Extract per-object point clouds from WAI-format scenes for SuperDec inference.

Back-projects depth maps using camera intrinsics, segments by instance ID,
aggregates across frames, and downsamples to a target number of points.

Output: one .npz per object with key 'points' (shape [N, 3]).
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def backproject_depth(depth, mask, fx, fy, cx, cy, c2w):
    """Back-project valid depth pixels to world-space 3D points."""
    h, w = depth.shape
    v, u = np.where(mask > 0)
    z = depth[v, u]

    # Camera-space points
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)  # (N, 4)

    # World-space
    pts_world = (c2w @ pts_cam.T).T[:, :3]  # (N, 3)
    return pts_world


def extract_scene_pointclouds(
    wai_path: Path,
    output_path: Path,
    target_points: int = 4096,
    min_points: int = 256,
    skip_classes: set = None,
    frame_stride: int = 1,
):
    """Extract per-object point clouds from a WAI scene."""
    if skip_classes is None:
        skip_classes = {"empty_space", "background", "ceiling"}

    # Load scene metadata
    with open(wai_path / "scene_meta.json") as f:
        meta = json.load(f)

    # Load object instance mapping
    obj_map_path = wai_path / "object_instances_to_classes.json"
    if obj_map_path.exists():
        with open(obj_map_path) as f:
            obj_map = json.load(f)
    else:
        obj_map = {}

    frames = meta["frames"]
    print(f"Scene has {len(frames)} frames, processing every {frame_stride}")

    # Accumulate points per instance ID
    instance_points = defaultdict(list)

    for i, frame in enumerate(tqdm(frames[::frame_stride], desc="Frames")):
        # Load depth
        depth_path = wai_path / frame["depth"]
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue

        # Load validity mask
        mask_path = wai_path / frame["mask_path"]
        validity = np.array(Image.open(mask_path)) > 0

        # Load instance segmentation
        inst_path = wai_path / frame["instance"]
        instances = np.array(Image.open(inst_path))

        # Camera intrinsics
        fx, fy = frame["fl_x"], frame["fl_y"]
        cx, cy = frame["cx"], frame["cy"]

        # Camera-to-world transform
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)

        # Combined valid mask (positive depth + validity mask)
        valid = validity & (depth > 0)

        # Get unique instance IDs in this frame
        unique_ids = np.unique(instances[valid])

        for inst_id in unique_ids:
            inst_id_int = int(inst_id)
            # Skip certain classes
            class_name = obj_map.get(str(inst_id_int), "unknown")
            if class_name in skip_classes:
                continue

            # Mask for this instance
            inst_mask = valid & (instances == inst_id)
            if inst_mask.sum() < 10:
                continue

            pts = backproject_depth(depth, inst_mask, fx, fy, cx, cy, c2w)
            instance_points[inst_id_int].append(pts)

    # Save per-object point clouds
    output_path.mkdir(parents=True, exist_ok=True)
    saved = 0

    for inst_id, pts_list in instance_points.items():
        all_pts = np.concatenate(pts_list, axis=0)

        if len(all_pts) < min_points:
            continue

        # Random subsample to target
        if len(all_pts) > target_points:
            idx = np.random.choice(len(all_pts), target_points, replace=False)
            all_pts = all_pts[idx]

        class_name = obj_map.get(str(inst_id), "unknown")
        fname = f"{inst_id:04d}_{class_name}.npz"
        np.savez_compressed(output_path / fname, points=all_pts.astype(np.float32))
        saved += 1

    print(f"Saved {saved} object point clouds to {output_path}")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract per-object point clouds from WAI scenes")
    parser.add_argument("--wai_path", type=str, required=True, help="Path to WAI scene directory")
    parser.add_argument("--output_path", type=str, default=None, help="Output directory for NPZ files")
    parser.add_argument("--target_points", type=int, default=4096)
    parser.add_argument("--min_points", type=int, default=256)
    parser.add_argument("--frame_stride", type=int, default=5, help="Process every Nth frame")
    parser.add_argument("--skip_classes", nargs="*", default=["empty_space", "background", "ceiling"])
    args = parser.parse_args()

    wai_path = Path(args.wai_path)
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        scene_name = wai_path.name
        output_path = Path(f"data/pointclouds/{scene_name}")

    extract_scene_pointclouds(
        wai_path=wai_path,
        output_path=output_path,
        target_points=args.target_points,
        min_points=args.min_points,
        frame_stride=args.frame_stride,
        skip_classes=set(args.skip_classes),
    )


if __name__ == "__main__":
    main()
