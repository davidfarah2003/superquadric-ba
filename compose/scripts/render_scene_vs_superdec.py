#!/usr/bin/env python3
"""
Render two matched images of an ASE scene for a slide deck:
  1. The original WAI RGB frame (ground-truth view).
  2. The SuperDec superquadric reconstruction rendered from the same
     camera pose and intrinsics.

Both images are saved side-by-side in a single PNG as well as individually.

Usage:
    python scripts/render_scene_vs_superdec.py \
        --scene 0 --frame 175 \
        --out_dir data/compare/
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "superdec"))

import json
import numpy as np
from PIL import Image

import open3d as o3d
import open3d.visualization.rendering as rendering

from superdec.utils.predictions_handler import PredictionHandler


def load_frame(wai_path: Path, frame_idx: int):
    with open(wai_path / "scene_meta.json") as f:
        meta = json.load(f)
    frame = meta["frames"][frame_idx]
    rgb_path = wai_path / frame["image"]
    w, h = int(frame["w"]), int(frame["h"])
    fx, fy = float(frame["fl_x"]), float(frame["fl_y"])
    cx, cy = float(frame["cx"]), float(frame["cy"])
    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    return rgb_path, (w, h, fx, fy, cx, cy), c2w


def render_superdec(npz_path: Path, intr, c2w, out_path: Path, resolution: int = 60):
    w, h, fx, fy, cx, cy = intr

    predictions = PredictionHandler.from_npz(str(npz_path))
    tri_meshes = predictions.get_meshes(resolution=resolution)

    renderer = rendering.OffscreenRenderer(w, h)
    scene = renderer.scene
    scene.set_background([1.0, 1.0, 1.0, 1.0])
    scene.scene.set_sun_light([-0.3, -0.7, -0.6], [1.0, 1.0, 1.0], 80000.0)
    scene.scene.enable_sun_light(True)
    scene.scene.enable_indirect_light(True)
    scene.scene.set_indirect_light_intensity(35000.0)

    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"

    added = 0
    for i, tm in enumerate(tri_meshes):
        if tm is None or len(tm.vertices) == 0:
            continue
        o3m = o3d.geometry.TriangleMesh()
        o3m.vertices = o3d.utility.Vector3dVector(np.asarray(tm.vertices, dtype=np.float64))
        o3m.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces, dtype=np.int32))
        if tm.visual.vertex_colors is not None and len(tm.visual.vertex_colors):
            vc = np.asarray(tm.visual.vertex_colors, dtype=np.float64)[:, :3] / 255.0
            o3m.vertex_colors = o3d.utility.Vector3dVector(vc)
        o3m.compute_vertex_normals()
        scene.add_geometry(f"obj_{i}", o3m, mat)
        added += 1
    print(f"Added {added} superquadric meshes to the renderer")

    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    extrinsic = np.linalg.inv(c2w)  # WAI stores OpenCV c2w, renderer wants w2c
    renderer.setup_camera(intrinsic, extrinsic)

    img = renderer.render_to_image()
    o3d.io.write_image(str(out_path), img)
    print(f"Wrote {out_path}")


def make_side_by_side(left: Path, right: Path, out: Path, gap: int = 20):
    a = Image.open(left).convert("RGB")
    b = Image.open(right).convert("RGB")
    h = max(a.height, b.height)
    canvas = Image.new("RGB", (a.width + b.width + gap, h), (255, 255, 255))
    canvas.paste(a, (0, (h - a.height) // 2))
    canvas.paste(b, (a.width + gap, (h - b.height) // 2))
    canvas.save(out)
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=str, default="0")
    p.add_argument("--frame", type=int, default=175)
    p.add_argument("--wai_root", type=str, default="/work/courses/3dv/team39/compose/data/wai")
    p.add_argument("--npz_dir", type=str, default="/work/courses/3dv/team39/compose/data/output_npz")
    p.add_argument("--out_dir", type=str, default="/work/courses/3dv/team39/compose/data/compare")
    p.add_argument("--resolution", type=int, default=60, help="Superquadric mesh tessellation")
    args = p.parse_args()

    wai_path = Path(args.wai_root) / args.scene
    npz_path = Path(args.npz_dir) / f"ase_scene_{args.scene}.npz"
    out_dir = Path(args.out_dir) / f"scene_{args.scene}_frame_{args.frame:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_src, intr, c2w = load_frame(wai_path, args.frame)
    rgb_dst = out_dir / "original.png"
    Image.open(rgb_src).convert("RGB").save(rgb_dst)
    print(f"Wrote {rgb_dst}")

    sq_dst = out_dir / "superdec.png"
    render_superdec(npz_path, intr, c2w, sq_dst, resolution=args.resolution)

    make_side_by_side(rgb_dst, sq_dst, out_dir / "side_by_side.png")


if __name__ == "__main__":
    main()
