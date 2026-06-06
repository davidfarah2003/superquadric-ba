"""
Export BA point cloud and SUPERDEC superquadrics from a saved benchmark
visualization sample to PLY files that can be opened locally in any 3-D viewer
(MeshLab, CloudCompare, VS Code with a PLY extension, etc.).

Outputs written next to the input data:
    <sample_dir>/ba_pointcloud.ply    — coloured BA point cloud
    <sample_dir>/superquadrics.ply    — superquadric mesh (one colour per object)

The depth maps (view_XX_depth.npz) contain only the Z-coordinate of each pixel
in camera frame.  Back-projection to full 3-D uses a pinhole model; adjust
--focal-length if the cloud looks distorted.

Usage
-----
    python visualize_benchmark_sample.py <sample_dir> [--focal-length FLOAT]

Example
-------
    python visualize_benchmark_sample.py \\
        "/work/courses/3dv/team39/logs/benchmark_ase_sparse_surface_cov06/viz/10 @ ASEWAI/sample_0"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/work/courses/3dv/team39/superdec")
from superdec.utils.visualizations import generate_ncolors


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def backproject_depth(depth: np.ndarray, focal_length: float):
    """Back-project a (H, W) Z-depth map to camera-frame points.

    Returns
    -------
    pts  : (N, 3) float32  camera-frame points for valid pixels
    mask : (H*W,) bool     flat validity mask (depth > 0), for colour indexing
    """
    depth = depth.squeeze().astype(np.float32)
    H, W = depth.shape
    cx, cy = W / 2.0, H / 2.0
    yi, xi = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    z = depth
    x = (xi - cx) * z / focal_length
    y = (yi - cy) * z / focal_length
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    mask = pts[:, 2] > 0
    return pts[mask], mask


def cam_to_world(cam_entry: dict) -> np.ndarray:
    """Return a (4, 4) camera-to-world matrix from a cameras.json view entry."""
    if "cam_to_world" in cam_entry:
        return np.array(cam_entry["cam_to_world"], dtype=np.float64)
    R = Rotation.from_quat(cam_entry["quat_xyzw"]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cam_entry["translation"]
    return T


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_pointcloud(stage_dir: Path, focal_length: float, max_points: int,
                    image_dir: Path | None = None):
    """Merge all views into a single world-frame point cloud with RGB colours.

    Parameters
    ----------
    stage_dir   : directory with view_XX_depth.npz + cameras.json
    focal_length: pinhole focal length in pixels
    max_points  : random subsample cap
    image_dir   : directory with view_XX_image.png for RGB colours;
                  falls back to uniform light-blue if absent

    Returns
    -------
    pts    : (N, 3) float32
    colors : (N, 3) uint8   values in [0, 255]
    """
    cameras = json.loads((stage_dir / "cameras.json").read_text())
    all_pts, all_colors = [], []

    for view_name in sorted(cameras):
        depth_file = stage_dir / f"{view_name}_depth.npz"
        if not depth_file.exists():
            continue
        depth = np.load(depth_file)["depth"]
        pts_cam, mask = backproject_depth(depth, focal_length)

        c2w = cam_to_world(cameras[view_name])
        pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]
        all_pts.append(pts_world.astype(np.float32))

        if image_dir is not None:
            img_file = image_dir / f"{view_name}_image.png"
            if img_file.exists():
                img = np.array(PILImage.open(img_file))   # (H, W, 3) uint8
                colors = img.reshape(-1, 3)[mask]
            else:
                colors = np.full((len(pts_world), 3), 90, dtype=np.uint8)
        else:
            colors = np.full((len(pts_world), 3), 90, dtype=np.uint8)
        all_colors.append(colors)

    if not all_pts:
        return None, None

    pts = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, colors = pts[idx], colors[idx]

    return pts, colors


# ---------------------------------------------------------------------------
# Superquadric mesh (adapted from PredictionHandler._superquadric_mesh)
# ---------------------------------------------------------------------------

def _sq_mesh(scale, exponents, rotation_aa, translation, N: int = 20):
    def f(o, m): return np.sign(np.sin(o)) * np.abs(np.sin(o)) ** m
    def g(o, m): return np.sign(np.cos(o)) * np.abs(np.cos(o)) ** m

    u = np.tile(np.linspace(-np.pi, np.pi, N, endpoint=True), N)
    v = np.repeat(np.linspace(-np.pi / 2.0, np.pi / 2.0, N, endpoint=True), N)

    x = scale[0] * g(v, exponents[0]) * g(u, exponents[1])
    y = scale[1] * g(v, exponents[0]) * f(u, exponents[1])
    z = scale[2] * f(v, exponents[0])
    x[:N] = 0.0
    x[-N:] = 0.0

    R = Rotation.from_rotvec(rotation_aa).as_matrix()
    vertices = (R @ np.stack([x, y, z])).T + translation

    tris = []
    for i in range(N - 1):
        for j in range(N - 1):
            tris += [[i*N+j, i*N+j+1, (i+1)*N+j],
                     [(i+1)*N+j, i*N+j+1, (i+1)*N+(j+1)]]
    for i in range(N - 1):
        tris += [[i*N+(N-1), i*N, (i+1)*N+(N-1)],
                 [(i+1)*N+(N-1), i*N, (i+1)*N]]
    tris += [[(N-1)*N+(N-1), (N-1)*N, N-1], [N-1, (N-1)*N, 0]]

    return vertices, np.array(tris)


def build_sq_mesh(sq_json: Path) -> trimesh.Trimesh | None:
    data = json.loads(sq_json.read_text())
    primitives = data["primitives"]
    if not primitives:
        return None

    colors = generate_ncolors(len(primitives))
    all_v, all_f, all_fc = [], [], []
    offset = 0
    for k, prim in enumerate(primitives):
        verts, faces = _sq_mesh(
            np.array(prim["scale"]),
            np.array(prim["exponents"]),
            np.array(prim["rotation_aa"]),
            np.array(prim["translation"]),
        )
        all_v.append(verts)
        all_f.append(faces + offset)
        all_fc.append(np.tile(colors[k], (len(faces), 1)))
        offset += len(verts)

    verts = np.concatenate(all_v)
    faces = np.concatenate(all_f)
    fcolors = np.concatenate(all_fc) / 255.0
    return trimesh.Trimesh(verts, faces, face_colors=fcolors)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export BA point cloud + superquadrics to PLY files",
    )
    parser.add_argument("sample_dir", help="Path to a sample_N/ directory")
    parser.add_argument(
        "--focal-length", type=float, default=450.0,
        help="Pinhole focal length in pixels for Z-depth back-projection "
             "(default 450, tuned for 518×518 ASE; adjust if cloud looks distorted)",
    )
    parser.add_argument(
        "--max-points", type=int, default=500_000,
        help="Point-cloud size cap across all views (default 500 000)",
    )
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    if not sample_dir.is_dir():
        sys.exit(f"Directory not found: {sample_dir}")

    # --- BA point cloud ---
    ba_dir = sample_dir / "ba"
    if ba_dir.exists():
        print("Loading BA point cloud …")
        pts, colors = load_pointcloud(
            ba_dir, args.focal_length, args.max_points,
            image_dir=sample_dir / "gt",
        )
        if pts is not None:
            out_pc = sample_dir / "ba_pointcloud.ply"
            pc = trimesh.PointCloud(vertices=pts, colors=colors)
            pc.export(str(out_pc))
            print(f"  saved {len(pts):,} points → {out_pc}")
    else:
        print("No ba/ directory found — skipping point cloud.")

    # --- Superquadrics ---
    sq_json = sample_dir / "superquadrics.json"
    if sq_json.exists():
        print("Building superquadric meshes …")
        sq_mesh = build_sq_mesh(sq_json)
        if sq_mesh is not None:
            out_sq = sample_dir / "superquadrics.ply"
            sq_mesh.export(str(out_sq))
            n_prims = json.loads(sq_json.read_text())["num_active_primitives"]
            print(f"  {n_prims} primitives → {out_sq}")
    else:
        print("No superquadrics.json found — skipping superquadric mesh.")


if __name__ == "__main__":
    main()
