"""Poster Fig. 1: clean 3D render of a scene's super-quadric decomposition + the
GT camera poses. Unlike show_scene.fig_sq3d (an analysis figure with axes, ticks,
title and legend), this writes a poster-ready image: axes/box hidden (no tick
numbers, no stray 3D-pane line) and larger, clearer camera markers.

Writes poster/figures/superquadrics_3d_clean.png directly.
"""
import sys
sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")
sys.path.insert(0, "/work/courses/3dv/team39/ba/python")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from ba.superdec import load_scene
from show_scene import match_views, _sq_surface

OUT = "/work/courses/3dv/team39/poster/figures/superquadrics_3d_clean.png"
CAM = "#D62728"   # strong red for the GT cameras


def main(scene_idx=6):
    label, gt_poses, _ = match_views(scene_idx)
    sq = load_scene(f"/work/courses/3dv/team39/compose/data/output_npz/ase_scene_{label}.npz")
    R = Rotation.from_rotvec(sq["rotation_aa"]).as_matrix()
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")
    sc = np.asarray(sq["scale"]); tr = np.asarray(sq["translation"])
    # Drop the big flat horizontal slabs (the floor/ceiling planes). In the
    # top-down view their large translucent surface washes over the whole scene.
    # Use rotation-aware world extents so rotated slabs are caught too; walls
    # (tall: large vertical extent) and furniture (small footprint) are kept.
    ext = np.abs(R) @ sc[..., None]          # (K,3,1) world half-extents
    ext = ext[..., 0]
    horiz_area = ext[:, 0] * ext[:, 1]
    slab = (ext[:, 2] < 0.30) & (horiz_area > 1.2)
    K = len(sc)
    drawn = 0
    for k in range(K):
        if slab[k]:
            continue
        drawn += 1
        x, y, z = _sq_surface(sq["scale"][k], sq["exponents"][k], n=12)
        P = R[k] @ np.stack([x.ravel(), y.ravel(), z.ravel()]) + sq["translation"][k][:, None]
        sh = x.shape
        ax.plot_surface(P[0].reshape(sh), P[1].reshape(sh), P[2].reshape(sh),
                        color=cmap(int(sq["object_idx"][k]) % 20), alpha=0.6,
                        linewidth=0, antialiased=False, shade=True)
    gc = gt_poses[:, :3, 3]
    gd = (gt_poses[:, :3, :3] @ np.array([0, 0, 1.0]))
    # larger, clearly visible cameras (positions + view directions)
    ax.scatter(gc[:, 0], gc[:, 1], gc[:, 2], c=CAM, s=190, marker="^",
               edgecolor="black", linewidth=1.6, depthshade=False, zorder=10)
    ax.quiver(gc[:, 0], gc[:, 1], gc[:, 2], gd[:, 0], gd[:, 1], gd[:, 2],
              length=1.0, color=CAM, linewidth=3.2, zorder=11)
    allp = np.concatenate([sq["translation"], gc], 0)
    mins = allp.min(0); maxs = allp.max(0); rng = np.maximum(maxs - mins, 1e-3)
    ax.set_xlim(mins[0], maxs[0]); ax.set_ylim(mins[1], maxs[1]); ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(rng)
    ax.set_axis_off()          # no panes, gridlines, ticks or bounding-box edges
    ax.view_init(elev=62, azim=-78)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    print(f"wrote {OUT}  ({drawn}/{K} super-quadrics drawn, {int(slab.sum())} floor/ceiling slabs dropped, {gc.shape[0]} cameras)")


if __name__ == "__main__":
    main(6)
