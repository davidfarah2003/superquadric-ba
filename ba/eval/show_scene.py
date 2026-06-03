"""Show an ACTUAL view of a scene: the real input photos the benchmark used, and
a 3D render of the superquadric decomposition with the camera poses.

fig8: the 10 input RGB views (matched from the WAI trajectory by camera pose).
fig9: 3D superquadric reconstruction (each primitive rendered as a surface,
      colored by object) + the 10 GT cameras (positions + view directions).
"""
import os, sys, json, glob
sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")
sys.path.insert(0, "/work/courses/3dv/team39/ba/python")
import numpy as np
import offline_eval as oe
from ba.superdec import load_scene
from scipy.spatial.transform import Rotation
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT = "/work/courses/3dv/team39/ba/eval/analysis"
CACHE = "/work/courses/3dv/team39/compose/data/ba_cache"


def match_views(scene_idx):
    """Match the cache's 10 GT cameras to WAI frames by camera-centre nearest match."""
    cache = oe.load_cache(sorted(glob.glob(os.path.join(CACHE, "*.npz")))[scene_idx])
    gt_poses = np.asarray(cache["gt_poses"], np.float64)
    gt_c = gt_poses[:, :3, 3]
    label = str(cache["scene_label"])
    meta = json.load(open(f"/work/courses/3dv/team39/compose/data/wai/{label}/scene_meta.json"))
    frames = meta["frames"]
    fc = np.array([np.asarray(f["transform_matrix"])[:3, 3] for f in frames])
    chosen = []
    for c in gt_c:
        j = int(np.argmin(np.linalg.norm(fc - c, axis=1)))
        chosen.append((frames[j]["image"], float(np.linalg.norm(fc[j] - c))))
    return label, gt_poses, chosen


def fig_views(scene_idx=6):
    label, gt_poses, chosen = match_views(scene_idx)
    root = f"/work/courses/3dv/team39/compose/data/wai/{label}"
    n = len(chosen)
    cols = 5; rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for k, (impath, d) in enumerate(chosen):
        ax = axs.flat[k]
        try:
            ax.imshow(Image.open(os.path.join(root, impath)))
        except Exception as e:
            ax.text(0.5, 0.5, f"missing\n{impath}", ha="center")
        ax.set_title(f"view {k}  ({impath.split('/')[-1]}, d={d:.2f}m)", fontsize=8)
        ax.axis("off")
    for k in range(n, rows * cols):
        axs.flat[k].axis("off")
    fig.suptitle(f"scene {label}: the {n} input photos the benchmark reconstructs from", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fig8_views_scene{scene_idx}.png"), dpi=110)
    print(f"fig8: scene {label}, max match dist {max(d for _,d in chosen):.3f}m")


def _sq_surface(a, eps, n=12):
    e1 = float(np.clip(eps[0], 0.1, 1.9)); e2 = float(np.clip(eps[1], 0.1, 1.9))
    eta = np.linspace(-np.pi / 2, np.pi / 2, n)
    om = np.linspace(-np.pi, np.pi, 2 * n)
    E, O = np.meshgrid(eta, om)
    c = lambda t, p: np.sign(np.cos(t)) * np.abs(np.cos(t)) ** p
    s = lambda t, p: np.sign(np.sin(t)) * np.abs(np.sin(t)) ** p
    x = a[0] * c(E, e1) * c(O, e2); y = a[1] * c(E, e1) * s(O, e2); z = a[2] * s(E, e1)
    return x, y, z


def fig_sq3d(scene_idx=6):
    label, gt_poses, _ = match_views(scene_idx)
    sq = load_scene(f"/work/courses/3dv/team39/compose/data/output_npz/ase_scene_{label}.npz")
    R = Rotation.from_rotvec(sq["rotation_aa"]).as_matrix()
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")
    K = len(sq["scale"])
    for k in range(K):
        x, y, z = _sq_surface(sq["scale"][k], sq["exponents"][k], n=10)
        P = R[k] @ np.stack([x.ravel(), y.ravel(), z.ravel()]) + sq["translation"][k][:, None]
        sh = x.shape
        ax.plot_surface(P[0].reshape(sh), P[1].reshape(sh), P[2].reshape(sh),
                        color=cmap(int(sq["object_idx"][k]) % 20), alpha=0.55,
                        linewidth=0, antialiased=False, shade=True)
    gc = gt_poses[:, :3, 3]
    gd = (gt_poses[:, :3, :3] @ np.array([0, 0, 1.0]))
    ax.scatter(gc[:, 0], gc[:, 1], gc[:, 2], c="red", s=70, marker="^",
               edgecolor="k", zorder=10, label="cameras (GT)")
    ax.quiver(gc[:, 0], gc[:, 1], gc[:, 2], gd[:, 0], gd[:, 1], gd[:, 2],
              length=0.6, color="red", linewidth=2)
    # TRUE proportions (apartment floor is wide + flat): per-axis limits + box aspect
    allp = np.concatenate([sq["translation"], gc], 0)
    mins = allp.min(0); maxs = allp.max(0); rng = np.maximum(maxs - mins, 1e-3)
    ax.set_xlim(mins[0], maxs[0]); ax.set_ylim(mins[1], maxs[1]); ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(rng)
    ax.set_title(f"scene {label}: superquadric decomposition ({K} primitives, "
                 f"{len(np.unique(sq['object_idx']))} objects) + camera poses")
    ax.legend(loc="upper left"); ax.view_init(elev=62, azim=-78)
    fig.savefig(os.path.join(OUT, f"fig9_sq3d_scene{scene_idx}.png"), dpi=130, bbox_inches="tight")
    print(f"fig9: rendered {K} superquadrics for scene {label}")


if __name__ == "__main__":
    fig_views(6); fig_sq3d(6)
