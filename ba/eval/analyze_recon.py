"""Reconstruction + Manhattan-structure visualizations (light; no Ceres).

fig4: top-down reconstruction for a scene — GT vs predicted (raw VGGT) camera
      positions AND heading arrows, over the triangulated points colored by SQ
      association. Heading mismatch = rotation error; centre mismatch = position
      error. Makes the rotation-vs-translation question concrete per scene.
fig5: Manhattan structure — histogram of per-SQ residual angle to the voted cube
      frame (the 5deg clustering), and a unit-sphere scatter of SQ axes showing
      they pile onto 3 orthogonal directions.
"""
import os, sys, glob
sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")
sys.path.insert(0, "/work/courses/3dv/team39/ba/python")
import numpy as np
import offline_eval as oe
import strat_common as sc
from ba.superdec import (load_scene, umeyama_sim3_pred_to_world, invert_sim3,
                         _OCTAHEDRAL, _nearest_box_symmetry, assign_points_to_sqs)
from scipy.spatial.transform import Rotation
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/work/courses/3dv/team39/ba/eval/analysis"
CACHE = "/work/courses/3dv/team39/compose/data/ba_cache"


def apply_sim3(sim3, pts):
    s, R, t = sim3
    return s * (pts @ R.T) + t


def _rel0(poses):
    """Re-express C2W poses relative to view-0 (view0 -> identity), exactly as the
    pose metric's _to_view0_relative does. This is the gauge the AUC scores in."""
    P0inv = np.linalg.inv(poses[0])
    rel = np.einsum("ij,vjk->vik", P0inv, poses)
    rel[0] = np.eye(4)
    return rel


def recon_fig(scene_idx=6):
    path = sorted(glob.glob(os.path.join(CACHE, "*.npz")))[scene_idx]
    cache = oe.load_cache(path)
    cams = np.asarray(cache["cameras"], np.float64)         # W2C, predicted frame
    gt_poses = np.asarray(cache["gt_poses"], np.float64)    # C2W, GT frame
    V = cams.shape[0]
    z = np.array([0, 0, 1.0])

    # Build predicted C2W, then put BOTH pred and GT in the metric's view-0 frame.
    pred = np.zeros((V, 4, 4))
    for v in range(V):
        R_cw = Rotation.from_rotvec(cams[v, :3]).as_matrix().T
        pred[v] = np.eye(4); pred[v, :3, :3] = R_cw; pred[v, :3, 3] = -R_cw @ cams[v, 3:6]
    G = _rel0(gt_poses); P = _rel0(pred)
    gc = G[:, :3, 3]; gf = G[:, :3, :3] @ z
    pc = P[:, :3, 3]; pf = P[:, :3, :3] @ z
    # scale-align pred centres to GT (translation_angle is scale-free; this is for display)
    s = (np.linalg.norm(gc[1:], axis=1).sum() / max(np.linalg.norm(pc[1:], axis=1).sum(), 1e-9))
    pc = pc * s

    # relative rotation error per camera (vs view0 frame) = what AUC's pairs see
    rerr = np.degrees([np.arccos(np.clip((np.trace(P[v, :3, :3].T @ G[v, :3, :3]) - 1) / 2, -1, 1))
                       for v in range(V)])

    spread = gc.std(0); a, b = [int(i) for i in np.argsort(spread)[-2:]]
    fig, ax = plt.subplots(figsize=(8, 7.5))
    al = 0.9
    ax.quiver(gc[:, a], gc[:, b], gf[:, a], gf[:, b], color="green", angles="xy",
              scale=10, width=0.006, alpha=al, label="GT orientation")
    ax.quiver(pc[:, a], pc[:, b], pf[:, a], pf[:, b], color="red", angles="xy",
              scale=10, width=0.006, alpha=al, label="pred orientation")
    ax.scatter(gc[:, a], gc[:, b], c="green", s=60, marker="^", edgecolor="k", zorder=5)
    ax.scatter(pc[:, a], pc[:, b], c="red", s=60, marker="o", edgecolor="k", zorder=5)
    for v in range(V):
        ax.plot([gc[v, a], pc[v, a]], [gc[v, b], pc[v, b]], "k-", lw=0.7, alpha=0.5)
        ax.annotate(f"{rerr[v]:.0f}", (pc[v, a], pc[v, b]), fontsize=7, color="darkred")
    ax.scatter([gc[0, a]], [gc[0, b]], c="gold", s=140, marker="*", zorder=6,
               edgecolor="k", label="view-0 (anchor)")
    ax.set_title(f"scene {scene_idx}: cameras in the AUC's view-0 frame "
                 f"(GT green ^, pred red o)\nred numbers = per-camera relative rotation "
                 f"error (deg); mean {rerr[1:].mean():.1f}deg")
    ax.set_xlabel(f"axis {a}"); ax.set_ylabel(f"axis {b}")
    ax.legend(fontsize=8, loc="best"); ax.set_aspect("equal", "datalim")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fig4_recon_scene{scene_idx}.png"), dpi=130)
    print(f"fig4 scene{scene_idx}: mean relative-rot err {rerr[1:].mean():.1f}deg")


def manhattan_fig():
    paths = sorted(glob.glob(os.path.join(CACHE, "*.npz")))
    all_res = []
    axes_demo = None
    for i, p in enumerate(paths):
        cache = oe.load_cache(p)
        sq = sc.surface_pred(cache)
        if sq is None:
            continue
        R = Rotation.from_rotvec(sq["rotation_aa"]).as_matrix()
        scale = np.abs(np.asarray(sq["scale"], np.float64))
        aniso = scale.max(1) / np.clip(scale.min(1), 1e-9, None) > 1.5
        Rs = R[aniso]
        if len(Rs) < 5:
            continue
        # vote frame
        R_m = np.eye(3)
        for _ in range(10):
            gi, _ = _nearest_box_symmetry(Rs, R_m)
            tg = np.einsum("nij,njk->nik", Rs, np.transpose(_OCTAHEDRAL[gi], (0, 2, 1)))
            U, _, Vt = np.linalg.svd(tg.sum(0))
            D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); R_m = U @ D @ Vt
        _, tr = _nearest_box_symmetry(Rs, R_m)
        ang = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
        all_res.append(ang)
        if axes_demo is None:
            # express axes in the voted frame so clusters land on +/-x,y,z
            axes_demo = np.concatenate([(R_m.T @ Rs[:, :, k].T).T for k in range(3)], 0)
    res = np.concatenate(all_res)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(res, bins=np.arange(0, 45, 1.5), color="#6a4c93")
    ax1.axvline(np.median(res), color="k", ls="--", label=f"median {np.median(res):.1f}deg")
    ax1.axvline(15, color="#d1495b", ls=":", label="snap threshold 15deg")
    ax1.set_xlabel("SQ orientation residual to voted Manhattan frame (deg)")
    ax1.set_ylabel("# anisotropic SQs (all scenes)")
    ax1.set_title(f"{100*np.mean(res<10):.0f}% within 10deg, {100*np.mean(res<15):.0f}% within 15deg")
    ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    f = axes_demo / np.linalg.norm(axes_demo, axis=1, keepdims=True)
    f *= np.sign(f[:, np.argmax(np.abs(f), axis=1)])[:, None] if False else 1
    nearest = np.argmax(np.abs(f), axis=1)
    for k, col in zip(range(3), ["#d1495b", "#2e86ab", "#3a8a3a"]):
        m = nearest == k
        ax2.scatter(f[m, 0], f[m, 1], f[m, 2], s=8, c=col, alpha=0.5)
    ax2.set_title("SQ axes in voted frame:\npile onto 3 orthogonal directions (Manhattan)")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_manhattan.png"), dpi=130)
    print(f"fig5: median residual {np.median(res):.1f}deg over {len(res)} SQ-orientations")


if __name__ == "__main__":
    recon_fig(6)
    recon_fig(1)
    manhattan_fig()
