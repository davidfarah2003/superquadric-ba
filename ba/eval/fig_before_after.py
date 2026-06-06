"""Poster figure: before/after camera poses (VGGT-only vs Ours).

Two top-down panels in the AUC's view-0 frame for one scene:
  left  = raw VGGT feed-forward camera poses vs ground truth
  right = Ours (surface BA + super-quadric prior) vs ground truth

Green = ground-truth cameras, red = estimated; the headings are the arrows.
The mean per-camera relative rotation error (what AUC@5 scores) is printed in
each panel title. Designed for the poster: big type, thick arrows, no axis
clutter, shared axis limits so the two panels are directly comparable.

Writes poster/figures/before_after_poses.png.
"""
import os, sys, glob, importlib
import numpy as np
sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")
sys.path.insert(0, "/work/courses/3dv/team39/ba/python")
import offline_eval as oe
from scipy.spatial.transform import Rotation
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = "/work/courses/3dv/team39/compose/data/ba_cache"
OUT = "/work/courses/3dv/team39/poster/figures/before_after_poses.png"
GT_C = "#2E7D32"      # ground truth (green)
PR_C = "#C0392B"      # estimate (red)
SCENE = 6


def _rel0(poses):
    P0inv = np.linalg.inv(poses[0])
    rel = np.einsum("ij,vjk->vik", P0inv, poses)
    rel[0] = np.eye(4)
    return rel


def _cams_to_c2w(cams):
    V = cams.shape[0]
    pred = np.zeros((V, 4, 4))
    for v in range(V):
        R_cw = Rotation.from_rotvec(cams[v, :3]).as_matrix().T
        pred[v] = np.eye(4)
        pred[v, :3, :3] = R_cw
        pred[v, :3, 3] = -R_cw @ cams[v, 3:6]
    return pred


def _frame(predc2w, gt, a, b):
    """Return GT/pred centres+headings in view-0 frame on display axes (a,b)."""
    z = np.array([0, 0, 1.0])
    G = _rel0(gt); P = _rel0(predc2w)
    gc = G[:, :3, 3]; gf = G[:, :3, :3] @ z
    pc = P[:, :3, 3]; pf = P[:, :3, :3] @ z
    s = np.linalg.norm(gc[1:], axis=1).sum() / max(np.linalg.norm(pc[1:], axis=1).sum(), 1e-9)
    pc = pc * s
    rerr = np.degrees([np.arccos(np.clip((np.trace(P[v, :3, :3].T @ G[v, :3, :3]) - 1) / 2, -1, 1))
                       for v in range(gt.shape[0])])
    return gc[:, [a, b]], gf[:, [a, b]], pc[:, [a, b]], pf[:, [a, b]], float(np.mean(rerr[1:]))


def _panel(ax, gc, gf, pc, pf, title, err, err_color):
    sc = 9
    ax.quiver(gc[:, 0], gc[:, 1], gf[:, 0], gf[:, 1], color=GT_C, angles="xy",
              scale=sc, width=0.011, zorder=3)
    ax.quiver(pc[:, 0], pc[:, 1], pf[:, 0], pf[:, 1], color=PR_C, angles="xy",
              scale=sc, width=0.011, zorder=3)
    for v in range(gc.shape[0]):
        ax.plot([gc[v, 0], pc[v, 0]], [gc[v, 1], pc[v, 1]], "-", color="0.55", lw=1.2, zorder=2)
    ax.scatter(gc[:, 0], gc[:, 1], c=GT_C, s=130, marker="^", edgecolor="k", lw=0.8, zorder=5)
    ax.scatter(pc[:, 0], pc[:, 1], c=PR_C, s=110, marker="o", edgecolor="k", lw=0.8, zorder=5)
    ax.scatter([gc[0, 0]], [gc[0, 1]], c="gold", s=320, marker="*", edgecolor="k", lw=1.0, zorder=6)
    ax.set_title(title, fontsize=23, fontweight="bold", color="#123362", pad=14)
    ax.text(0.04, 0.97, f"{err:.0f}$^\\circ$", transform=ax.transAxes, ha="left",
            va="top", fontsize=40, fontweight="bold", color=err_color)
    ax.text(0.045, 0.80, "mean\nrotation error", transform=ax.transAxes, ha="left",
            va="top", fontsize=15, color="#6F6F6F", linespacing=1.1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#DEE0E4"); s.set_linewidth(1.5)


def main():
    path = sorted(glob.glob(os.path.join(CACHE, "*.npz")))[SCENE]
    c = oe.load_cache(path)
    gt = np.asarray(c["gt_poses"], np.float64)
    vggt_c2w = _cams_to_c2w(np.asarray(c["cameras"], np.float64))
    strat = importlib.import_module("strategies.em_reassoc")
    ours_cams = strat.refine(c, {"lambda_surface": 15.0, "assoc_max_distance": 0.15,
                                 "max_points": 5000, "seed": 0})
    ours_c2w = _cams_to_c2w(np.asarray(ours_cams, np.float64))

    # pick the two display axes with the most GT spread (top-down)
    spread = _rel0(gt)[:, :3, 3].std(0)
    a, b = sorted(int(i) for i in np.argsort(spread)[-2:])

    gcL, gfL, pcL, pfL, eL = _frame(vggt_c2w, gt, a, b)
    gcR, gfR, pcR, pfR, eR = _frame(ours_c2w, gt, a, b)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 6.8))
    _panel(axL, gcL, gfL, pcL, pfL, "VGGT-only (feed-forward)", eL, PR_C)
    _panel(axR, gcR, gfR, pcR, pfR, "Ours  (BA + super-quadric prior)", eR, GT_C)

    # shared limits so the panels are directly comparable
    xs = np.r_[gcL[:, 0], pcL[:, 0], gcR[:, 0], pcR[:, 0]]
    ys = np.r_[gcL[:, 1], pcL[:, 1], gcR[:, 1], pcR[:, 1]]
    mx = 0.18 * (xs.max() - xs.min()); my = 0.18 * (ys.max() - ys.min())
    for ax in (axL, axR):
        ax.set_xlim(xs.min() - mx, xs.max() + mx)
        ax.set_ylim(ys.min() - my, ys.max() + my)

    # one shared legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="^", color="w", markerfacecolor=GT_C, markeredgecolor="k", markersize=15, label="ground-truth camera"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor=PR_C, markeredgecolor="k", markersize=14, label="estimated camera"),
               Line2D([0], [0], marker="*", color="w", markerfacecolor="gold", markeredgecolor="k", markersize=20, label="anchor (view 0)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=17, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}  (VGGT {eL:.1f} deg -> Ours {eR:.1f} deg)")


if __name__ == "__main__":
    main()
