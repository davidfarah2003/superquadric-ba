"""Poster figure (honest, full pipeline): VGGT-only -> + bundle adjustment ->
+ super-quadric prior, on the SAME real scene-6 6-view problem.

Three benchmark runs select the SAME 6 views (deterministic seed) and differ
only in what runs after VGGT:
  VGGT  : logs/viz_vggt_v6_s6              (bundle_adjustment=none) -> vggt/cameras.json
  BASE  : ...surface_em_cov06_v6_lam0      (BA, lambda_surface=0)   -> ba/cameras.json
  OURS  : ...surface_em_cov06_v6_lam15.0   (BA, lambda_surface=15)  -> ba/cameras.json

Honest attribution: VGGT->BA is bundle adjustment's contribution; BA->Ours is the
prior's. The poster's claim (the prior) is the LAST step only. Each panel shows
estimated cameras (red) vs ground truth (green) top-down in the AUC's view-0
frame, scale-aligned for display, annotated with pose AUC@5 (the poster metric).

NB: the surface-BA runs' own vggt/cameras.json is aliased to the BA result
(`preds_vggt = preds` before in-place BA), so genuine VGGT poses come from the
separate no-BA run.

Writes poster/figures/poses_v6_s6.png.
"""
import os, sys, json, glob
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

VGGT_RUN = "/work/courses/3dv/team39/logs/viz_vggt_v6_s6"
LAM0 = "/work/courses/3dv/team39/logs/benchmark_ase_sparse_surface_em_cov06_v6_lam0"
LAM15 = "/work/courses/3dv/team39/logs/benchmark_ase_sparse_surface_em_cov06_v6_lam15.0"
OUT = "/work/courses/3dv/team39/poster/figures/poses_v6_s6.png"
SCENE = "6"
GT_C, PR_C, DARK = "#2E7D32", "#C0392B", "#123362"


def _sample_dir(run):
    hits = glob.glob(os.path.join(run, "viz", "*", "sample_*"))
    if not hits:
        raise FileNotFoundError(f"no viz sample under {run}")
    return sorted(hits)[0]


def _load_cams(path):
    d = json.load(open(path)); out = []
    for v in sorted(d.keys()):
        e = d[v]
        if "cam_to_world" in e:
            out.append(np.array(e["cam_to_world"], float))
        else:
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(e["quat_xyzw"]).as_matrix()
            T[:3, 3] = e["translation"]
            out.append(T)
    return np.array(out)


def _rel0(P):
    P0 = np.linalg.inv(P[0]); R = np.einsum("ij,vjk->vik", P0, P); R[0] = np.eye(4)
    return R


def _frame(pred_c2w, gt_c2w, a, b):
    """view-0 frame, scale-aligned (global) so all panels are comparable to GT."""
    z = np.array([0, 0, 1.0])
    G, P = _rel0(gt_c2w), _rel0(pred_c2w)
    gc, gf = G[:, :3, 3], G[:, :3, :3] @ z
    pc, pf = P[:, :3, 3], P[:, :3, :3] @ z
    s = np.linalg.norm(gc[1:], axis=1).sum() / max(np.linalg.norm(pc[1:], axis=1).sum(), 1e-9)
    pc = pc * s
    return gc[:, [a, b]], gf[:, [a, b]], pc[:, [a, b]], pf[:, [a, b]]


def _auc(run):
    f = glob.glob(os.path.join(run, "*per_scene_results.json"))
    if not f:
        return None
    v = json.load(open(f[0]))[SCENE]["pose_auc_5"]
    return v[0] if isinstance(v, list) else v


def _panel(ax, fr, title, auc, badge_color):
    gc, gf, pc, pf = fr
    sc = 9
    for v in range(gc.shape[0]):
        ax.plot([gc[v, 0], pc[v, 0]], [gc[v, 1], pc[v, 1]], "-", color="0.6", lw=1.5, zorder=2)
    ax.quiver(gc[:, 0], gc[:, 1], gf[:, 0], gf[:, 1], color=GT_C, angles="xy", scale=sc, width=0.013, zorder=3)
    ax.quiver(pc[:, 0], pc[:, 1], pf[:, 0], pf[:, 1], color=PR_C, angles="xy", scale=sc, width=0.013, zorder=3)
    ax.scatter(gc[:, 0], gc[:, 1], c=GT_C, s=150, marker="^", edgecolor="k", lw=0.8, zorder=5)
    ax.scatter(pc[:, 0], pc[:, 1], c=PR_C, s=130, marker="o", edgecolor="k", lw=0.8, zorder=5)
    ax.scatter([gc[0, 0]], [gc[0, 1]], c="gold", s=330, marker="*", edgecolor="k", lw=1.0, zorder=6)
    ax.set_title(title, fontsize=21, fontweight="bold", color=DARK, pad=10)
    ax.text(0.05, 0.97, f"{auc:.0f}", transform=ax.transAxes, ha="left", va="top",
            fontsize=40, fontweight="bold", color=badge_color)
    ax.text(0.06, 0.80, "AUC@5", transform=ax.transAxes, ha="left", va="top",
            fontsize=14, color="#6F6F6F")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#DEE0E4"); s.set_linewidth(1.5)


def main():
    dv, d0, d15 = _sample_dir(VGGT_RUN), _sample_dir(LAM0), _sample_dir(LAM15)
    gt = _load_cams(os.path.join(d0, "gt", "cameras.json"))
    vggt = _load_cams(os.path.join(dv, "vggt", "cameras.json"))
    base = _load_cams(os.path.join(d0, "ba", "cameras.json"))
    ours = _load_cams(os.path.join(d15, "ba", "cameras.json"))
    av, ab, ao = _auc(VGGT_RUN), _auc(LAM0), _auc(LAM15)
    if None in (av, ab, ao):
        raise RuntimeError(f"AUC not ready: vggt={av} base={ab} ours={ao}")

    spread = _rel0(gt)[:, :3, 3].std(0)
    a, b = sorted(int(i) for i in np.argsort(spread)[-2:])
    fv, fb, fo = (_frame(p, gt, a, b) for p in (vggt, base, ours))

    fig, axs = plt.subplots(1, 3, figsize=(19.8, 6.4))
    _panel(axs[0], fv, "VGGT-only (feed-forward)", av, PR_C)
    _panel(axs[1], fb, "+ bundle adjustment", ab, "#555555")
    _panel(axs[2], fo, "+ super-quadric prior (Ours)", ao, GT_C)

    allx = np.concatenate([np.r_[f[0][:, 0], f[2][:, 0]] for f in (fv, fb, fo)])
    ally = np.concatenate([np.r_[f[0][:, 1], f[2][:, 1]] for f in (fv, fb, fo)])
    mx, my = 0.16 * (allx.max() - allx.min()), 0.16 * (ally.max() - ally.min())
    for ax in axs:
        ax.set_xlim(allx.min() - mx, allx.max() + mx); ax.set_ylim(ally.min() - my, ally.max() + my)

    handles = [Line2D([0], [0], marker="^", color="w", markerfacecolor=GT_C, markeredgecolor="k", markersize=15, label="ground-truth camera"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor=PR_C, markeredgecolor="k", markersize=14, label="estimated camera"),
               Line2D([0], [0], color="0.6", lw=2, label="position error"),
               Line2D([0], [0], marker="*", color="w", markerfacecolor="gold", markeredgecolor="k", markersize=20, label="anchor (view 0)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=16, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")
    print(f"  AUC@5: VGGT {av} -> +BA {ab} -> +prior {ao}")


if __name__ == "__main__":
    main()
