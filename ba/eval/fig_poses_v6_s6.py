"""Poster figure: VGGT-only -> + bundle adjustment -> + super-quadric prior, on
the SAME real scene-6 6-view problem (qualitative pose visualization).

Three benchmark runs select the SAME 6 views (deterministic seed):
  VGGT  : logs/viz_vggt_v6_s6          (bundle_adjustment=none) -> vggt/cameras.json
  BASE  : ...surface_em_cov06_v6_lam0   (BA, lambda_surface=0)   -> ba/cameras.json
  OURS  : ...surface_em_cov06_v6_lam15  (BA, lambda_surface=15)  -> ba/cameras.json

IMPORTANT — faithful display. AUC@5 scores pairwise relative pose, not absolute
position, and the three configs live in different gauges (the BA runs are
GT-aligned, raw VGGT is not). To make the PICTURE agree with the metric we
Sim(3)-align EACH config's camera centres to ground truth (the standard
trajectory-comparison alignment) and rotate its headings by the same R. The
residual *rotation* error then ranks VGGT > BA > Ours monotonically, exactly like
AUC@5 (29 < 48 < 57) — so no panel can look better than its score. Each panel is
badged with that mean rotation residual; AUC@5 is given in the caption.

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


def _umeyama(src, dst):
    """similarity transform s,R,t minimising ||dst - (s R src + t)||."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    H = S.T @ D / len(src)
    U, d, Vt = np.linalg.svd(H)
    E = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        E[2, 2] = -1
    R = Vt.T @ E @ U.T
    var = (S ** 2).sum() / len(src)
    s = np.trace(np.diag(d) @ E) / var
    t = mu_d - s * R @ mu_s
    return s, R, t


def _frame(pred_c2w, gt_c2w, a, b):
    """Sim(3)-align pred camera centres to GT, rotate headings by R; project to (a,b)."""
    z = np.array([0, 0, 1.0])
    pc, gc = pred_c2w[:, :3, 3], gt_c2w[:, :3, 3]
    s, R, t = _umeyama(pc, gc)
    pc_a = (s * (R @ pc.T).T) + t
    gf = (gt_c2w[:, :3, :3] @ z)
    pf = ((R @ pred_c2w[:, :3, :3]) @ z)
    rot = np.degrees([np.arccos(np.clip((np.trace((R @ pred_c2w[v, :3, :3]).T @ gt_c2w[v, :3, :3]) - 1) / 2, -1, 1))
                      for v in range(gt_c2w.shape[0])])
    return gc[:, [a, b]], gf[:, [a, b]], pc_a[:, [a, b]], pf[:, [a, b]], float(np.mean(rot))


def _auc(run):
    f = glob.glob(os.path.join(run, "*per_scene_results.json"))
    if not f:
        return None
    v = json.load(open(f[0]))[SCENE]["pose_auc_5"]
    return v[0] if isinstance(v, list) else v


def _panel(ax, fr, title, badge_color):
    gc, gf, pc, pf, rot = fr
    sc = 9
    for v in range(gc.shape[0]):
        ax.plot([gc[v, 0], pc[v, 0]], [gc[v, 1], pc[v, 1]], "-", color="0.65", lw=1.3, zorder=2)
    ax.quiver(gc[:, 0], gc[:, 1], gf[:, 0], gf[:, 1], color=GT_C, angles="xy", scale=sc, width=0.016, zorder=3)
    ax.quiver(pc[:, 0], pc[:, 1], pf[:, 0], pf[:, 1], color=PR_C, angles="xy", scale=sc, width=0.016, zorder=3)
    ax.scatter(gc[:, 0], gc[:, 1], c=GT_C, s=150, marker="^", edgecolor="k", lw=0.8, zorder=5)
    ax.scatter(pc[:, 0], pc[:, 1], c=PR_C, s=130, marker="o", edgecolor="k", lw=0.8, zorder=5)
    ax.set_title(title, fontsize=21, fontweight="bold", color=DARK, pad=10)
    ax.text(0.05, 0.97, f"{rot:.0f}$^\\circ$", transform=ax.transAxes, ha="left", va="top",
            fontsize=40, fontweight="bold", color=badge_color)
    ax.text(0.06, 0.80, "mean\nrotation error", transform=ax.transAxes, ha="left", va="top",
            fontsize=13.5, color="#6F6F6F", linespacing=1.1)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#DEE0E4"); s.set_linewidth(1.5)


def main():
    dv, d0, d15 = _sample_dir(VGGT_RUN), _sample_dir(LAM0), _sample_dir(LAM15)
    gt = _load_cams(os.path.join(d0, "gt", "cameras.json"))
    vggt = _load_cams(os.path.join(dv, "vggt", "cameras.json"))
    base = _load_cams(os.path.join(d0, "ba", "cameras.json"))
    ours = _load_cams(os.path.join(d15, "ba", "cameras.json"))

    spread = gt[:, :3, 3].std(0)
    a, b = sorted(int(i) for i in np.argsort(spread)[-2:])
    fv, fb, fo = (_frame(p, gt, a, b) for p in (vggt, base, ours))

    fig, axs = plt.subplots(1, 3, figsize=(19.8, 6.4))
    _panel(axs[0], fv, "VGGT-only (feed-forward)", PR_C)
    _panel(axs[1], fb, "+ bundle adjustment", "#555555")
    _panel(axs[2], fo, "+ super-quadric prior (Ours)", GT_C)

    allx = np.concatenate([np.r_[f[0][:, 0], f[2][:, 0]] for f in (fv, fb, fo)])
    ally = np.concatenate([np.r_[f[0][:, 1], f[2][:, 1]] for f in (fv, fb, fo)])
    mx, my = 0.16 * (allx.max() - allx.min()), 0.16 * (ally.max() - ally.min())
    for ax in axs:
        ax.set_xlim(allx.min() - mx, allx.max() + mx); ax.set_ylim(ally.min() - my, ally.max() + my)

    handles = [Line2D([0], [0], marker="^", color="w", markerfacecolor=GT_C, markeredgecolor="k", markersize=15, label="ground-truth camera"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor=PR_C, markeredgecolor="k", markersize=14, label="estimated camera"),
               Line2D([0], [0], color="0.65", lw=2, label="position error")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=16, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")
    print(f"  mean rot residual: VGGT {fv[4]:.1f} -> +BA {fb[4]:.1f} -> +prior {fo[4]:.1f}")
    print(f"  AUC@5 (caption): VGGT {_auc(VGGT_RUN)} -> +BA {_auc(LAM0)} -> +prior {_auc(LAM15)}")


if __name__ == "__main__":
    main()
