"""Validate what 'sparser' means on this data BEFORE running the experiment.

Discovery: every triangulated point is seen by EXACTLY 2 views (MASt3R pairwise
triangulation). So the BA structure is a camera-pair graph: point count on edge
(i,j) = #correspondences MASt3R found for that pair. Dropping a VIEW removes every
edge (point set) touching it.

Figures:
  fig7: (a) covisibility heatmap = #points per camera pair;
        (b) surviving points vs #kept views (greedy best-K subset);
        (c) top-down full 10-view cloud+cameras;
        (d) top-down K=4 view subset -> surviving points only.
This shows whether view-subsampling genuinely starves the reconstruction.
"""
import os, sys, glob
sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")
sys.path.insert(0, "/work/courses/3dv/team39/ba/python")
import numpy as np
import offline_eval as oe
from scipy.spatial.transform import Rotation
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/work/courses/3dv/team39/ba/eval/analysis"
CACHE = "/work/courses/3dv/team39/compose/data/ba_cache"


def pair_counts(cache):
    """V x V matrix: #points shared by each camera pair (each point has 2 views)."""
    ci = np.asarray(cache["cam_indices"]); pi = np.asarray(cache["pt_indices"])
    V = int(cache["cameras"].shape[0])
    order = np.argsort(pi, kind="stable")
    ci_s = ci[order]; pi_s = pi[order]
    M = np.zeros((V, V), int)
    # each point occupies 2 consecutive rows after sort-by-point
    for k in range(0, len(pi_s), 2):
        a, b = ci_s[k], ci_s[k + 1]
        M[a, b] += 1; M[b, a] += 1
    return M


def greedy_subsets(M):
    """Nested camera subsets maximizing surviving points: start from the best pair,
    add the view that adds the most edges. Returns list of (K, set, surviving)."""
    V = M.shape[0]
    i, j = np.unravel_index(np.argmax(M), M.shape)
    S = [int(i), int(j)]
    seq = []
    def surv(s):
        return sum(M[a, b] for a in s for b in s if a < b)
    seq.append((2, list(S), surv(S)))
    while len(S) < V:
        best, bestgain = None, -1
        for c in range(V):
            if c in S:
                continue
            g = sum(M[c, a] for a in S)
            if g > bestgain:
                bestgain, best = g, c
        S = S + [best]
        seq.append((len(S), list(S), surv(S)))
    return seq


def cam_centres_pred(cache):
    cams = np.asarray(cache["cameras"], np.float64); V = cams.shape[0]
    pc = np.zeros((V, 3))
    for v in range(V):
        R_cw = Rotation.from_rotvec(cams[v, :3]).as_matrix().T
        pc[v] = -R_cw @ cams[v, 3:6]
    return pc


def topdown(ax, pts, pc, keep_pts_mask, keep_cams, a, b, title):
    sub = np.random.default_rng(0).choice(len(pts), min(5000, len(pts)), replace=False)
    m = keep_pts_mask[sub]
    ax.scatter(pts[sub][~m][:, a], pts[sub][~m][:, b], s=2, c="#e8e8e8")
    ax.scatter(pts[sub][m][:, a], pts[sub][m][:, b], s=2, c="#2e86ab", alpha=0.5)
    for v in range(len(pc)):
        on = v in keep_cams
        ax.scatter([pc[v, a]], [pc[v, b]], s=90 if on else 40,
                   c="red" if on else "#bbbbbb", marker="^", edgecolor="k", zorder=5)
        ax.annotate(str(v), (pc[v, a], pc[v, b]), fontsize=8, zorder=6)
    ax.set_title(title); ax.set_aspect("equal", "datalim")


def fig(scene_idx=6):
    cache = oe.load_cache(sorted(glob.glob(os.path.join(CACHE, "*.npz")))[scene_idx])
    pts = np.asarray(cache["points"], np.float64)
    ci = np.asarray(cache["cam_indices"]); pi = np.asarray(cache["pt_indices"])
    V = int(cache["cameras"].shape[0])
    M = pair_counts(cache)
    seq = greedy_subsets(M)
    pc = cam_centres_pred(cache)
    spread = pc.std(0); a, b = [int(i) for i in np.argsort(spread)[-2:]]

    # point->pair lookup for masks
    order = np.argsort(pi, kind="stable")
    pair_of_pt = ci[order].reshape(-1, 2)  # (npts,2) the 2 cams of each point (sorted-by-pt)
    pt_ids = pi[order].reshape(-1, 2)[:, 0]
    cams_per_pt = np.full((pts.shape[0], 2), -1)
    cams_per_pt[pt_ids] = pair_of_pt

    def mask_for(keepset):
        ks = set(keepset)
        return np.array([cams_per_pt[i, 0] in ks and cams_per_pt[i, 1] in ks
                         for i in range(pts.shape[0])])

    fig = plt.figure(figsize=(13, 11))
    axA = fig.add_subplot(2, 2, 1)
    im = axA.imshow(M, cmap="viridis"); plt.colorbar(im, ax=axA, fraction=0.046)
    axA.set_title(f"(a) covisibility: #points per camera pair (scene {scene_idx})")
    axA.set_xlabel("camera"); axA.set_ylabel("camera")
    for i in range(V):
        for j in range(V):
            if M[i, j] > 0:
                axA.text(j, i, M[i, j], ha="center", va="center",
                         color="w" if M[i, j] < M.max() * 0.6 else "k", fontsize=6)

    axB = fig.add_subplot(2, 2, 2)
    Ks = [s[0] for s in seq]; surv = [s[2] for s in seq]
    axB.plot(Ks, surv, "o-", color="#d1495b")
    for K, s in zip(Ks, surv):
        axB.annotate(f"{s}", (K, s), fontsize=7)
    axB.set_xlabel("# kept views"); axB.set_ylabel("# surviving 2-view points")
    axB.set_title("(b) surviving points vs #views (greedy best-K subset)")
    axB.grid(alpha=0.3)

    axC = fig.add_subplot(2, 2, 3)
    topdown(axC, pts, pc, np.ones(len(pts), bool), set(range(V)), a, b,
            f"(c) FULL: 10 views, {len(pts)} points")
    # K=4 subset
    sub4 = next(s for s in seq if s[0] == 4)[1]
    axD = fig.add_subplot(2, 2, 4)
    m4 = mask_for(sub4)
    topdown(axD, pts, pc, m4, set(sub4), a, b,
            f"(d) K=4 views {sub4}: {int(m4.sum())} surviving points "
            f"({100*m4.mean():.0f}% of full)")

    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fig7_sparsity_scene{scene_idx}.png"), dpi=125)
    print(f"scene {scene_idx}: {len(pts)} pts, pair-count range {M[M>0].min()}-{M.max()}")
    print("surviving points by #views:", {K: s for K, _, s in seq})
    print(f"wrote {OUT}/fig7_sparsity_scene{scene_idx}.png")


if __name__ == "__main__":
    for s in (6, 1):
        fig(s)
