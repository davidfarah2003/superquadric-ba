"""Additional pose metrics from ALREADY-SAVED data (CPU only, no pipeline re-run).

For the three reported configs:
    VGGT-only    = "raw"            (cached pre-BA VGGT cameras)
    BA baseline  = "reproj"         (reprojection-only BA, lambda_surface=0)
    Ours         = "surface_snap"   (surface prior + Manhattan snap = live default)
                   (also report "surface" = surface prior, no snap, for completeness)

Metrics computed:
  (1) pose AUC @ {1,3,5,10} deg   -- combined, rotation-only, translation-only
  (2) rotation-only vs translation-only AUC@5
  (3) ATE (RMSE of camera centres) after Sim(3) alignment to GT centres

(1)+(2) are derived from the cached per-view-pair errors in
analysis/pose_decomp.npz (8 arrays of length 450 = 10 scenes x 45 pairs).
The HEADLINE AUC is the MEAN over per-scene AUCs (matching offline_eval/the
live harness), so we split the 450-pair pool into 10 contiguous 45-pair scene
blocks and average per-scene AUCs. We also report the pooled AUC for reference.

(3) ATE needs camera CENTRES, which are NOT in pose_decomp.npz. We rebuild them:
  - raw  : directly from cache['cameras'] (no BA).
  - reproj / surface_snap : by RE-SOLVING BA per scene on the CPU cache (Ceres),
    using the same recipe as analyze_pose.py (WIN config). ATE is the RMSE of
    Sim(3)-aligned predicted centres vs GT centres (umeyama with scale), averaged
    over scenes. This is the standard "ATE after Sim(3) alignment" for a
    gauge/scale-free reconstruction.
"""
import os, sys, json
sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")
sys.path.insert(0, "/work/courses/3dv/team39/ba/python")
import glob
import numpy as np
import offline_eval as oe
from mapanything.utils.metrics import calculate_auc_np

OUT = "/work/courses/3dv/team39/ba/eval/analysis"
os.makedirs(OUT, exist_ok=True)
CACHE_DIR = "/work/courses/3dv/team39/compose/data/ba_cache"

PAIRS_PER_SCENE = 45  # 10 views -> 45 unordered pairs
THRESHOLDS = (1, 3, 5, 10)

# config key in pose_decomp.npz -> human label / role
CONFIGS = [
    ("raw",          "VGGT-only"),
    ("reproj",       "BA baseline (reproj)"),
    ("surface",      "Ours (surface)"),
    ("surface_snap", "Ours (surface+snap)"),
]

# Winning recipe (mirrors analyze_pose.WIN -> the live default).
WIN = dict(max_points=20000, function_tolerance=1e-6, num_threads=1, fix_first_camera=True,
           huber_threshold=1.0, assoc_max_distance=0.0372, surface_huber=2.749,
           n_outer=2, inner_iters=41, warmup=True, residual_mode=1, lambda_surface=15)


# ---------------------------------------------------------------------------
# (1) + (2)  AUC from cached per-pair errors
# ---------------------------------------------------------------------------
def _auc(r, t, thr):
    """calculate_auc_np wrapper -> percent."""
    return 100.0 * calculate_auc_np(np.asarray(r), np.asarray(t), thr)[0]


def auc_decomp(r, t, thr):
    """(combined, rot-only, trans-only) AUC@thr in %."""
    z = np.zeros_like(r)
    return _auc(r, t, thr), _auc(r, z, thr), _auc(z, t, thr)


def per_scene_mean_auc(R, T, thr):
    """Mean over per-scene AUCs (headline protocol). Returns (comb,rot,trans)."""
    n = len(R)
    assert n % PAIRS_PER_SCENE == 0, f"len {n} not multiple of {PAIRS_PER_SCENE}"
    blocks = n // PAIRS_PER_SCENE
    accs = np.zeros((blocks, 3))
    for b in range(blocks):
        s = slice(b * PAIRS_PER_SCENE, (b + 1) * PAIRS_PER_SCENE)
        accs[b] = auc_decomp(R[s], T[s], thr)
    return accs.mean(0)


def compute_auc_table():
    z = np.load(os.path.join(OUT, "pose_decomp.npz"))
    table = {}
    for key, label in CONFIGS:
        R, T = z[f"{key}_r"], z[f"{key}_t"]
        entry = {"label": label, "n_pairs": int(len(R)), "per_scene_mean": {}, "pooled": {}}
        for thr in THRESHOLDS:
            c, ro, tr = per_scene_mean_auc(R, T, thr)
            entry["per_scene_mean"][str(thr)] = {
                "combined": float(c), "rot_only": float(ro), "trans_only": float(tr)}
            pc, pro, ptr = auc_decomp(R, T, thr)
            entry["pooled"][str(thr)] = {
                "combined": float(pc), "rot_only": float(pro), "trans_only": float(ptr)}
        table[key] = entry
    return table


# ---------------------------------------------------------------------------
# (3)  ATE after Sim(3) alignment of camera centres
# ---------------------------------------------------------------------------
def _centres_from_cameras(cameras):
    """W2C (V,10) -> camera centres C2W (V,3): centre = -R_wc^T @ t."""
    from scipy.spatial.transform import Rotation
    V = cameras.shape[0]
    C = np.zeros((V, 3))
    for v in range(V):
        R_wc = Rotation.from_rotvec(cameras[v, 0:3]).as_matrix()
        C[v] = -R_wc.T @ cameras[v, 3:6]
    return C


def _sim3_align_rmse(P, G):
    """Umeyama Sim(3) (with scale) aligning P onto G; return RMSE of aligned P vs G."""
    P = np.asarray(P, np.float64); G = np.asarray(G, np.float64)
    V = P.shape[0]
    muP, muG = P.mean(0), G.mean(0)
    Pc, Gc = P - muP, G - muG
    var_P = np.mean(np.sum(Pc ** 2, 1))
    H = (Gc.T @ Pc) / V
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    R = U @ D @ Vt
    s = float(np.sum(S * np.diag(D))) / var_P
    t = muG - s * R @ muP
    Pa = (s * (R @ P.T)).T + t
    return float(np.sqrt(np.mean(np.sum((Pa - G) ** 2, 1))))


def _refined_cameras(cache, extra):
    import importlib
    em = importlib.import_module("strategies.em_reassoc")
    p = dict(WIN); p.update(extra)
    return em.refine(cache, p)


def compute_ate(do_ba=True):
    """Per-scene ATE (Sim3-aligned centre RMSE) for each config; mean over scenes."""
    paths = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")),
                   key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    roles = {"raw": None}
    if do_ba:
        roles["reproj"] = dict(lambda_surface=0.0)
        roles["surface_snap"] = dict(manhattan_snap_deg=15.0)
    per_scene = {k: [] for k in roles}
    for path in paths:
        cache = oe.load_cache(path)
        G = np.asarray(cache["gt_centres"], np.float64)
        for key, extra in roles.items():
            if key == "raw":
                cams = np.asarray(cache["cameras"], np.float64)
            else:
                cams = np.ascontiguousarray(_refined_cameras(cache, extra), np.float64)
            C = _centres_from_cameras(cams)
            per_scene[key].append(_sim3_align_rmse(C, G))
        print(f"  {os.path.basename(path)}: " +
              "  ".join(f"{k}={per_scene[k][-1]:.4f}" for k in roles), flush=True)
    return {k: {"per_scene": [float(x) for x in v],
                "mean": float(np.mean(v)), "median": float(np.median(v))}
            for k, v in per_scene.items()}


# ---------------------------------------------------------------------------
# Figure + table
# ---------------------------------------------------------------------------
def build_figure(auc_table, ate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k, _ in CONFIGS]
    labels = [l for _, l in CONFIGS]
    colors = {"1": "#a8dadc", "3": "#457b9d", "5": "#1d3557", "10": "#e63946"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # Panel A: combined AUC across thresholds (per-scene mean)
    ax = axes[0]
    x = np.arange(len(keys)); w = 0.2
    for i, thr in enumerate(THRESHOLDS):
        vals = [auc_table[k]["per_scene_mean"][str(thr)]["combined"] for k in keys]
        ax.bar(x + (i - 1.5) * w, vals, w, label=f"@{thr}deg", color=colors[str(thr)])
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("combined pose AUC (%)")
    ax.set_title("(1) Combined AUC @ 1/3/5/10 deg\n(mean over per-scene AUCs)")
    ax.legend(fontsize=8)

    # Panel B: rot-only vs trans-only vs combined AUC@5
    ax = axes[1]
    w = 0.26
    rot = [auc_table[k]["per_scene_mean"]["5"]["rot_only"] for k in keys]
    trn = [auc_table[k]["per_scene_mean"]["5"]["trans_only"] for k in keys]
    cmb = [auc_table[k]["per_scene_mean"]["5"]["combined"] for k in keys]
    ax.bar(x - w, rot, w, label="rotation-only", color="#d1495b")
    ax.bar(x, trn, w, label="translation-only", color="#2e86ab")
    ax.bar(x + w, cmb, w, label="combined", color="#444")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("AUC@5deg (%)")
    ax.set_title("(2) Rotation-only vs translation-only AUC@5")
    ax.legend(fontsize=8)

    # Panel C: ATE (Sim3-aligned centre RMSE)
    ax = axes[2]
    ate_keys = [k for k in keys if k in ate]
    ate_labels = [dict(CONFIGS)[k] for k in ate_keys]
    means = [ate[k]["mean"] for k in ate_keys]
    bars = ax.bar(np.arange(len(ate_keys)), means, color="#6a994e")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(ate_keys)))
    ax.set_xticklabels(ate_labels, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("ATE RMSE (Sim3-aligned, scene units)")
    ax.set_title("(3) ATE after Sim(3) alignment\n(mean over scenes)")

    fig.tight_layout()
    p = os.path.join(OUT, "fig_extra_metrics.png")
    fig.savefig(p, dpi=130)
    print("figure ->", p)


def write_table(auc_table, ate):
    lines = []
    lines.append("=== ADDITIONAL POSE METRICS (CPU, from saved poses) ===\n")
    lines.append("(1)+(2) AUC: per-scene-mean protocol (headline). Pooled in parens.\n")
    hdr = f"{'config':<22}" + "".join(f"{'AUC@'+str(t):>10}" for t in THRESHOLDS)
    lines.append("-- Combined AUC (%) --")
    lines.append(hdr)
    for k, lab in CONFIGS:
        row = f"{lab:<22}"
        for t in THRESHOLDS:
            ps = auc_table[k]["per_scene_mean"][str(t)]["combined"]
            po = auc_table[k]["pooled"][str(t)]["combined"]
            row += f"{ps:>6.2f}({po:>4.1f})"[:10].rjust(10)
        lines.append(row)
    lines.append("")
    lines.append("-- AUC@5: rotation-only / translation-only / combined (per-scene-mean %) --")
    lines.append(f"{'config':<22}{'rot-only':>10}{'trans-only':>12}{'combined':>10}")
    for k, lab in CONFIGS:
        e = auc_table[k]["per_scene_mean"]["5"]
        lines.append(f"{lab:<22}{e['rot_only']:>10.2f}{e['trans_only']:>12.2f}{e['combined']:>10.2f}")
    lines.append("")
    lines.append("-- (3) ATE RMSE after Sim(3) alignment (scene units) --")
    lines.append(f"{'config':<22}{'mean':>10}{'median':>10}")
    for k, lab in CONFIGS:
        if k in ate:
            lines.append(f"{lab:<22}{ate[k]['mean']:>10.4f}{ate[k]['median']:>10.4f}")
        else:
            lines.append(f"{lab:<22}{'(skipped)':>10}")
    txt = "\n".join(lines)
    p = os.path.join(OUT, "extra_metrics_table.txt")
    open(p, "w").write(txt + "\n")
    print("\n" + txt)
    print("\ntable ->", p)
    return txt


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ba", action="store_true",
                    help="skip BA re-solve for ATE (only raw ATE)")
    a = ap.parse_args()

    print("Computing AUC table from pose_decomp.npz ...")
    auc_table = compute_auc_table()

    print("\nComputing ATE (Sim3-aligned centre RMSE) ...")
    ate = compute_ate(do_ba=not a.no_ba)

    out = {"auc": auc_table, "ate": ate,
           "notes": {
               "pairs_per_scene": PAIRS_PER_SCENE, "n_scenes": 10,
               "auc_protocol": "mean over per-scene AUCs (headline); pooled also stored",
               "ate_units": "scene/world units (gauge-free; Sim3 with scale aligns to GT centres)",
               "config_roles": {k: l for k, l in CONFIGS},
               "ours_is": "surface_snap (live default)"}}
    json.dump(out, open(os.path.join(OUT, "extra_metrics.json"), "w"), indent=2)
    print("json ->", os.path.join(OUT, "extra_metrics.json"))

    write_table(auc_table, ate)
    build_figure(auc_table, ate)


if __name__ == "__main__":
    main()
