"""Diagnostic analysis + visualizations for the sparse-view pose result.

Central question: is pose_auc_5 ROTATION-bound? The metric is the AUC of
max(rotation_err, translation_angle_err) per view-pair up to 5deg, so we can
decompose it: rotation-only AUC, translation-only AUC, combined. If
combined ~= rotation-only << translation-only, rotation is the binding constraint.

Modes:
  raw            : decompose the cached PRE-BA (raw VGGT) cameras  [LIGHT, no Ceres]
  --with-ba      : ALSO run reproj-only + surface-hinge(no-snap) + surface-hinge(snap)
                   BA per scene and decompose each  [HEAVY -> run as a batch job]

Saves per-pair errors to analysis/pose_decomp.npz and figures to analysis/*.png.
Plots use Agg (headless). Run plotting (build_figures) anywhere; it only needs the npz.
"""
import os, sys, json, argparse
sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")
sys.path.insert(0, "/work/courses/3dv/team39/ba/python")
import glob
import numpy as np
import offline_eval as oe
from mapanything.utils.metrics import se3_to_relative_pose_error, calculate_auc_np

OUT = "/work/courses/3dv/team39/ba/eval/analysis"
os.makedirs(OUT, exist_ok=True)

# Winning recipe (matches the live default / 29.6 config). num_threads=1 for
# determinism + scene-level parallelism; max_points capped for a fast, qualitative
# rot/trans decomposition (same cap on all arms -> the comparison is faithful).
WIN = dict(max_points=20000, function_tolerance=1e-6, num_threads=1, fix_first_camera=True,
           huber_threshold=1.0, assoc_max_distance=0.0372, surface_huber=2.749,
           n_outer=2, inner_iters=41, warmup=True, residual_mode=1, lambda_surface=15)


def per_pair_errors(cameras, gt_poses, gt_centres):
    """Return per view-pair (rot_err_deg, trans_angle_err_deg) for these cameras."""
    pred = oe.cameras_to_pred_poses(np.ascontiguousarray(cameras, np.float64), gt_centres)
    pr = oe._to_view0_relative(pred)
    gt = oe._to_view0_relative(gt_poses)
    r, t = se3_to_relative_pose_error(pr, gt, pr.shape[0])
    return r.cpu().numpy(), t.cpu().numpy()


def auc3(r, t, thr=5):
    """(combined, rot-only, trans-only) AUC@thr in %."""
    z = np.zeros_like(r)
    return (100 * calculate_auc_np(r, t, thr)[0],
            100 * calculate_auc_np(r, z, thr)[0],
            100 * calculate_auc_np(z, t, thr)[0])


def refined_cameras(cache, extra):
    import importlib
    em = importlib.import_module("strategies.em_reassoc")
    p = dict(WIN); p.update(extra)
    return em.refine(cache, p)


def _scene_worker(task):
    """Process ONE scene: return (name, {config: (r_errs, t_errs)}). Module-level
    so it is picklable for ProcessPoolExecutor."""
    path, with_ba = task
    cache = oe.load_cache(path)
    gt_poses = np.asarray(cache["gt_poses"], np.float64)
    gt_centres = np.asarray(cache["gt_centres"], np.float64)
    cam_sets = {"raw": np.asarray(cache["cameras"], np.float64)}
    if with_ba:
        cam_sets["reproj"] = refined_cameras(cache, dict(lambda_surface=0.0))
        cam_sets["surface"] = refined_cameras(cache, dict(manhattan_snap_deg=0.0))
        cam_sets["surface_snap"] = refined_cameras(cache, dict(manhattan_snap_deg=15.0))
    out = {c: per_pair_errors(cam, gt_poses, gt_centres) for c, cam in cam_sets.items()}
    return os.path.basename(path), out


def _save(results, configs):
    """Aggregate whatever scenes are done so far and write npz + json (incremental,
    so a timeout still leaves usable partial output)."""
    npz = {}; res = {"configs": configs, "n_scenes_done": len(results), "summary": {}}
    for c in configs:
        R = np.concatenate([out[c][0] for _, out in results])
        T = np.concatenate([out[c][1] for _, out in results])
        ps = np.array([auc3(*out[c]) for _, out in results])  # per-scene comb,rot,trans
        npz[f"{c}_r"] = R; npz[f"{c}_t"] = T
        res["summary"][c] = {
            "auc_combined": float(ps[:, 0].mean()), "auc_rot_only": float(ps[:, 1].mean()),
            "auc_trans_only": float(ps[:, 2].mean()), "median_rot_err": float(np.median(R)),
            "median_trans_err": float(np.median(T)), "frac_pairs_rot_binding": float(np.mean(R > T)),
            "frac_rot_under5": float(np.mean(R < 5)), "frac_trans_under5": float(np.mean(T < 5))}
    np.savez(os.path.join(OUT, "pose_decomp.npz"), **npz)
    json.dump(res, open(os.path.join(OUT, "pose_decomp.json"), "w"), indent=2)
    return res


def collect(cache_dir, with_ba, jobs=2):
    import concurrent.futures as cf
    paths = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
    configs = ["raw"] + (["reproj", "surface", "surface_snap"] if with_ba else [])
    tasks = [(p, with_ba) for p in paths]
    results = []
    with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
        for name, out in ex.map(_scene_worker, tasks):
            results.append((name, out))
            print(f"  {name}: " + "  ".join(f"{c}={auc3(*out[c])[0]:.1f}" for c in configs), flush=True)
            _save(results, configs)  # incremental
    return _save(results, configs)


def build_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(os.path.join(OUT, "pose_decomp.npz"))
    res = json.load(open(os.path.join(OUT, "pose_decomp.json")))
    configs = res["configs"]
    labels = {"raw": "raw VGGT", "reproj": "reproj-BA", "surface": "surface-BA",
              "surface_snap": "surface+snap"}

    # --- Fig 1: AUC decomposition bars (rot-only / trans-only / combined) ---
    fig, ax = plt.subplots(figsize=(1.6 * len(configs) + 2, 4.2))
    x = np.arange(len(configs)); w = 0.26
    rot = [res["summary"][c]["auc_rot_only"] for c in configs]
    trn = [res["summary"][c]["auc_trans_only"] for c in configs]
    cmb = [res["summary"][c]["auc_combined"] for c in configs]
    ax.bar(x - w, rot, w, label="rotation-only AUC", color="#d1495b")
    ax.bar(x, trn, w, label="translation-only AUC", color="#2e86ab")
    ax.bar(x + w, cmb, w, label="combined (pose_auc_5)", color="#444")
    for xi, v in zip(x - w, rot): ax.text(xi, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    for xi, v in zip(x + w, cmb): ax.text(xi, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([labels.get(c, c) for c in configs])
    ax.set_ylabel("AUC@5deg (%)"); ax.legend(fontsize=8)
    ax.set_title("pose_auc_5 is bound by BOTH rot AND trans (combined < either);\n"
                 "surface term barely moves it vs plain reproj-BA")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_auc_decomp.png"), dpi=130)

    # --- Fig 2: pooled per-pair error histograms (rot vs trans), 5deg line ---
    base = "surface" if "surface" in configs else "raw"
    R, T = z[f"{base}_r"], z[f"{base}_t"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, 30, 61)
    ax.hist(R, bins=bins, alpha=0.6, color="#d1495b", label=f"rotation err (med {np.median(R):.1f}deg)")
    ax.hist(T, bins=bins, alpha=0.6, color="#2e86ab", label=f"translation-angle err (med {np.median(T):.1f}deg)")
    ax.axvline(5, color="k", ls="--", lw=1.2, label="5deg AUC cutoff")
    ax.set_xlabel("per-view-pair error (deg)"); ax.set_ylabel("# pairs")
    ax.set_title(f"{labels.get(base, base)}: {100*np.mean(R<5):.0f}% of pairs <5deg rot, "
                 f"{100*np.mean(T<5):.0f}% <5deg trans")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_error_hist.png"), dpi=130)

    # --- Fig 3: per-pair scatter rot vs trans, binding region ---
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(R, T, s=6, alpha=0.35, color="#555")
    lim = min(30, max(R.max(), T.max()))
    ax.plot([0, lim], [0, lim], "k:", lw=1)
    ax.axvline(5, color="#d1495b", ls="--", lw=1); ax.axhline(5, color="#2e86ab", ls="--", lw=1)
    ax.add_patch(plt.Rectangle((0, 0), 5, 5, fill=True, color="green", alpha=0.08))
    ax.set_xlabel("rotation err (deg)"); ax.set_ylabel("translation-angle err (deg)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title(f"{labels.get(base, base)}: pairs scored 'good' only inside the 5x5 box\n"
                 f"rotation is the binding error in {100*np.mean(R>T):.0f}% of pairs")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_rot_vs_trans.png"), dpi=130)
    print(f"figures written to {OUT}/fig1_auc_decomp.png, fig2_error_hist.png, fig3_rot_vs_trans.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="/work/courses/3dv/team39/compose/data/ba_cache")
    ap.add_argument("--with-ba", action="store_true", help="also run BA variants (HEAVY)")
    ap.add_argument("--plot-only", action="store_true", help="just rebuild figures from npz")
    a = ap.parse_args()
    if not a.plot_only:
        res = collect(a.cache_dir, a.with_ba)
        print("\n=== SUMMARY (AUC@5: combined / rot-only / trans-only) ===")
        for c in res["configs"]:
            s = res["summary"][c]
            print(f"  {c:13s}  comb={s['auc_combined']:5.2f}  rot={s['auc_rot_only']:5.2f}  "
                  f"trans={s['auc_trans_only']:5.2f}   median_rot={s['median_rot_err']:.1f}deg  "
                  f"median_trans={s['median_trans_err']:.1f}deg  rot_binds={100*s['frac_pairs_rot_binding']:.0f}%")
    build_figures()


if __name__ == "__main__":
    main()
