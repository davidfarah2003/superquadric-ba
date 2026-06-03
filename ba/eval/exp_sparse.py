"""Does the superquadric surface prior gain LEVERAGE as the point cloud gets
sparse? Decomposition showed surface~=reproj at full density (40k-117k pts pin
the 10 cameras, leaving the prior nothing to do). Hypothesis: starve the points
and reprojection under-constrains the cameras, so the surface prior helps more —
possibly sparse+surface >= dense plain-BA (the 29.42 bar), a less-complexity win.

Sweep max_points; at each density run reproj-only vs surface(hinge) BA; report
pose_auc_5 and the surface-minus-reproj delta. num_threads=1 (determinism),
incremental JSON save (timeout-safe). Plot: AUC vs density + delta vs density.
"""
import os, sys, json, argparse
sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")
sys.path.insert(0, "/work/courses/3dv/team39/ba/python")
import numpy as np
import run_strategy as rs

OUT = "/work/courses/3dv/team39/ba/eval/analysis"
os.makedirs(OUT, exist_ok=True)
BAR = 29.42
NS = [100, 250, 500, 1000, 2500, 5000, 10000, 20000]
CONFIGS = {
    "reproj": dict(lambda_surface=0.0),
    "surf15": dict(lambda_surface=15.0, residual_mode=1),
    "surf50": dict(lambda_surface=50.0, residual_mode=1),
}
COMMON = dict(function_tolerance=1e-6, num_threads=1, fix_first_camera=True,
              huber_threshold=1.0, assoc_max_distance=0.0372, surface_huber=2.749,
              n_outer=2, inner_iters=41, warmup=True)


def run(cache_dir, jobs):
    results = {}  # results[N][cfg] = auc
    for N in NS:
        results[N] = {}
        for cname, cfg in CONFIGS.items():
            p = dict(COMMON); p.update(cfg); p["max_points"] = N
            auc = rs.score_strategy("em_reassoc", cache_dir, p, jobs=jobs)["pose_auc_5"]
            results[N][cname] = auc
            d = auc - results[N].get("reproj", auc)
            print(f"  N={N:6d}  {cname:7s}  auc={auc:6.3f}"
                  f"{'  (surf-reproj='+format(d,'+.3f')+')' if cname!='reproj' else ''}",
                  flush=True)
            json.dump(results, open(os.path.join(OUT, "sparse_sweep.json"), "w"), indent=2)
    return results


def plot():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    results = json.load(open(os.path.join(OUT, "sparse_sweep.json")))
    Ns = sorted(int(k) for k in results)
    cfgs = list(CONFIGS)
    col = {"reproj": "#444", "surf15": "#2e86ab", "surf50": "#d1495b"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for c in cfgs:
        y = [results[str(N)].get(c, np.nan) for N in Ns]
        ax1.plot(Ns, y, "o-", color=col[c], label=c)
    ax1.axhline(BAR, color="green", ls="--", lw=1, label=f"dense regular-BA bar ({BAR})")
    ax1.set_xscale("log"); ax1.set_xlabel("# triangulated points (BA)"); ax1.set_ylabel("pose_auc_5")
    ax1.set_title("pose_auc_5 vs point density"); ax1.legend(fontsize=8)
    for c in cfgs:
        if c == "reproj":
            continue
        d = [results[str(N)].get(c, np.nan) - results[str(N)].get("reproj", np.nan) for N in Ns]
        ax2.plot(Ns, d, "o-", color=col[c], label=f"{c} - reproj")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xscale("log"); ax2.set_xlabel("# triangulated points (BA)")
    ax2.set_ylabel("surface gain over reproj (pose_auc_5)")
    ax2.set_title("does the surface prior gain leverage as points get sparse?")
    ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig6_sparse_sweep.png"), dpi=130)
    print("wrote", os.path.join(OUT, "fig6_sparse_sweep.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="/work/courses/3dv/team39/compose/data/ba_cache")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if not a.plot_only:
        run(a.cache_dir, a.jobs)
    plot()
