"""Exp7: does Manhattan-snapping SQ orientations (denoise the diagnosed
geometry ceiling) help? And does it unlock the normal residual (mode5) that only
TIED the hinge with noisy normals? Offline ranking vs the mode1 hinge reference
(live 29.6). num_threads=1 for determinism (CAVEAT 2), full fidelity."""
import sys, json, argparse
sys.path.insert(0, "/work/courses/3dv/team39/ba/eval"); sys.path.insert(0, "/work/courses/3dv/team39/ba/python")
import run_strategy as rs
BAR = 29.42
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--jobs", type=int, default=10); ap.add_argument("--num_threads", type=int, default=1)
    a = ap.parse_args()
    # max_points=30000 (deterministic seed) caps the slowest 117k-pt scenes so 7
    # configs finish on the plugin-imposed 2 CPUs. Same cap on snap & no-snap arms
    # -> the RELATIVE ranking is preserved; absolute won't match the full-fidelity
    # anchor (so compare snap vs the in-experiment REF, not vs 29.24). 30k >> the
    # 5k proxy that misranked; still ranking-only -> confirm the top LIVE.
    common = dict(max_points=30000, function_tolerance=1e-6, num_threads=a.num_threads,
                  fix_first_camera=True, huber_threshold=1.0, assoc_max_distance=0.0372,
                  surface_huber=2.749, n_outer=2, inner_iters=41, warmup=True)
    rows = []
    def run(tag, extra):
        p = dict(common); p.update(extra)
        auc = rs.score_strategy("em_reassoc", a.cache_dir, p, jobs=a.jobs)["pose_auc_5"]
        rows.append((tag, auc))
        print(f"  {auc:.3f}  {tag}{'  >=BAR' if auc >= BAR else ''}", flush=True)
    # --- hinge (mode1): reference vs snapped ---
    run("hinge lam15            (REF no-snap)", dict(residual_mode=1, lambda_surface=15))
    run("hinge lam15 snap15",                   dict(residual_mode=1, lambda_surface=15, manhattan_snap_deg=15.0))
    run("hinge lam15 snap10",                   dict(residual_mode=1, lambda_surface=15, manhattan_snap_deg=10.0))
    run("hinge lam30 snap15",                   dict(residual_mode=1, lambda_surface=30, manhattan_snap_deg=15.0))
    # --- normal residual (mode5): does snap unlock the tied lever? ---
    run("normal lam100          (no-snap)",     dict(residual_mode=5, lambda_surface=100))
    run("normal lam100 snap15",                 dict(residual_mode=5, lambda_surface=100, manhattan_snap_deg=15.0))
    run("normal lam200 snap15",                 dict(residual_mode=5, lambda_surface=200, manhattan_snap_deg=15.0))
    rows.sort(key=lambda t: -t[1])
    print("\n=== RANKED (offline; ranking-only, confirm top LIVE) ===")
    [print(f"  {v:.3f}  {n}") for n, v in rows]
    print(json.dumps({"ranked": [{"name": n, "auc": v} for n, v in rows]}))
main()
