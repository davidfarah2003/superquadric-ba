"""Exp3: full-fidelity ranking of the new hinge-capable strategies.

Compares (offline ranking; live validates the top — see EXPERIMENTS.md caveats):
  em_reassoc mode1 lam15 h1.0   -- the live-winning reference (offline ~29.82)
  sq_softweight                 -- soft per-point confidence weights + hinge
  sq_gated (residual_mode=1)    -- percentile/consistency-gated + hinge
  sq_em_soft (residual_mode=1)  -- annealed lambda + MAD pruning + hinge
  sq_outlier_filter             -- SQ-as-outlier-filter -> plain BA (no surface)

All at full fidelity. num_threads default 1 for a DETERMINISTIC ranking (the
offset diagnosis showed num_threads>1 adds ~0.5 noise); raise it for speed if the
ranking gaps are large.
"""
import sys
import json
import argparse

sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")
sys.path.insert(0, "/work/courses/3dv/team39/ba/python")

import run_strategy as rs  # noqa: E402

BAR = 29.42


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--num_threads", type=int, default=1)
    args = ap.parse_args(argv)

    base = dict(max_points=None, function_tolerance=1e-6,
                num_threads=args.num_threads, fix_first_camera=True)
    rows = []

    def run(name, extra):
        p = dict(base); p.update(extra)
        auc = rs.score_strategy(name, args.cache_dir, p, jobs=args.jobs)["pose_auc_5"]
        rows.append((f"{name}:{json.dumps(extra, separators=(',',':'))}", auc))
        print(f"  {name:18s} {json.dumps(extra, separators=(',',':'))[:60]:60s} "
              f"{auc:.3f}{'  >=BAR' if auc >= BAR else ''}", flush=True)
        return auc

    print("=== reference: em_reassoc hinge (live winner) ===", flush=True)
    run("em_reassoc", dict(residual_mode=1, lambda_surface=15.0,
        huber_threshold=1.0, assoc_max_distance=0.0372, surface_huber=2.749,
        n_outer=2, inner_iters=41, warmup=True))

    print("=== sq_softweight (hinge + soft weights) sigma sweep ===", flush=True)
    for sigma in [0.04, 0.06, 0.09]:
        run("sq_softweight", dict(residual_mode=1, lambda_surface=15.0,
            huber_threshold=1.0, sigma=sigma, assoc_max_distance=0.12,
            surface_huber=2.749, n_outer=2, inner_iters=41, warmup=True))

    print("=== sq_gated (hinge) ===", flush=True)
    for gp in [0.3, 0.5]:
        run("sq_gated", dict(residual_mode=1, lambda_surface=15.0,
            huber_threshold=1.0, gate_percentile=gp, assoc_max_distance=0.08,
            surface_huber=2.749, n_outer=2, inner_iters=41, warmup=True))

    print("=== sq_em_soft (hinge, annealed) ===", flush=True)
    run("sq_em_soft", dict(residual_mode=1, lambda_start=5.0, lambda_end=30.0,
        huber_threshold=1.0, assoc_start=0.0372, assoc_end=0.10,
        surface_huber=2.749, n_outer=3, inner_iters=41, warmup=True, outlier_k=3.0))

    print("=== sq_outlier_filter (no surface; SQ as filter) ===", flush=True)
    for rp in [85, 92]:
        run("sq_outlier_filter", dict(huber_threshold=1.0, reject_percentile=rp,
            assoc_max_distance=0.5, min_keep_fraction=0.5, max_iterations=200))

    rows.sort(key=lambda t: -t[1])
    print(f"\n=== RANKED (offline; bar≈{BAR}, live offset ~+0.9) ===")
    for n, a in rows:
        print(f"  {a:8.3f}  {n}")
    print(json.dumps({"ranked": [{"name": n, "auc": a} for n, a in rows]}, indent=2))


if __name__ == "__main__":
    main()
