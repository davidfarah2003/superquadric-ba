"""Measure (not guess) how the mast3r_sq Ceres solve scales with num_threads,
and where the time goes, so claims about the parallel ceiling are data-backed.

For one cached scene it builds the full surface-BA problem once, then runs a
fixed-iteration solve at num_threads in {1,2,4,8,16,28}, 3 reps each (min taken).
It also runs one verbose solve to dump Ceres' FullReport, which breaks time into
'Jacobian & residual evaluation' (parallel, NOT GPU-offloadable) vs 'Linear
solver' (the only part a CUDA Ceres build would accelerate).

  python thread_scaling.py --cache compose/data/ba_cache/5.npz --iters 40
"""
import sys
import time
import argparse

sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")
sys.path.insert(0, "/work/courses/3dv/team39/ba/python")

import numpy as np  # noqa: E402
import offline_eval as oe  # noqa: E402
import strat_common as sc  # noqa: E402
import ba  # noqa: E402


def build(cache, assoc):
    a = sc.prepare(cache, max_points=None)          # full problem
    sqp = sc.surface_pred(cache)
    sq_params, p2sq, _ = sc.associate(a["points"], sqp, assoc)
    return a, sq_params, p2sq


def one_solve(cache, sq_params, p2sq, nt, iters, verbose=False):
    a = sc.prepare(cache, max_points=None)          # fresh arrays each run
    t = time.perf_counter()
    ba.run_bundle_adjustment_mast3r_sq(
        a["cameras"], a["points"], a["observations"],
        a["cam_indices"], a["pt_indices"],
        fix_first_camera=True, huber_threshold=0.738, verbose=verbose,
        fix_points=False, sq_params=sq_params, point_to_sq=p2sq,
        lambda_surface=3.347, surface_huber=2.749,
        max_num_iterations=iters, function_tolerance=1e-12,  # force full iters
        num_threads=nt)
    return time.perf_counter() - t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--assoc", type=float, default=0.0372)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--threads", default="1,2,4,8,16,28")
    args = ap.parse_args(argv)

    cache = oe.load_cache(args.cache)
    a, sq_params, p2sq = build(cache, args.assoc)
    npts = a["points"].shape[0]
    nobs = a["observations"].shape[0]
    nass = int((p2sq >= 0).sum())
    print(f"scene={args.cache}  points={npts}  obs={nobs}  assigned_to_sq={nass}  "
          f"iters={args.iters}", flush=True)

    thread_list = [int(x) for x in args.threads.split(",")]
    base = None
    print(f"{'threads':>8} {'min_s':>8} {'speedup':>8} {'efficiency':>10}", flush=True)
    for nt in thread_list:
        reps = [one_solve(cache, sq_params, p2sq, nt, args.iters) for _ in range(3)]
        t = min(reps)
        if base is None:
            base = t
        sp = base / t
        print(f"{nt:>8} {t:>8.2f} {sp:>8.2f} {sp/nt:>10.2f}", flush=True)

    print("\n=== Ceres FullReport (verbose) at 28 threads — phase breakdown ===",
          flush=True)
    one_solve(cache, sq_params, p2sq, 28, args.iters, verbose=True)
    print("\n=== Ceres FullReport (verbose) at 1 thread — phase breakdown ===",
          flush=True)
    one_solve(cache, sq_params, p2sq, 1, args.iters, verbose=True)


if __name__ == "__main__":
    main()
