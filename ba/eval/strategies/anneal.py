"""Strategy: lambda annealing with re-association.

Motivation
----------
The single-shot surface solve (baseline.py, pose_auc_5 ~19.4) applies the full
surface lambda from iteration 0. When the initial triangulated points are noisy
or the nearest-superquadric associations are wrong, that strong surface pull
drags the cameras off before the reprojection term can settle the geometry.

This strategy ramps the surface influence in gradually. We run an outer loop of
``n_steps`` solves. At step k (0-indexed) the surface weight is

    lambda_k = lambda_max * (k + 1) / n_steps          # 1/n .. n/n  ramp

so the first solve sees only a fraction of the surface term (mostly reprojection
BA, which stabilises the structure) and the final solve sees the full lambda_max.
Crucially, before EACH solve we RE-ASSOCIATE the *current* (moving, subsampled)
working points to the superquadrics: as the points migrate during the ramp, their
nearest-SQ assignments are refreshed, so the surface term acts on up-to-date,
better-conditioned correspondences instead of the stale day-0 ones.

Surface predictions are derived once (sc.surface_pred), since the SQs live in the
fixed predicted-world frame; only the point->SQ assignment changes per step.

A strategy module must define:  refine(cache, params) -> cameras (V,10) float64
"""
import sys

sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")

import numpy as np  # noqa: E402
import strat_common as sc  # noqa: E402


def refine(cache, params):
    p = dict(params or {})

    # ---- tunables (every one exposed as a params key with a default) --------
    lambda_max = float(p.get("lambda_max", 50.0))
    assoc = float(p.get("assoc_max_distance", 0.15))
    surface_huber = float(p.get("surface_huber", 0.0))
    huber_threshold = float(p.get("huber_threshold", 2.0))
    n_steps = int(p.get("n_steps", 4))
    inner_iters = int(p.get("inner_iters", 25))
    max_points = p.get("max_points")
    function_tolerance = float(p.get("function_tolerance", 1e-3))
    num_threads = int(p.get("num_threads", 4))
    fix_first_camera = bool(p.get("fix_first_camera", True))
    seed = int(p.get("seed", 0))

    # ---- fresh mutable working set (deterministic subsample) ----------------
    work = sc.prepare(cache, max_points=max_points, seed=seed)
    cams, pts = work["cameras"], work["points"]

    # ---- superquadrics in the predicted-world frame (fixed across steps) ----
    sqp = None
    if lambda_max > 0.0 and n_steps > 0:
        sqp = sc.surface_pred(cache)

    # Degenerate Sim3 (or surface disabled): fall back to a single plain solve.
    if sqp is None:
        sc.solve(cams, pts, work["observations"],
                 work["cam_indices"], work["pt_indices"],
                 lambda_surface=0.0,
                 surface_huber=surface_huber, huber_threshold=huber_threshold,
                 fix_first_camera=fix_first_camera,
                 sq_params=None, point_to_sq=None,
                 max_iterations=inner_iters,
                 function_tolerance=function_tolerance,
                 num_threads=num_threads)
        return cams

    # ---- annealed outer loop ------------------------------------------------
    for k in range(n_steps):
        lambda_k = lambda_max * float(k + 1) / float(n_steps)

        # Re-associate the CURRENT working points (length matches pts, which is
        # exactly what sc.solve expects). As points move, assignments refresh.
        sq_params, point_to_sq, _ = sc.associate(work["points"], sqp, assoc)

        sc.solve(cams, pts, work["observations"],
                 work["cam_indices"], work["pt_indices"],
                 lambda_surface=lambda_k,
                 surface_huber=surface_huber, huber_threshold=huber_threshold,
                 fix_first_camera=fix_first_camera,
                 sq_params=sq_params, point_to_sq=point_to_sq,
                 max_iterations=inner_iters,
                 function_tolerance=function_tolerance,
                 num_threads=num_threads)

    return cams
