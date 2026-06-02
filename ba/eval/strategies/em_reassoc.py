"""EM-style iterated hard re-association surface BA.

Motivation: the one-shot surface BA (strategies/baseline.py) associates each
triangulated point to its nearest superquadric ONCE, at the initial (noisy)
geometry, then pins that association for the whole solve. When the initial
points are off or the nearest-SQ guess is wrong, those frozen residuals pull
the cameras in the wrong direction -> pose_auc_5 drops (19.4 vs plain 29.5).

This strategy treats association as a latent variable and alternates, EM-style:

    E-step:  hard-assign the CURRENT (moving) working points to their nearest SQ
    M-step:  run a short surface BA (Ceres) that moves cameras+points

Because we re-associate the working points after every short solve, points that
drifted toward the wrong SQ get re-pointed (or dropped past assoc_max_distance)
before they can accumulate a large wrong pull. Optionally a warmup pass runs
reprojection-only (lambda_surface=0) first, so the surface term sees cleaner
structure before its first association.

A strategy module must define:  refine(cache, params) -> cameras (V,10) float64
"""
import sys

sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")

import numpy as np  # noqa: E402
import strat_common as sc  # noqa: E402


def refine(cache, params):
    p = dict(params or {})

    # --- tunables (every one is a params key with a default) -----------------
    lam = float(p.get("lambda_surface", 50.0))
    assoc = float(p.get("assoc_max_distance", 0.15))
    surface_huber = float(p.get("surface_huber", 0.0))
    huber_threshold = float(p.get("huber_threshold", 2.0))
    n_outer = int(p.get("n_outer", 3))
    inner_iters = int(p.get("inner_iters", 30))
    warmup = bool(p.get("warmup", True))
    residual_mode = int(p.get("residual_mode", 0))
    refine_sq = bool(p.get("refine_sq", False))
    sq_anchor_weight = float(p.get("sq_anchor_weight", 10.0))
    fix_first_camera = bool(p.get("fix_first_camera", True))
    function_tolerance = float(p.get("function_tolerance", 1e-3))
    num_threads = int(p.get("num_threads", 4))

    # --- subsample once; sq_pred once ---------------------------------------
    a = sc.prepare(cache, max_points=p.get("max_points"), seed=int(p.get("seed", 0)))
    cams, pts = a["cameras"], a["points"]
    obs, ci, pi = a["observations"], a["cam_indices"], a["pt_indices"]

    sqp = sc.surface_pred(cache) if lam > 0.0 else None
    if sqp is None:
        # Degenerate Sim3 (or surface disabled): plain reprojection BA, with the
        # same total iteration budget so it's a fair fallback.
        sc.solve(cams, pts, obs, ci, pi,
                 lambda_surface=0.0, surface_huber=surface_huber,
                 huber_threshold=huber_threshold,
                 fix_first_camera=fix_first_camera,
                 sq_params=None, point_to_sq=None,
                 max_iterations=max(inner_iters * max(n_outer, 1), inner_iters),
                 function_tolerance=function_tolerance, num_threads=num_threads)
        return cams

    # --- optional reproj-only warmup: clean structure before associating -----
    if warmup:
        sc.solve(cams, pts, obs, ci, pi,
                 lambda_surface=0.0, surface_huber=surface_huber,
                 huber_threshold=huber_threshold,
                 fix_first_camera=fix_first_camera,
                 sq_params=None, point_to_sq=None,
                 max_iterations=inner_iters,
                 function_tolerance=function_tolerance, num_threads=num_threads)

    # --- EM outer loop: re-associate MOVING working points, then short solve -
    for _ in range(max(n_outer, 1)):
        # E-step: associate the CURRENT working points directly. Returned
        # point_to_sq length == len(pts), exactly what sc.solve expects.
        sq_params, point_to_sq, _ = sc.associate(pts, sqp, assoc)
        # M-step: short surface BA that moves cameras + points in place.
        sc.solve(cams, pts, obs, ci, pi,
                 lambda_surface=lam, surface_huber=surface_huber,
                 huber_threshold=huber_threshold,
                 fix_first_camera=fix_first_camera,
                 sq_params=sq_params, point_to_sq=point_to_sq,
                 residual_mode=residual_mode,
                 refine_sq=refine_sq, sq_anchor_weight=sq_anchor_weight,
                 max_iterations=inner_iters,
                 function_tolerance=function_tolerance, num_threads=num_threads)

    return cams
