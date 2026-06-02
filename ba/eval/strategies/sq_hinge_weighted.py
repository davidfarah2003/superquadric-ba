"""EM hinge surface BA with SOFT per-point confidence weights (point_weights).

Complements the hinge grid (mode1/4 x lambda x huber): instead of a hard
assoc_max_distance cutoff (a point is fully in or fully out of the surface term),
weight each point's one-sided hinge residual smoothly by its association
distance, w_i = exp(-(d_i / sigma)^2). Near-surface points keep full weight;
points drifting outward fade out continuously rather than snapping off at the
threshold. The research pass flagged confidence weighting as a top lever, and the
C++ backend now multiplies each surface residual by sqrt(point_weights[i]).

Built on the live-winning recipe (EM re-association + hinge mode1). With
residual_mode=0 and sigma=inf this reduces to plain radial EM, so it is a strict
superset of em_reassoc.

params (defaults in []):
  residual_mode [1]      surface form (1=hinge-outside, 4=hinge-outside-normalized)
  lambda_surface [15.0]  surface weight (hinge tolerates higher than radial)
  sigma [0.05]           weight falloff (m); larger = softer cutoff
  assoc_max_distance [0.15]  hard cap on association (generous; sigma does the gating)
  surface_huber [2.749]  surface Huber delta
  huber_threshold [1.0]  reprojection Huber delta (px); 1.0 matches mast3r backend
  n_outer [2]  inner_iters [41]  warmup [True]
  fix_first_camera [True]  function_tolerance [1e-6]  num_threads [4]  max_points
"""
import sys

sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")

import numpy as np  # noqa: E402
import strat_common as sc  # noqa: E402


def refine(cache, params):
    p = dict(params or {})
    mode = int(p.get("residual_mode", 1))
    lam = float(p.get("lambda_surface", 15.0))
    sigma = float(p.get("sigma", 0.05))
    assoc = float(p.get("assoc_max_distance", 0.15))
    surface_huber = float(p.get("surface_huber", 2.749))
    huber_threshold = float(p.get("huber_threshold", 1.0))
    n_outer = int(p.get("n_outer", 2))
    inner_iters = int(p.get("inner_iters", 41))
    warmup = bool(p.get("warmup", True))
    fix_first_camera = bool(p.get("fix_first_camera", True))
    function_tolerance = float(p.get("function_tolerance", 1e-6))
    num_threads = int(p.get("num_threads", 4))

    a = sc.prepare(cache, max_points=p.get("max_points"), seed=int(p.get("seed", 0)))
    cams, pts = a["cameras"], a["points"]
    obs, ci, pi = a["observations"], a["cam_indices"], a["pt_indices"]

    sqp = sc.surface_pred(cache) if lam > 0.0 else None
    if sqp is None:
        sc.solve(cams, pts, obs, ci, pi, lambda_surface=0.0,
                 surface_huber=surface_huber, huber_threshold=huber_threshold,
                 fix_first_camera=fix_first_camera, sq_params=None,
                 point_to_sq=None, max_iterations=max(inner_iters * max(n_outer, 1),
                                                       inner_iters),
                 function_tolerance=function_tolerance, num_threads=num_threads)
        return cams

    if warmup:
        sc.solve(cams, pts, obs, ci, pi, lambda_surface=0.0,
                 surface_huber=surface_huber, huber_threshold=huber_threshold,
                 fix_first_camera=fix_first_camera, sq_params=None,
                 point_to_sq=None, max_iterations=inner_iters,
                 function_tolerance=function_tolerance, num_threads=num_threads)

    for _ in range(max(n_outer, 1)):
        sq_params, point_to_sq, dists = sc.associate(pts, sqp, assoc)
        # Soft confidence weight from association distance; 0 weight where there
        # is no association (skipped in the backend anyway). float64, contiguous.
        w = np.exp(-(np.asarray(dists, np.float64) / max(sigma, 1e-6)) ** 2)
        w = np.ascontiguousarray(np.where(point_to_sq >= 0, w, 0.0), np.float64)
        sc.solve(cams, pts, obs, ci, pi, lambda_surface=lam,
                 surface_huber=surface_huber, huber_threshold=huber_threshold,
                 fix_first_camera=fix_first_camera, sq_params=sq_params,
                 point_to_sq=point_to_sq, residual_mode=mode, point_weights=w,
                 max_iterations=inner_iters,
                 function_tolerance=function_tolerance, num_threads=num_threads)

    return cams
