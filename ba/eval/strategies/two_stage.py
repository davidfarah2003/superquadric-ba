"""Two-stage surface BA: clean structure first, then associate + surface solve.

Rationale for why plain surface BA (one-shot, fixed lambda, association on the
INITIAL triangulated points) hurts pose_auc_5: the nearest-superquadric
association is computed against noisy initial points, so many points are
attached to the wrong SQ (or to a plausible SQ at a wrong location). Pulling
those points to their assigned surfaces then drags the cameras off.

This strategy decouples the two jobs:

  Stage 1 -- reprojection-only ``sc.solve`` (lambda_surface=0). This cleans both
             structure (3D points) and poses using only the well-conditioned
             reprojection residuals, exactly like plain BA (which scores 29.5).

  Stage 2 -- ``sc.associate`` the now-CLEANED working points against the
             predicted-frame superquadrics, then run surface BA
             (lambda_surface>0). Because association happens on clean geometry,
             point->SQ assignments are far more trustworthy, so the surface term
             refines rather than corrupts the poses.

A strategy module must define:  refine(cache, params) -> cameras (V,10) float64
"""
import sys

sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")

import numpy as np  # noqa: E402
import strat_common as sc  # noqa: E402


def refine(cache, params):
    p = dict(params or {})

    # Shared tunables (sensible defaults; every knob is sweepable via params).
    max_points = p.get("max_points")
    lam = float(p.get("lambda_surface", 50.0))
    assoc = float(p.get("assoc_max_distance", 0.15))
    surface_huber = float(p.get("surface_huber", 0.0))
    huber_threshold = float(p.get("huber_threshold", 2.0))
    fix_first_camera = bool(p.get("fix_first_camera", True))
    stage1_iters = int(p.get("stage1_iters", 40))
    stage2_iters = int(p.get("stage2_iters", 40))
    function_tolerance = float(p.get("function_tolerance", 1e-3))
    num_threads = int(p.get("num_threads", 4))

    work = sc.prepare(cache, max_points=max_points)
    cams, pts = work["cameras"], work["points"]

    # ---- Stage 1: reprojection-only BA to clean structure + poses ----------
    if stage1_iters > 0:
        sc.solve(cams, pts, work["observations"],
                 work["cam_indices"], work["pt_indices"],
                 lambda_surface=0.0,
                 surface_huber=surface_huber,
                 huber_threshold=huber_threshold,
                 fix_first_camera=fix_first_camera,
                 sq_params=None, point_to_sq=None,
                 max_iterations=stage1_iters,
                 function_tolerance=function_tolerance,
                 num_threads=num_threads)

    # ---- Stage 2: associate CLEANED points, then surface BA ----------------
    # Only run a surface stage if it is actually enabled and the Sim3 / SQ
    # prediction is well-defined; otherwise the cleaned reproj-only result is
    # already a strong (plain-BA-quality) answer.
    sq_params = point_to_sq = None
    run_surface = lam > 0.0 and stage2_iters > 0
    if run_surface:
        sq_pred = sc.surface_pred(cache)
        if sq_pred is None:
            run_surface = False
        else:
            # Associate the CURRENT, moving, subsampled working points so the
            # returned point_to_sq lines up 1:1 with `pts` (what sc.solve wants).
            sq_params, point_to_sq, _ = sc.associate(pts, sq_pred, assoc)

    if run_surface:
        sc.solve(cams, pts, work["observations"],
                 work["cam_indices"], work["pt_indices"],
                 lambda_surface=lam,
                 surface_huber=surface_huber,
                 huber_threshold=huber_threshold,
                 fix_first_camera=fix_first_camera,
                 sq_params=sq_params, point_to_sq=point_to_sq,
                 max_iterations=stage2_iters,
                 function_tolerance=function_tolerance,
                 num_threads=num_threads)

    return np.ascontiguousarray(cams, np.float64)
