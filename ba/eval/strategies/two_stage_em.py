"""Strategy: two_stage_em — clean structure first, then EM surface refinement.

Motivation: the one-shot surface BA (baseline.py) scores pose_auc_5=19.4, WORSE
than plain reprojection BA (29.5). The culprit is that it associates points to
superquadrics ONCE, at the noisy initial triangulation, then drags cameras
toward those (often wrong) associations at a fixed lambda. Bad associations on
bad points pull poses off.

This strategy combines the two strongest fixes:

  Stage 1 — clean the structure. Run a pure reprojection BA
            (lambda_surface=0) for ``stage1_iters`` iterations. This refines
            cameras AND points using image evidence only, so the points are
            close to their true 3D location before any surface term sees them.

  Stage 2 — EM over the surface term. Repeat ``n_outer`` times:
              E-step: re-associate the CURRENT (moving, subsampled) working
                      points to the superquadrics.
              M-step: surface BA (lambda_surface>0) for ``inner_iters``.
            Because association is recomputed on the now-clean, post-solve
            points each outer iteration, wrong assignments self-correct instead
            of being frozen in at the noisy start.

If the Sim3 used to bring superquadrics into the predicted frame is degenerate
(surface_pred returns None), the surface term is skipped and we simply run
reprojection BA for stage1_iters + n_outer*inner_iters iterations — i.e. plain
BA, which is already the thing we're trying to beat, never worse.

Defines:  refine(cache, params) -> cameras (V,10) float64
"""
import sys

sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")

import numpy as np  # noqa: E402
import strat_common as sc  # noqa: E402


def refine(cache, params):
    p = dict(params or {})

    # --- tunables (every one exposed, with a sensible default) ---------------
    lam = float(p.get("lambda_surface", 50.0))
    assoc = float(p.get("assoc_max_distance", 0.15))
    surface_huber = float(p.get("surface_huber", 0.0))
    huber_threshold = float(p.get("huber_threshold", 2.0))
    stage1_iters = int(p.get("stage1_iters", 40))
    n_outer = int(p.get("n_outer", 2))
    inner_iters = int(p.get("inner_iters", 25))
    max_points = p.get("max_points")
    fix_first_camera = bool(p.get("fix_first_camera", True))
    function_tolerance = float(p.get("function_tolerance", 1e-3))
    num_threads = int(p.get("num_threads", 4))

    # --- fresh mutable working copies (deterministic subsample) --------------
    work = sc.prepare(cache, max_points=max_points)
    cams = work["cameras"]

    def _solve(lambda_surface, sq_params, point_to_sq, max_iterations):
        if max_iterations <= 0:
            return
        sc.solve(
            cams, work["points"], work["observations"],
            work["cam_indices"], work["pt_indices"],
            lambda_surface=lambda_surface,
            surface_huber=surface_huber,
            huber_threshold=huber_threshold,
            fix_first_camera=fix_first_camera,
            sq_params=sq_params, point_to_sq=point_to_sq,
            max_iterations=int(max_iterations),
            function_tolerance=function_tolerance,
            num_threads=num_threads,
        )

    # Superquadrics in the predicted frame (None => degenerate, skip surface).
    sq_pred = sc.surface_pred(cache) if lam > 0.0 else None

    # --- Stage 1: pure reprojection BA to clean structure --------------------
    _solve(0.0, None, None, stage1_iters)

    # --- Stage 2: EM loop over the surface term ------------------------------
    if lam > 0.0 and sq_pred is not None and n_outer > 0 and inner_iters > 0:
        for _ in range(n_outer):
            # E-step: re-associate the CURRENT working points directly (its
            # length matches work['points'], exactly what sc.solve needs).
            sq_params, point_to_sq, _ = sc.associate(work["points"], sq_pred, assoc)
            # M-step: surface BA on the freshly-associated, cleaned points.
            _solve(lam, sq_params, point_to_sq, inner_iters)
    else:
        # No usable surface term: spend the remaining budget on reprojection BA
        # so behaviour degrades to plain BA (the >=29.5 target) rather than worse.
        _solve(0.0, None, None, n_outer * inner_iters)

    return cams
