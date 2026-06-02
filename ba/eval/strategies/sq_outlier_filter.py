"""Superquadrics as a STRUCTURE outlier filter — NOT an extra residual.

Idea
----
Every prior SQ variant adds a point->surface *pull* to the cost and loses to
plain reprojection BA (one-shot surface 19.42, best EM surface 28.93 < plain
29.42). The surface term, however tuned, drags the cameras off their good
reprojection optimum.

This strategy uses the superquadric decomposition for the OPPOSITE job: as a
clean geometric prior on where the scene surfaces are, to *clean the structure*
before optimisation. Many triangulated 3D points are noisy / mis-triangulated
outliers; such points sit far from EVERY superquadric surface. We therefore:

    1. associate each triangulated point to its nearest SQ (radial distance),
    2. DROP (hard) or DOWN-WEIGHT (soft) points that lie farther than a
       threshold from all surfaces — i.e. likely triangulation errors,
    3. run PLAIN reprojection BA (lambda_surface = 0, no surface pull) on the
       cleaned observation set.

Hypothesis: cleaner structure -> better-constrained cameras -> beats 29.42,
with far less machinery than the surface residual, and ZERO risk of the surface
term yanking the poses (there is no surface term in the solve).

The SQ geometry only ever touches the problem by *removing* observations; it
never enters the cost function. So this can only help if the dropped points were
genuinely hurting the reprojection optimum.

Filtering rule
--------------
``dists[i]`` = radial distance of point i to its nearest SQ (from
``strat_common.associate``; finite even when "unassigned", it is the distance to
the nearest SQ). We keep point i if EITHER

    * hard mode (default): dists[i] <= reject_distance, OR if ``reject_distance``
      is None, dists[i] <= the ``reject_percentile``-th percentile of dists
      (keep the closest fraction of points).

Points with no SQ within ``assoc_max_distance`` (point_to_sq == -1, i.e. far
from every surface) are the prime outlier suspects and are dropped iff
``keep_unassigned`` is False (the default — dropping them IS the whole point).
``min_keep_fraction`` is a safety floor: if the rule would drop more than
``1 - min_keep_fraction`` of the points, the threshold is relaxed to keep at
least that fraction (closest-first), so a pathological scene can never collapse
the structure.

Soft mode (``mode='soft'``) keeps all points but is a no-op for camera pose in a
pure reprojection BA without per-residual weights, so it just falls back to
plain BA on all points; hard drop is the meaningful lever here.

Params (read from ``params`` with defaults)
-------------------------------------------
    mode               : 'hard' (default) | 'soft'
    reject_distance    : float meters or None. If set, drop points whose nearest
                         -SQ distance exceeds it. Default 0.30.
    reject_percentile  : float in (0,100], used only when reject_distance is
                         None: keep the closest ``reject_percentile``% of points.
                         Default 90.0 (drop the farthest 10%).
    keep_unassigned    : bool, keep points with NO SQ within assoc_max_distance.
                         Default False (drop them — they are the outliers).
    assoc_max_distance : float meters for the association call. Default 0.50
                         (generous, so "unassigned" really means far from ALL
                         surfaces, not just past a tight surface-pull radius).
    min_keep_fraction  : float in (0,1], safety floor on surviving points.
                         Default 0.50.
    huber_threshold    : reprojection Huber delta (pixels). Default 2.0.
    fix_first_camera   : bool. Default True.
    max_iterations     : Ceres max iters. Default 100.
    function_tolerance : Ceres ftol. Default 1e-3.
    num_threads        : Ceres threads. Default 4.
    max_points         : optional point subsample (passed to prepare).
    seed               : subsample seed (deterministic). Default 0.

Suggested sweeps
----------------
    reject_distance    : {None, 0.15, 0.20, 0.30, 0.50, 0.75}  (None => use pct)
    reject_percentile  : {80, 85, 90, 95, 98}                  (when dist None)
    keep_unassigned    : {False, True}
    assoc_max_distance : {0.30, 0.50, 0.75}
    min_keep_fraction  : {0.30, 0.50, 0.70}

A strategy module defines:  refine(cache, params) -> cameras (V,10) float64
"""
import sys

sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")

import numpy as np  # noqa: E402
import strat_common as sc  # noqa: E402


def _plain_ba(a, p):
    """Plain reprojection BA on the (already prepared) arrays, in place."""
    cams, pts = a["cameras"], a["points"]
    sc.solve(cams, pts, a["observations"], a["cam_indices"], a["pt_indices"],
             lambda_surface=0.0, surface_huber=0.0,
             huber_threshold=float(p.get("huber_threshold", 2.0)),
             fix_first_camera=bool(p.get("fix_first_camera", True)),
             sq_params=None, point_to_sq=None,
             max_iterations=int(p.get("max_iterations", 100)),
             function_tolerance=float(p.get("function_tolerance", 1e-3)),
             num_threads=int(p.get("num_threads", 4)))
    return cams


def _keep_mask(dists, point_to_sq, p):
    """Boolean (M,) mask of points to KEEP, per the filtering rule.

    dists[i]        radial distance of point i to its NEAREST SQ (finite even
                    when unassigned).
    point_to_sq[i]  index of assigned SQ, or -1 if none within assoc_max_distance.
    """
    M = dists.shape[0]
    dists = np.asarray(dists, np.float64)
    finite = np.isfinite(dists)

    reject_distance = p.get("reject_distance", 0.30)
    reject_percentile = float(p.get("reject_percentile", 90.0))
    keep_unassigned = bool(p.get("keep_unassigned", False))
    min_keep_fraction = float(p.get("min_keep_fraction", 0.50))

    # --- distance threshold: explicit meters, else a percentile of dists ----
    if reject_distance is not None:
        thr = float(reject_distance)
    else:
        # percentile over FINITE distances only (inf -> unassigned handled below)
        base = dists[finite]
        if base.size == 0:
            thr = np.inf
        else:
            pct = float(np.clip(reject_percentile, 0.0, 100.0))
            thr = float(np.percentile(base, pct))

    keep = finite & (dists <= thr)

    # --- unassigned (far from EVERY surface) -> drop unless explicitly kept --
    unassigned = np.asarray(point_to_sq, np.int64) < 0
    if keep_unassigned:
        keep = keep | unassigned
    # else: leave unassigned points as dropped (default — they are the outliers)

    # --- safety floor: never drop below min_keep_fraction (closest-first) ----
    n_floor = int(np.ceil(np.clip(min_keep_fraction, 0.0, 1.0) * M))
    if keep.sum() < n_floor and M > 0:
        # relax: keep the n_floor closest points by distance (inf sorts last).
        order = np.argsort(np.where(finite, dists, np.inf))
        relaxed = np.zeros(M, dtype=bool)
        relaxed[order[:n_floor]] = True
        keep = keep | relaxed
    return keep


def _compact(a, keep_pts):
    """Drop points where keep_pts is False and remap observations.

    Mirrors strat_common.prepare's remap logic: build old->new index map over
    the surviving points, keep only observations whose point survives, renumber
    their pt_indices into the compacted array. Cameras are left intact.
    """
    pts = a["points"]
    M = pts.shape[0]
    old_to_new = np.full(M, -1, np.int64)
    new_idx = np.nonzero(keep_pts)[0]
    old_to_new[new_idx] = np.arange(new_idx.shape[0])

    obs_mask = old_to_new[a["pt_indices"]] >= 0
    out = dict(a)
    out["points"] = np.ascontiguousarray(pts[keep_pts], np.float64).copy()
    out["observations"] = np.ascontiguousarray(a["observations"][obs_mask], np.float64)
    out["cam_indices"] = np.ascontiguousarray(a["cam_indices"][obs_mask], np.int32)
    out["pt_indices"] = np.ascontiguousarray(
        old_to_new[a["pt_indices"][obs_mask]], np.int32)
    return out


def refine(cache, params):
    p = dict(params or {})
    a = sc.prepare(cache, max_points=p.get("max_points"), seed=int(p.get("seed", 0)))

    mode = str(p.get("mode", "hard")).lower()

    # Surface prior. None => degenerate Sim3: fall back to plain BA, no filter.
    sqp = sc.surface_pred(cache)
    if sqp is None or mode == "soft":
        # soft mode is a no-op for pose in unweighted reprojection BA -> plain BA
        # on all points (kept for interface symmetry / future per-obs weights).
        return _plain_ba(a, p)

    assoc = float(p.get("assoc_max_distance", 0.50))
    # Associate the CURRENT working points -> dists/point_to_sq align with pts.
    _sq_params, point_to_sq, dists = sc.associate(a["points"], sqp, assoc)

    keep = _keep_mask(dists, point_to_sq, p)

    # Degenerate keep (nothing or everything) -> plain BA on current arrays.
    if keep.all() or not keep.any():
        return _plain_ba(a, p)

    a = _compact(a, keep)

    # If filtering somehow stranded a camera (no observations), the solve still
    # runs; pose for unconstrained cams is undefined but pose_auc_5 alignment is
    # global. In practice the safety floor keeps plenty of points per camera.
    return _plain_ba(a, p)
