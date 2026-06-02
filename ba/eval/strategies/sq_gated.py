"""Strategy: sq_gated — confidence-GATED surface BA.

Motivation
----------
The one-shot point->SQ surface BA scores pose_auc_5 = 19.42, WORSE than plain
reprojection BA (29.42 = BAR). Even the best EM re-association variant only
reaches 28.93 (loses by 0.49). The diagnosis behind every failed variant is the
same: a *minority of WRONG / low-quality* point->SQ associations pull the cameras
off their good reprojection optimum. The surface term is applied indiscriminately
to every associated point, so the few bad pulls outweigh the many good ones.

Hypothesis (this strategy): the surface term is net-positive *if you only trust
the associations you have evidence for*. Concretely, keep a surface residual ONLY
for high-confidence point->SQ associations and DROP the rest (set
``point_to_sq[i] = -1`` so they contribute pure reprojection only). If the kept
subset is dominated by correct associations on well-triangulated points sitting
on big reliable primitives (floors/walls/tables), the surface term should pin the
gauge a little tighter than reprojection alone and cross 29.42.

Gating rules (all combine; each is independently switchable via params)
-----------------------------------------------------------------------
1. **Distance-percentile gate (primary).** ``associate`` returns the radial
   distance ``dists[i]`` from each point to its nearest SQ. Among the points that
   are associated at all (``point_to_sq >= 0``), keep a surface residual only for
   those whose distance is in the lowest ``gate_percentile`` fraction (e.g. the
   closest 30%). Everything above that per-iteration threshold -> -1. Close
   points are the ones most likely sitting *on* a real surface; far ones are the
   noisy/ambiguous tail that does the damage.

2. **Per-SQ support gate (optional).** Drop any SQ that fewer than
   ``min_sq_support`` of the kept points associate to — such SQs are likely
   spurious decomposition primitives (slivers/clutter). Points pointing at a
   dropped SQ -> -1. Recomputed every outer iteration on the kept set.

3. **Warmup-consistency gate (optional).** With ``warmup=True`` we run a short
   reprojection-only solve first, associating BEFORE and AFTER it. Keep a point
   only if its nearest-SQ distance did not grow during the warmup
   (``dist_after <= dist_before * (1 + consistency_slack)``) — i.e. cleaning the
   structure with image evidence alone pulled it *toward* (not away from) its SQ,
   evidence the association is real rather than an artifact of the noisy start.
   This mask is intersected with the distance-percentile gate.

EM wrapper
----------
Gating is re-evaluated inside a small EM loop (``n_outer`` iterations): as points
move under the surface+reprojection solve, we re-associate the *current* points,
re-rank distances, and re-gate. A point that drifts out of the trusted set is
dropped before it can accumulate a large wrong pull; a point that settles onto a
surface can re-enter. Lambda is kept modest so reprojection still dominates.

Degeneracy: if ``surface_pred`` returns None (degenerate Sim3) or no point
survives gating, we fall back to plain reprojection BA over the full iteration
budget — never worse than the 29.42 bar.

Params (read from ``params`` with defaults)
-------------------------------------------
    gate_percentile     0.30   keep closest-this-fraction of associated points
    min_sq_support      0      drop SQs with < this many kept points (0=off)
    warmup              True   run reproj-only warmup + consistency gate
    consistency_slack   0.0    allowed relative distance growth in warmup
    warmup_iters        25     iterations for the reprojection-only warmup
    lambda_surface      8.0    surface weight (MODEST — reproj must dominate)
    assoc_max_distance  0.05   nearest-SQ cutoff (m) for association
    surface_huber       1.0    surface Huber delta (0 disables robustifier)
    huber_threshold     1.0    reprojection Huber delta (px)
    n_outer             2      EM outer iterations (re-gate as points move)
    inner_iters         30     Ceres iters per surface (M-step) solve
    max_iterations      50     iters for the plain-BA fallback path
    function_tolerance  1e-3   Ceres function tolerance
    fix_first_camera    True
    num_threads         4
    max_points          None   deterministic point subsample (seed below)
    seed                0      deterministic subsample seed

Sweep ranges (for the orchestrator's BO)
-----------------------------------------
    gate_percentile    [0.10, 0.60]   (sharper gate vs more support)
    lambda_surface     [1.0, 30.0]    (modest; reproj must keep the gauge)
    assoc_max_distance [0.02, 0.15]
    surface_huber      [0.0, 3.0]
    huber_threshold    [0.5, 2.0]
    min_sq_support     {0, 3, 5, 10}
    consistency_slack  [0.0, 0.25]
    n_outer            {1, 2, 3}
    inner_iters        [20, 45]
    warmup             {True, False}

Defines:  refine(cache, params) -> cameras (V,10) float64
"""
import sys

sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")

import numpy as np  # noqa: E402
import strat_common as sc  # noqa: E402


def _percentile_keep_mask(point_to_sq, dists, gate_percentile):
    """Boolean mask over points to KEEP a surface residual for.

    Keep only ASSOCIATED points (point_to_sq >= 0) whose nearest-SQ distance is
    in the lowest ``gate_percentile`` fraction *of the associated set*. Returns
    an all-False mask if nothing is associated. gate_percentile >= 1.0 keeps
    every associated point (gate disabled).
    """
    assoc_mask = point_to_sq >= 0
    n_assoc = int(assoc_mask.sum())
    if n_assoc == 0:
        return np.zeros_like(assoc_mask)
    if gate_percentile >= 1.0:
        return assoc_mask

    d_assoc = dists[assoc_mask]
    # Distance threshold at the requested percentile of the associated set.
    thr = np.quantile(d_assoc, float(gate_percentile))
    keep = assoc_mask & (dists <= thr)
    if not np.any(keep):
        # Percentile too tight (e.g. ties / tiny set): keep at least the single
        # closest associated point so the surface term is not silently disabled.
        idx = np.where(assoc_mask)[0]
        keep = np.zeros_like(assoc_mask)
        keep[idx[np.argmin(dists[idx])]] = True
    return keep


def _apply_support_gate(point_to_sq, keep_mask, min_sq_support):
    """Drop SQs supported by fewer than ``min_sq_support`` KEPT points.

    Mutates nothing; returns an updated keep_mask with under-supported SQs'
    points removed. ``min_sq_support <= 1`` is a no-op.
    """
    if min_sq_support is None or int(min_sq_support) <= 1:
        return keep_mask
    kept_sqs = point_to_sq[keep_mask]
    if kept_sqs.size == 0:
        return keep_mask
    counts = np.bincount(kept_sqs.astype(np.int64))
    weak = np.where(counts < int(min_sq_support))[0]
    if weak.size == 0:
        return keep_mask
    weak_set = np.isin(point_to_sq, weak)
    return keep_mask & ~weak_set


def _gated_p2sq(point_to_sq, keep_mask):
    """Return a fresh int32 point_to_sq with non-kept points set to -1."""
    g = np.full(point_to_sq.shape[0], -1, np.int32)
    g[keep_mask] = point_to_sq[keep_mask]
    return np.ascontiguousarray(g, np.int32)


def refine(cache, params):
    p = dict(params or {})

    # --- tunables (every one a params key with a default) --------------------
    gate_percentile = float(p.get("gate_percentile", 0.30))
    min_sq_support = int(p.get("min_sq_support", 0))
    warmup = bool(p.get("warmup", True))
    consistency_slack = float(p.get("consistency_slack", 0.0))
    warmup_iters = int(p.get("warmup_iters", 25))
    lam = float(p.get("lambda_surface", 8.0))
    assoc = float(p.get("assoc_max_distance", 0.05))
    surface_huber = float(p.get("surface_huber", 1.0))
    huber_threshold = float(p.get("huber_threshold", 1.0))
    n_outer = int(p.get("n_outer", 2))
    inner_iters = int(p.get("inner_iters", 30))
    max_iterations = int(p.get("max_iterations", 50))
    function_tolerance = float(p.get("function_tolerance", 1e-3))
    fix_first_camera = bool(p.get("fix_first_camera", True))
    num_threads = int(p.get("num_threads", 4))
    residual_mode = int(p.get("residual_mode", 0))

    # --- fresh mutable working copies (deterministic subsample) --------------
    a = sc.prepare(cache, max_points=p.get("max_points"), seed=int(p.get("seed", 0)))
    cams, pts = a["cameras"], a["points"]
    obs, ci, pi = a["observations"], a["cam_indices"], a["pt_indices"]

    def _solve(lambda_surface, sq_params, point_to_sq, max_iters):
        if max_iters <= 0:
            return
        sc.solve(cams, pts, obs, ci, pi,
                 lambda_surface=lambda_surface, surface_huber=surface_huber,
                 huber_threshold=huber_threshold,
                 fix_first_camera=fix_first_camera,
                 sq_params=sq_params, point_to_sq=point_to_sq,
                 residual_mode=residual_mode,
                 max_iterations=int(max_iters),
                 function_tolerance=function_tolerance, num_threads=num_threads)

    # Superquadrics in the predicted frame (None => degenerate Sim3 -> skip).
    sq_pred = sc.surface_pred(cache) if lam > 0.0 else None
    if sq_pred is None:
        # Plain reprojection BA fallback over the full budget (>= 29.42 bar).
        budget = max(max_iterations, warmup_iters + n_outer * inner_iters)
        _solve(0.0, None, None, budget)
        return cams

    # --- optional warmup + consistency gate ----------------------------------
    # dist_before: nearest-SQ distance at the noisy start; dist_after: after a
    # reprojection-only solve cleaned the structure. A point is "consistent" if
    # cleaning did NOT push it away from its SQ.
    consistency_mask = None
    if warmup and warmup_iters > 0:
        _, _, dist_before = sc.associate(pts, sq_pred, assoc)
        _solve(0.0, None, None, warmup_iters)
        _, _, dist_after = sc.associate(pts, sq_pred, assoc)
        consistency_mask = dist_after <= dist_before * (1.0 + consistency_slack)

    # --- EM loop with re-gating ----------------------------------------------
    any_surface_solve = False
    for _ in range(max(n_outer, 1)):
        # E-step: associate the CURRENT (moving) working points.
        sq_params, point_to_sq, dists = sc.associate(pts, sq_pred, assoc)

        # Gate 1: distance percentile over the associated set.
        keep = _percentile_keep_mask(point_to_sq, dists, gate_percentile)
        # Gate 3: warmup consistency (intersect).
        if consistency_mask is not None:
            keep = keep & consistency_mask
        # Gate 2: per-SQ support (drop spurious primitives).
        keep = _apply_support_gate(point_to_sq, keep, min_sq_support)

        if np.any(keep):
            gated = _gated_p2sq(point_to_sq, keep)
            _solve(lam, sq_params, gated, inner_iters)
            any_surface_solve = True
        else:
            # Nothing trusted this round -> spend the budget on reprojection.
            _solve(0.0, None, None, inner_iters)

    if not any_surface_solve and warmup:
        # Surface never fired and we already spent warmup_iters on reproj; the
        # EM loop also ran reproj, so cams are a valid plain-BA result. Nothing
        # more to do (kept explicit for clarity).
        pass

    return cams
