"""Strategy: sq_em_soft — annealed, outlier-aware EM surface BA.

Idea
----
``em_reassoc`` (the current best SQ variant, live pose_auc_5 = 28.93) does HARD
EM: a reprojection-only warmup, then for ``n_outer`` iterations it hard-assigns
every moving point to its nearest superquadric and runs a short surface solve at
a CONSTANT surface weight. Two things still hurt it relative to plain BA (29.42):

  1. The full surface weight is applied from the very first outer iteration, when
     the structure is still settling. Early, noisy geometry gets dragged by the
     prior before reprojection has pinned it down.
  2. A handful of *wrong* associations (a point whose true surface SUPERDEC never
     modelled, or that sits between two primitives) are kept as long as they fall
     under a single fixed ``assoc_max_distance``. Those few large, wrong pulls are
     exactly what nudges the pose AUC below the reprojection optimum.

``sq_em_soft`` keeps the EM machinery of ``em_reassoc`` but adds three
deterministic refinements, all driven by the OUTER iteration index:

  A. ANNEAL the surface weight. The per-outer surface lambda ramps geometrically
     from ``lambda_start`` to ``lambda_end`` over the outer loop, so early
     iterations trust the prior little and let reprojection clean the structure,
     while later iterations — operating on stabilised points — trust it fully.
     (``solve`` takes a scalar lambda, so "annealing" = a different lambda per
     outer iteration.)

  B. ANNEAL the association gate. The nearest-SQ acceptance radius ramps linearly
     from ``assoc_start`` to ``assoc_end`` across outer iterations. Starting tight
     admits only confident, near-surface points (the most trustworthy pulls);
     loosening later re-admits points once the geometry has moved onto the
     surfaces. Set ``assoc_start == assoc_end`` to disable (constant gate).

  C. OUTLIER-AWARE re-association (robust MAD pruning). After the nearest-SQ
     assignment each outer iteration, the residual association distances of the
     *currently assigned* points have a median ``m`` and median-absolute-deviation
     ``MAD``. Any association whose distance exceeds ``m + outlier_k * 1.4826*MAD``
     is dropped (``point_to_sq = -1``) for THAT iteration — there is no per-point
     weight, so de-associating is how we down-weight a suspect pull. This prunes
     the few wrong, far pulls before the solve instead of letting them bend the
     cameras. ``outlier_k = 0`` disables pruning (pure annealed EM).

  D. Optional early stop. After each outer solve we measure how far the camera
     CENTRES moved (in the cameras' own pre-alignment frame). When the max centre
     shift drops below ``conv_eps`` the structure has stabilised and further
     surface iterations would only risk overfitting the prior, so we stop.
     ``conv_eps = 0`` disables the check (always runs all ``n_outer`` iters).

Schedule summary (k = 0 .. n_outer-1, n = n_outer):
    lambda_k = lambda_start * (lambda_end / lambda_start) ** (k / (n-1))   (geom)
    assoc_k  = assoc_start  + (assoc_end - assoc_start) * (k / (n-1))      (lin)
    keep association i  iff  dist_i <= assoc_k  AND
                            dist_i <= median(dist_assigned) + outlier_k*1.4826*MAD
(for n_outer == 1 both schedules collapse to their *end* values.)

Fallbacks
---------
If the predicted->GT Sim3 is degenerate (``surface_pred`` returns None) or the
surface term is disabled (``lambda_end <= 0`` and ``lambda_start <= 0``), we run
a single plain reprojection BA with the SAME total iteration budget
(``inner_iters * n_outer`` + warmup), so this strategy never does worse than the
plain-BA bar it is trying to beat.

Params (read from ``params`` with defaults)
-------------------------------------------
    lambda_start      (3.0)    surface weight at the first outer iter (low)
    lambda_end        (8.0)    surface weight at the last outer iter (high)
    assoc_start       (0.037)  association radius (m) at the first outer iter (tight)
    assoc_end         (0.060)  association radius (m) at the last outer iter (loose)
    n_outer           (3)      number of EM outer iterations
    inner_iters       (30)     Ceres iterations per inner (M-step) solve
    warmup            (True)   run a reprojection-only solve before the EM loop
    outlier_k         (3.0)    MAD multiplier for pruning (0 = off)
    conv_eps          (0.0)    early-stop on max camera-centre shift (0 = off)
    surface_huber     (2.0)    Huber delta on the surface residual (0 = off)
    huber_threshold   (1.0)    Huber delta on the reprojection residual (px)
    max_iterations    (None)   alias for inner_iters if inner_iters absent
    function_tolerance(1e-3)   Ceres function tolerance per solve
    num_threads       (4)      Ceres threads per solve
    fix_first_camera  (True)   gauge-fix the first camera
    max_points        (None)   deterministic point subsample (None = all)
    seed              (0)      subsample seed (deterministic)

Defaults sit in the known-good ``em_reassoc`` regime (small lambda, tight assoc,
warmup on) so the annealing/pruning are refinements ON TOP of a configuration
already at parity with plain BA, rather than a fresh untuned point.

A strategy module must define:  refine(cache, params) -> cameras (V,10) float64
"""
import sys

sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")

import numpy as np  # noqa: E402
import strat_common as sc  # noqa: E402


def _cam_centres(cameras):
    """Camera centres C = -R_wc^T @ t from W2C (angle_axis, trans) rows.

    Mirrors offline_eval.cameras_to_pred_poses' W2C->C2W step; used only to
    measure inter-iteration camera motion for the convergence check, so it does
    not need the Sim3 alignment (that is monotone in centre displacement).
    """
    from scipy.spatial.transform import Rotation

    cams = np.asarray(cameras, np.float64)
    V = cams.shape[0]
    C = np.empty((V, 3), np.float64)
    for v in range(V):
        R_wc = Rotation.from_rotvec(cams[v, 0:3]).as_matrix()
        C[v] = -R_wc.T @ cams[v, 3:6]
    return C


def _geom_ramp(a, b, k, n):
    """Geometric interpolation a -> b at step k of n (k in [0, n-1]).

    Falls back to linear if either endpoint is non-positive (geom undefined).
    For n <= 1 returns the END value b (the fully-annealed weight).
    """
    if n <= 1:
        return float(b)
    f = float(k) / float(n - 1)
    a = float(a)
    b = float(b)
    if a > 0.0 and b > 0.0:
        return float(a * (b / a) ** f)
    return float(a + (b - a) * f)


def _lin_ramp(a, b, k, n):
    """Linear interpolation a -> b at step k of n. END value b when n <= 1."""
    if n <= 1:
        return float(b)
    f = float(k) / float(n - 1)
    return float(a) + (float(b) - float(a)) * f


def _prune_outliers(point_to_sq, dists, outlier_k):
    """Drop associations whose distance is a robust outlier (in place copy).

    Keep i iff dist_i <= median(d_assigned) + outlier_k * 1.4826 * MAD, where the
    statistics are taken over the CURRENTLY ASSIGNED points only (point_to_sq>=0).
    Returns a new int32 array; the input is not mutated. ``outlier_k <= 0`` is a
    no-op (returns the array unchanged). If fewer than a handful of points are
    assigned the distribution is too small to estimate robustly, so we skip.
    """
    if outlier_k <= 0.0:
        return point_to_sq
    p2sq = np.array(point_to_sq, np.int32, copy=True)
    assigned = p2sq >= 0
    n_assigned = int(assigned.sum())
    if n_assigned < 8:
        return p2sq
    d = np.asarray(dists, np.float64)[assigned]
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    # 1.4826 makes MAD a consistent estimator of sigma for Gaussian data.
    thresh = med + float(outlier_k) * 1.4826 * mad
    # If MAD == 0 (all distances identical) nothing is an outlier.
    if mad <= 0.0:
        return p2sq
    drop = assigned & (np.asarray(dists, np.float64) > thresh)
    p2sq[drop] = -1
    return p2sq


def refine(cache, params):
    p = dict(params or {})

    # ---- tunables -----------------------------------------------------------
    lambda_start = float(p.get("lambda_start", 3.0))
    lambda_end = float(p.get("lambda_end", 8.0))
    assoc_start = float(p.get("assoc_start", 0.037))
    assoc_end = float(p.get("assoc_end", 0.060))
    n_outer = int(p.get("n_outer", 3))
    inner_iters = int(p.get("inner_iters", p.get("max_iterations", 30)))
    warmup = bool(p.get("warmup", True))
    outlier_k = float(p.get("outlier_k", 3.0))
    conv_eps = float(p.get("conv_eps", 0.0))
    surface_huber = float(p.get("surface_huber", 2.0))
    huber_threshold = float(p.get("huber_threshold", 1.0))
    function_tolerance = float(p.get("function_tolerance", 1e-3))
    num_threads = int(p.get("num_threads", 4))
    fix_first_camera = bool(p.get("fix_first_camera", True))
    residual_mode = int(p.get("residual_mode", 0))
    seed = int(p.get("seed", 0))

    n_outer = max(n_outer, 1)
    surface_on = (lambda_start > 0.0) or (lambda_end > 0.0)

    # ---- fresh mutable working set (deterministic subsample) ----------------
    a = sc.prepare(cache, max_points=p.get("max_points"), seed=seed)
    cams, pts = a["cameras"], a["points"]
    obs, ci, pi = a["observations"], a["cam_indices"], a["pt_indices"]

    def _solve(lambda_surface, sq_params, point_to_sq, max_iterations):
        if max_iterations <= 0:
            return
        sc.solve(cams, pts, obs, ci, pi,
                 lambda_surface=lambda_surface, surface_huber=surface_huber,
                 huber_threshold=huber_threshold,
                 fix_first_camera=fix_first_camera,
                 sq_params=sq_params, point_to_sq=point_to_sq,
                 residual_mode=residual_mode,
                 max_iterations=int(max_iterations),
                 function_tolerance=function_tolerance, num_threads=num_threads)

    # ---- superquadrics in predicted frame (None => degenerate Sim3) ---------
    sqp = sc.surface_pred(cache) if surface_on else None

    # ---- fallback: plain reprojection BA with the SAME total budget ---------
    if sqp is None:
        budget = (inner_iters if warmup else 0) + inner_iters * n_outer
        _solve(0.0, None, None, max(budget, inner_iters))
        return cams

    # ---- optional reproj-only warmup: clean structure before associating ----
    if warmup:
        _solve(0.0, None, None, inner_iters)

    # ---- annealed, outlier-aware EM outer loop ------------------------------
    prev_centres = _cam_centres(cams) if conv_eps > 0.0 else None
    for k in range(n_outer):
        lam_k = _geom_ramp(lambda_start, lambda_end, k, n_outer)
        assoc_k = _lin_ramp(assoc_start, assoc_end, k, n_outer)
        if lam_k <= 0.0:
            # No surface pull this round -> a reprojection-only clean-up step.
            _solve(0.0, None, None, inner_iters)
        else:
            # E-step: hard-associate the CURRENT (moving) points, then robustly
            # prune outlier pulls. ``dists`` covers every point (incl. dropped),
            # so the MAD statistics are taken over the assigned subset only.
            sq_params, point_to_sq, dists = sc.associate(pts, sqp, assoc_k)
            point_to_sq = _prune_outliers(point_to_sq, dists, outlier_k)
            # M-step: short surface BA at the annealed weight.
            _solve(lam_k, sq_params, point_to_sq, inner_iters)

        # ---- optional convergence check on camera-centre motion -------------
        if conv_eps > 0.0:
            cur = _cam_centres(cams)
            shift = float(np.max(np.linalg.norm(cur - prev_centres, axis=1)))
            prev_centres = cur
            if shift < conv_eps:
                break

    return cams
