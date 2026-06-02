"""Strategy: sq_best — the UNION of everything that has helped, one tunable knob set.

This is the consolidation strategy: every lever that moved pose_auc_5 in our
favour, folded into a single ``refine`` so the orchestrator can sweep them
jointly. It is deliberately a *strict superset* of the first live winner
(em-hinge: residual_mode=1, lambda=15, huber=1.0, warmup, n_outer=2,
inner_iters=41 -> live pose_auc_5 = 29.6 > 29.42 bar). With the defaults below
every added lever is in its "off / neutral" position, so ``refine(cache, {})``
reproduces that winner's solve graph and only departs from it when a knob is
turned. That gives the BO a search space whose origin is already a win.

What is combined (each independently switchable)
------------------------------------------------
1. One-sided HINGE surface residual (``residual_mode`` default 1 = HINGE_OUTSIDE):
   penalise only points OUTSIDE their superquadric, never the (correct) interior.
   This is the change that flipped the surface term from net-negative (symmetric
   radial, 19.42) to net-positive (29.6). mode 4 = HINGE_OUTSIDE_NORMALIZED is the
   natural sweep neighbour; modes 0/2/3 remain reachable.

2. EM iterated re-association: optional reprojection-only ``warmup`` (clean the
   structure with image evidence before the prior ever associates), then
   ``n_outer`` outer iterations that RE-associate the CURRENT (moving) points to
   their nearest SQ each round. A point that drifts toward a wrong primitive is
   re-pointed (or dropped past ``assoc_max_distance``) before it accrues a large
   wrong pull. This is what lifted the one-shot surface from 19.42 to ~parity.

3. SOFT per-point weights ``w_i = exp(-(d_i / sigma)^2)`` passed as
   ``point_weights`` to ``sc.solve`` (Ceres scales each surface residual by
   sqrt(w_i), so the squared term is scaled by w_i). Replaces the hard on/off
   association cliff with a smooth, evidence-proportional pull: a point on the
   surface keeps w=1, one ``sigma`` away keeps ~0.37, ``2*sigma`` away ~0.018.
   The trustworthy near-surface mass (floors/walls/tables) keeps full weight and
   tightens the gauge; the suspect cutoff tail is continuously suppressed.
   ``soft_weights=False`` reverts to hard 0/1 weighting (pass None).

4. Percentile GATING: optionally drop the worst-distance tail of the ASSOCIATED
   set to ``point_to_sq = -1`` each outer iteration, keeping a surface residual
   only for the closest ``gate_percentile`` fraction. ``gate_percentile = 1.0``
   (default) keeps every associated point -> gate OFF. Gating composes with soft
   weighting: gating removes the far tail entirely, soft-weighting shapes the
   survivors.

5. Optional lambda ANNEALING across outer iterations: ramp the surface weight
   geometrically from ``lambda_start`` to ``lambda_end`` so early (still-settling)
   iterations trust the prior little and late (stabilised) iterations trust it
   fully. ANNEALING IS OFF BY DEFAULT (``anneal=False``); when off, a single
   constant ``lambda_surface`` (default 15.0, the live winner) is used every
   round. Turn it on and set lambda_start<lambda_end to engage.

Schedule (k = 0 .. n_outer-1, n = n_outer), only when ``anneal=True``:
    lambda_k = lambda_start * (lambda_end / lambda_start) ** (k/(n-1))   (geom)
    assoc_k  = assoc_start  + (assoc_end - assoc_start) * (k/(n-1))      (lin)
With ``anneal=False`` both collapse to (lambda_surface, assoc_max_distance) for
every k, exactly reproducing em-hinge.

Per-outer E-step (with surface_on and lambda_k > 0):
    associate current points -> (point_to_sq, dists)  [cutoff = assoc_k]
    gate: keep only closest gate_percentile of associated -> rest to -1
    weights: w_i = exp(-(d_i/sigma)^2) on kept points, 0 elsewhere (or None if
             soft_weights=False, i.e. hard 0/1 via point_to_sq mask alone)
M-step: short surface solve (residual_mode, point_weights=w) moving cams+points.

Fallbacks (never worse than the plain-BA bar)
---------------------------------------------
If the predicted->GT Sim3 is degenerate (``surface_pred`` -> None) or the surface
is disabled (max scheduled lambda <= 0), run ONE plain reprojection BA over the
same total iteration budget (warmup + n_outer*inner_iters).

Params (read from ``params`` with defaults at the live winner / neutral levers)
-------------------------------------------------------------------------------
    residual_mode      (1)      1=HINGE_OUTSIDE (live winner); 4=normalized hinge
    lambda_surface     (15.0)   constant surface weight when anneal=False (winner)
    huber_threshold    (1.0)    reprojection Huber delta (px) (winner)
    surface_huber      (0.0)    surface Huber delta (0 disables robustifier)
    assoc_max_distance (0.0372) nearest-SQ cutoff (m); winner used ~0.0372..0.10
    sigma              (0.06)   Gaussian soft-weight width (m); only if soft_weights
    soft_weights       (True)   exp(-(d/sigma)^2) per-point weights vs hard 0/1
    hard_cutoff        (True)   also zero soft weights past assoc cutoff (belt+brace)
    gate_percentile    (1.0)    keep closest-this-fraction of associated (1.0 = OFF)
    min_keep           (1)      gate floor: always keep >= this many closest points
    anneal             (False)  ramp lambda/assoc across outer iters (OFF = winner)
    lambda_start       (3.0)    annealed surface weight at first outer iter
    lambda_end         (15.0)   annealed surface weight at last outer iter
    assoc_start        (0.0372) annealed assoc radius at first outer iter
    assoc_end          (0.0372) annealed assoc radius at last outer iter
    n_outer            (2)      EM outer iterations (winner)
    inner_iters        (41)     Ceres iters per inner (M-step) solve (winner)
    warmup             (True)   reprojection-only solve before the EM loop (winner)
    function_tolerance (1e-3)   Ceres function tolerance per solve
    num_threads        (4)      Ceres threads per solve (set 1 for det. ranking)
    fix_first_camera   (True)   gauge-fix the first camera
    max_iterations     (—)      alias for inner_iters if inner_iters absent
    max_points         (None)   deterministic point subsample (None = full fidelity)
    seed               (0)      deterministic subsample seed

Suggested sweep ranges (for the orchestrator's BO; origin = a live win)
-----------------------------------------------------------------------
    residual_mode      {1, 4}
    lambda_surface     [8.0, 25.0]        (>=30 hurt live; keep modest)
    huber_threshold    [0.5, 1.5]
    surface_huber      [0.0, 3.0]
    assoc_max_distance [0.03, 0.12]
    sigma              [0.02, 0.10]
    soft_weights       {True, False}
    gate_percentile    [0.5, 1.0]         (1.0 = off; below ~0.5 starves the term)
    anneal             {False, True}
    lambda_start       [2.0, 10.0]        (only if anneal=True; < lambda_end)
    n_outer            {2, 3}
    inner_iters        [35, 45]
    warmup             {True}             (warmup off regressed in every test)

A strategy module must define:  refine(cache, params) -> cameras (V,10) float64
"""
import sys

sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")

import numpy as np  # noqa: E402
import strat_common as sc  # noqa: E402


def _geom_ramp(a, b, k, n):
    """Geometric a->b at step k of n (k in [0, n-1]); END value b when n<=1.

    Falls back to linear if either endpoint is non-positive (geom undefined).
    """
    if n <= 1:
        return float(b)
    f = float(k) / float(n - 1)
    a, b = float(a), float(b)
    if a > 0.0 and b > 0.0:
        return float(a * (b / a) ** f)
    return float(a + (b - a) * f)


def _lin_ramp(a, b, k, n):
    """Linear a->b at step k of n (k in [0, n-1]); END value b when n<=1."""
    if n <= 1:
        return float(b)
    f = float(k) / float(n - 1)
    return float(a) + (float(b) - float(a)) * f


def _gate_keep_mask(point_to_sq, dists, gate_percentile, min_keep):
    """Boolean (M,) mask: associated points whose distance is in the closest
    ``gate_percentile`` fraction of the associated set.

    gate_percentile >= 1.0 keeps every associated point (gate OFF). Always keeps
    at least ``min_keep`` closest associated points so a tight percentile cannot
    silently disable the surface term.
    """
    assoc_mask = point_to_sq >= 0
    n_assoc = int(assoc_mask.sum())
    if n_assoc == 0:
        return assoc_mask  # all-False
    if gate_percentile >= 1.0:
        return assoc_mask
    d_assoc = dists[assoc_mask]
    thr = float(np.quantile(d_assoc, float(gate_percentile)))
    keep = assoc_mask & (dists <= thr)
    floor = max(int(min_keep), 1)
    if int(keep.sum()) < floor:
        # Percentile too tight (ties / tiny set): keep the ``floor`` closest
        # associated points so the term is never silently switched off.
        idx = np.where(assoc_mask)[0]
        order = idx[np.argsort(dists[idx])][:floor]
        keep = np.zeros_like(assoc_mask)
        keep[order] = True
    return keep


def _soft_weights(dists, keep_mask, sigma, hard_cutoff, assoc_cutoff, soft):
    """Full-length (M,) float64 surface weights, or None for hard 0/1 weighting.

    With ``soft`` True: w_i = exp(-(d_i/sigma)^2) on kept points, 0 elsewhere
    (and additionally 0 past ``assoc_cutoff`` when ``hard_cutoff``). With ``soft``
    False return None -> ``sc.solve`` applies an implicit weight of 1 to every
    point still associated (point_to_sq>=0), i.e. the original hard 0/1 gate.
    Length equals len(dists) == len(pts), exactly what sc.solve expects.
    """
    if not soft:
        return None
    d = np.asarray(dists, np.float64)
    sig = max(float(sigma), 1e-9)  # guard sigma=0 division
    w = np.exp(-(d / sig) ** 2)
    w[~keep_mask] = 0.0            # gated-out + unassigned points get no pull
    if hard_cutoff:
        w[d > float(assoc_cutoff)] = 0.0
    return np.ascontiguousarray(w, np.float64)


def refine(cache, params):
    p = dict(params or {})

    # ---- residual / robustifier --------------------------------------------
    residual_mode = int(p.get("residual_mode", 1))
    lambda_surface = float(p.get("lambda_surface", 15.0))
    huber_threshold = float(p.get("huber_threshold", 1.0))
    surface_huber = float(p.get("surface_huber", 0.0))

    # ---- association / soft-weight / gate ----------------------------------
    assoc_max_distance = float(p.get("assoc_max_distance", 0.0372))
    sigma = float(p.get("sigma", 0.06))
    soft_weights = bool(p.get("soft_weights", True))
    hard_cutoff = bool(p.get("hard_cutoff", True))
    gate_percentile = float(p.get("gate_percentile", 1.0))
    min_keep = int(p.get("min_keep", 1))

    # ---- annealing (OFF by default -> constant lambda/assoc = live winner) --
    anneal = bool(p.get("anneal", False))
    lambda_start = float(p.get("lambda_start", 3.0))
    lambda_end = float(p.get("lambda_end", 15.0))
    assoc_start = float(p.get("assoc_start", 0.0372))
    assoc_end = float(p.get("assoc_end", 0.0372))

    # ---- EM loop / solver budget -------------------------------------------
    n_outer = max(int(p.get("n_outer", 2)), 1)
    inner_iters = int(p.get("inner_iters", p.get("max_iterations", 41)))
    warmup = bool(p.get("warmup", True))
    function_tolerance = float(p.get("function_tolerance", 1e-3))
    num_threads = int(p.get("num_threads", 4))
    fix_first_camera = bool(p.get("fix_first_camera", True))
    seed = int(p.get("seed", 0))

    # Surface is "on" iff any scheduled lambda is positive.
    if anneal:
        surface_on = (lambda_start > 0.0) or (lambda_end > 0.0)
    else:
        surface_on = lambda_surface > 0.0

    # ---- fresh mutable working set (deterministic subsample) ----------------
    a = sc.prepare(cache, max_points=p.get("max_points"), seed=seed)
    cams, pts = a["cameras"], a["points"]
    obs, ci, pi = a["observations"], a["cam_indices"], a["pt_indices"]

    def _solve(lam, sq_params, point_to_sq, point_weights, max_iters):
        if max_iters <= 0:
            return
        sc.solve(cams, pts, obs, ci, pi,
                 lambda_surface=lam, surface_huber=surface_huber,
                 huber_threshold=huber_threshold,
                 fix_first_camera=fix_first_camera,
                 sq_params=sq_params, point_to_sq=point_to_sq,
                 residual_mode=residual_mode, point_weights=point_weights,
                 max_iterations=int(max_iters),
                 function_tolerance=function_tolerance, num_threads=num_threads)

    # ---- superquadrics in predicted frame (None => degenerate Sim3) ---------
    sqp = sc.surface_pred(cache) if surface_on else None

    # ---- fallback: plain reprojection BA with the SAME total budget ---------
    if sqp is None:
        budget = (inner_iters if warmup else 0) + inner_iters * n_outer
        _solve(0.0, None, None, None, max(budget, inner_iters))
        return cams

    # ---- optional reproj-only warmup: clean structure before associating ----
    if warmup:
        _solve(0.0, None, None, None, inner_iters)

    # ---- EM outer loop: re-associate moving points, gate, weight, solve -----
    for k in range(n_outer):
        if anneal:
            lam_k = _geom_ramp(lambda_start, lambda_end, k, n_outer)
            assoc_k = _lin_ramp(assoc_start, assoc_end, k, n_outer)
        else:
            lam_k, assoc_k = lambda_surface, assoc_max_distance

        if lam_k <= 0.0:
            # No surface pull this round -> reprojection-only clean-up step.
            _solve(0.0, None, None, None, inner_iters)
            continue

        # E-step: associate the CURRENT (moving) points at this round's cutoff.
        sq_params, point_to_sq, dists = sc.associate(pts, sqp, assoc_k)

        # Percentile gate: drop the far tail of the associated set to -1.
        keep = _gate_keep_mask(point_to_sq, dists, gate_percentile, min_keep)
        if not np.any(keep):
            # Nothing trusted this round -> spend the budget on reprojection.
            _solve(0.0, None, None, None, inner_iters)
            continue
        if gate_percentile < 1.0:
            gated = np.full(point_to_sq.shape[0], -1, np.int32)
            gated[keep] = point_to_sq[keep]
            point_to_sq = np.ascontiguousarray(gated, np.int32)

        # Soft per-point weights (or None for hard 0/1 via the point_to_sq mask).
        w = _soft_weights(dists, keep, sigma, hard_cutoff, assoc_k, soft_weights)

        # M-step: short hinge surface BA at this round's weight.
        _solve(lam_k, sq_params, point_to_sq, w, inner_iters)

    return cams
