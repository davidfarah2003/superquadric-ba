"""Strategy: sq_softweight — SOFT confidence-weighted EM surface BA.

Motivation
----------
Every prior SQ variant uses a HARD association gate: a point either gets the
full surface residual (``point_to_sq >= 0``) or none at all (``-1``), decided by
a single ``assoc_max_distance`` cutoff. That cliff is exactly where the damage
comes from — a point sitting *just* inside the cutoff at a noisy distance gets
the same full pull as a point sitting dead-on the surface, and a point sitting
*just* outside contributes nothing even though it is almost certainly on the
surface. The few wrong, far-but-accepted pulls are what dragged earlier surface
BA below the plain-BA bar (29.42); the best EM hard variant only reaches 28.93.

This strategy replaces the hard gate with a SOFT, smoothly-decaying per-point
weight, using the new ``point_weights`` argument of ``sc.solve`` (which scales
each point's surface residual by ``sqrt(w_i)`` inside Ceres, so the squared
residual is scaled by ``w_i``). Combined with the one-sided HINGE residual
(``residual_mode=1``, penalize only points OUTSIDE the SQ), the net effect is:

    every point near a surface contributes, but its pull is down-weighted
    continuously by how confidently it is associated — no cliff.

Weighting rule
--------------
After each (re-)association we have, per point i, the nearest-SQ distance d_i and
the assignment ``point_to_sq[i]`` (-1 if the nearest SQ is beyond
``assoc_max_distance``). The soft weight is a Gaussian in the association
distance::

    w_i = exp( -(d_i / sigma)^2 )                       for associated points
    w_i = 0                                             for unassigned points
          (point_to_sq[i] == -1, i.e. d_i > assoc_max_distance)

``sigma`` (meters) sets how fast confidence falls with distance: a point right on
the surface has w=1, a point ``sigma`` away keeps w=exp(-1)~=0.37, ``2*sigma``
away keeps w~=0.018. With ``hard_cutoff=True`` (default) weights past
``assoc_max_distance`` are additionally forced to exactly 0 (belt-and-braces with
the association ``-1``); with ``hard_cutoff=False`` only the association's own
``-1`` zeroes them and ``assoc_max_distance`` purely controls which points enter
the surface term at all. Because ``solve`` reads ``point_weights`` for EVERY
point in ``pts`` (length must equal ``len(pts)``), w is built as a full-length
``(M,)`` float64 array each E-step; unassigned entries are 0 so their surface
residual vanishes regardless of ``point_to_sq``.

EM loop (mirrors em_reassoc)
----------------------------
  warmup (optional): one reprojection-only solve (lambda=0) to clean structure
                     before the first association sees it.
  for n_outer iterations:
      E-step: associate the CURRENT (moving) points -> (point_to_sq, dists),
              then compute the soft weights w from dists + the association mask.
      M-step: a short surface solve with residual_mode (default 1=hinge),
              passing point_weights=w so each point's pull is soft-scaled.

Re-associating + re-weighting the moving points each round means a point that
drifts toward a wrong SQ has its distance grow and its weight decay smoothly,
self-attenuating before it can bend the cameras, while a point that settles onto
a surface earns weight back — a soft analogue of em_reassoc's hard re-pointing.

Why this could push past 29.42
------------------------------
The hard variants lose because they treat marginal associations (d near the
cutoff) as fully trusted, so a minority of wrong pulls overpower the many good
ones. Soft weighting makes each pull proportional to its evidence: the large,
trustworthy mass of near-surface points (floors/walls/tables) keeps almost full
weight and tightens the gauge, while the suspect tail near the cutoff is
continuously suppressed (w -> 0) instead of being either fully on or fully off.
Paired with the one-sided hinge (which never penalizes points correctly *inside*
a primitive), the surface term adds a clean, low-variance constraint on top of
the reprojection optimum — the regime in which the first live win (29.6) was
already observed. Soft gating is a strict generalization of that win: with a
small ``sigma`` it approaches the hard gate, with a larger ``sigma`` it admits
more support, so there is room to find a setting that dominates 29.42.

Fallbacks
---------
If the predicted->GT Sim3 is degenerate (``surface_pred`` -> None) or the surface
term is disabled (``lambda_surface <= 0``), run a single plain reprojection BA
with the same total iteration budget (warmup + n_outer*inner_iters), so this
strategy never does worse than the plain-BA bar.

Params (read from ``params`` with defaults)
-------------------------------------------
    sigma              (0.05)   Gaussian width (m) of the soft weight in distance
    lambda_surface     (15.0)   surface weight (live-win regime)
    huber_threshold    (1.0)    reprojection Huber delta (px)
    assoc_max_distance (0.10)   nearest-SQ cutoff (m) -> association mask
    surface_huber      (2.749)  surface Huber delta (0 disables robustifier)
    residual_mode      (1)      1=HINGE_OUTSIDE (penalize only points outside SQ)
    n_outer            (2)      EM outer iterations (re-associate + re-weight)
    inner_iters        (41)     Ceres iters per inner (M-step) solve
    warmup             (True)   reprojection-only solve before the EM loop
    hard_cutoff        (True)   also zero weights past assoc_max_distance
    max_iterations     (None)   alias for inner_iters if inner_iters absent;
                                also the plain-BA fallback budget floor
    function_tolerance (1e-3)   Ceres function tolerance per solve
    num_threads        (4)      Ceres threads per solve
    fix_first_camera   (True)   gauge-fix the first camera
    max_points         (None)   deterministic point subsample (None = all)
    seed               (0)      subsample seed (deterministic)

Suggested sweep ranges (for the orchestrator's BO)
--------------------------------------------------
    sigma              [0.01, 0.12]    (tight ~ hard gate; loose ~ more support)
    lambda_surface     [5.0, 30.0]
    assoc_max_distance [0.04, 0.15]
    surface_huber      [0.0, 3.0]
    huber_threshold    [0.5, 2.0]
    residual_mode      {1, 4}          (hinge / normalized hinge)
    n_outer            {1, 2, 3}
    inner_iters        [25, 45]
    warmup             {True, False}
    hard_cutoff        {True, False}

A strategy module must define:  refine(cache, params) -> cameras (V,10) float64
"""
import sys

sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")

import numpy as np  # noqa: E402
import strat_common as sc  # noqa: E402


def _soft_weights(point_to_sq, dists, sigma, assoc_max_distance, hard_cutoff):
    """Full-length (M,) float64 soft surface weights.

    w_i = exp(-(d_i/sigma)^2) for associated points (point_to_sq >= 0), else 0.
    With hard_cutoff, weights are additionally zeroed where d_i exceeds
    assoc_max_distance (redundant with the association's own -1, but explicit).
    Length equals len(point_to_sq) == len(pts), exactly what sc.solve expects.
    """
    d = np.asarray(dists, np.float64)
    p2sq = np.asarray(point_to_sq)
    sig = max(float(sigma), 1e-9)  # guard against sigma=0 division
    w = np.exp(-(d / sig) ** 2)
    # Unassigned points (nearest SQ beyond assoc_max_distance) get no pull.
    w[p2sq < 0] = 0.0
    if hard_cutoff:
        w[d > float(assoc_max_distance)] = 0.0
    return np.ascontiguousarray(w, np.float64)


def refine(cache, params):
    p = dict(params or {})

    # --- tunables (every one a params key with a default) --------------------
    sigma = float(p.get("sigma", 0.05))
    lam = float(p.get("lambda_surface", 15.0))
    huber_threshold = float(p.get("huber_threshold", 1.0))
    assoc = float(p.get("assoc_max_distance", 0.10))
    surface_huber = float(p.get("surface_huber", 2.749))
    residual_mode = int(p.get("residual_mode", 1))
    n_outer = int(p.get("n_outer", 2))
    inner_iters = int(p.get("inner_iters", p.get("max_iterations", 41)))
    warmup = bool(p.get("warmup", True))
    hard_cutoff = bool(p.get("hard_cutoff", True))
    function_tolerance = float(p.get("function_tolerance", 1e-3))
    num_threads = int(p.get("num_threads", 4))
    fix_first_camera = bool(p.get("fix_first_camera", True))
    seed = int(p.get("seed", 0))

    n_outer = max(n_outer, 1)
    surface_on = lam > 0.0

    # --- fresh mutable working set (deterministic subsample) -----------------
    a = sc.prepare(cache, max_points=p.get("max_points"), seed=seed)
    cams, pts = a["cameras"], a["points"]
    obs, ci, pi = a["observations"], a["cam_indices"], a["pt_indices"]

    def _solve(lambda_surface, sq_params, point_to_sq, point_weights, max_iters):
        if max_iters <= 0:
            return
        sc.solve(cams, pts, obs, ci, pi,
                 lambda_surface=lambda_surface, surface_huber=surface_huber,
                 huber_threshold=huber_threshold,
                 fix_first_camera=fix_first_camera,
                 sq_params=sq_params, point_to_sq=point_to_sq,
                 residual_mode=residual_mode, point_weights=point_weights,
                 max_iterations=int(max_iters),
                 function_tolerance=function_tolerance, num_threads=num_threads)

    # --- superquadrics in predicted frame (None => degenerate Sim3) ----------
    sqp = sc.surface_pred(cache) if surface_on else None

    # --- fallback: plain reprojection BA with the SAME total budget ----------
    if sqp is None:
        budget = (inner_iters if warmup else 0) + inner_iters * n_outer
        _solve(0.0, None, None, None, max(budget, inner_iters))
        return cams

    # --- optional reproj-only warmup: clean structure before associating -----
    if warmup:
        _solve(0.0, None, None, None, inner_iters)

    # --- soft-weighted EM outer loop -----------------------------------------
    for _ in range(n_outer):
        # E-step: associate the CURRENT (moving) points -> mask + distances.
        sq_params, point_to_sq, dists = sc.associate(pts, sqp, assoc)
        # Soft per-point confidence weights from the association distances.
        w = _soft_weights(point_to_sq, dists, sigma, assoc, hard_cutoff)
        # M-step: short hinge surface BA with smooth per-point weighting.
        _solve(lam, sq_params, point_to_sq, w, inner_iters)

    return cams
