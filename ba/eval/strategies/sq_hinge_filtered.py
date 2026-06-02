"""EM one-sided-hinge surface BA with degenerate-superquadric pre-filtering.

The winning recipe (``strategies/em_reassoc.py`` with residual_mode=1 HINGE,
lambda=15, assoc=0.0372, surface_huber=2.749, n_outer=2, inner_iters=41,
warmup=True) reaches pose_auc_5 = 29.6, beating plain BA (29.42). Sweeping the
surface-residual FORM (radial-hinge mode1 vs normal/tangent mode5/6) leaves the
score flat at ~29.24 offline -> the residual form has hit its ceiling. The real
limiter is the FROZEN, mis-registered SQ geometry: SUPERDEC primitives are
loaded, Sim3-fitted into the predicted frame from camera centres, then never
optimized. A measurable ~7.5% of those SQs are degenerate -- razor-thin slivers,
near-zero axes, or implausibly huge boxes (aspect ratio p99 = 115). A sliver
whose smallest axis is thinner than the association band has a surface that wraps
the wrong way through nearby points, so it generates *wrong* hinge pulls no
matter how the residual is shaped. Those bad primitives are also what forces
lambda to stay low (lambda > 15 HURTS): the trustworthy planar structure cannot
be weighted up without simultaneously weighting up the slivers.

This strategy is the winner recipe, UNCHANGED, except the SQ dict is filtered
ONCE up front to drop degenerate primitives BEFORE the EM loop ever associates:

    keep SQ k  iff   min_axis_k  >= min_axis   (drop near-zero / sliver axes)
                AND  max_axis_k  <= max_axis   (drop implausibly huge boxes)
                AND  aspect_k = max_axis_k/min_axis_k <= max_aspect  (drop razors)

Filtering at the SQ-dict level (not at association) makes the dropped primitives
invisible to BOTH the nearest-SQ association and the Ceres cost, so a point that
used to snap onto a sliver now associates to the nearest *valid* surface (or to
nothing, past assoc). If the thresholds would drop every SQ we keep all of them,
so the surface term is never silently disabled.

A strategy module must define:  refine(cache, params) -> cameras (V,10) float64
"""
import sys

sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")

import numpy as np  # noqa: E402
import strat_common as sc  # noqa: E402

# Per-SQ arrays in the sq_pred dict that must be subset in lockstep when we drop
# degenerate primitives. ``names`` is NOT here: it is an (O,) per-object table
# indexed indirectly via object_idx, so it is passed through unchanged.
_PER_SQ_KEYS = ("scale", "exponents", "rotation_aa", "translation",
                "object_idx", "primitive_idx")


def _filter_degenerate(sqp, min_axis, max_axis, max_aspect):
    """Return (filtered sqp, n_kept, n_total) dropping degenerate superquadrics.

    Keep SQ k iff its smallest scale axis >= min_axis (not a sliver / not near
    zero), its largest scale axis <= max_axis (not implausibly huge), and its
    aspect ratio max_axis/min_axis <= max_aspect (not razor-thin). Scale axes are
    SUPERDEC half-extents in meters; ``scale`` is (K, 3). If the mask would drop
    everything, keep all SQs (no-op) so the surface term is never disabled.
    """
    scale = np.abs(np.asarray(sqp["scale"], np.float64))   # (K, 3) half-extents
    K = scale.shape[0]
    if K == 0:
        return sqp, 0, 0

    min_ax = scale.min(axis=1)                             # (K,)
    max_ax = scale.max(axis=1)                             # (K,)
    # Guard against exact-zero min axis before dividing for the aspect ratio.
    aspect = max_ax / np.clip(min_ax, 1e-12, None)         # (K,)

    keep = (min_ax >= float(min_axis)) \
        & (max_ax <= float(max_axis)) \
        & (aspect <= float(max_aspect))

    n_kept = int(keep.sum())
    if n_kept == 0 or n_kept == K:
        # Over-aggressive (kills the surface term) or no-op -> keep everything.
        return sqp, K, K

    out = dict(sqp)
    for k in _PER_SQ_KEYS:
        if k in out:
            out[k] = np.ascontiguousarray(np.asarray(out[k])[keep])
    return out, n_kept, K


def refine(cache, params):
    p = dict(params or {})

    # --- winner-recipe tunables (defaults = the verified 29.6 config) --------
    lam = float(p.get("lambda_surface", 15.0))
    assoc = float(p.get("assoc_max_distance", 0.0372))
    surface_huber = float(p.get("surface_huber", 2.749))
    huber_threshold = float(p.get("huber_threshold", 1.0))
    n_outer = int(p.get("n_outer", 2))
    inner_iters = int(p.get("inner_iters", 41))
    warmup = bool(p.get("warmup", True))
    residual_mode = int(p.get("residual_mode", 1))
    fix_first_camera = bool(p.get("fix_first_camera", True))
    max_iterations = int(p.get("max_iterations", inner_iters))
    function_tolerance = float(p.get("function_tolerance", 1e-3))
    num_threads = int(p.get("num_threads", 4))

    # --- degenerate-SQ filter thresholds (meters / dimensionless) ------------
    min_axis = float(p.get("min_axis", 0.01))     # drop axes thinner than 1 cm
    max_axis = float(p.get("max_axis", 2.0))      # drop boxes larger than 2 m
    max_aspect = float(p.get("max_aspect", 20.0))  # drop razor-thin primitives

    # --- subsample once; sq_pred once ---------------------------------------
    a = sc.prepare(cache, max_points=p.get("max_points"), seed=int(p.get("seed", 0)))
    cams, pts = a["cameras"], a["points"]
    obs, ci, pi = a["observations"], a["cam_indices"], a["pt_indices"]

    sqp = sc.surface_pred(cache) if lam > 0.0 else None
    if sqp is None:
        # Degenerate Sim3 (or surface disabled): plain reprojection BA, with the
        # same total iteration budget so it's a fair fallback (matches em_reassoc).
        sc.solve(cams, pts, obs, ci, pi,
                 lambda_surface=0.0, surface_huber=surface_huber,
                 huber_threshold=huber_threshold,
                 fix_first_camera=fix_first_camera,
                 sq_params=None, point_to_sq=None,
                 max_iterations=max(inner_iters * max(n_outer, 1), inner_iters),
                 function_tolerance=function_tolerance, num_threads=num_threads)
        return cams

    # --- drop degenerate superquadrics BEFORE any association ----------------
    sqp, _n_kept, _n_total = _filter_degenerate(sqp, min_axis, max_axis, max_aspect)

    # --- optional reproj-only warmup: clean structure before associating -----
    if warmup:
        sc.solve(cams, pts, obs, ci, pi,
                 lambda_surface=0.0, surface_huber=surface_huber,
                 huber_threshold=huber_threshold,
                 fix_first_camera=fix_first_camera,
                 sq_params=None, point_to_sq=None,
                 max_iterations=inner_iters,
                 function_tolerance=function_tolerance, num_threads=num_threads)

    # --- EM outer loop: re-associate MOVING points against the FILTERED sqp --
    for _ in range(max(n_outer, 1)):
        # E-step: associate the CURRENT working points directly against the
        # filtered SQ set. Returned point_to_sq length == len(pts).
        sq_params, point_to_sq, _ = sc.associate(pts, sqp, assoc)
        # M-step: short one-sided-hinge surface BA moving cameras + points.
        sc.solve(cams, pts, obs, ci, pi,
                 lambda_surface=lam, surface_huber=surface_huber,
                 huber_threshold=huber_threshold,
                 fix_first_camera=fix_first_camera,
                 sq_params=sq_params, point_to_sq=point_to_sq,
                 residual_mode=residual_mode,
                 max_iterations=inner_iters,
                 function_tolerance=function_tolerance, num_threads=num_threads)

    return cams
