"""Reliability-filtered surface BA.

The plain surface baseline associates points to the *nearest* superquadric
regardless of how trustworthy that SQ is. Tiny SUPERDEC primitives (thin slivers,
clutter, decomposition noise) create spurious surface residuals that yank the
cameras off their good reprojection optimum -> pose_auc_5 drops (19.4 < 29.5).

Big primitives -- floors, walls, table tops -- have large physical extent and
are the most reliable geometry. This strategy keeps only the reliable SQs before
association: an SQ survives if its volume (product of its three scale axes) is
>= ``min_volume``, OR it is among the ``top_k`` largest-volume SQs (default keep
the largest ~50%). Points then associate only against this trustworthy subset,
so weak primitives can no longer corrupt the poses, while the dominant planar
structure still pins the solution.

A strategy module defines:  refine(cache, params) -> cameras (V,10) float64
"""
import sys

sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")

import numpy as np  # noqa: E402
import strat_common as sc  # noqa: E402

# Per-SQ arrays in the sq_pred dict that must be subset in lockstep when we drop
# unreliable primitives. ``names`` is NOT here: it is an (O,) per-object table
# indexed indirectly via object_idx, so it is passed through unchanged.
_PER_SQ_KEYS = ("scale", "exponents", "rotation_aa", "translation",
                "object_idx", "primitive_idx")


def _filter_sqs(sq_pred, min_volume, top_k):
    """Return a copy of sq_pred keeping only reliable (large) superquadrics.

    Keep rule (union): volume >= min_volume OR among the top_k by volume.
    ``top_k`` defaults to ~50% of the SQs when None. Returns (filtered_dict,
    n_kept). If nothing would survive, falls back to keeping all SQs so the
    surface term is never silently disabled by an over-aggressive threshold.
    """
    scale = np.asarray(sq_pred["scale"], np.float64)         # (K, 3)
    K = scale.shape[0]
    if K == 0:
        return sq_pred, 0

    volume = np.abs(np.prod(scale, axis=1))                  # (K,)

    if top_k is None:
        top_k = max(1, int(round(0.5 * K)))
    top_k = int(np.clip(top_k, 0, K))

    keep = np.zeros(K, dtype=bool)
    if min_volume is not None:
        keep |= volume >= float(min_volume)
    if top_k > 0:
        # indices of the top_k largest volumes
        order = np.argsort(volume)[::-1]
        keep[order[:top_k]] = True

    n_kept = int(keep.sum())
    if n_kept == 0:
        # Over-aggressive filter would kill the surface term entirely; keep all.
        return sq_pred, K
    if n_kept == K:
        return sq_pred, K

    out = dict(sq_pred)
    for k in _PER_SQ_KEYS:
        if k in out:
            out[k] = np.ascontiguousarray(np.asarray(out[k])[keep])
    return out, n_kept


def refine(cache, params):
    p = dict(params or {})
    a = sc.prepare(cache, max_points=p.get("max_points"))
    cams, pts = a["cameras"], a["points"]

    lam = float(p.get("lambda_surface", 50.0))
    assoc = float(p.get("assoc_max_distance", 0.15))
    min_volume = p.get("min_volume", 1e-3)   # m^3; ~floors/walls/tables survive
    top_k = p.get("top_k", None)             # None -> keep ~50% by volume

    sq_params = point_to_sq = None
    if lam > 0.0:
        sqp = sc.surface_pred(cache)
        if sqp is not None:
            # Filter sq_pred to the reliable (large) primitives BEFORE assoc so
            # the dropped SQs are invisible to both association and Ceres.
            sqp_f, _ = _filter_sqs(sqp, min_volume, top_k)
            # Associate the CURRENT working points -> length matches pts, which
            # is exactly what sc.solve consumes (no full-then-subsample needed).
            sq_params, point_to_sq, _ = sc.associate(pts, sqp_f, assoc)
            if not np.any(point_to_sq >= 0):
                # No point survived association -> degrade to plain reprojection.
                lam = 0.0
                sq_params = point_to_sq = None
        else:
            lam = 0.0

    sc.solve(cams, pts, a["observations"], a["cam_indices"], a["pt_indices"],
             lambda_surface=lam,
             surface_huber=float(p.get("surface_huber", 0.0)),
             huber_threshold=float(p.get("huber_threshold", 2.0)),
             fix_first_camera=bool(p.get("fix_first_camera", True)),
             sq_params=sq_params, point_to_sq=point_to_sq,
             max_iterations=int(p.get("max_iterations", 30)),
             function_tolerance=float(p.get("function_tolerance", 1e-3)),
             num_threads=int(p.get("num_threads", 4)))
    return cams
