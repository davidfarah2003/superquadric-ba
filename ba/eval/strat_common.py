"""Shared primitives for structural surface-BA strategies.

Each strategy lives in ``ba/eval/strategies/<name>.py`` and defines:

    def refine(cache, params) -> np.ndarray   # refined cameras (V, 10) float64

reusing the helpers here, then is scored with ``run_strategy.py`` (which turns
the refined cameras into pose_auc_5 via the validated offline_eval pose path).

The point of this module is that the surface-association pipeline (Sim3 ->
transform SQs into predicted frame -> nearest-SQ assignment -> pack for Ceres)
is replicated ONCE here, exactly as the live benchmark and offline_eval do it,
so strategies can re-associate against *moving* points (EM-style) without
re-deriving the math.
"""
import sys

sys.path.insert(0, "/work/courses/3dv/team39/ba/python")

import numpy as np  # noqa: E402
import ba  # noqa: E402
from ba.superdec import (  # noqa: E402
    load_scene,
    umeyama_sim3_pred_to_world,
    invert_sim3,
    transform_sqs,
    assign_points_to_sqs,
    pack_for_ceres,
)


def prepare(cache, max_points=None, seed=0):
    """Copy the cached BA arrays and (optionally) subsample points.

    Returns a dict of fresh, contiguous arrays safe to mutate in place:
        cameras (V,10) points (M,3) observations (K,2) cam_indices (K,)
        pt_indices (K,) keep (indices into the FULL cache points, or None)
    Subsampling is deterministic (seed) so every strategy/param set sees the
    same points -> fair comparison. ``keep`` lets you subsample a per-point
    quantity (e.g. point_to_sq computed on full points) in lockstep.
    """
    cams = np.ascontiguousarray(cache["cameras"], np.float64).copy()
    pts = np.ascontiguousarray(cache["points"], np.float64).copy()
    obs = np.ascontiguousarray(cache["observations"], np.float64)
    ci = np.ascontiguousarray(cache["cam_indices"], np.int32)
    pi = np.ascontiguousarray(cache["pt_indices"], np.int32)
    keep = None
    if max_points and pts.shape[0] > int(max_points):
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(pts.shape[0], int(max_points), replace=False))
        o2n = np.full(pts.shape[0], -1, np.int64)
        o2n[keep] = np.arange(keep.shape[0])
        m = o2n[pi] >= 0
        pts = np.ascontiguousarray(pts[keep], np.float64).copy()
        obs = np.ascontiguousarray(obs[m], np.float64)
        ci = np.ascontiguousarray(ci[m], np.int32)
        pi = np.ascontiguousarray(o2n[pi[m]], np.int32)
    return dict(cameras=cams, points=pts, observations=obs,
                cam_indices=ci, pt_indices=pi, keep=keep)


def surface_pred(cache):
    """Return the superquadrics in the PREDICTED-world frame, or None.

    None means the Sim3 (predicted->GT camera centres) was degenerate, in which
    case the surface term must be skipped (matches live behaviour).
    """
    sim3 = umeyama_sim3_pred_to_world(cache["cam_centres"], cache["gt_centres"])
    if sim3 is None:
        return None
    sq_world = load_scene(cache["superdec_npz_path"])
    return transform_sqs(sq_world, invert_sim3(sim3))


def associate(points, sq_pred, assoc_max_distance):
    """Nearest-SQ association for ``points`` against ``sq_pred``.

    Returns (sq_params (K,11), point_to_sq (M,) int32, dists (M,)).
    point_to_sq[i] == -1 means point i is unassigned (nearest SQ farther than
    assoc_max_distance) and gets no surface residual.
    """
    point_to_sq, dists = assign_points_to_sqs(
        points, sq_pred, max_distance=assoc_max_distance)
    sq_params, _meta = pack_for_ceres(sq_pred)
    return sq_params, np.ascontiguousarray(point_to_sq, np.int32), np.asarray(dists)


def solve(cameras, points, observations, cam_indices, pt_indices, *,
          lambda_surface=0.0, surface_huber=0.0, huber_threshold=2.0,
          fix_first_camera=True, sq_params=None, point_to_sq=None,
          residual_mode=0, point_weights=None,
          refine_sq=False, sq_anchor_weight=10.0,
          max_iterations=50, function_tolerance=1e-3, num_threads=4):
    """Run one Ceres mast3r_sq solve IN PLACE on cameras/points.

    Returns (final_cost, num_successful_steps). Pass sq_params/point_to_sq with
    lambda_surface>0 to enable the surface term; omit them for plain reprojection.
    ``residual_mode`` selects the surface-residual form (0=RADIAL default,
    1=HINGE_OUTSIDE, 2=HINGE_INSIDE, 3=RADIAL_NORMALIZED, 4=HINGE_OUTSIDE_NORMALIZED);
    ``point_weights`` (M,) optionally soft-weights each point's surface term.
    """
    return ba.run_bundle_adjustment_mast3r_sq(
        cameras, points, observations, cam_indices, pt_indices,
        fix_first_camera=fix_first_camera, huber_threshold=huber_threshold,
        verbose=False, fix_points=False, sq_params=sq_params,
        point_to_sq=point_to_sq, lambda_surface=lambda_surface,
        surface_huber=surface_huber, residual_mode=residual_mode,
        point_weights=point_weights, refine_sq=refine_sq,
        sq_anchor_weight=sq_anchor_weight, max_num_iterations=max_iterations,
        function_tolerance=function_tolerance, num_threads=num_threads)
