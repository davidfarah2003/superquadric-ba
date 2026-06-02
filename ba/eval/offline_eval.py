#!/usr/bin/env python3
"""
Standalone OFFLINE scorer + Bayesian-opt scaffold for the SUPERDEC surface-BA
sweep (Phase 2).

This module deliberately imports NO torch model / VGGT / MASt3R. It depends only
on:
    * numpy
    * the ``ba`` package (Ceres bindings) at /work/courses/3dv/team39/ba/python
    * ``ba.superdec`` (SUPERDEC scene loader + point<->SQ association)
    * ``mapanything.utils.metrics`` (pose metrics)

It consumes the per-scene caches dumped by the BA_DUMP_DIR guard inside
``ba/python/ba/__init__.py:mast3r_bundle_adjust`` and re-runs *only* the
parameter-dependent part of the pipeline offline:

    surface association (depends on assoc_max_distance)
        -> Ceres BA (depends on lambda_surface, surface_huber, huber_threshold,
           fix_first_camera)
        -> pose AUC@5 / ATE-RMSE

so a Bayesian optimizer can sweep BA hyper-parameters without ever touching a
GPU, VGGT, or MASt3R.

Cache npz schema (per (scene,b), written by the stage-1 dump):
    cameras           float64  (V,10)   PRE-BA [angle_axis(3), trans_wc(3), fx,fy,cx,cy]
    points            float64  (M,3)
    observations      float64  (K,2)
    cam_indices       int32    (K,)
    pt_indices        int32    (K,)
    cam_centres       float64  (V,3)    predicted camera centres (world)
    gt_centres        float64  (V,3)    gt_poses[:, :3, 3]
    gt_poses          float64  (V,4,4)  GT C2W
    gt_quats          float64  (V,4)    xyzw (optional)
    gt_trans          float64  (V,3)    (optional)
    scene_label       <U..     ()
    superdec_npz_path <U..     ()

CLI:
    python offline_eval.py score    --cache_dir DIR --lam 50 --huber 0 --assoc 0.15
    python offline_eval.py validate --cache_dir DIR
    python offline_eval.py bayes    --cache_dir DIR --n_trials 50
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Make ``import ba`` resolve to /work/courses/3dv/team39/ba/python regardless of
# the caller's PYTHONPATH. mapanything + superdec are installed editable in the
# shared venv, so they need no path surgery.
# ---------------------------------------------------------------------------
_BA_PY = "/work/courses/3dv/team39/ba/python"
if _BA_PY not in sys.path:
    sys.path.insert(0, _BA_PY)

import ba  # noqa: E402  (must come after sys.path injection)
from ba.superdec import (  # noqa: E402
    load_scene,
    transform_sqs,
    invert_sim3,
    umeyama_sim3_pred_to_world,
    assign_points_to_sqs,
    pack_for_ceres,
)

# scipy.spatial.transform mirrors the rotation math used in ba/__init__.py
from scipy.spatial.transform import Rotation  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Cache loading
# ---------------------------------------------------------------------------

def load_cache(npz_path) -> dict:
    """Load a single per-scene cache npz into a plain dict.

    String 0-d arrays (scene_label, superdec_npz_path) are converted to Python
    str. All other arrays are returned as-is (correct dtypes from the dump).
    """
    z = np.load(str(npz_path), allow_pickle=False)
    cache: dict = {}
    for k in z.files:
        v = z[k]
        if v.dtype.kind in ("U", "S") and v.ndim == 0:
            cache[k] = str(v)
        else:
            cache[k] = v
    cache["_npz_path"] = str(npz_path)
    return cache


# ---------------------------------------------------------------------------
# 2. Surface inputs (replicates ba/__init__.py:581-600)
# ---------------------------------------------------------------------------

def build_surface_inputs(cache: dict, assoc_max_distance: float):
    """Rebuild (sq_params, point_to_sq) for the cached BA point cloud.

    Exact replication of the use_surface branch in
    ``mast3r_bundle_adjust`` (ba/__init__.py:581-600):

        sim3_p2g = umeyama_sim3_pred_to_world(cam_centres, gt_centres)
        if sim3_p2g is None: -> (None, None)   # degenerate Sim3
        sim3_g2p = invert_sim3(sim3_p2g)
        sq_pred  = transform_sqs(load_scene(superdec_npz_path), sim3_g2p)
        point_to_sq, _ = assign_points_to_sqs(points, sq_pred,
                                              max_distance=assoc_max_distance)
        sq_params, _   = pack_for_ceres(sq_pred)

    Returns
    -------
    sq_params   : (K, 11) float64 or None
    point_to_sq : (M,)    int32   or None
    """
    cam_centres = cache["cam_centres"]
    gt_centres = cache["gt_centres"]
    points = cache["points"]
    superdec_npz_path = cache["superdec_npz_path"]

    sim3_p2g = umeyama_sim3_pred_to_world(cam_centres, gt_centres)
    if sim3_p2g is None:
        # Degenerate Sim3 — matches the in-tree behaviour of skipping the
        # surface term for this element.
        return None, None

    sq_world = load_scene(superdec_npz_path)
    sim3_g2p = invert_sim3(sim3_p2g)
    sq_pred = transform_sqs(sq_world, sim3_g2p)
    point_to_sq, _dists = assign_points_to_sqs(
        points, sq_pred, max_distance=assoc_max_distance)
    sq_params, _meta = pack_for_ceres(sq_pred)
    return sq_params, point_to_sq


# ---------------------------------------------------------------------------
# 3. Run BA offline
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS = {
    "lambda_surface": 0.0,
    "surface_huber": 0.0,
    "assoc_max_distance": 0.15,
    "huber_threshold": 2.0,
    "fix_first_camera": True,
}


def run_ba(cache: dict, params: dict):
    """Re-run the Ceres mast3r_sq backend on a cached BA problem.

    ``params`` keys (missing keys fall back to _DEFAULT_PARAMS):
        lambda_surface, surface_huber, assoc_max_distance,
        huber_threshold, fix_first_camera

    The cached arrays are COPIED before the call because
    ``run_bundle_adjustment_mast3r_sq`` mutates cameras/points in place.

    Surface association is rebuilt only when lambda_surface > 0 (mirrors
    ``use_surface`` gating in the live pipeline). When lambda_surface == 0 the
    backend is called with sq_params/point_to_sq = None and lambda_surface 0.0,
    i.e. a pure reprojection BA, exactly as the live code does when the surface
    term is disabled.

    Returns
    -------
    dict with keys:
        cameras     : (V,10) float64  refined (W2C + intrinsics)
        points      : (M,3)  float64  refined structure
        final_cost  : float
        iters       : int
    """
    p = dict(_DEFAULT_PARAMS)
    p.update(params or {})

    lambda_surface = float(p["lambda_surface"])
    surface_huber = float(p["surface_huber"])
    assoc_max_distance = float(p["assoc_max_distance"])
    huber_threshold = float(p["huber_threshold"])
    fix_first_camera = bool(p["fix_first_camera"])
    max_num_iterations = int(p.get("max_iterations") or 200)
    num_threads = int(p.get("num_threads") or 4)
    function_tolerance = float(p.get("function_tolerance") or 1e-6)

    # Copy the in-place arrays (Ceres writes back into cameras/points).
    cameras = np.ascontiguousarray(cache["cameras"], dtype=np.float64).copy()
    points = np.ascontiguousarray(cache["points"], dtype=np.float64).copy()
    observations = np.ascontiguousarray(cache["observations"], dtype=np.float64)
    cam_indices = np.ascontiguousarray(cache["cam_indices"], dtype=np.int32)
    pt_indices = np.ascontiguousarray(cache["pt_indices"], dtype=np.int32)

    # Optional point subsampling. The 10 cameras are massively over-constrained
    # (40k-117k points), so keeping a few thousand points pins them just as well
    # while cutting the Ceres solve from minutes to seconds. pose_auc_5 depends
    # only on the cameras, so it is nearly invariant to this. Deterministic
    # (seed 0) so every param set sees the same points -> fair comparisons.
    # max_points=None (default) keeps full fidelity for the final validation.
    keep_idx = None
    max_points = p.get("max_points")
    if max_points and points.shape[0] > int(max_points):
        rng = np.random.default_rng(0)
        keep_idx = np.sort(rng.choice(points.shape[0], int(max_points), replace=False))
        old_to_new = np.full(points.shape[0], -1, dtype=np.int64)
        old_to_new[keep_idx] = np.arange(keep_idx.shape[0])
        obs_mask = old_to_new[pt_indices] >= 0
        points = np.ascontiguousarray(points[keep_idx], dtype=np.float64).copy()
        observations = np.ascontiguousarray(observations[obs_mask], dtype=np.float64)
        cam_indices = np.ascontiguousarray(cam_indices[obs_mask], dtype=np.int32)
        pt_indices = np.ascontiguousarray(old_to_new[pt_indices[obs_mask]], dtype=np.int32)

    use_surface = lambda_surface > 0.0
    if use_surface:
        sq_params, point_to_sq = build_surface_inputs(cache, assoc_max_distance)
        # build_surface_inputs may return (None, None) on a degenerate Sim3;
        # in that case the backend runs as pure reprojection (lambda gated to 0).
        if sq_params is None or point_to_sq is None:
            lambda_surface = 0.0
        elif keep_idx is not None:
            point_to_sq = np.ascontiguousarray(point_to_sq[keep_idx],
                                               dtype=point_to_sq.dtype)
    else:
        sq_params, point_to_sq = None, None

    final_cost, iters = ba.run_bundle_adjustment_mast3r_sq(
        cameras, points, observations, cam_indices, pt_indices,
        fix_first_camera=fix_first_camera,
        huber_threshold=huber_threshold,
        verbose=False,
        fix_points=False,
        sq_params=sq_params,
        point_to_sq=point_to_sq,
        lambda_surface=lambda_surface,
        surface_huber=surface_huber,
        max_num_iterations=max_num_iterations,
        num_threads=num_threads,
        function_tolerance=function_tolerance,
    )

    return {
        "cameras": cameras,
        "points": points,
        "final_cost": float(final_cost),
        "iters": int(iters),
    }


# ---------------------------------------------------------------------------
# 4. Refined cameras -> aligned predicted C2W poses
#    (replicates _update_preds / _quat_xyzw_to_angleaxis + _align_preds_to_gt)
# ---------------------------------------------------------------------------

def cameras_to_pred_poses(refined_cameras: np.ndarray, gt_centres: np.ndarray):
    """Convert refined W2C ``cameras`` (V,10) into per-view *aligned* C2W 4x4
    predicted poses.

    Step A — W2C -> C2W (mirrors ba/__init__.py:_update_preds:115-123):
        aa  = cameras[v, 0:3]                 # angle-axis W2C
        t   = cameras[v, 3:6]                 # translation W2C
        R_wc = Rotation.from_rotvec(aa)
        R_cw = R_wc.T
        t_cw = -R_cw @ t                      # C2W translation = camera centre
        (the C2W quaternion is Rotation.from_matrix(R_cw), but we keep R_cw
         directly to build the 4x4 pose.)

    Step B — Sim3 align predicted centres to GT centres (mirrors
        ba/__init__.py:_align_preds_to_gt:200-235), using gt_centres as the GT
        target (= gt_poses[:, :3, 3]). The same Umeyama fit
        G ≈ s*R*P + t is applied to centres AND orientations.

    Returns
    -------
    pred_poses : (V,4,4) float64  aligned C2W predicted poses
    """
    V = refined_cameras.shape[0]

    # --- Step A: build per-view C2W (centre P[v], rotation R_cw[v]) ---
    P = np.zeros((V, 3), dtype=np.float64)       # predicted camera centres (C2W t)
    R_cw_list = np.zeros((V, 3, 3), dtype=np.float64)
    for v in range(V):
        aa = refined_cameras[v, 0:3]
        t = refined_cameras[v, 3:6]
        R_wc = Rotation.from_rotvec(aa).as_matrix()
        R_cw = R_wc.T
        t_cw = -R_cw @ t
        P[v] = t_cw
        R_cw_list[v] = R_cw

    G = np.asarray(gt_centres, dtype=np.float64)  # (V,3) GT centres

    # --- Step B: Umeyama Sim3  G ≈ s*R_align*P + t_align (== _align_preds_to_gt) ---
    n_views = V
    mu_P = P.mean(0)
    mu_G = G.mean(0)
    P_c = P - mu_P
    G_c = G - mu_G
    var_P = np.mean(np.sum(P_c ** 2, axis=1))
    if var_P < 1e-10:
        # all cameras coincide — skip alignment (matches the early return)
        s_align = 1.0
        R_align = np.eye(3)
        t_align = np.zeros(3)
    else:
        H = (G_c.T @ P_c) / n_views
        U, S, Vt = np.linalg.svd(H)
        det_sign = np.sign(np.linalg.det(U @ Vt))
        D = np.diag([1.0, 1.0, float(det_sign)])
        R_align = U @ D @ Vt
        s_align = float(np.sum(S * np.diag(D))) / var_P
        t_align = mu_G - s_align * R_align @ mu_P

    pred_poses = np.zeros((V, 4, 4), dtype=np.float64)
    for v in range(V):
        c_old = P[v]
        c_new = s_align * R_align @ c_old + t_align          # aligned centre
        R_new = R_align @ R_cw_list[v]                       # aligned C2W rot
        pred_poses[v] = np.eye(4)
        pred_poses[v, :3, :3] = R_new
        pred_poses[v, :3, 3] = c_new
    return pred_poses


# ---------------------------------------------------------------------------
# 5. Pose metrics (replicates benchmark.py:694-716 + view0-relative re-expr.)
# ---------------------------------------------------------------------------

def _to_view0_relative(poses: np.ndarray) -> "object":
    """Re-express a stack of C2W 4x4 poses in view-0's frame, forcing view0 to
    identity — mirrors ``get_all_info_for_metric_computation`` where every pose
    is transformed into view0's frame (view0 initialised to identity).

        P0 = poses[0]
        rel[i] = inv(P0) @ poses[i]          (rel[0] == I exactly)

    Returns a torch tensor (V,4,4) float64 on CPU (metric fns expect torch).
    """
    import torch

    poses = np.asarray(poses, dtype=np.float64)
    P0_inv = np.linalg.inv(poses[0])
    rel = np.einsum("ij,vjk->vik", P0_inv, poses)
    rel[0] = np.eye(4)  # force view0 to exact identity
    return torch.from_numpy(rel)


def pose_auc_5(pred_poses: np.ndarray, gt_poses: np.ndarray) -> float:
    """Pose AUC@5 (%), replicating benchmark.py:694-716.

    Both pose stacks are first re-expressed in view-0's frame (view0 forced to
    identity), exactly as the live metric harness does, then fed to
    ``se3_to_relative_pose_error`` -> ``calculate_auc_np(..., max_threshold=5)``.

    NOTE: AUC@5 is built from *relative* pose errors between view pairs, and the
    rotation/translation angle errors are scale- and global-frame-invariant, so
    the view0 re-expression does not change the number — it is included purely
    to match the reference path bit-for-bit.
    """
    import torch
    from mapanything.utils.metrics import (
        se3_to_relative_pose_error,
        calculate_auc_np,
    )

    pr = _to_view0_relative(pred_poses)
    gt = _to_view0_relative(gt_poses)
    num_frames = pr.shape[0]

    rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
        pred_se3=pr, gt_se3=gt, num_frames=num_frames,
    )
    rError = rel_rangle_deg.cpu().numpy()
    tError = rel_tangle_deg.cpu().numpy()
    auc, _ = calculate_auc_np(rError, tError, max_threshold=5)
    return float(auc * 100.0)


def pose_ate_rmse(pred_poses: np.ndarray, gt_poses: np.ndarray) -> float:
    """APPROXIMATE pose ATE-RMSE (reference only), via mapanything ``evaluate_ate``.

    !!! APPROXIMATE !!!
    The live benchmark feeds NORMALISED poses (pr_pose_trans / pr_norm_factor,
    gt_pose_trans / gt_norm_factor) into evaluate_ate. Those per-sample
    normalisation factors come from ``normalize_multiple_pointclouds`` over the
    full GT/pred pointmaps, which are NOT in the cache. Here we feed the
    Sim3-aligned (to gt_centres) predicted poses and raw GT poses directly, so
    the absolute scale of this ATE will differ from the benchmark's reported
    pose_ate_rmse. evaluate_ate itself does a Horn alignment, so the *shape*
    error is meaningful, but the magnitude is not directly comparable. Use
    pose_auc_5 as the optimisation target; this is diagnostic only.
    """
    import torch
    from mapanything.utils.metrics import evaluate_ate

    pr = _to_view0_relative(pred_poses)
    gt = _to_view0_relative(gt_poses)
    gt_list = [gt[i] for i in range(gt.shape[0])]
    pr_list = [pr[i] for i in range(pr.shape[0])]
    return float(evaluate_ate(gt_traj=gt_list, est_traj=pr_list))


# ---------------------------------------------------------------------------
# 6. Per-scene + aggregate scoring
# ---------------------------------------------------------------------------

def _score_one_cache(cache: dict, params: dict) -> dict:
    """Run BA on one cache and return its per-scene metrics."""
    out = run_ba(cache, params)
    gt_poses = np.asarray(cache["gt_poses"], dtype=np.float64)
    gt_centres = np.asarray(cache["gt_centres"], dtype=np.float64)
    pred_poses = cameras_to_pred_poses(out["cameras"], gt_centres)

    auc = pose_auc_5(pred_poses, gt_poses)
    try:
        ate = pose_ate_rmse(pred_poses, gt_poses)
    except Exception:  # noqa: BLE001  diagnostic-only, never fail the score
        ate = float("nan")

    return {
        "pose_auc_5": auc,
        "pose_ate_rmse_APPROX": ate,
        "final_cost": out["final_cost"],
        "iters": out["iters"],
        "n_points": int(out["points"].shape[0]),
        "n_views": int(out["cameras"].shape[0]),
    }


def _score_worker(task):
    """Picklable module-level worker: score one scene cache.

    task = (path, params) -> (label, per_scene_result).
    """
    path, params = task
    cache = load_cache(path)
    label = cache.get("scene_label")
    if label is None or (hasattr(label, "size") and label.size == 0):
        label = Path(path).stem
    if hasattr(label, "item"):
        label = label.item()
    return str(label), _score_one_cache(cache, params)


def score(params: dict, cache_dir, jobs: int = None) -> dict:
    """Score a parameter set across every cache npz in ``cache_dir``.

    Scenes are independent, so they are scored in parallel across ``jobs``
    worker processes (default: min(n_scenes, cpu_count)). Each Ceres solve also
    uses its own internal threads, so a few scene-workers saturate a CPU node.

    Returns
    -------
    dict:
        pose_auc_5            : float  (mean across scene caches)
        pose_ate_rmse_APPROX  : float  (mean, ignoring NaN)
        per_scene             : {scene_label: {pose_auc_5, ...}}
        n_scenes              : int
        params                : the effective parameter dict used
    """
    cache_paths = sorted(glob.glob(os.path.join(str(cache_dir), "*.npz")))
    if not cache_paths:
        raise FileNotFoundError(
            f"No .npz caches found in {cache_dir!r}. "
            "Populate it first by running a superbundle benchmark with "
            "BA_DUMP_DIR set (see validation_procedure)."
        )

    eff = dict(_DEFAULT_PARAMS)
    eff.update(params or {})

    if jobs is None:
        jobs = min(len(cache_paths), os.cpu_count() or 1)

    tasks = [(p, eff) for p in cache_paths]
    if jobs and int(jobs) > 1:
        import concurrent.futures as _cf
        with _cf.ProcessPoolExecutor(max_workers=int(jobs)) as ex:
            results = list(ex.map(_score_worker, tasks))
    else:
        results = [_score_worker(t) for t in tasks]

    per_scene: dict = {}
    aucs = []
    ates = []
    for label, res in results:
        per_scene[label] = res
        aucs.append(res["pose_auc_5"])
        if np.isfinite(res["pose_ate_rmse_APPROX"]):
            ates.append(res["pose_ate_rmse_APPROX"])

    return {
        "pose_auc_5": float(np.mean(aucs)),
        "pose_ate_rmse_APPROX": float(np.mean(ates)) if ates else float("nan"),
        "per_scene": per_scene,
        "n_scenes": len(cache_paths),
        "params": eff,
    }


# ---------------------------------------------------------------------------
# 8. Bayesian optimisation (optuna preferred, scipy fallback)
# ---------------------------------------------------------------------------

# Search space for the four tunable BA hyper-parameters. fix_first_camera is
# kept fixed (True) to match the live benchmark config.
_SEARCH_SPACE = {
    "lambda_surface":    (0.0, 200.0),   # pixels-per-meter
    "surface_huber":     (0.0, 5.0),     # pixel-equivalent Huber delta (0 disables)
    "assoc_max_distance": (0.02, 0.50),  # meters
    "huber_threshold":   (0.5, 8.0),     # pixels
}


def _ensure_optuna():
    """Import optuna, pip-installing it into the shared venv if absent.

    The install is pinned via a constraints file so torch (2.10.0+cu130) and
    numpy (2.3.5) are never upgraded/downgraded by optuna's dependency solve.
    """
    try:
        import optuna  # noqa: F401
        return True
    except ImportError:
        pass

    import subprocess
    import tempfile

    constraints = "torch==2.10.0+cu130\nnumpy==2.3.5\n"
    fd, cpath = tempfile.mkstemp(prefix="optuna_constraints_", suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write(constraints)

    py = "/work/courses/3dv/team39/envs/3dv/bin/python"
    cmd = [py, "-m", "pip", "install", "optuna", "-c", cpath]
    print(f"[offline_eval] optuna not found; installing: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
    finally:
        try:
            os.remove(cpath)
        except OSError:
            pass

    try:
        import optuna  # noqa: F401
        return True
    except ImportError:
        return False


def bayes_opt(cache_dir, n_trials: int = 50, jobs: int = None, max_points: int = None,
              max_iterations: int = 50, function_tolerance: float = 1e-3,
              num_threads: int = None):
    """Maximise pose_auc_5 over the 4-D BA hyper-parameter space.

    Prefers optuna (TPE sampler). If optuna cannot be imported and cannot be
    installed, falls back to scipy.optimize (differential_evolution, which is
    derivative-free and handles the noisy objective). DOES NOT run unless called
    explicitly — the CLI 'bayes' subcommand triggers it.

    Returns
    -------
    dict: best_params, best_pose_auc_5, n_trials, backend
    """
    have_optuna = _ensure_optuna()

    if have_optuna:
        import optuna

        def objective(trial):
            params = {
                "lambda_surface": trial.suggest_float(
                    "lambda_surface", *_SEARCH_SPACE["lambda_surface"]),
                "surface_huber": trial.suggest_float(
                    "surface_huber", *_SEARCH_SPACE["surface_huber"]),
                "assoc_max_distance": trial.suggest_float(
                    "assoc_max_distance", *_SEARCH_SPACE["assoc_max_distance"]),
                "huber_threshold": trial.suggest_float(
                    "huber_threshold", *_SEARCH_SPACE["huber_threshold"]),
                "fix_first_camera": True,
                "max_points": max_points,
                "max_iterations": max_iterations,
                "function_tolerance": function_tolerance,
                "num_threads": num_threads,
            }
            return score(params, cache_dir, jobs=jobs)["pose_auc_5"]

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")

        def _cb(study, trial):
            b = study.best_trial
            print(f"[trial {trial.number:3d}] auc={trial.value:7.3f}  "
                  f"best={b.value:7.3f} @ "
                  f"lam={b.params['lambda_surface']:.1f} "
                  f"huber={b.params['surface_huber']:.2f} "
                  f"assoc={b.params['assoc_max_distance']:.3f} "
                  f"rhuber={b.params['huber_threshold']:.2f}", flush=True)

        study.optimize(objective, n_trials=int(n_trials), callbacks=[_cb])
        best = dict(study.best_params)
        best["fix_first_camera"] = True
        return {
            "backend": "optuna",
            "best_params": best,
            "best_pose_auc_5": float(study.best_value),
            "n_trials": int(n_trials),
        }

    # ---- scipy fallback: differential evolution (maximise => minimise -auc) ----
    from scipy.optimize import differential_evolution

    keys = ["lambda_surface", "surface_huber",
            "assoc_max_distance", "huber_threshold"]
    bounds = [_SEARCH_SPACE[k] for k in keys]

    def neg_objective(x):
        params = {k: float(v) for k, v in zip(keys, x)}
        params["fix_first_camera"] = True
        params["max_points"] = max_points
        params["max_iterations"] = max_iterations
        params["function_tolerance"] = function_tolerance
        params["num_threads"] = num_threads
        return -score(params, cache_dir, jobs=jobs)["pose_auc_5"]

    # popsize*len(bounds) ~ evals/iter; cap maxiter so total evals ~ n_trials.
    popsize = 8
    maxiter = max(1, int(n_trials) // (popsize * len(bounds)) or 1)
    result = differential_evolution(
        neg_objective, bounds, maxiter=maxiter, popsize=popsize,
        polish=False, seed=0, tol=1e-3,
    )
    best = {k: float(v) for k, v in zip(keys, result.x)}
    best["fix_first_camera"] = True
    return {
        "backend": "scipy.differential_evolution",
        "best_params": best,
        "best_pose_auc_5": float(-result.fun),
        "n_trials": int(n_trials),
    }


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

def _cmd_score(args):
    params = {
        "lambda_surface": args.lam,
        "surface_huber": args.huber,
        "assoc_max_distance": args.assoc,
        "huber_threshold": args.huber_threshold,
        "fix_first_camera": not args.free_first_camera,
        "max_points": args.max_points,
    }
    out = score(params, args.cache_dir, jobs=args.jobs)
    print(json.dumps({
        "pose_auc_5": out["pose_auc_5"],
        "pose_ate_rmse_APPROX": out["pose_ate_rmse_APPROX"],
        "n_scenes": out["n_scenes"],
        "params": out["params"],
        "per_scene": {k: v["pose_auc_5"] for k, v in out["per_scene"].items()},
    }, indent=2))


def _cmd_validate(args):
    """Print pose_auc_5 for the two reference configs.

    Expected (from the reference live runs over the ASE sparse benchmark):
        (lam=50, assoc=0.15, huber=0)  -> pose_auc_5 ~ 19.4
        (lam=0)                        -> pose_auc_5 ~ 29.5
    """
    surf = score({
        "lambda_surface": 50.0,
        "surface_huber": 0.0,
        "assoc_max_distance": 0.15,
        "max_points": args.max_points,
    }, args.cache_dir, jobs=args.jobs)
    base = score({
        "lambda_surface": 0.0,
        "max_points": args.max_points,
    }, args.cache_dir, jobs=args.jobs)
    print("=== offline_eval validate ===")
    print(f"n_scenes = {surf['n_scenes']}")
    print(f"[surface] lam=50 assoc=0.15 huber=0 -> "
          f"pose_auc_5 = {surf['pose_auc_5']:.4f}   (expected ~19.4)")
    print(f"[baseline] lam=0                    -> "
          f"pose_auc_5 = {base['pose_auc_5']:.4f}   (expected ~29.5)")
    print(f"delta (baseline - surface) = "
          f"{base['pose_auc_5'] - surf['pose_auc_5']:.4f}")


def _cmd_bayes(args):
    out = bayes_opt(args.cache_dir, n_trials=args.n_trials, jobs=args.jobs,
                    max_points=args.max_points, max_iterations=args.max_iterations,
                    function_tolerance=args.function_tolerance,
                    num_threads=args.num_threads)
    print(json.dumps(out, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Offline SUPERDEC surface-BA scorer + Bayesian-opt scaffold.")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("score", help="Score one parameter set over a cache dir.")
    ps.add_argument("--cache_dir", required=True)
    ps.add_argument("--lam", type=float, default=50.0,
                    help="lambda_surface (pixels-per-meter)")
    ps.add_argument("--huber", type=float, default=0.0,
                    help="surface_huber (pixel-equivalent; 0 disables)")
    ps.add_argument("--assoc", type=float, default=0.15,
                    help="assoc_max_distance (meters)")
    ps.add_argument("--huber_threshold", type=float, default=2.0,
                    help="reprojection Huber delta (pixels)")
    ps.add_argument("--free_first_camera", action="store_true",
                    help="do NOT fix the first camera")
    ps.add_argument("--jobs", type=int, default=None,
                    help="parallel scene workers (default: min(n_scenes, ncpu))")
    ps.add_argument("--max_points", type=int, default=None,
                    help="subsample points per scene for speed (None=full)")
    ps.set_defaults(func=_cmd_score)

    pv = sub.add_parser("validate", help="Print pose_auc_5 for two reference configs.")
    pv.add_argument("--cache_dir", required=True)
    pv.add_argument("--jobs", type=int, default=None,
                    help="parallel scene workers (default: min(n_scenes, ncpu))")
    pv.add_argument("--max_points", type=int, default=None,
                    help="subsample points per scene for speed (None=full)")
    pv.set_defaults(func=_cmd_validate)

    pb = sub.add_parser("bayes", help="Bayesian opt over BA hyper-parameters.")
    pb.add_argument("--cache_dir", required=True)
    pb.add_argument("--n_trials", type=int, default=50)
    pb.add_argument("--jobs", type=int, default=None,
                    help="parallel scene workers per trial (default: min(n_scenes, ncpu))")
    pb.add_argument("--max_points", type=int, default=5000,
                    help="subsample points per scene for speed (BO default 5000)")
    pb.add_argument("--max_iterations", type=int, default=50,
                    help="Ceres max iterations per solve (BO default 50)")
    pb.add_argument("--function_tolerance", type=float, default=1e-3,
                    help="Ceres function tolerance (BO default 1e-3)")
    pb.add_argument("--num_threads", type=int, default=None,
                    help="Ceres threads per solve (default 4)")
    pb.set_defaults(func=_cmd_bayes)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
