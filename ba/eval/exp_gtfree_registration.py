#!/usr/bin/env python3
"""
GT-FREE superquadric-prior registration (offline, CPU-only) — Variant B (ICP).

=============================================================================
WHAT THIS DOES (journal blocker #1)
=============================================================================
The SuperBundle surface prior needs the SUPERDEC superquadrics expressed in the
PREDICTED reconstruction frame before it can pull triangulated points onto them.
The live pipeline (ba/python/ba/__init__.py:629-647), the offline scorer
(ba/eval/offline_eval.py:124-141) and the harness (ba/eval/strat_common.py:73-77)
ALL do this the same GT-DEPENDENT way:

    sim3 = umeyama_sim3_pred_to_world(cam_centres, gt_centres)   # <-- uses GT poses
    sq_pred = transform_sqs(sq_world, invert_sim3(sim3))

i.e. they fit a Sim3 from PREDICTED camera centres to GROUND-TRUTH camera
centres. That GT dependence is exactly what makes the surface-prior numbers not
publishable.

This script implements a GT-FREE replacement. Instead of using GT camera
centres, it registers the SUPERDEC reference point cloud (the `pc` array stored
in the superdec npz — the cloud the SQs were fit to, in "SQ-world" frame) DIRECTLY
to the PREDICTED triangulated points (cache['points']) via a robust Sim3-ICP:

    sim3_pred_to_sqworld = ICP( pred_points -> sq_reference_cloud )
    sq_pred = transform_sqs(sq_world, invert_sim3(sim3_pred_to_sqworld))

No ground-truth camera poses enter the prior. (Ground truth is still used by the
SCORING path — pose AUC@5 is computed after the standard Sim3-to-GT alignment of
predicted CAMERA CENTRES, offline_eval.cameras_to_pred_poses — that is evaluation
protocol, identical for both arms, and is NOT the prior; it stays.)

The script then:
  1. runs offline BA with the GT-REGISTERED prior   (baseline, = offline_eval)
  2. runs offline BA with the GT-FREE  ICP prior     (this variant)
  3. also reports lam=0 pure-reprojection            (reference floor)
and prints pose AUC@5 per scene + the GT-free vs GT-registered delta.

It is fully self-contained: it imports the existing harness (offline_eval +
ba.superdec) and does NOT modify any tracked file. The only NEW logic here is the
GT-free ICP registration (build_surface_inputs_gtfree).

=============================================================================
HOW TO RUN  (CPU ONLY — no GPU / VGGT / MASt3R needed)
=============================================================================
Everything runs on the cached BA problems in compose/data/ba_cache/*.npz on a
plain CPU. The Ceres solve is CPU. NO sbatch / GPU is required.

    PY=/work/courses/3dv/team39/envs/3dv/bin/python
    $PY /work/courses/3dv/team39/ba/eval/exp_gtfree_registration.py \
        --cache_dir /work/courses/3dv/team39/compose/data/ba_cache \
        --lam 50 --assoc 0.15 --max_points 4000 --jobs 10

Flags:
    --cache_dir   dir of per-scene cache npz (default: compose/data/ba_cache)
    --lam         lambda_surface for the surface arms (default 50)
    --assoc       assoc_max_distance, meters (default 0.15)
    --huber       surface_huber, px-equiv (default 0)
    --max_points  subsample pred points per scene for the Ceres solve (None=full).
                  ICP always uses its own (denser) subsample, see --icp_points.
    --icp_points  max pred/ref points used inside ICP (default 6000)
    --icp_iters   ICP iterations (default 40)
    --icp_trim    fraction of worst NN correspondences trimmed each iter (default 0.3)
    --jobs        parallel scene workers (default min(n_scenes, ncpu))
    --json_out    optional path to dump the full per-scene result dict

NOTE on the cluster: account 3dv is capped at 2 CPUs/GPU and the Lua plugin
forces jobs onto GPU nodes, so if you want many parallel workers either run on
the login node briefly (small/fast) or wrap in run.sh:

    sbatch /work/courses/3dv/team39/run.sh \
        /work/courses/3dv/team39/envs/3dv/bin/python \
        /work/courses/3dv/team39/ba/eval/exp_gtfree_registration.py \
        --cache_dir /work/courses/3dv/team39/compose/data/ba_cache --lam 50

=============================================================================
CAVEATS / HONEST LIMITS
=============================================================================
- ICP can fall into local minima / scale or axis flips on symmetric scenes. We
  mitigate with a centroid + RMS-radius scale init and several rotation seeds
  (identity + axis flips + 90deg yaw hypotheses), robust trimmed NN, and keep the
  best-residual seed. It is still a heuristic — treat per-scene deltas, not a
  single hero number, as the signal.
- The SUPERDEC `pc` reference cloud is itself GT-derived (the SQs were fit to GT
  geometry). What this script removes is the dependence on GT CAMERA POSES for
  registration; it does NOT claim to remove the GT-geometry origin of the SQs.
  Removing that is a separate (harder) step. This is the registration-only
  ablation.
- Like offline_eval, this does a ONE-SHOT association + single Ceres solve (no EM
  re-association). Both arms use the identical solve, so the comparison is fair.
"""

from __future__ import annotations

import argparse
import json
import glob
import os
import sys

import numpy as np

# Resolve `ba` + reuse the validated offline harness (no tracked file touched).
_EVAL_DIR = "/work/courses/3dv/team39/ba/eval"
_BA_PY = "/work/courses/3dv/team39/ba/python"
for _p in (_EVAL_DIR, _BA_PY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import offline_eval as oe  # noqa: E402  (load_cache/run_ba/score path/metrics)
from ba.superdec import (  # noqa: E402
    load_scene,
    transform_sqs,
    invert_sim3,
    assign_points_to_sqs,
    pack_for_ceres,
)

from scipy.spatial import cKDTree  # noqa: E402


# ===========================================================================
# GT-FREE Sim3-ICP:  pred_points  ->  sq_reference_cloud  (SQ-world frame)
# ===========================================================================

def _umeyama_corr(P: np.ndarray, Q: np.ndarray):
    """Closed-form Sim3 (s,R,t) s.t. Q ~= s*R@P + t for CORRESPONDING rows.

    Same math as superdec.umeyama_sim3_pred_to_world but on point
    correspondences (not camera centres). Returns (s,R,t) or None if degenerate.
    """
    P = np.asarray(P, np.float64)
    Q = np.asarray(Q, np.float64)
    n = P.shape[0]
    if n < 3:
        return None
    muP = P.mean(0)
    muQ = Q.mean(0)
    Pc = P - muP
    Qc = Q - muQ
    varP = (Pc ** 2).sum() / n
    if varP < 1e-12:
        return None
    H = (Qc.T @ Pc) / n
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / varP)
    t = muQ - s * R @ muP
    return s, R, t


def _apply_sim3(sim3, X: np.ndarray) -> np.ndarray:
    s, R, t = sim3
    return s * (X @ R.T) + t


def _ref_cloud_from_superdec(superdec_npz_path: str, max_points: int,
                             seed: int = 0) -> np.ndarray:
    """The SUPERDEC reference cloud in SQ-world frame.

    Uses the `pc` array stored in the superdec npz (per-object input point cloud
    the SQs were fit to). Flattens objects, drops NaNs, deterministically
    subsamples to <= max_points.
    """
    npz = np.load(str(superdec_npz_path), allow_pickle=True)
    pc = np.asarray(npz["pc"], np.float64)            # (O, P, 3)
    pc = pc.reshape(-1, 3)
    pc = pc[np.isfinite(pc).all(1)]
    if pc.shape[0] > int(max_points):
        rng = np.random.default_rng(seed)
        idx = rng.choice(pc.shape[0], int(max_points), replace=False)
        pc = pc[idx]
    return np.ascontiguousarray(pc, np.float64)


def _seed_sim3s(P: np.ndarray, Q: np.ndarray):
    """Coarse Sim3 hypotheses mapping pred cloud P onto ref cloud Q.

    Init: match centroids, match RMS-radius scale, try several rotations
    (identity + 3 axis-flips + 3 cardinal yaw/pitch/roll 90deg). This breaks the
    symmetric-scene ICP local minima the understand phase warned about.
    """
    muP = P.mean(0)
    muQ = Q.mean(0)
    rP = np.sqrt(((P - muP) ** 2).sum(1).mean()) + 1e-9
    rQ = np.sqrt(((Q - muQ) ** 2).sum(1).mean()) + 1e-9
    s0 = rQ / rP

    def rot(axis, deg):
        from scipy.spatial.transform import Rotation
        v = np.zeros(3)
        v[axis] = np.deg2rad(deg)
        return Rotation.from_rotvec(v).as_matrix()

    Rs = [np.eye(3)]
    # axis flips (det +1: flip two axes)
    Rs.append(np.diag([1.0, -1.0, -1.0]))
    Rs.append(np.diag([-1.0, 1.0, -1.0]))
    Rs.append(np.diag([-1.0, -1.0, 1.0]))
    # cardinal 90 deg yaw hypotheses about each axis
    for ax in (0, 1, 2):
        Rs.append(rot(ax, 90.0))
        Rs.append(rot(ax, 180.0))

    seeds = []
    for R in Rs:
        t0 = muQ - s0 * R @ muP
        seeds.append((s0, R, t0))
    return seeds


def _icp_sim3(P: np.ndarray, Q: np.ndarray, *, iters: int, trim: float,
              seed_sim3):
    """Trimmed Sim3-ICP refining a single seed. Returns (sim3, mean_trim_dist)."""
    tree = cKDTree(Q)
    sim3 = seed_sim3
    last = np.inf
    for _ in range(int(iters)):
        Pt = _apply_sim3(sim3, P)
        d, j = tree.query(Pt, k=1)
        # robust trimming: keep the best (1 - trim) correspondences
        if trim and trim > 0.0:
            k = max(3, int(round((1.0 - trim) * len(d))))
            order = np.argsort(d)[:k]
        else:
            order = np.arange(len(d))
        src = P[order]
        dst = Q[j[order]]
        upd = _umeyama_corr(src, dst)
        if upd is None:
            break
        sim3 = upd
        m = float(d[order].mean())
        if abs(last - m) < 1e-6 * (last + 1e-9):
            last = m
            break
        last = m
    return sim3, last


def register_pred_to_sqworld_gtfree(cache: dict, *, icp_points: int,
                                    icp_iters: int, icp_trim: float):
    """GT-FREE Sim3 mapping PREDICTED points -> SQ-world frame, via robust ICP.

    Returns (sim3_pred_to_sqworld, best_mean_dist) or (None, inf) if no usable
    cloud. Uses NO ground-truth poses.
    """
    P = np.asarray(cache["points"], np.float64)
    P = P[np.isfinite(P).all(1)]
    if P.shape[0] < 10:
        return None, float("inf")
    if P.shape[0] > int(icp_points):
        rng = np.random.default_rng(0)
        P = P[rng.choice(P.shape[0], int(icp_points), replace=False)]
    Q = _ref_cloud_from_superdec(cache["superdec_npz_path"], int(icp_points))
    if Q.shape[0] < 10:
        return None, float("inf")

    best = None
    best_d = float("inf")
    for seed in _seed_sim3s(P, Q):
        sim3, d = _icp_sim3(P, Q, iters=icp_iters, trim=icp_trim, seed_sim3=seed)
        if d < best_d:
            best_d = d
            best = sim3
    return best, best_d


def build_surface_inputs_gtfree(cache: dict, assoc_max_distance: float, *,
                                icp_points: int, icp_iters: int, icp_trim: float):
    """GT-FREE analogue of offline_eval.build_surface_inputs.

    Replaces umeyama(cam_centres, gt_centres) with ICP(pred_points -> sq_world).
    Returns (sq_params, point_to_sq, icp_mean_dist) or (None, None, inf).
    """
    sim3_p2sq, icp_d = register_pred_to_sqworld_gtfree(
        cache, icp_points=icp_points, icp_iters=icp_iters, icp_trim=icp_trim)
    if sim3_p2sq is None:
        return None, None, float("inf")
    sq_world = load_scene(cache["superdec_npz_path"])
    # sim3_p2sq maps pred -> sq_world; bring SQs into the predicted frame:
    sq_pred = transform_sqs(sq_world, invert_sim3(sim3_p2sq))
    points = np.asarray(cache["points"], np.float64)
    point_to_sq, _d = assign_points_to_sqs(
        points, sq_pred, max_distance=assoc_max_distance)
    sq_params, _meta = pack_for_ceres(sq_pred)
    return sq_params, np.ascontiguousarray(point_to_sq, np.int32), icp_d


# ===========================================================================
# Offline BA with an EXTERNALLY-SUPPLIED (sq_params, point_to_sq)
# ===========================================================================
# offline_eval.run_ba rebuilds the GT surface inputs internally. To inject our
# GT-free inputs we replicate its (copy arrays -> optional subsample -> Ceres)
# body, but pass in OUR sq_params/point_to_sq. This is the minimal duplication
# needed without touching the tracked offline_eval source.

import ba as _ba  # noqa: E402


def _run_ba_with_surface(cache: dict, params: dict, sq_params, point_to_sq):
    """Run the Ceres mast3r_sq backend with an explicit surface association.

    Mirrors offline_eval.run_ba's array prep + subsampling. If sq_params is None
    (or lambda 0) it degrades to pure reprojection, exactly like the GT path.
    """
    p = dict(oe._DEFAULT_PARAMS)
    p.update(params or {})
    lambda_surface = float(p["lambda_surface"])
    surface_huber = float(p["surface_huber"])
    huber_threshold = float(p["huber_threshold"])
    fix_first_camera = bool(p["fix_first_camera"])
    residual_mode = int(p.get("residual_mode") or 0)
    max_num_iterations = int(p.get("max_iterations") or 200)
    num_threads = int(p.get("num_threads") or 4)
    function_tolerance = float(p.get("function_tolerance") or 1e-6)

    cameras = np.ascontiguousarray(cache["cameras"], np.float64).copy()
    points = np.ascontiguousarray(cache["points"], np.float64).copy()
    observations = np.ascontiguousarray(cache["observations"], np.float64)
    cam_indices = np.ascontiguousarray(cache["cam_indices"], np.int32)
    pt_indices = np.ascontiguousarray(cache["pt_indices"], np.int32)

    keep_idx = None
    max_points = p.get("max_points")
    if max_points and points.shape[0] > int(max_points):
        rng = np.random.default_rng(0)
        keep_idx = np.sort(rng.choice(points.shape[0], int(max_points), replace=False))
        o2n = np.full(points.shape[0], -1, np.int64)
        o2n[keep_idx] = np.arange(keep_idx.shape[0])
        m = o2n[pt_indices] >= 0
        points = np.ascontiguousarray(points[keep_idx], np.float64).copy()
        observations = np.ascontiguousarray(observations[m], np.float64)
        cam_indices = np.ascontiguousarray(cam_indices[m], np.int32)
        pt_indices = np.ascontiguousarray(o2n[pt_indices[m]], np.int32)

    use_surface = lambda_surface > 0.0 and sq_params is not None and point_to_sq is not None
    if use_surface:
        pts = point_to_sq
        if keep_idx is not None:
            pts = np.ascontiguousarray(point_to_sq[keep_idx], point_to_sq.dtype)
    else:
        sq_params = None
        pts = None
        lambda_surface = 0.0

    final_cost, iters = _ba.run_bundle_adjustment_mast3r_sq(
        cameras, points, observations, cam_indices, pt_indices,
        fix_first_camera=fix_first_camera, huber_threshold=huber_threshold,
        verbose=False, fix_points=False, sq_params=sq_params, point_to_sq=pts,
        lambda_surface=lambda_surface, surface_huber=surface_huber,
        residual_mode=residual_mode, max_num_iterations=max_num_iterations,
        num_threads=num_threads, function_tolerance=function_tolerance)
    return {"cameras": cameras, "points": points,
            "final_cost": float(final_cost), "iters": int(iters)}


def _auc_from_cameras(cache: dict, cameras: np.ndarray) -> float:
    gt_poses = np.asarray(cache["gt_poses"], np.float64)
    gt_centres = np.asarray(cache["gt_centres"], np.float64)
    pred_poses = oe.cameras_to_pred_poses(cameras, gt_centres)
    return oe.pose_auc_5(pred_poses, gt_poses)


# ===========================================================================
# Per-scene worker: three arms (lam0 baseline / GT-registered / GT-free ICP)
# ===========================================================================

def _score_one(path: str, params: dict, icp_kwargs: dict) -> tuple:
    cache = oe.load_cache(path)
    label = cache.get("scene_label")
    if label is None or (hasattr(label, "size") and getattr(label, "size", 1) == 0):
        label = os.path.splitext(os.path.basename(path))[0]
    if hasattr(label, "item"):
        label = label.item()

    lam = float(params["lambda_surface"])
    assoc = float(params["assoc_max_distance"])

    # --- arm 0: pure reprojection (lam=0) reference floor ---
    p0 = dict(params); p0["lambda_surface"] = 0.0
    out0 = _run_ba_with_surface(cache, p0, None, None)
    auc0 = _auc_from_cameras(cache, out0["cameras"])

    # --- arm 1: GT-REGISTERED prior (identical to offline_eval) ---
    sqg, ptsg = oe.build_surface_inputs(cache, assoc)
    out_gt = _run_ba_with_surface(cache, params, sqg, ptsg)
    auc_gt = _auc_from_cameras(cache, out_gt["cameras"])
    n_assoc_gt = int((ptsg >= 0).sum()) if ptsg is not None else 0

    # --- arm 2: GT-FREE ICP prior ---
    sqf, ptsf, icp_d = build_surface_inputs_gtfree(cache, assoc, **icp_kwargs)
    out_gf = _run_ba_with_surface(cache, params, sqf, ptsf)
    auc_gf = _auc_from_cameras(cache, out_gf["cameras"])
    n_assoc_gf = int((ptsf >= 0).sum()) if ptsf is not None else 0

    return str(label), {
        "auc_lam0": auc0,
        "auc_gt_registered": auc_gt,
        "auc_gtfree_icp": auc_gf,
        "delta_gtfree_minus_gt": auc_gf - auc_gt,
        "delta_gt_minus_lam0": auc_gt - auc0,
        "delta_gtfree_minus_lam0": auc_gf - auc0,
        "icp_mean_dist": float(icp_d),
        "n_assoc_gt": n_assoc_gt,
        "n_assoc_gtfree": n_assoc_gf,
        "n_points": int(out0["points"].shape[0]),
    }


def _worker(task):
    path, params, icp_kwargs = task
    return _score_one(path, params, icp_kwargs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir",
                    default="/work/courses/3dv/team39/compose/data/ba_cache")
    ap.add_argument("--lam", type=float, default=50.0)
    ap.add_argument("--assoc", type=float, default=0.15)
    ap.add_argument("--huber", type=float, default=0.0)
    ap.add_argument("--max_points", type=int, default=4000)
    ap.add_argument("--icp_points", type=int, default=6000)
    ap.add_argument("--icp_iters", type=int, default=40)
    ap.add_argument("--icp_trim", type=float, default=0.3)
    ap.add_argument("--max_iterations", type=int, default=100)
    ap.add_argument("--num_threads", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.cache_dir, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"No .npz caches in {args.cache_dir!r}")

    params = {
        "lambda_surface": args.lam,
        "surface_huber": args.huber,
        "assoc_max_distance": args.assoc,
        "huber_threshold": 2.0,
        "fix_first_camera": True,
        "max_points": args.max_points,
        "max_iterations": args.max_iterations,
        "num_threads": args.num_threads,
    }
    icp_kwargs = dict(icp_points=args.icp_points, icp_iters=args.icp_iters,
                      icp_trim=args.icp_trim)

    jobs = args.jobs or min(len(paths), os.cpu_count() or 1)
    tasks = [(p, params, icp_kwargs) for p in paths]
    if jobs > 1:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(_worker, tasks))
    else:
        results = [_worker(t) for t in tasks]

    per_scene = dict(results)

    # ---- report ----
    print("=" * 88)
    print("GT-FREE superquadric-prior REGISTRATION ablation (pose AUC@5, %)")
    print(f"  lam={args.lam}  assoc={args.assoc}  surface_huber={args.huber}  "
          f"max_points={args.max_points}")
    print(f"  ICP: points={args.icp_points} iters={args.icp_iters} "
          f"trim={args.icp_trim}")
    print("=" * 88)
    hdr = (f"{'scene':>12} | {'lam0':>7} | {'GT-reg':>7} | {'GT-free':>8} | "
           f"{'GF-GTreg':>9} | {'icp_d':>7} | {'assoc gt/gf':>13}")
    print(hdr)
    print("-" * len(hdr))
    a0, agt, agf = [], [], []
    for label in sorted(per_scene):
        r = per_scene[label]
        a0.append(r["auc_lam0"]); agt.append(r["auc_gt_registered"])
        agf.append(r["auc_gtfree_icp"])
        print(f"{label:>12} | {r['auc_lam0']:7.3f} | {r['auc_gt_registered']:7.3f} | "
              f"{r['auc_gtfree_icp']:8.3f} | {r['delta_gtfree_minus_gt']:+9.3f} | "
              f"{r['icp_mean_dist']:7.4f} | "
              f"{r['n_assoc_gt']:6d}/{r['n_assoc_gtfree']:<6d}")
    print("-" * len(hdr))
    a0 = np.array(a0); agt = np.array(agt); agf = np.array(agf)
    print(f"{'MEAN':>12} | {a0.mean():7.3f} | {agt.mean():7.3f} | "
          f"{agf.mean():8.3f} | {(agf - agt).mean():+9.3f} |")
    print()
    print(f"mean pose_auc_5:  lam0={a0.mean():.3f}  "
          f"GT-registered={agt.mean():.3f}  GT-free-ICP={agf.mean():.3f}")
    print(f"mean delta (GT-free  - GT-registered) = {(agf - agt).mean():+.3f}  "
          f"(>=0 means GT-free matches/beats the GT prior)")
    print(f"mean delta (GT-free  - lam0)          = {(agf - a0).mean():+.3f}")
    print(f"mean delta (GT-reg   - lam0)          = {(agt - a0).mean():+.3f}")
    n_gf_ge_gt = int((agf >= agt - 1e-6).sum())
    print(f"scenes where GT-free >= GT-registered: {n_gf_ge_gt}/{len(agf)}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({
                "params": params, "icp": icp_kwargs,
                "per_scene": per_scene,
                "mean": {"lam0": float(a0.mean()),
                         "gt_registered": float(agt.mean()),
                         "gtfree_icp": float(agf.mean()),
                         "delta_gtfree_minus_gt": float((agf - agt).mean())},
            }, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
