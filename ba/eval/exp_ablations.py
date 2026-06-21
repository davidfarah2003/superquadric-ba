#!/usr/bin/env python3
r"""
exp_ablations.py — offline hyper-parameter + prior-robustness ablations for the
SuperBundle surface-prior bundle adjustment.

Everything here runs on CPU from the cached pre-BA Ceres problems in
``compose/data/ba_cache/*.npz`` (no GPU / VGGT / MASt3R). It re-uses the offline
harness in ``ba/eval/offline_eval.py`` (load_cache, build_surface_inputs,
cameras_to_pred_poses, pose_auc_5) and the SUPERDEC helpers in
``ba/python/ba/superdec.py``.

It sweeps the key BA hyper-parameters one-at-a-time around a baseline config:

    * lambda_surface     surface-prior weight (px/m)
    * huber_threshold    reprojection robustifier (px)
    * assoc_max_distance point<->SQ association cutoff (m)
    * surface_huber      surface-residual robustifier (px-equiv; bonus axis)
    * EM outer/inner     iterated re-association (E-step) + short solves (M-step)
    * n_primitives       superquadric count, controlled by the SUPERDEC
                         existence-probability threshold used in load_scene
                         (higher threshold -> fewer active SQs)

plus a ROBUSTNESS-TO-PRIOR-QUALITY ablation that DEGRADES the superquadrics
before BA and measures graceful degradation:

    * jitter   : Gaussian perturbation of SQ translation / scale / orientation
    * drop     : randomly remove a fraction of primitives

NOTE on offline EM and n_primitives: ``offline_eval.run_ba`` does a single
one-shot solve and only loads SQs at the default exist_threshold=0.5. This
script therefore carries its own ``run_ba_ext`` that (a) optionally runs the
EM outer/inner re-association loop ported from ba/__init__.py:714-749, and
(b) accepts an arbitrary pre-built ``sq`` dict so we can change the primitive
count or perturb parameters. Both paths call the SAME Ceres binding
``ba.run_bundle_adjustment_mast3r_sq`` used live, so numbers are comparable to
the live mast3r_sq backend.

CAVEAT (journal blocker #1): the surface prior is still registered into the
predicted frame with a Sim3 fit from PREDICTED to GROUND-TRUTH camera centres
(superdec.umeyama_sim3_pred_to_world inside build_surface_inputs). These
ablations vary the prior, not that GT dependence. pose_auc_5 is the headline
metric (mean of per-scene AUC@5, %); higher is better.

------------------------------------------------------------------------------
RUN COMMANDS
------------------------------------------------------------------------------
PY=/work/courses/3dv/team39/envs/3dv/bin/python
S=/work/courses/3dv/team39/ba/eval/exp_ablations.py
CACHE=/work/courses/3dv/team39/compose/data/ba_cache
OUT=/work/courses/3dv/team39/ba/eval/analysis

# Quick smoke test: baseline only, 1 scene, subsampled points (~10-20 s)
mkdir -p /tmp/ba_one && cp $CACHE/0.npz /tmp/ba_one/
$PY $S --cache_dir /tmp/ba_one --only baseline --max_points 3000 --jobs 1

# Single ablation axis over all 10 scenes, subsampled (fast iteration)
$PY $S --cache_dir $CACHE --only lambda --max_points 5000 --jobs 10

# FULL ablation suite (all axes + robustness), subsampled points
$PY $S --cache_dir $CACHE --max_points 5000 --jobs 10 \
       --out $OUT/ablations.json

# Final full-fidelity numbers (no point subsampling — slow, minutes/scene)
$PY $S --cache_dir $CACHE --jobs 10 --out $OUT/ablations_full.json

------------------------------------------------------------------------------
ROUGH RUNTIME  (10 ASE scenes, shared venv, account 3dv = 2 CPU/GPU cap)
------------------------------------------------------------------------------
With --max_points 5000 and --jobs 10 a single config over 10 scenes is ~5-15 s.
The full suite has ~50-70 configs (lambda 7, huber 5, assoc 6, surface_huber 4,
EM ~6, n_prim 5, jitter 4x3 levels, drop 4) -> ~10-25 min subsampled.
Without --max_points each config is minutes/scene; the full suite is hours and
should be staged on Slurm. EM configs run em_outer solves each, so multiply
their cost by em_outer. Use --only <axis> to run one axis at a time.

Axes for --only: baseline, lambda, huber, assoc, surface_huber, em,
n_primitives, jitter, drop, all (default).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# --- resolve offline_eval (same dir) and the ba package -----------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_BA_PY = "/work/courses/3dv/team39/ba/python"
if _BA_PY not in sys.path:
    sys.path.insert(0, _BA_PY)

import offline_eval as oe  # noqa: E402
import ba  # noqa: E402
from ba.superdec import (  # noqa: E402
    load_scene,
    transform_sqs,
    invert_sim3,
    umeyama_sim3_pred_to_world,
    assign_points_to_sqs,
    pack_for_ceres,
)
from scipy.spatial.transform import Rotation  # noqa: E402


# =============================================================================
# Baseline config (the reference around which each axis is swept).
# Mirrors the live superbundle_surface defaults: lam=15, assoc=0.15,
# huber_threshold=2.0, one-shot (no EM), exist_threshold=0.5.
# =============================================================================
BASELINE = {
    "lambda_surface": 15.0,
    "surface_huber": 0.0,
    "assoc_max_distance": 0.15,
    "huber_threshold": 2.0,
    "fix_first_camera": True,
    "exist_threshold": 0.5,   # SUPERDEC existence cutoff -> primitive count
    "em_outer": 1,            # 1 == one-shot (no re-association)
    "em_inner_iters": 50,
    "em_warmup": False,
    "refine_sq": False,
    "sq_anchor_weight": 10.0,
    # solver budget / fidelity
    "max_iterations": 200,
    "num_threads": 3,
    "function_tolerance": 1e-6,
    "residual_mode": 0,
    "max_points": None,       # overridden by --max_points on the CLI
}


# =============================================================================
# SQ preparation: build the (sq dict in predicted frame) so we can vary the
# primitive set BEFORE association. Registration is the same GT-centre Sim3 used
# everywhere else (build_surface_inputs / live path).
# =============================================================================

def _prepare_sq_pred(cache: dict, exist_threshold: float):
    """Load SUPERDEC SQs at a given existence threshold and map them into the
    predicted reconstruction frame via the GT-centre Sim3 (same as
    offline_eval.build_surface_inputs, but with a tunable exist_threshold so we
    can control the primitive count).

    Returns (sq_pred dict, sim3_g2p) or (None, None) on a degenerate Sim3.
    """
    sim3_p2g = umeyama_sim3_pred_to_world(cache["cam_centres"], cache["gt_centres"])
    if sim3_p2g is None:
        return None, None
    sq_world = load_scene(cache["superdec_npz_path"], exist_threshold=float(exist_threshold))
    sim3_g2p = invert_sim3(sim3_p2g)
    sq_pred = transform_sqs(sq_world, sim3_g2p)
    return sq_pred, sim3_g2p


def _subset_sq(sq: dict, keep_mask: np.ndarray) -> dict:
    """Subset every per-SQ array of an sq dict in lockstep."""
    K = len(sq["scale"])
    out = {}
    for k, v in sq.items():
        arr = np.asarray(v)
        out[k] = arr[keep_mask] if (arr.ndim >= 1 and arr.shape[0] == K) else v
    return out


def perturb_sq(sq: dict, *, trans_sigma=0.0, scale_frac=0.0, rot_sigma_deg=0.0,
               drop_frac=0.0, seed=0) -> dict:
    """Degrade an sq dict to simulate a lower-quality prior.

    trans_sigma   : std (m) of additive Gaussian jitter on SQ centres
    scale_frac    : std of MULTIPLICATIVE log-normal jitter on scales
                    (scale *= exp(N(0, scale_frac)))
    rot_sigma_deg : std (deg) of a random small rotation pre-multiplied onto
                    each SQ orientation
    drop_frac     : fraction of primitives to randomly remove (0..1)
    seed          : RNG seed (deterministic per scene if combined with offset)
    """
    rng = np.random.default_rng(seed)
    out = {k: (np.asarray(v).copy() if hasattr(v, "copy") else v) for k, v in sq.items()}
    K = len(out["scale"])
    if K == 0:
        return out

    if trans_sigma and trans_sigma > 0:
        out["translation"] = out["translation"] + rng.normal(
            0.0, float(trans_sigma), size=out["translation"].shape)

    if scale_frac and scale_frac > 0:
        factor = np.exp(rng.normal(0.0, float(scale_frac), size=out["scale"].shape))
        out["scale"] = out["scale"] * factor

    if rot_sigma_deg and rot_sigma_deg > 0:
        # small random rotation per SQ: axis uniform, angle ~ N(0, sigma)
        ang = np.radians(rng.normal(0.0, float(rot_sigma_deg), size=K))
        axis = rng.normal(0.0, 1.0, size=(K, 3))
        axis /= np.clip(np.linalg.norm(axis, axis=1, keepdims=True), 1e-9, None)
        dR = Rotation.from_rotvec(axis * ang[:, None]).as_matrix()       # (K,3,3)
        R = Rotation.from_rotvec(out["rotation_aa"]).as_matrix()
        R_new = np.einsum("kij,kjl->kil", dR, R)
        out["rotation_aa"] = Rotation.from_matrix(R_new).as_rotvec()

    if drop_frac and drop_frac > 0 and K > 1:
        n_keep = max(1, int(round(K * (1.0 - float(drop_frac)))))
        keep_idx = np.sort(rng.choice(K, n_keep, replace=False))
        keep_mask = np.zeros(K, dtype=bool)
        keep_mask[keep_idx] = True
        out = _subset_sq(out, keep_mask)

    return out


# =============================================================================
# Extended offline BA: supports an explicit sq dict + EM re-association.
# Falls back to offline_eval.run_ba semantics when em_outer<=1 and no custom sq.
# =============================================================================

def run_ba_ext(cache: dict, params: dict, sq_pred: dict | None = None) -> dict:
    """Re-run Ceres mast3r_sq offline with optional EM and an explicit sq dict.

    If ``sq_pred`` is None it is built from the cache at params['exist_threshold']
    via _prepare_sq_pred (so the default reproduces build_surface_inputs at
    exist_threshold=0.5). Pass a (possibly perturbed / re-thresholded) sq_pred to
    override the prior.

    EM: when params['em_outer'] > 1 the E-step (assign_points_to_sqs on the
    CURRENT moving points) alternates with an M-step (a short surface solve),
    ported from ba/__init__.py:714-749. em_warmup runs a reprojection-only solve
    first. When em_outer<=1 a single solve is done (one-shot association).

    Returns {cameras, points, final_cost, iters, n_sq, n_assigned}.
    """
    p = dict(BASELINE)
    p.update(params or {})

    lambda_surface = float(p["lambda_surface"])
    surface_huber = float(p["surface_huber"])
    assoc_max_distance = float(p["assoc_max_distance"])
    huber_threshold = float(p["huber_threshold"])
    fix_first_camera = bool(p["fix_first_camera"])
    residual_mode = int(p["residual_mode"])
    max_num_iterations = int(p["max_iterations"])
    num_threads = int(p["num_threads"])
    function_tolerance = float(p["function_tolerance"])
    em_outer = int(p["em_outer"])
    em_inner_iters = int(p["em_inner_iters"])
    em_warmup = bool(p["em_warmup"])
    refine_sq = bool(p["refine_sq"])
    sq_anchor_weight = float(p["sq_anchor_weight"])

    cameras = np.ascontiguousarray(cache["cameras"], dtype=np.float64).copy()
    points = np.ascontiguousarray(cache["points"], dtype=np.float64).copy()
    observations = np.ascontiguousarray(cache["observations"], dtype=np.float64)
    cam_indices = np.ascontiguousarray(cache["cam_indices"], dtype=np.int32)
    pt_indices = np.ascontiguousarray(cache["pt_indices"], dtype=np.int32)

    # Optional point subsampling (deterministic, seed 0) — same scheme as
    # offline_eval.run_ba. Done BEFORE association so point_to_sq matches.
    max_points = p.get("max_points")
    if max_points and points.shape[0] > int(max_points):
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(points.shape[0], int(max_points), replace=False))
        old_to_new = np.full(points.shape[0], -1, dtype=np.int64)
        old_to_new[keep] = np.arange(keep.shape[0])
        obs_mask = old_to_new[pt_indices] >= 0
        points = np.ascontiguousarray(points[keep], dtype=np.float64).copy()
        observations = np.ascontiguousarray(observations[obs_mask], dtype=np.float64)
        cam_indices = np.ascontiguousarray(cam_indices[obs_mask], dtype=np.int32)
        pt_indices = np.ascontiguousarray(old_to_new[pt_indices[obs_mask]], dtype=np.int32)

    use_surface = lambda_surface > 0.0
    n_sq = 0
    n_assigned = 0

    if use_surface:
        if sq_pred is None:
            sq_pred, _ = _prepare_sq_pred(cache, p["exist_threshold"])
        if sq_pred is None or len(sq_pred["scale"]) == 0:
            use_surface = False

    if not use_surface:
        # Pure reprojection BA.
        final_cost, iters = ba.run_bundle_adjustment_mast3r_sq(
            cameras, points, observations, cam_indices, pt_indices,
            fix_first_camera=fix_first_camera, huber_threshold=huber_threshold,
            verbose=False, fix_points=False, sq_params=None, point_to_sq=None,
            lambda_surface=0.0, surface_huber=surface_huber,
            residual_mode=residual_mode, max_num_iterations=max_num_iterations,
            num_threads=num_threads, function_tolerance=function_tolerance)
        return {"cameras": cameras, "points": points,
                "final_cost": float(final_cost), "iters": int(iters),
                "n_sq": 0, "n_assigned": 0}

    sq_params, _meta = pack_for_ceres(sq_pred)
    n_sq = int(len(sq_pred["scale"]))

    if em_outer <= 1:
        # One-shot association (matches offline_eval / live non-EM path).
        point_to_sq, _d = assign_points_to_sqs(points, sq_pred,
                                               max_distance=assoc_max_distance)
        point_to_sq = np.ascontiguousarray(point_to_sq, np.int32)
        n_assigned = int((point_to_sq >= 0).sum())
        final_cost, iters = ba.run_bundle_adjustment_mast3r_sq(
            cameras, points, observations, cam_indices, pt_indices,
            fix_first_camera=fix_first_camera, huber_threshold=huber_threshold,
            verbose=False, fix_points=False, sq_params=sq_params,
            point_to_sq=point_to_sq, lambda_surface=lambda_surface,
            surface_huber=surface_huber, residual_mode=residual_mode,
            refine_sq=refine_sq, sq_anchor_weight=sq_anchor_weight,
            max_num_iterations=max_num_iterations, num_threads=num_threads,
            function_tolerance=function_tolerance)
        return {"cameras": cameras, "points": points,
                "final_cost": float(final_cost), "iters": int(iters),
                "n_sq": n_sq, "n_assigned": n_assigned}

    # --- EM loop (ported from ba/__init__.py:714-749) ---
    final_cost, iters = 0.0, 0
    if em_warmup:
        final_cost, iters = ba.run_bundle_adjustment_mast3r_sq(
            cameras, points, observations, cam_indices, pt_indices,
            fix_first_camera=fix_first_camera, huber_threshold=huber_threshold,
            verbose=False, fix_points=False, sq_params=None, point_to_sq=None,
            lambda_surface=0.0, surface_huber=surface_huber,
            max_num_iterations=em_inner_iters, num_threads=num_threads,
            function_tolerance=function_tolerance)
    for _it in range(em_outer):
        point_to_sq, _d = assign_points_to_sqs(points, sq_pred,
                                               max_distance=assoc_max_distance)
        point_to_sq = np.ascontiguousarray(point_to_sq, np.int32)
        n_assigned = int((point_to_sq >= 0).sum())
        final_cost, iters = ba.run_bundle_adjustment_mast3r_sq(
            cameras, points, observations, cam_indices, pt_indices,
            fix_first_camera=fix_first_camera, huber_threshold=huber_threshold,
            verbose=False, fix_points=False, sq_params=sq_params,
            point_to_sq=point_to_sq, lambda_surface=lambda_surface,
            surface_huber=surface_huber, residual_mode=residual_mode,
            refine_sq=refine_sq, sq_anchor_weight=sq_anchor_weight,
            max_num_iterations=em_inner_iters, num_threads=num_threads,
            function_tolerance=function_tolerance)

    return {"cameras": cameras, "points": points,
            "final_cost": float(final_cost), "iters": int(iters),
            "n_sq": n_sq, "n_assigned": n_assigned}


# =============================================================================
# Scoring (per-scene pose_auc_5 -> mean), with optional SQ-perturbation hook.
# =============================================================================

def _score_one(cache: dict, params: dict, perturb: dict | None) -> dict:
    """Score one scene cache for a config (params), optionally degrading the
    prior with ``perturb`` kwargs (passed to perturb_sq with a per-scene seed)."""
    sq_pred = None
    if float(params.get("lambda_surface", BASELINE["lambda_surface"])) > 0.0:
        et = params.get("exist_threshold", BASELINE["exist_threshold"])
        sq_pred, _ = _prepare_sq_pred(cache, et)
        if sq_pred is not None and perturb:
            # deterministic seed derived from the scene path for reproducibility
            seed = (abs(hash(cache.get("_npz_path", ""))) % (2 ** 31))
            sq_pred = perturb_sq(sq_pred, seed=seed, **perturb)

    out = run_ba_ext(cache, params, sq_pred=sq_pred)
    gt_poses = np.asarray(cache["gt_poses"], dtype=np.float64)
    gt_centres = np.asarray(cache["gt_centres"], dtype=np.float64)
    pred_poses = oe.cameras_to_pred_poses(out["cameras"], gt_centres)
    auc = oe.pose_auc_5(pred_poses, gt_poses)
    return {
        "pose_auc_5": auc,
        "final_cost": out["final_cost"],
        "iters": out["iters"],
        "n_sq": out["n_sq"],
        "n_assigned": out["n_assigned"],
        "n_points": int(out["points"].shape[0]),
    }


def _worker(task):
    path, params, perturb = task
    cache = oe.load_cache(path)
    label = cache.get("scene_label")
    if label is None or (hasattr(label, "size") and getattr(label, "size", 1) == 0):
        label = Path(path).stem
    if hasattr(label, "item"):
        label = label.item()
    return str(label), _score_one(cache, params, perturb)


def score_config(cache_paths, params: dict, perturb: dict | None = None,
                 jobs: int | None = None) -> dict:
    """Score one config (and optional perturbation) across all scene caches.

    Returns {pose_auc_5 (mean), per_scene, n_sq_mean, n_assigned_mean}.
    """
    if jobs is None:
        jobs = min(len(cache_paths), os.cpu_count() or 1)
    tasks = [(p, params, perturb) for p in cache_paths]
    if jobs and int(jobs) > 1:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=int(jobs)) as ex:
            results = list(ex.map(_worker, tasks))
    else:
        results = [_worker(t) for t in tasks]

    per_scene = {}
    aucs, nsq, nass = [], [], []
    for label, res in results:
        per_scene[label] = res
        aucs.append(res["pose_auc_5"])
        nsq.append(res["n_sq"])
        nass.append(res["n_assigned"])
    return {
        "pose_auc_5": float(np.mean(aucs)),
        "pose_auc_5_std": float(np.std(aucs)),
        "n_sq_mean": float(np.mean(nsq)),
        "n_assigned_mean": float(np.mean(nass)),
        "per_scene": {k: v["pose_auc_5"] for k, v in per_scene.items()},
    }


# =============================================================================
# Ablation axis definitions.  Each axis is a list of (tag, overrides[, perturb]).
# =============================================================================

def _cfg(**over):
    c = dict(BASELINE)
    c.update(over)
    return c


def build_axes(max_points):
    base_over = {"max_points": max_points}

    axes = {}

    axes["baseline"] = [("baseline", _cfg(**base_over), None)]

    axes["lambda"] = [
        (f"lam={v}", _cfg(lambda_surface=v, **base_over), None)
        for v in [0.0, 5.0, 15.0, 30.0, 50.0, 100.0, 200.0]
    ]

    axes["huber"] = [
        (f"huber_thr={v}", _cfg(huber_threshold=v, **base_over), None)
        for v in [0.0, 1.0, 2.0, 4.0, 8.0]
    ]

    axes["assoc"] = [
        (f"assoc={v}", _cfg(assoc_max_distance=v, **base_over), None)
        for v in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    ]

    axes["surface_huber"] = [
        (f"surf_huber={v}", _cfg(surface_huber=v, **base_over), None)
        for v in [0.0, 0.5, 1.0, 2.0]
    ]

    # EM: vary outer (with a fixed short inner) and a couple inner settings.
    em_list = []
    for outer in [1, 2, 3, 5]:
        em_list.append((f"em_outer={outer}",
                        _cfg(em_outer=outer, em_inner_iters=20, **base_over), None))
    em_list.append(("em_outer=3_warmup",
                    _cfg(em_outer=3, em_inner_iters=20, em_warmup=True, **base_over), None))
    em_list.append(("em_outer=3_inner50",
                    _cfg(em_outer=3, em_inner_iters=50, **base_over), None))
    axes["em"] = em_list

    # n_primitives via SUPERDEC existence threshold (higher -> fewer SQs).
    axes["n_primitives"] = [
        (f"exist_thr={v}", _cfg(exist_threshold=v, **base_over), None)
        for v in [0.1, 0.3, 0.5, 0.7, 0.9]
    ]

    # Robustness to prior quality — jitter (3 levels per channel).
    jitter = []
    for s in [0.02, 0.05, 0.10]:
        jitter.append((f"jit_trans={s}", _cfg(**base_over), {"trans_sigma": s}))
    for s in [0.10, 0.25, 0.50]:
        jitter.append((f"jit_scale={s}", _cfg(**base_over), {"scale_frac": s}))
    for s in [5.0, 15.0, 30.0]:
        jitter.append((f"jit_rot_deg={s}", _cfg(**base_over), {"rot_sigma_deg": s}))
    axes["jitter"] = jitter

    # Robustness to prior quality — randomly drop primitives.
    axes["drop"] = [
        (f"drop={f}", _cfg(**base_over), {"drop_frac": f})
        for f in [0.1, 0.25, 0.5, 0.75]
    ]

    return axes


# =============================================================================
# Driver
# =============================================================================

def run(cache_dir, only, max_points, jobs, out_path):
    cache_paths = sorted(glob.glob(os.path.join(str(cache_dir), "*.npz")))
    if not cache_paths:
        raise FileNotFoundError(f"No .npz caches in {cache_dir!r}")

    axes = build_axes(max_points)
    if only and only != "all":
        if only not in axes:
            raise SystemExit(f"--only must be one of {list(axes)} or 'all'")
        axes = {only: axes[only]}

    results = {
        "meta": {
            "cache_dir": str(cache_dir),
            "n_scenes": len(cache_paths),
            "max_points": max_points,
            "baseline": {k: BASELINE[k] for k in
                         ("lambda_surface", "surface_huber", "assoc_max_distance",
                          "huber_threshold", "exist_threshold", "em_outer")},
        },
        "axes": {},
    }

    for axis_name, configs in axes.items():
        print(f"\n=== axis: {axis_name} ({len(configs)} configs) ===", flush=True)
        axis_res = []
        for tag, params, perturb in configs:
            t0 = time.time()
            r = score_config(cache_paths, params, perturb=perturb, jobs=jobs)
            dt = time.time() - t0
            entry = {
                "tag": tag,
                "pose_auc_5": r["pose_auc_5"],
                "pose_auc_5_std": r["pose_auc_5_std"],
                "n_sq_mean": r["n_sq_mean"],
                "n_assigned_mean": r["n_assigned_mean"],
                "perturb": perturb,
                "per_scene": r["per_scene"],
                "seconds": round(dt, 2),
            }
            axis_res.append(entry)
            print(f"  {tag:24s} auc5={r['pose_auc_5']:7.3f} "
                  f"(+/-{r['pose_auc_5_std']:5.2f})  n_sq={r['n_sq_mean']:.0f}  "
                  f"n_assoc={r['n_assigned_mean']:.0f}  [{dt:.1f}s]", flush=True)
        results["axes"][axis_name] = axis_res

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[exp_ablations] wrote {out_path}", flush=True)

    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", default="/work/courses/3dv/team39/compose/data/ba_cache")
    ap.add_argument("--only", default="all",
                    help="run a single axis: baseline, lambda, huber, assoc, "
                         "surface_huber, em, n_primitives, jitter, drop, all")
    ap.add_argument("--max_points", type=int, default=None,
                    help="subsample points per scene for speed (None=full fidelity)")
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel scene workers (default: min(n_scenes, ncpu))")
    ap.add_argument("--out", default=None, help="write full results JSON here")
    args = ap.parse_args(argv)
    run(args.cache_dir, args.only, args.max_points, args.jobs, args.out)


if __name__ == "__main__":
    main()
