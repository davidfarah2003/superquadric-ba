"""
SUPERDEC scene loader and point-to-superquadric association for surface-residual BA.

NPZ schema (per `compose/data/output_npz/ase_scene_K.npz`):
    names         (O,)          U30        per-object label
    pc            (O, P, 3)     float32    per-object input pointcloud (world frame)
    assign_matrix (O, P, S)     float32    soft pt->sq assignment, intra-object
    scale         (O, S, 3)     float32    (a1, a2, a3) — physical scales (meters)
    rotation      (O, S, 3, 3)  float32    rotation matrix per SQ (world<-canonical)
    translation   (O, S, 3)     float32    SQ centre in world coords
    exponents     (O, S, 2)     float32    (eps1, eps2)
    exist         (O, S, 1)     float32    existence probability per SQ

We flatten all (object, primitive) pairs that pass `exist > exist_threshold`
into a single array of K active SQs, dropping the per-object structure for BA.

Surface residual (Solina inverse form, matches SUPERDEC's LM refinement at
`superdec/superdec/lm_optimization/lm_optimizer.py:99-136`):

    q   = R^T (p - t)              # transform point into SQ canonical frame
    F   = (|q_x/a_1|^{2/eps_2} + |q_y/a_2|^{2/eps_2})^{eps_2/eps_1}
        + |q_z/a_3|^{2/eps_1}
    r(p) = ||q|| * | F^{-eps_1/2} - 1 |    # radial Euclidean distance, meters

`r=0` exactly on the surface, positive elsewhere. Used as a single scalar
residual per associated point in the Ceres cost.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation


def load_scene(npz_path, exist_threshold: float = 0.5) -> dict:
    """
    Load a SUPERDEC scene NPZ and return only the active superquadrics,
    flattened across objects.

    Returns a dict with arrays of length K = number of active SQs:
        scale        (K, 3)   float64  (a1, a2, a3)
        exponents    (K, 2)   float64  (eps1, eps2)
        rotation_aa  (K, 3)   float64  Rodrigues angle-axis (world<-canonical)
        translation  (K, 3)   float64  world frame
        object_idx   (K,)     int32    source object index in the NPZ
        primitive_idx (K,)    int32    source primitive index within the object
        names        (O,)     U30      original object names (kept for debug)
    """
    npz = np.load(str(npz_path), allow_pickle=True)

    exist = npz["exist"][..., 0]                   # (O, S)
    active = exist > exist_threshold               # (O, S) bool

    obj_idx, prim_idx = np.where(active)           # (K,), (K,)

    scale       = npz["scale"][obj_idx, prim_idx].astype(np.float64)        # (K, 3)
    exponents   = npz["exponents"][obj_idx, prim_idx].astype(np.float64)    # (K, 2)
    rotation_m  = npz["rotation"][obj_idx, prim_idx].astype(np.float64)     # (K, 3, 3)
    translation = npz["translation"][obj_idx, prim_idx].astype(np.float64)  # (K, 3)

    rotation_aa = Rotation.from_matrix(rotation_m).as_rotvec()              # (K, 3)

    return {
        "scale":         scale,
        "exponents":     exponents,
        "rotation_aa":   rotation_aa,
        "translation":   translation,
        "object_idx":    obj_idx.astype(np.int32),
        "primitive_idx": prim_idx.astype(np.int32),
        "names":         npz["names"],
    }


def filter_degenerate_sqs(sq: dict, min_axis: float = 0.01,
                          max_axis: float = 2.0,
                          max_aspect: float = 20.0) -> dict:
    """Drop degenerate superquadrics (slivers / razor-thin / huge boxes).

    Keep SQ k iff min-scale-axis >= ``min_axis`` AND max-scale-axis <= ``max_axis``
    AND aspect (max/min) <= ``max_aspect``. ~7.5% of SUPERDEC primitives are
    slivers (aspect p99 ~115) that create wrong surface pulls. Subsets every
    per-SQ array in lockstep so association/pack stay consistent. Never drops all
    (returns the input unchanged if the mask would empty the set). Pass
    ``max_aspect<=0`` upstream to disable.
    """
    scale = np.abs(np.asarray(sq["scale"], np.float64))   # (K, 3) half-extents
    K = scale.shape[0]
    if K == 0:
        return sq
    min_ax = scale.min(axis=1)
    max_ax = scale.max(axis=1)
    aspect = max_ax / np.clip(min_ax, 1e-12, None)
    keep = (min_ax >= float(min_axis)) & (max_ax <= float(max_axis)) \
        & (aspect <= float(max_aspect))
    if not keep.any() or keep.all():
        return sq
    out = {}
    for k, v in sq.items():
        arr = np.asarray(v)
        out[k] = arr[keep] if (arr.ndim >= 1 and arr.shape[0] == K) else v
    return out


def _octahedral_group() -> np.ndarray:
    """The 24 proper rotations (det = +1) that map an axis-aligned box onto
    itself — the orientation symmetries of a superquadric box. Used to fold SQ
    orientations before voting a shared Manhattan frame."""
    from itertools import permutations, product
    mats = []
    for perm in permutations(range(3)):
        for signs in product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            if abs(np.linalg.det(M) - 1.0) < 1e-9:
                mats.append(M)
    return np.asarray(mats)                                 # (24, 3, 3)


_OCTAHEDRAL = _octahedral_group()


def _nearest_box_symmetry(Rs: np.ndarray, R_m: np.ndarray):
    """For each rotation in ``Rs`` (N,3,3), the box symmetry g minimising the
    geodesic angle between Rs and ``R_m @ g``. Returns (g_index (N,), trace (N,))
    where trace = 1 + 2 cos(angle), so larger trace == smaller angle."""
    cand = np.einsum("ij,gjk->gik", R_m, _OCTAHEDRAL)       # (24,3,3)  R_m g
    Rrel = np.einsum("nij,gkj->ngik", Rs, cand)             # Rs cand^T -> (N,24,3,3)
    tr = np.trace(Rrel, axis1=-2, axis2=-1)                 # (N,24)
    gi = np.argmax(tr, axis=1)                              # max trace == min angle
    return gi, tr[np.arange(len(Rs)), gi]


def manhattan_snap_sqs(sq: dict, max_snap_deg: float = 15.0,
                       min_aspect: float = 1.5, iters: int = 10) -> dict:
    """Snap near-axis-aligned SQ orientations onto a per-scene shared cube
    ("Manhattan") frame, denoising SUPERDEC's orientation fit.

    ASE indoor scenes are Manhattan: anisotropic SQ axes cluster a median ~5deg
    from a single orthonormal frame (verified — ba/eval/manhattan_check.py). On
    large flat primitives (walls/floor) that ~5deg is mostly fit noise, and it
    tilts the surface a point is pulled toward by up to ~scale*sin(5deg) at the
    edges. This votes the shared frame R_m by chordal-L2 rotation averaging over
    the anisotropic SQs (octahedral-symmetry assignment), then replaces every SQ
    whose orientation is within ``max_snap_deg`` of R_m with the exact snapped
    orientation R_m @ g*. SQs farther than the threshold (genuinely off-axis
    structure) are left untouched. Only the rotation changes — scales, exponents
    and translation are preserved, so each snapped SQ body is merely re-oriented
    by <= max_snap_deg, never reshaped.

    ``max_snap_deg <= 0`` disables (returns the input unchanged). Never empties.
    """
    if not max_snap_deg or max_snap_deg <= 0.0:
        return sq
    R = Rotation.from_rotvec(sq["rotation_aa"]).as_matrix()  # (K,3,3)
    K = R.shape[0]
    if K == 0:
        return sq

    # Vote the shared frame from anisotropic SQs (orientation is meaningful there).
    scale = np.abs(np.asarray(sq["scale"], np.float64))
    aspect = scale.max(1) / np.clip(scale.min(1), 1e-9, None)
    aniso = aspect > float(min_aspect)
    R_vote = R[aniso] if int(aniso.sum()) >= 5 else R

    R_m = np.eye(3)
    for _ in range(int(iters)):
        gi, _ = _nearest_box_symmetry(R_vote, R_m)
        targets = np.einsum("nij,njk->nik", R_vote,
                            np.transpose(_OCTAHEDRAL[gi], (0, 2, 1)))  # Rs g^-1
        U, _, Vt = np.linalg.svd(targets.sum(0))
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
        R_m = U @ D @ Vt

    # Snap every SQ within threshold; leave the rest exactly as-is.
    gi, tr = _nearest_box_symmetry(R, R_m)
    ang = np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))   # (K,)
    snap = ang <= float(max_snap_deg)
    R_snapped = np.einsum("ij,njk->nik", R_m, _OCTAHEDRAL[gi])          # R_m g*
    R_out = R.copy()
    R_out[snap] = R_snapped[snap]

    out = dict(sq)
    out["rotation_aa"] = Rotation.from_matrix(R_out).as_rotvec()
    return out


def _radial_distance_world(points, sq) -> np.ndarray:
    """
    Compute SUPERDEC's radial-distance residual r(p) (meters) for every
    (point, sq) pair, vectorised. Mirrors the inverse-form Solina residual
    in `lm_optimizer.compute_residuals_points_unweighted`.

    Parameters
    ----------
    points : (N, 3) float64   world-frame 3-D points
    sq     : dict from `load_scene`, contains arrays of length K

    Returns
    -------
    r : (N, K) float64   radial Euclidean distance from each point to each SQ
    """
    N = len(points)
    K = len(sq["scale"])

    # Build R^T per SQ once: (K, 3, 3), where R maps canonical -> world,
    # so R^T maps world -> canonical.
    R = Rotation.from_rotvec(sq["rotation_aa"]).as_matrix()        # (K, 3, 3)
    Rt = np.transpose(R, (0, 2, 1))                                # (K, 3, 3)

    diff = points[:, None, :] - sq["translation"][None, :, :]      # (N, K, 3)
    q = np.einsum("kij,nkj->nki", Rt, diff)                        # (N, K, 3)

    # Numerical safeguarding (mirrors superdec.utils.safe_operations.safe_pow).
    a   = np.clip(sq["scale"],    1e-3, 5e2)                       # (K, 3)
    eps = np.clip(sq["exponents"], 0.1, 1.9)                       # (K, 2) per SUPERDEC clamp

    qa = np.clip(np.abs(q) / a[None, :, :], 1e-3, 5e2)             # (N, K, 3)

    e2_inv = 1.0 / eps[None, :, 1]                                 # (1, K)
    e1_inv = 1.0 / eps[None, :, 0]                                 # (1, K)
    ratio  = eps[None, :, 1] / eps[None, :, 0]                     # (1, K)

    Fxy = qa[..., 0] ** (2.0 * e2_inv) + qa[..., 1] ** (2.0 * e2_inv)
    Fxy = np.clip(Fxy, 1e-3, 5e2) ** ratio
    Fz  = qa[..., 2] ** (2.0 * e1_inv)
    F   = np.clip(Fxy + Fz, 1e-3, 5e2)

    inside_outside = F ** (-eps[None, :, 0] / 2.0) - 1.0           # (N, K)

    r_norm = np.clip(np.linalg.norm(q, axis=-1), 1e-4, 1e6)        # (N, K)
    return r_norm * np.abs(inside_outside)                         # (N, K)


def assign_points_to_sqs(points, sq, max_distance: float = 0.15):
    """
    Assign each world-frame BA point to the closest superquadric, dropping
    points whose nearest-SQ radial distance exceeds `max_distance` (meters).

    Returns
    -------
    point_to_sq : (N,) int32   index into sq arrays, or -1 if unassigned
    nearest_dist : (N,) float64  radial distance to the chosen SQ (or to
                                 the nearest SQ when -1, kept for diagnostics)
    """
    if len(sq["scale"]) == 0:
        return np.full(len(points), -1, dtype=np.int32), \
               np.full(len(points), np.inf)

    r = _radial_distance_world(points, sq)                         # (N, K)
    nearest = np.argmin(r, axis=1)                                 # (N,)
    nearest_dist = r[np.arange(len(points)), nearest]              # (N,)

    point_to_sq = nearest.astype(np.int32)
    point_to_sq[nearest_dist > max_distance] = -1
    return point_to_sq, nearest_dist


def transform_sqs(sq: dict, sim3) -> dict:
    """
    Apply a Sim3 transform (s * R * x + t) to every superquadric in `sq`.

    Sim3 acts on:
      - SQ centre:     t' = s * R_sim @ t + t_sim
      - SQ rotation:   R' = R_sim @ R   (composition of canonical->world rots)
      - SQ scale:      a' = s * a       (linear meter scale)
      - exponents:     unchanged (dimensionless shape)

    Parameters
    ----------
    sq    : dict from `load_scene`
    sim3  : tuple (s, R_sim, t_sim) where R_sim is (3,3), t_sim is (3,)

    Returns
    -------
    new sq dict with transformed parameters; other keys passed through.
    """
    s, R_sim, t_sim = sim3

    R = Rotation.from_rotvec(sq["rotation_aa"]).as_matrix()        # (K, 3, 3)
    R_new = np.einsum("ij,kjl->kil", R_sim, R)                     # (K, 3, 3)
    aa_new = Rotation.from_matrix(R_new).as_rotvec()               # (K, 3)

    t_new = s * (sq["translation"] @ R_sim.T) + t_sim              # (K, 3)
    a_new = s * sq["scale"]                                        # (K, 3)

    return {
        "scale":         a_new,
        "exponents":     sq["exponents"].copy(),
        "rotation_aa":   aa_new,
        "translation":   t_new,
        "object_idx":    sq["object_idx"].copy(),
        "primitive_idx": sq["primitive_idx"].copy(),
        "names":         sq["names"],
    }


def invert_sim3(sim3):
    """
    Invert a Sim3 (s, R, t) such that the inverse maps `y = s R x + t`
    back to `x = (1/s) R^T (y - t) = (1/s) R^T y + t_inv`.

    Returns (1/s, R^T, t_inv) where t_inv = -(1/s) * R^T @ t.
    """
    s, R, t = sim3
    s_inv = 1.0 / s
    R_inv = R.T
    t_inv = -s_inv * R_inv @ t
    return s_inv, R_inv, t_inv


def umeyama_sim3_pred_to_world(pred_centres: np.ndarray,
                               world_centres: np.ndarray):
    """
    Fit Sim3 (s, R, t) such that `world_centres ≈ s * R @ pred_centres + t`.
    This is the same Umeyama fit used in `_align_preds_to_gt` in __init__.py;
    extracted here so we can apply the same transform to SQs (instead of just
    to camera poses).

    Returns (s, R, t) or None if degenerate.
    """
    P = np.asarray(pred_centres,  dtype=np.float64)                # (N, 3)
    G = np.asarray(world_centres, dtype=np.float64)                # (N, 3)
    n = len(P)
    if n < 2:
        return None

    mu_P = P.mean(0); mu_G = G.mean(0)
    Pc = P - mu_P;    Gc = G - mu_G
    var_P = (Pc ** 2).sum() / n
    if var_P < 1e-10:
        return None

    H = Gc.T @ Pc / n
    U, S, Vt = np.linalg.svd(H)
    det = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, float(det)])
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / var_P)
    t = mu_G - s * R @ mu_P
    return s, R, t


def pack_for_ceres(sq: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Pack active-SQ params into the (K, 11) float64 layout consumed by the
    C++ surface-residual cost functor.

    Layout per row:
        [0:3]  scale       (a1, a2, a3)
        [3:5]  exponents   (eps1, eps2)
        [5:8]  rotation    Rodrigues angle-axis (world <- canonical)
        [8:11] translation world-frame centre

    Returns
    -------
    sq_params  : (K, 11) float64
    meta       : dict with diagnostic metadata for write-back / logging
    """
    K = len(sq["scale"])
    sq_params = np.zeros((K, 11), dtype=np.float64)
    sq_params[:, 0:3]  = sq["scale"]
    sq_params[:, 3:5]  = sq["exponents"]
    sq_params[:, 5:8]  = sq["rotation_aa"]
    sq_params[:, 8:11] = sq["translation"]
    meta = {
        "object_idx":    sq["object_idx"].copy(),
        "primitive_idx": sq["primitive_idx"].copy(),
        "names":         sq["names"],
    }
    return sq_params, meta
