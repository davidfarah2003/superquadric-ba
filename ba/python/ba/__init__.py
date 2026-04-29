"""
Bundle adjustment package — wraps Ceres-based pybind11 extensions.

Two backends are exposed with an identical Python API:
    ba.baseline  —  standard Ceres BA (reference)
    ba.custom    —  custom research BA (diverges from baseline over time)

Camera convention (10 params per camera, float64):
    [0:3]  angle-axis rotation  (world-to-camera, Rodrigues)
    [3:6]  translation          (world-to-camera)
    [6]    fx
    [7]    fy
    [8]    cx
    [9]    cy

Typical mapanything usage — call bundle_adjust() right after model inference:

    from ba import bundle_adjust, mast3r_bundle_adjust
    preds = bundle_adjust(preds, batch)                   # GT-structure baseline
    preds = mast3r_bundle_adjust(preds, batch, mast3r)    # MASt3R-match baseline
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
from collections import defaultdict
from scipy.spatial.transform import Rotation


# Add the directory containing the compiled .so files to sys.path
_pkg_dir = Path(__file__).parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

_PYCOLMAP_LIB = None  # pycolmap is installed via pip

# Lazy-load the C extensions to give a clear error if the build is missing.
def _load_core(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Could not import '{name}'. "
            "Did you build the C++ extensions?\n"
            "  cd /work/courses/3dv/team39/ba/build && cmake .. && make -j$(nproc)"
        ) from e


# ---------------------------------------------------------------------------
# Low-level interface (mirrors the C++ signature exactly)
# ---------------------------------------------------------------------------

def run_bundle_adjustment_baseline(cameras, points, observations,
                                   cam_indices, pt_indices, verbose=False):
    """Run baseline Ceres BA. All arrays are modified **in place**."""
    core = _load_core("baseline_ba_core")
    return core.run_bundle_adjustment(cameras, points, observations,
                                      cam_indices, pt_indices, verbose)


def run_bundle_adjustment_custom(cameras, points, observations,
                                 cam_indices, pt_indices,
                                 fix_first_camera=True,
                                 huber_threshold=2.0,
                                 verbose=False,
                                 fix_points=False):
    """Run custom Ceres BA. All arrays are modified **in place**."""
    core = _load_core("custom_ba_core")
    return core.run_bundle_adjustment(cameras, points, observations,
                                      cam_indices, pt_indices,
                                      fix_first_camera, huber_threshold,
                                      verbose, fix_points)


# ---------------------------------------------------------------------------
# High-level mapanything adapter
# ---------------------------------------------------------------------------

def _quat_xyzw_to_angleaxis(quats_xyzw: np.ndarray) -> np.ndarray:
    """(N, 4) xyzw quaternions → (N, 3) angle-axis."""
    return Rotation.from_quat(quats_xyzw).inv().as_rotvec()


def _angleaxis_to_quat_xyzw(rotvecs: np.ndarray) -> np.ndarray:
    """(N, 3) angle-axis → (N, 4) xyzw quaternions (camera-to-world)."""
    return Rotation.from_rotvec(rotvecs).inv().as_quat()


def _pycolmap_align_sim3(pts_world_raw, gt_c2w, pred_c2w, K_arr, H, W,
                          max_pts=8000,
                          min_inlier_observations=0.3,
                          max_reproj_error=8.0):
    """
    Robust Sim3 alignment (GT world frame → predicted world frame) via
    pycolmap.align_reconstructions_via_reprojections (RANSAC-based).

    Returns a pycolmap.Sim3d such that p_pred = sim3 * p_gt, or None if
    alignment fails or pycolmap is unavailable.
    """
    try:
        import pycolmap
    except ImportError:
        return None

    n_views = len(gt_c2w)
    n_pts   = len(pts_world_raw)

    if n_pts > max_pts:
        rng  = np.random.default_rng(42)
        sel  = rng.choice(n_pts, max_pts, replace=False)
        pts_sub = pts_world_raw[sel]
    else:
        sel     = np.arange(n_pts)
        pts_sub = pts_world_raw

    src = pycolmap.Reconstruction()
    tgt = pycolmap.Reconstruction()

    per_image_obs: dict = {}

    for vi in range(n_views):
        image_id = vi + 1
        fx, fy = float(K_arr[vi, 0, 0]), float(K_arr[vi, 1, 1])
        cx, cy = float(K_arr[vi, 0, 2]), float(K_arr[vi, 1, 2])

        R_wc_gt = gt_c2w[vi, :3, :3].T
        t_wc_gt = -R_wc_gt @ gt_c2w[vi, :3, 3]

        p_cam = (R_wc_gt @ pts_sub.T).T + t_wc_gt
        depth = p_cam[:, 2]
        vis   = depth > 0
        u = np.where(vis, fx * p_cam[:, 0] / np.where(vis, depth, 1.0) + cx, -1.0)
        v = np.where(vis, fy * p_cam[:, 1] / np.where(vis, depth, 1.0) + cy, -1.0)
        vis &= (u >= 0) & (u < W) & (v >= 0) & (v < H)

        idx_vis = np.where(vis)[0]
        if len(idx_vis) == 0:
            continue

        kp = np.column_stack([u[idx_vis], v[idx_vis]]).astype(np.float64)
        per_image_obs[vi] = (idx_vis, kp)

        R_wc_pred = pred_c2w[vi, :3, :3].T
        t_wc_pred = -R_wc_pred @ pred_c2w[vi, :3, 3]

        for recon, R_wc, t_wc in [(src, R_wc_gt, t_wc_gt),
                                   (tgt, R_wc_pred, t_wc_pred)]:
            cam = pycolmap.Camera.create_from_model_name(image_id, "PINHOLE", fx, W, H)
            cam.params = np.array([fx, fy, cx, cy])
            recon.add_camera_with_trivial_rig(cam)

            img = pycolmap.Image(
                name=str(vi), keypoints=kp,
                camera_id=image_id, image_id=image_id)
            recon.add_image_with_trivial_frame(
                img,
                pycolmap.Rigid3d(
                    np.column_stack([R_wc, t_wc.reshape(3, 1)]).astype(np.float64)))

    pt_to_track: dict = defaultdict(list)
    for vi, (idx_vis, _) in per_image_obs.items():
        image_id = vi + 1
        for kp_idx, sub_pt_idx in enumerate(idx_vis):
            pt_to_track[int(sub_pt_idx)].append((image_id, kp_idx))

    for sub_pi, elements in pt_to_track.items():
        if len(elements) < 2:
            continue
        track = pycolmap.Track()
        for img_id, kp_idx in elements:
            track.add_element(img_id, kp_idx)
        xyz = pts_sub[sub_pi].astype(np.float64)
        src.add_point3D(xyz, track)
        tgt.add_point3D(xyz, track)

    if src.num_points3D() == 0:
        return None

    return pycolmap.align_reconstructions_via_reprojections(
        src, tgt, min_inlier_observations, max_reproj_error)


def _build_ba_arrays(preds, batch, sample_stride: int = 8,
                     points_world: np.ndarray | None = None,
                     b: int = 0):
    """
    Convert mapanything preds + batch into BA arrays.

    preds : list of per-view dicts produced by the model
        preds[i]["cam_quats"]  : (B, 4) xyzw, camera-to-world
        preds[i]["cam_trans"]  : (B, 3), camera-to-world
        preds[i]["pts3d"]      : (B, H, W, 3) world-frame dense pointmap

    batch : list of per-view dicts from the dataloader
        batch[i]["camera_intrinsics"] : (B, 3, 3)
        batch[i]["camera_pose"]       : (B, 4, 4) C2W — only used when
                                        points_world is provided

    sample_stride : sub-sample every `sample_stride`-th pixel to keep
                    the problem tractable. Ignored when points_world is given.

    points_world : optional (N, 3) float64 array of world-frame 3-D points.
        When provided, these replace the predicted pts3d as the BA structure
        and observations are derived from the GT camera poses in `batch`
        (batch[i]["camera_pose"]) so that reprojection residuals are
        meaningful.  When None (default), the predicted pts3d is used and
        observations are derived from the predicted cameras.

    b : batch element index to process.

    Returns
    -------
    cameras      : (N_cams, 10) float64  — world-to-camera angle-axis + intrinsics
    points       : (N_pts,   3) float64  — 3-D world positions
    observations : (N_obs,   2) float64  — (u, v) pixel coordinates
    cam_indices  : (N_obs,)    int32
    pt_indices   : (N_obs,)    int32
    meta         : dict with info needed to map results back to preds
    """
    n_views = len(preds)

    # --- cameras (always from predicted poses) ---
    cam_quats  = np.stack([preds[i]["cam_quats"][b].cpu().numpy()
                           for i in range(n_views)])          # (V, 4)
    cam_trans  = np.stack([preds[i]["cam_trans"][b].cpu().numpy()
                           for i in range(n_views)])          # (V, 3)
    K_arr      = np.stack([batch[i]["camera_intrinsics"][b].cpu().numpy()
                           for i in range(n_views)])          # (V, 3, 3)

    angle_axis = _quat_xyzw_to_angleaxis(cam_quats)           # (V, 3) W2C

    # W2C translation: t_wc = -R_wc @ t_cw  where t_cw = cam_trans (C2W)
    R_wc = Rotation.from_rotvec(angle_axis).as_matrix()       # (V, 3, 3)
    trans_wc = np.einsum("vij,vj->vi", R_wc, -cam_trans)      # (V, 3)

    fx = K_arr[:, 0, 0]   # (V,)
    fy = K_arr[:, 1, 1]   # (V,)
    cx = K_arr[:, 0, 2]   # (V,)
    cy = K_arr[:, 1, 2]   # (V,)

    cameras = np.column_stack([angle_axis, trans_wc,
                                fx, fy, cx, cy]).astype(np.float64)  # (V, 10)

    # Image dimensions for bounds checking
    img_shape = batch[0]["img"].shape  # (B, C, H, W)
    H, W = int(img_shape[-2]), int(img_shape[-1])

    # --- points ---
    if points_world is not None:
        pts_world_raw = np.asarray(points_world, dtype=np.float64)

        # Build C2W pose arrays for all views (GT and predicted).
        gt_c2w_all   = np.stack([batch[vi]["camera_pose"][b].cpu().numpy()
                                  for vi in range(n_views)])   # (V, 4, 4)
        pred_c2w_all = np.zeros((n_views, 4, 4)); pred_c2w_all[:, 3, 3] = 1.0
        pred_c2w_all[:, :3, :3] = R_wc.transpose(0, 2, 1)
        pred_c2w_all[:, :3,  3] = -(R_wc.transpose(0, 2, 1) @ trans_wc[:, :, None])[:, :, 0]

        sim3 = _pycolmap_align_sim3(pts_world_raw, gt_c2w_all, pred_c2w_all,
                                     K_arr, H, W)
        if sim3 is None:
            raise RuntimeError("pycolmap Sim3 alignment failed")
        pts_world = sim3 * pts_world_raw

        p_in_cam0 = (R_wc[0] @ pts_world.T).T + trans_wc[0]
        valid     = np.isfinite(pts_world).all(axis=1) & (p_in_cam0[:, 2] > 0)
        pts_world    = pts_world[valid]
        pts_world_gt = pts_world_raw[valid]   # GT-frame subset — used for GT-cam observations
        # No pixel-grid indices — point writeback to pts3d is skipped.
        sample_rows, sample_cols = None, None
    else:
        # Fall back to strided sub-sample of the predicted view-0 pointmap.
        pts3d_v0 = preds[0]["pts3d"][b].cpu().numpy()   # (H, W, 3)
        H, W, _ = pts3d_v0.shape

        rows = np.arange(0, H, sample_stride)
        cols = np.arange(0, W, sample_stride)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        rr = rr.ravel()
        cc = cc.ravel()

        pts_world = pts3d_v0[rr, cc]                    # (N, 3)

        # Filter invalid points (NaN, inf, or z <= 0 in view-0's camera frame).
        p_in_cam0 = (R_wc[0] @ pts_world.T).T + trans_wc[0]   # (N, 3) in cam-0
        valid = np.isfinite(pts_world).all(axis=1) & (p_in_cam0[:, 2] > 0)
        pts_world = pts_world[valid]
        sample_rows, sample_cols = rr[valid], cc[valid]

    points = pts_world.astype(np.float64)

    # --- observations ---
    # When using an external GT point cloud, project through GT cameras
    # (batch[i]["camera_pose"] is C2W) so reprojection residuals are
    # meaningful and BA actually refines the predicted camera poses.
    # When using the predicted point cloud, project through predicted cameras
    # (self-consistent initial state).
    obs_list, cam_idx_list, pt_idx_list = [], [], []
    for vi in range(n_views):
        if points_world is not None:
            R_obs = gt_c2w_all[vi, :3, :3].T
            t_obs = -R_obs @ gt_c2w_all[vi, :3, 3]
            p_cam = (R_obs @ pts_world_gt.T).T + t_obs
        else:
            R_obs = R_wc[vi]
            t_obs = trans_wc[vi]
            p_cam = (R_obs @ pts_world.T).T + t_obs   # (N, 3)
        depth = p_cam[:, 2]
        in_front = depth > 0

        u = np.where(in_front, fx[vi] * (p_cam[:, 0] / np.where(in_front, depth, 1)) + cx[vi], -1)
        v = np.where(in_front, fy[vi] * (p_cam[:, 1] / np.where(in_front, depth, 1)) + cy[vi], -1)

        in_bounds = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.where(in_bounds)[0]

        obs_list.append(np.column_stack([u[idx], v[idx]]))
        cam_idx_list.append(np.full(len(idx), vi, dtype=np.int32))
        pt_idx_list.append(idx.astype(np.int32))

    observations = np.concatenate(obs_list,   axis=0).astype(np.float64)
    cam_indices  = np.concatenate(cam_idx_list).astype(np.int32)
    pt_indices   = np.concatenate(pt_idx_list).astype(np.int32)

    meta = {
        "n_views": n_views,
        "H": H, "W": W,
        "sample_rows": sample_rows,
        "sample_cols": sample_cols,
        "cam_quats_orig": cam_quats,
        "cam_trans_orig": cam_trans,
        "R_wc": R_wc,
    }

    return cameras, points, observations, cam_indices, pt_indices, meta


def _update_preds(preds, cameras, points, meta, b: int = 0):
    """
    Write the refined camera params and 3-D points back into the preds dicts
    (in-place, for batch element b).
    """
    import torch

    n_views = meta["n_views"]
    for vi in range(n_views):
        aa  = cameras[vi, 0:3]   # angle-axis  W2C
        t   = cameras[vi, 3:6]   # translation W2C

        # Convert W2C → C2W for the quats/trans convention used by mapanything
        R_wc_vi = Rotation.from_rotvec(aa).as_matrix()  # (3,3)
        R_cw    = R_wc_vi.T
        t_cw    = -R_cw @ t                             # C2W translation

        quat_cw = Rotation.from_matrix(R_cw).as_quat()  # xyzw

        dev   = preds[vi]["cam_quats"].device
        dtype = preds[vi]["cam_quats"].dtype

        preds[vi]["cam_quats"][b] = torch.tensor(quat_cw, device=dev, dtype=dtype)
        preds[vi]["cam_trans"][b] = torch.tensor(t_cw,    device=dev, dtype=dtype)

    # Write refined world-frame 3-D positions back into view-0's pointmap.
    # Only the sampled pixels are updated; the rest remain from the model.
    # Skipped when an external point cloud was passed (no pixel-grid indices).
    if meta["sample_rows"] is not None:
        rr    = meta["sample_rows"]
        cc    = meta["sample_cols"]
        dev   = preds[0]["pts3d"].device
        dtype = preds[0]["pts3d"].dtype
        preds[0]["pts3d"][b, rr, cc] = torch.tensor(
            points, device=dev, dtype=dtype
        )

    return preds



def _triangulate_midpoint(C1, d1, C2, d2):
    """
    Vectorized midpoint triangulation of N ray pairs.

    For each pair of rays (C1+s*d1, C2+t*d2), finds the 3-D midpoint of the
    closest approach.  Parallel rays are flagged as invalid.

    Parameters
    ----------
    C1, C2 : (N, 3) camera centres in world coords
    d1, d2 : (N, 3) ray directions in world coords (need not be unit-length)

    Returns
    -------
    pts   : (N, 3)  triangulated world-frame points
    valid : (N,)    bool — False where rays are nearly parallel (|sin θ| < 1e-3)
    """
    d1 = d1 / np.linalg.norm(d1, axis=-1, keepdims=True)
    d2 = d2 / np.linalg.norm(d2, axis=-1, keepdims=True)

    w   = C2 - C1                                       # (N, 3)  C1→C2
    a   = np.einsum("ni,ni->n", d1, d2)                 # cos(angle between rays)
    denom = 1.0 - a * a                                 # sin²(angle)
    valid = denom > 1e-6

    safe = np.where(valid, denom, 1.0)
    wd1  = np.einsum("ni,ni->n", w, d1)
    wd2  = np.einsum("ni,ni->n", w, d2)

    # Parameters of the closest-approach points on each ray
    s = (wd1 - a * wd2) / safe   # along d1 from C1
    t = (a * wd1 - wd2) / safe   # along d2 from C2

    P1 = C1 + s[:, None] * d1
    P2 = C2 + t[:, None] * d2
    pts = (P1 + P2) * 0.5

    return pts, valid


def bundle_adjust(preds, batch, backend: str = "baseline",
                  sample_stride: int = 8,
                  fix_first_camera: bool = True,
                  huber_threshold: float = 2.0,
                  verbose: bool = False,
                  points_world=None):
    """
    Run bundle adjustment on mapanything model predictions and return
    updated preds with refined camera poses and (optionally) view-0 pointmap.

    Parameters
    ----------
    preds             : list of per-view prediction dicts from the model
    batch             : list of per-view batch dicts from the dataloader
    backend           : "baseline" or "custom"
    sample_stride     : pixel sub-sampling stride (larger = faster but less accurate).
                        Ignored when points_world is provided.
    fix_first_camera  : fix camera 0 constant to remove gauge freedom (custom only)
    huber_threshold   : Huber loss delta in pixels (custom only)
    verbose           : print Ceres solver output
    points_world      : optional world-frame 3-D points.  Either a single
                        (N, 3) float64 array applied to all batch elements, or
                        a list of per-element (N_b, 3) arrays.  When provided,
                        observations are derived from the GT camera poses in
                        batch, giving meaningful reprojection residuals.
                        pts3d in preds is NOT updated in this case.
                        When None (default), the predicted pts3d is used.

    Returns
    -------
    preds with cam_quats / cam_trans updated in-place for all batch elements.
    pts3d is also updated when points_world is None.
    """
    batch_size = preds[0]["cam_quats"].shape[0]

    for b in range(batch_size):
        if isinstance(points_world, list):
            pw = points_world[b]
        else:
            pw = points_world  # single array or None — same for all elements

        cameras, points, observations, cam_indices, pt_indices, meta = \
            _build_ba_arrays(preds, batch, sample_stride, pw, b=b)

        if backend == "baseline":
            final_cost, iters = run_bundle_adjustment_baseline(
                cameras, points, observations, cam_indices, pt_indices, verbose)
        elif backend == "custom":
            # When GT points are supplied as structure, freeze them to remove
            # the scale degree of freedom (see fix_points in custom_ba.cpp).
            fix_pts = pw is not None
            final_cost, iters = run_bundle_adjustment_custom(
                cameras, points, observations, cam_indices, pt_indices,
                fix_first_camera, huber_threshold, verbose, fix_pts)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'baseline' or 'custom'.")

        if verbose:
            print(f"[BA/{backend}] b={b}  final_cost={final_cost:.4f}  iters={iters}")

        _update_preds(preds, cameras, points, meta, b=b)

    return preds


# ---------------------------------------------------------------------------
# Sim3 alignment helper
# ---------------------------------------------------------------------------

def _align_preds_to_gt(preds, batch, b: int = 0):
    """
    Align predicted camera poses to GT via a Sim3 (scale+rotation+translation)
    Umeyama fit on the camera centres, then write back into preds in-place.

    preds[v]["cam_trans"][b] and preds[v]["cam_quats"][b] are C2W, xyzw.
    batch[v]["camera_pose_trans"][b] and ["camera_pose_quats"][b] are GT C2W.
    """
    import torch

    n_views = len(preds)
    P = np.stack([preds[v]["cam_trans"][b].cpu().float().numpy()
                  for v in range(n_views)], axis=0)   # (N,3) predicted centres
    G = np.stack([batch[v]["camera_pose_trans"][b].cpu().float().numpy()
                  for v in range(n_views)], axis=0)   # (N,3) GT centres

    # Umeyama: find s, R, t  s.t.  G ≈ s*R*P + t
    mu_P  = P.mean(0);  mu_G = G.mean(0)
    P_c   = P - mu_P;   G_c  = G - mu_G
    var_P = np.mean(np.sum(P_c ** 2, axis=1))
    if var_P < 1e-10:
        return   # all cameras coincide — skip
    H = (G_c.T @ P_c) / n_views
    U, S, Vt = np.linalg.svd(H)
    det_sign = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, float(det_sign)])
    R_align = U @ D @ Vt                          # (3,3) rotation
    s_align = float(np.sum(S * np.diag(D))) / var_P
    t_align = mu_G - s_align * R_align @ mu_P     # (3,)

    # Apply Sim3 to every view in the batch element
    for v in range(n_views):
        dev   = preds[v]["cam_trans"].device
        dtype = preds[v]["cam_trans"].dtype

        # Transform camera centre
        c_old = preds[v]["cam_trans"][b].cpu().float().numpy()
        c_new = s_align * R_align @ c_old + t_align
        preds[v]["cam_trans"][b] = torch.tensor(c_new, device=dev, dtype=dtype)

        # Rotate camera orientation (C2W rotation: R_cw_new = R_align @ R_cw_old)
        q_old  = preds[v]["cam_quats"][b].cpu().float().numpy()   # xyzw
        R_old  = Rotation.from_quat(q_old).as_matrix()
        R_new  = R_align @ R_old
        q_new  = Rotation.from_matrix(R_new).as_quat()            # xyzw
        preds[v]["cam_quats"][b] = torch.tensor(q_new, device=dev, dtype=dtype)


# ---------------------------------------------------------------------------
# MASt3R-feature-matching bundle adjustment
# ---------------------------------------------------------------------------

def mast3r_bundle_adjust(
        preds, batch, mast3r_model,
        device: str = "cuda",
        backend: str = "custom",
        conf_threshold: float = 0.5,
        subsample: int = 8,
        min_matches: int = 8,
        fix_first_camera: bool = True,
        huber_threshold: float = 2.0,
        verbose: bool = False,
        align_to_gt: bool = False):
    """
    Refine VGGT-predicted poses with Ceres BA, using MASt3R as feature matcher.

    Pipeline per batch element
    --------------------------
    1. For every ordered pair (i, j) of views, run MASt3R symmetric inference
       to extract dense descriptor maps.
    2. Extract reciprocal 2-D correspondences (xy_i, xy_j) via fast_reciprocal_NNs.
    3. Triangulate each matched pixel pair to a world-frame 3-D point using the
       predicted camera poses (midpoint method).
    4. Build the BA problem from all triangulated points and their pixel
       observations, then solve with the Ceres backend.
    5. Write refined cam_quats / cam_trans back into preds.

    Parameters
    ----------
    preds          : list of per-view prediction dicts (cam_quats, cam_trans, …)
    batch          : list of per-view batch dicts (img, camera_intrinsics, …)
    mast3r_model   : AsymmetricMASt3R model, or a wrapper that exposes .model
    device         : torch device string
    backend        : "baseline" or "custom" Ceres backend
    conf_threshold : minimum MASt3R correspondence confidence to keep
    subsample      : pixel sub-sampling stride passed to fast_reciprocal_NNs
    min_matches    : skip a pair if fewer valid matches survive triangulation
    fix_first_camera : fix camera 0 (custom backend only)
    huber_threshold  : Huber loss delta in pixels (custom backend only)
    verbose        : print per-pair match counts and Ceres summary
    align_to_gt    : after BA, align refined cameras to GT via Sim3 (needs pycolmap)

    Returns
    -------
    preds with cam_quats / cam_trans updated in-place for every batch element.
    pts3d is NOT modified (no dense pointmap available for matched structure).
    """
    import torch as _torch

    # Import only from fast_nn to avoid pulling in sparse_ga (which needs roma).
    try:
        from mast3r.fast_nn import fast_reciprocal_NNs, merge_corres
        from dust3r.utils.device import todevice
    except ImportError as e:
        raise ImportError(
            "mast3r_bundle_adjust requires the mast3r package on sys.path.\n"
            "Add /work/courses/3dv/team39/mast3r to sys.path before importing ba."
        ) from e

    def _symmetric_inference(model, img1, img2, dev):
        shape1 = _torch.from_numpy(img1["true_shape"]).to(dev, non_blocking=True)
        shape2 = _torch.from_numpy(img2["true_shape"]).to(dev, non_blocking=True)
        i1 = img1["img"].to(dev, non_blocking=True)
        i2 = img2["img"].to(dev, non_blocking=True)
        feat1, feat2, pos1, pos2 = model._encode_image_pairs(i1, i2, shape1, shape2)
        def _dec(f1, f2, p1, p2, s1, s2):
            d1, d2 = model._decoder(f1, p1, f2, p2)
            with _torch.cuda.amp.autocast(enabled=False):
                r1 = model._downstream_head(1, [t.float() for t in d1], s1)
                r2 = model._downstream_head(2, [t.float() for t in d2], s2)
            return r1, r2
        r11, r21 = _dec(feat1, feat2, pos1, pos2, shape1, shape2)
        r22, r12 = _dec(feat2, feat1, pos2, pos1, shape2, shape1)
        return r11, r21, r22, r12

    def _extract_correspondences(feats, qonfs, dev, subsample=8):
        feat11, feat21, feat22, feat12 = [f.float() for f in feats]
        qonf11, qonf21, qonf22, qonf12 = [q.float() for q in qonfs]
        H1, W1 = feat11.shape[:2]
        H2, W2 = feat22.shape[:2]
        opt = dict(device=dev, dist="dot", block_size=2**13)
        idx1, idx2, q1, q2 = [], [], [], []
        # Disable autocast: bruteforce_reciprocal_nns uses in-place index_put
        # on float32 accumulators; autocast would re-cast matmul output to
        # bfloat16, causing a dtype mismatch at that index_put.
        with _torch.autocast(device_type="cuda", enabled=False):
            for A, B, QA, QB in [(feat11, feat21, qonf11.cpu(), qonf21.cpu()),
                                  (feat12, feat22, qonf12.cpu(), qonf22.cpu())]:
                nn12 = fast_reciprocal_NNs(A, B, subsample_or_initxy1=subsample,
                                           ret_xy=False, **opt)
                nn21 = fast_reciprocal_NNs(B, A, subsample_or_initxy1=subsample,
                                           ret_xy=False, **opt)
                idx1.append(np.r_[nn12[0], nn21[1]])
                idx2.append(np.r_[nn12[1], nn21[0]])
                q1.append(QA.ravel()[idx1[-1]])
                q2.append(QB.ravel()[idx2[-1]])
        cat = np.concatenate
        xy1, xy2, sel = merge_corres(cat(idx1), cat(idx2),
                                     (H1, W1), (H2, W2),
                                     ret_xy=True, ret_index=True)
        corres = (xy1.copy(), xy2.copy(),
                  np.sqrt(cat(q1)[sel] * cat(q2)[sel]))
        return todevice(corres, dev)

    # Accept both MASt3RSGAWrapper (which holds .model) and raw AsymmetricMASt3R.
    raw_model = getattr(mast3r_model, "model", mast3r_model)
    raw_model.eval()

    batch_size = preds[0]["cam_quats"].shape[0]
    n_views    = len(preds)

    img_shape  = batch[0]["img"].shape           # (B, C, H, W)
    H_img, W_img = int(img_shape[-2]), int(img_shape[-1])

    # MASt3R requires image dims to be multiples of its patch size (16).
    # Resize to the nearest smaller multiple; scale correspondences back afterwards.
    PATCH = 16
    H_m = (H_img // PATCH) * PATCH
    W_m = (W_img // PATCH) * PATCH
    scale_u = W_img / W_m   # multiply MASt3R u-coords by this to get original u
    scale_v = H_img / H_m

    true_shape_np = np.tile(np.array([[H_m, W_m]], dtype=np.int32),
                            (batch_size, 1))

    import torch.nn.functional as _F

    def _resize_batch(imgs):
        if H_m == H_img and W_m == W_img:
            return imgs
        return _F.interpolate(imgs, size=(H_m, W_m), mode="bilinear",
                              align_corners=False)

    # ------------------------------------------------------------------
    # Step 1-2: Run MASt3R on every pair and cache per-batch-element
    #           correspondences.
    # ------------------------------------------------------------------
    all_pairs = [(i, j) for i in range(n_views) for j in range(i + 1, n_views)]

    # pair_corres[(i,j)] = list (length batch_size) of (xy1, xy2) numpy arrays.
    # xy arrays have shape (N, 2): [:, 0] = x (col = u), [:, 1] = y (row = v),
    # already scaled back to the original image coordinate system.
    pair_corres: dict = {}

    with _torch.no_grad():
        for (vi, vj) in all_pairs:
            img_i = {
                "img":        _resize_batch(batch[vi]["img"]),
                "true_shape": true_shape_np,
                "instance":   str(vi),
            }
            img_j = {
                "img":        _resize_batch(batch[vj]["img"]),
                "true_shape": true_shape_np,
                "instance":   str(vj),
            }
            # sym_res = (res_ii, res_ji, res_jj, res_ij)
            sym_res = _symmetric_inference(raw_model, img_i, img_j, device)

            corres_batch = []
            for b in range(batch_size):
                descs = [r["desc"][b]      for r in sym_res]
                qonfs = [r["desc_conf"][b] for r in sym_res]
                xy1, xy2, conf = _extract_correspondences(
                    descs, qonfs, device, subsample=subsample)

                def _to_np(x):
                    if isinstance(x, _torch.Tensor):
                        return x.cpu().numpy()
                    return np.asarray(x)

                xy1, xy2, conf = _to_np(xy1), _to_np(xy2), _to_np(conf)

                # Scale correspondences back to original image pixel coords
                if scale_u != 1.0 or scale_v != 1.0:
                    xy1 = xy1 * np.array([[scale_u, scale_v]])
                    xy2 = xy2 * np.array([[scale_u, scale_v]])

                if conf_threshold > 0 and len(conf) > 0:
                    mask = conf > conf_threshold
                    xy1, xy2 = xy1[mask], xy2[mask]

                corres_batch.append((xy1, xy2))

            pair_corres[(vi, vj)] = corres_batch

    # ------------------------------------------------------------------
    # Step 3-5: Per batch element — triangulate, build BA, solve, write back.
    # ------------------------------------------------------------------
    for b in range(batch_size):
        # --- Predicted cameras (W2C) for this element ---
        cam_quats = np.stack([preds[vi]["cam_quats"][b].cpu().numpy()
                              for vi in range(n_views)])   # (V, 4) xyzw, C2W
        cam_trans = np.stack([preds[vi]["cam_trans"][b].cpu().numpy()
                              for vi in range(n_views)])   # (V, 3) C2W
        K_arr     = np.stack([batch[vi]["camera_intrinsics"][b].cpu().numpy()
                              for vi in range(n_views)])   # (V, 3, 3)

        angle_axis = _quat_xyzw_to_angleaxis(cam_quats)    # (V, 3) W2C rotvec
        R_wc       = Rotation.from_rotvec(angle_axis).as_matrix()  # (V, 3, 3) W2C
        trans_wc   = np.einsum("vij,vj->vi", R_wc, -cam_trans)    # (V, 3) W2C t

        # Camera centres in world frame: C_v = -R_wc[v]^T t_wc[v]
        cam_centres = np.einsum("vij,vj->vi",
                                R_wc.transpose(0, 2, 1), -trans_wc)  # (V, 3)

        fx = K_arr[:, 0, 0]; fy = K_arr[:, 1, 1]
        cx = K_arr[:, 0, 2]; cy = K_arr[:, 1, 2]

        cameras = np.column_stack(
            [angle_axis, trans_wc, fx, fy, cx, cy]).astype(np.float64)  # (V, 10)

        # --- Accumulate BA structure from all pairs ---
        all_pts_list   = []
        all_obs_list   = []
        all_cam_i_list = []
        all_pt_i_list  = []
        pt_offset = 0

        for (vi, vj) in all_pairs:
            xy1, xy2 = pair_corres[(vi, vj)][b]
            if len(xy1) < min_matches:
                if verbose:
                    print(f"[mast3r_BA] b={b} pair ({vi},{vj}): "
                          f"only {len(xy1)} matches — skipping")
                continue

            # xy[:, 0] = u (col), xy[:, 1] = v (row)
            u1 = xy1[:, 0].astype(np.float64)
            v1 = xy1[:, 1].astype(np.float64)
            u2 = xy2[:, 0].astype(np.float64)
            v2 = xy2[:, 1].astype(np.float64)
            N  = len(u1)

            # --- Unproject to ray directions in world frame ---
            x1n = (u1 - cx[vi]) / fx[vi]
            y1n = (v1 - cy[vi]) / fy[vi]
            x2n = (u2 - cx[vj]) / fx[vj]
            y2n = (v2 - cy[vj]) / fy[vj]

            r1_cam = np.column_stack([x1n, y1n, np.ones(N)])   # (N, 3)
            r2_cam = np.column_stack([x2n, y2n, np.ones(N)])   # (N, 3)

            # R_wc maps world→cam; R_wc^T maps cam→world.
            # Row-vector convention: r_world = r_cam @ R_wc
            d1_world = r1_cam @ R_wc[vi]   # (N, 3)
            d2_world = r2_cam @ R_wc[vj]   # (N, 3)

            C1 = np.broadcast_to(cam_centres[[vi]], (N, 3)).copy()
            C2 = np.broadcast_to(cam_centres[[vj]], (N, 3)).copy()

            pts, tri_valid = _triangulate_midpoint(C1, d1_world, C2, d2_world)

            # --- Filter: point must be in front of both cameras ---
            p_cam_i = (R_wc[vi] @ pts.T).T + trans_wc[vi]
            p_cam_j = (R_wc[vj] @ pts.T).T + trans_wc[vj]
            depth_i = p_cam_i[:, 2]
            depth_j = p_cam_j[:, 2]
            in_front = tri_valid & (depth_i > 0) & (depth_j > 0)

            # --- Filter: reprojected observation must land within the image ---
            safe_di = np.where(depth_i > 0, depth_i, 1.0)
            safe_dj = np.where(depth_j > 0, depth_j, 1.0)
            u1r = fx[vi] * p_cam_i[:, 0] / safe_di + cx[vi]
            v1r = fy[vi] * p_cam_i[:, 1] / safe_di + cy[vi]
            u2r = fx[vj] * p_cam_j[:, 0] / safe_dj + cx[vj]
            v2r = fy[vj] * p_cam_j[:, 1] / safe_dj + cy[vj]
            in_bounds = (in_front
                         & (u1r >= 0) & (u1r < W_img)
                         & (v1r >= 0) & (v1r < H_img)
                         & (u2r >= 0) & (u2r < W_img)
                         & (v2r >= 0) & (v2r < H_img))

            pts_ok = pts[in_bounds]
            u1_ok  = u1[in_bounds]; v1_ok = v1[in_bounds]
            u2_ok  = u2[in_bounds]; v2_ok = v2[in_bounds]
            M = len(pts_ok)

            if verbose:
                print(f"[mast3r_BA] b={b} pair ({vi},{vj}): "
                      f"{N} matches → {M} valid after triangulation")

            if M < 4:
                continue

            all_pts_list.append(pts_ok)
            # Two observations per 3-D point: one in view vi, one in view vj
            all_obs_list.append(np.column_stack([u1_ok, v1_ok]))
            all_obs_list.append(np.column_stack([u2_ok, v2_ok]))
            all_cam_i_list.append(np.full(M, vi, dtype=np.int32))
            all_cam_i_list.append(np.full(M, vj, dtype=np.int32))
            pt_range = np.arange(pt_offset, pt_offset + M, dtype=np.int32)
            all_pt_i_list.append(pt_range)
            all_pt_i_list.append(pt_range)
            pt_offset += M

        if pt_offset == 0:
            if verbose:
                print(f"[mast3r_BA] b={b}: no valid matches from any pair — "
                      "skipping BA for this element")
            continue

        points       = np.concatenate(all_pts_list, axis=0).astype(np.float64)
        observations = np.concatenate(all_obs_list, axis=0).astype(np.float64)
        cam_indices  = np.concatenate(all_cam_i_list).astype(np.int32)
        pt_indices   = np.concatenate(all_pt_i_list).astype(np.int32)

        if verbose:
            print(f"[mast3r_BA] b={b}: total {len(points)} 3-D pts, "
                  f"{len(observations)} obs — running Ceres ({backend})")

        # --- Solve BA ---
        if backend == "baseline":
            final_cost, iters = run_bundle_adjustment_baseline(
                cameras, points, observations, cam_indices, pt_indices, verbose)
        elif backend == "custom":
            final_cost, iters = run_bundle_adjustment_custom(
                cameras, points, observations, cam_indices, pt_indices,
                fix_first_camera, huber_threshold, verbose,
                False)   # fix_points=False — refine structure jointly
        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'baseline' or 'custom'.")

        if verbose:
            print(f"[mast3r_BA/{backend}] b={b}  "
                  f"final_cost={final_cost:.4f}  iters={iters}")

        # Write refined poses back.  pts3d is untouched (no pixel-grid indices
        # for sparse matched structure).
        meta = {
            "n_views":        n_views,
            "H":              H_img,
            "W":              W_img,
            "sample_rows":    None,   # disables pts3d writeback in _update_preds
            "sample_cols":    None,
            "cam_quats_orig": cam_quats,
            "cam_trans_orig": cam_trans,
            "R_wc":           R_wc,
        }
        _update_preds(preds, cameras, points, meta, b=b)

        if align_to_gt:
            _align_preds_to_gt(preds, batch, b=b)

    return preds
