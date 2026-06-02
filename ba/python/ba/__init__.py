"""
Bundle adjustment package — wraps Ceres-based pybind11 extensions.

Three backends:
    mast3r_ba_core      —  MASt3R BA (reprojection only)
    mast3r_sq_ba_core   —  MASt3R BA + superquadric surface residual
    vggt_sq_ba_core     —  VGGT BA + superquadric surface residual (WIP)

Camera convention (10 params per camera, float64):
    [0:3]  angle-axis rotation  (world-to-camera, Rodrigues)
    [3:6]  translation          (world-to-camera)
    [6]    fx
    [7]    fy
    [8]    cx
    [9]    cy

Entry point:
    from ba import mast3r_bundle_adjust
    preds = mast3r_bundle_adjust(preds, batch, mast3r)
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


# Add the directory containing the compiled .so files to sys.path
_pkg_dir = Path(__file__).parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

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
# Low-level interface
# ---------------------------------------------------------------------------

def run_bundle_adjustment_mast3r(cameras, points, observations,
                                 cam_indices, pt_indices, verbose=False):
    """Run MASt3R Ceres BA. All arrays are modified **in place**."""
    core = _load_core("mast3r_ba_core")
    return core.run_bundle_adjustment(cameras, points, observations,
                                      cam_indices, pt_indices, verbose)


def run_bundle_adjustment_mast3r_sq(cameras, points, observations,
                                    cam_indices, pt_indices,
                                    fix_first_camera=True,
                                    huber_threshold=2.0,
                                    verbose=False,
                                    fix_points=False,
                                    sq_params=None,
                                    point_to_sq=None,
                                    lambda_surface=0.0,
                                    surface_huber=0.0,
                                    residual_mode=0,
                                    point_weights=None,
                                    max_num_iterations=200,
                                    num_threads=4,
                                    function_tolerance=1e-6,
                                    refine_sq=False,
                                    sq_anchor_weight=10.0):
    """Run MASt3R + superquadric Ceres BA. All arrays are modified **in place**.

    Surface-residual mode is engaged when ``lambda_surface > 0`` and both
    ``sq_params`` (K, 11) and ``point_to_sq`` (num_points,) are supplied.
    See ``ba.superdec.pack_for_ceres`` for the expected sq_params layout.

    ``max_num_iterations`` / ``num_threads`` / ``function_tolerance`` expose the
    Ceres solver budget (defaults match the live benchmark; lower iterations or
    a looser tolerance trade a little accuracy for large speed-ups in sweeps).

    ``refine_sq`` (default False -> identical to the frozen-pose behaviour)
    lets Ceres optimise each used SQ's rigid pose ([aa(3), t(3)]) jointly with
    cameras+points. ``sq_anchor_weight`` is the stiffness of a soft prior
    pulling each refined SQ pose back to its SUPERDEC init (keeps the gauge
    well-posed; ignored when refine_sq is False).
    """
    core = _load_core("mast3r_sq_ba_core")
    return core.run_bundle_adjustment(cameras, points, observations,
                                      cam_indices, pt_indices,
                                      fix_first_camera, huber_threshold,
                                      verbose, fix_points,
                                      sq_params, point_to_sq,
                                      lambda_surface, surface_huber,
                                      residual_mode, point_weights,
                                      max_num_iterations, num_threads,
                                      function_tolerance,
                                      refine_sq, sq_anchor_weight)


def run_bundle_adjustment_vggt_sq(cameras, points, observations,
                                  cam_indices, pt_indices,
                                  fix_first_camera=True,
                                  huber_threshold=2.0,
                                  verbose=False,
                                  fix_points=False,
                                  sq_params=None,
                                  point_to_sq=None,
                                  lambda_surface=0.0,
                                  surface_huber=0.0):
    """Run VGGT + superquadric Ceres BA. All arrays are modified **in place**."""
    raise NotImplementedError("vggt_sq_ba_core is not yet implemented")


def _quat_xyzw_to_angleaxis(quats_xyzw: np.ndarray) -> np.ndarray:
    """(N, 4) xyzw quaternions → (N, 3) angle-axis."""
    return Rotation.from_quat(quats_xyzw).inv().as_rotvec()


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

    s = (wd1 - a * wd2) / safe   # along d1 from C1
    t = (a * wd1 - wd2) / safe   # along d2 from C2

    P1 = C1 + s[:, None] * d1
    P2 = C2 + t[:, None] * d2
    pts = (P1 + P2) * 0.5

    return pts, valid


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
        backend: str = "mast3r_sq",
        conf_threshold: float = 0.5,
        subsample: int = 8,
        min_matches: int = 8,
        fix_first_camera: bool = True,
        huber_threshold: float = 2.0,
        verbose: bool = False,
        align_to_gt: bool = False,
        superdec_npz_path: str | None = None,
        lambda_surface: float = 0.0,
        surface_huber: float = 0.0,
        assoc_max_distance: float = 0.15,
        em_outer: int = 1,
        em_inner_iters: int = 200,
        em_warmup: bool = False,
        residual_mode: int = 0,
        filter_max_aspect: float = 0.0,
        filter_min_axis: float = 0.01,
        filter_max_axis: float = 2.0,
        refine_sq: bool = False,
        sq_anchor_weight: float = 10.0,
        num_threads: int = 4):
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
    backend        : "mast3r" or "mast3r_sq" Ceres backend
    conf_threshold : minimum MASt3R correspondence confidence to keep
    subsample      : pixel sub-sampling stride passed to fast_reciprocal_NNs
    min_matches    : skip a pair if fewer valid matches survive triangulation
    fix_first_camera : fix camera 0 (custom backend only)
    huber_threshold  : Huber loss delta in pixels (custom backend only)
    verbose        : print per-pair match counts and Ceres summary
    align_to_gt    : after BA, align refined cameras to GT via Sim3 (needs pycolmap)

    Surface-augmented BA (optional):
    superdec_npz_path  : path to a SUPERDEC scene NPZ. When set together with
                         lambda_surface > 0, each triangulated 3-D point is
                         associated to its nearest SQ surface (within
                         assoc_max_distance) and a radial-distance residual is
                         added to the Ceres problem.
    lambda_surface     : pixels-per-meter weight on the surface residual.
    surface_huber      : Huber delta in pixel-equivalent units (<=0 disables).
    assoc_max_distance : drop points whose nearest-SQ radial distance exceeds
                         this threshold (meters). Default 0.15 m.

    EM-style iterated re-association (surface BA only):
    em_outer           : number of outer EM iterations. <=1 keeps the original
                         one-shot behaviour (associate once, solve once). >1
                         alternates E-step (re-assign the CURRENT moving points
                         to their nearest SQ) and M-step (a short surface solve),
                         so points that drift toward the wrong SQ get re-pointed
                         before they accumulate a wrong pull. Tuned best: 2.
    em_inner_iters     : Ceres max iterations per inner (M-step) solve. Tuned: 41.
    em_warmup          : if True, run one reprojection-only solve (lambda=0)
                         before the first association, so the surface term sees
                         cleaner structure. Tuned best: True.

    Returns
    -------
    preds with cam_quats / cam_trans updated in-place for every batch element.
    pts3d is NOT modified (no dense pointmap available for matched structure).
    """
    import torch as _torch
    use_surface = (superdec_npz_path is not None) and (lambda_surface > 0.0)
    if use_surface and backend != "mast3r_sq":
        raise ValueError("Surface residual requires backend='mast3r_sq'.")
    if use_surface:
        from .superdec import (load_scene, transform_sqs, invert_sim3,
                               umeyama_sim3_pred_to_world,
                               assign_points_to_sqs, pack_for_ceres,
                               filter_degenerate_sqs)
        sq_world = load_scene(superdec_npz_path)
        if verbose:
            print(f"[mast3r_BA/surface] loaded {len(sq_world['scale'])} SQs from "
                  f"{superdec_npz_path}")

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

        # --- Optional: bring SQs from world frame into predicted-world frame
        # via Sim3 fitted from GT camera centres -> predicted camera centres,
        # then build (point -> SQ) associations on the BA point cloud. ---
        sq_params_arg = None
        point_to_sq_arg = None
        sq_pred_arg = None       # SQs in predicted frame, kept for EM re-assoc
        if use_surface:
            gt_centres = np.stack([batch[v]["camera_pose"][b].cpu().numpy()[:3, 3]
                                   for v in range(n_views)], axis=0)
            sim3_p2g = umeyama_sim3_pred_to_world(cam_centres, gt_centres)
            if sim3_p2g is None:
                if verbose:
                    print(f"[mast3r_BA/surface] b={b}: degenerate Sim3 — "
                          "skipping surface term for this element")
            else:
                sim3_g2p = invert_sim3(sim3_p2g)
                sq_pred = transform_sqs(sq_world, sim3_g2p)
                if filter_max_aspect and filter_max_aspect > 0.0:
                    sq_pred = filter_degenerate_sqs(
                        sq_pred, filter_min_axis, filter_max_axis, filter_max_aspect)
                sq_pred_arg = sq_pred
                point_to_sq_arg, dists = assign_points_to_sqs(
                    points, sq_pred, max_distance=assoc_max_distance)
                sq_params_arg, _ = pack_for_ceres(sq_pred)
                if verbose:
                    n_assigned = int((point_to_sq_arg >= 0).sum())
                    print(f"[mast3r_BA/surface] b={b}: assigned "
                          f"{n_assigned}/{len(points)} points to SQs "
                          f"(median dist {np.median(dists)*1000:.1f} mm, "
                          f"threshold {assoc_max_distance*1000:.0f} mm)")

        # --- Optional: dump parameter-independent BA inputs for offline eval ---
        # Guarded by BA_DUMP_DIR. Captures the PRE-BA state (cameras via .copy()
        # before Ceres mutates them in place). Surface association (point_to_sq /
        # sq_params) is deliberately NOT dumped: it depends on assoc_max_distance
        # and is recomputed offline. Behavior is unchanged when BA_DUMP_DIR unset.
        _ba_dump_dir = os.environ.get("BA_DUMP_DIR")
        if _ba_dump_dir:
            scene_label = str(batch[0]["label"][0])
            gt_poses = np.stack(
                [batch[v]["camera_pose"][b].cpu().numpy() for v in range(n_views)],
                axis=0).astype(np.float64)                      # (V, 4, 4) C2W
            gt_centres_dump = gt_poses[:, :3, 3].astype(np.float64)  # (V, 3)
            dump_kw = dict(
                cameras=cameras.copy(),                         # pre-BA (V, 10)
                points=points,                                  # (M, 3)
                observations=observations,                      # (K, 2)
                cam_indices=cam_indices,                        # (K,) int32
                pt_indices=pt_indices,                          # (K,) int32
                cam_centres=cam_centres,                        # (V, 3) predicted
                gt_centres=gt_centres_dump,                     # (V, 3) GT
                gt_poses=gt_poses,                              # (V, 4, 4)
                scene_label=scene_label,
                superdec_npz_path=str(superdec_npz_path),
            )
            try:
                dump_kw["gt_quats"] = np.stack(
                    [batch[v]["camera_pose_quats"][b].cpu().numpy()
                     for v in range(n_views)], axis=0).astype(np.float64)  # (V,4) xyzw
                dump_kw["gt_trans"] = np.stack(
                    [batch[v]["camera_pose_trans"][b].cpu().numpy()
                     for v in range(n_views)], axis=0).astype(np.float64)  # (V,3)
            except (KeyError, TypeError):
                pass
            os.makedirs(_ba_dump_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(_ba_dump_dir, f"{scene_label}.npz"), **dump_kw)
            if verbose:
                print(f"[mast3r_BA/dump] b={b}: wrote "
                      f"{os.path.join(_ba_dump_dir, scene_label + '.npz')}")

        # --- Solve BA ---
        em_on = (backend == "mast3r_sq" and use_surface
                 and sq_pred_arg is not None and int(em_outer) > 1)
        if backend == "mast3r":
            final_cost, iters = run_bundle_adjustment_mast3r(
                cameras, points, observations, cam_indices, pt_indices, verbose)
        elif backend == "mast3r_sq" and not em_on:
            final_cost, iters = run_bundle_adjustment_mast3r_sq(
                cameras, points, observations, cam_indices, pt_indices,
                fix_first_camera=fix_first_camera,
                huber_threshold=huber_threshold,
                verbose=verbose,
                fix_points=False,    # refine structure jointly
                sq_params=sq_params_arg,
                point_to_sq=point_to_sq_arg,
                lambda_surface=lambda_surface if use_surface else 0.0,
                surface_huber=surface_huber,
                residual_mode=residual_mode,
                num_threads=num_threads)
        elif backend == "mast3r_sq" and em_on:
            # EM-style iterated re-association: alternate E-step (re-assign the
            # CURRENT moving points to their nearest SQ) with M-step (a short
            # surface solve that moves cameras+points in place). Optionally warm
            # up with a reprojection-only solve so the first association sees
            # cleaner structure. Ceres mutates `cameras`/`points` in place, so
            # each E-step re-associates against the updated geometry.
            if em_warmup:
                final_cost, iters = run_bundle_adjustment_mast3r_sq(
                    cameras, points, observations, cam_indices, pt_indices,
                    fix_first_camera=fix_first_camera,
                    huber_threshold=huber_threshold, verbose=verbose,
                    fix_points=False, sq_params=None, point_to_sq=None,
                    lambda_surface=0.0, surface_huber=surface_huber,
                    max_num_iterations=int(em_inner_iters),
                    num_threads=num_threads)
            for _it in range(int(em_outer)):
                point_to_sq_arg, _d = assign_points_to_sqs(
                    points, sq_pred_arg, max_distance=assoc_max_distance)
                point_to_sq_arg = np.ascontiguousarray(point_to_sq_arg, np.int32)
                final_cost, iters = run_bundle_adjustment_mast3r_sq(
                    cameras, points, observations, cam_indices, pt_indices,
                    fix_first_camera=fix_first_camera,
                    huber_threshold=huber_threshold, verbose=verbose,
                    fix_points=False, sq_params=sq_params_arg,
                    point_to_sq=point_to_sq_arg,
                    lambda_surface=lambda_surface, surface_huber=surface_huber,
                    residual_mode=residual_mode,
                    refine_sq=refine_sq, sq_anchor_weight=sq_anchor_weight,
                    max_num_iterations=int(em_inner_iters),
                    num_threads=num_threads)
                if verbose:
                    n_assigned = int((point_to_sq_arg >= 0).sum())
                    print(f"[mast3r_BA/surface] b={b} EM iter {_it+1}/{em_outer}: "
                          f"{n_assigned}/{len(points)} assigned  "
                          f"final_cost={final_cost:.4f}")
        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'mast3r' or 'mast3r_sq'.")

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


# ---------------------------------------------------------------------------
# VGGT + superquadric BA
# ---------------------------------------------------------------------------

def vggt_sq_bundle_adjust(
        preds, batch,
        superdec_npz_path: str,
        device: str = "cuda",
        fix_first_camera: bool = True,
        huber_threshold: float = 2.0,
        verbose: bool = False,
        lambda_surface: float = 1.0,
        surface_huber: float = 0.0,
        assoc_max_distance: float = 0.15):
    """NOT YET IMPLEMENTED — raises NotImplementedError."""
    raise NotImplementedError(
        "vggt_sq_bundle_adjust is not yet implemented. "
        "Implement vggt_sq_ba_core in ba/src/vggt_sq_ba.cpp and wire it up here."
    )
