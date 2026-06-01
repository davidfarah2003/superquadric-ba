

```python
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
```


```python
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

```