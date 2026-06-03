"""Manhattan-frame viability check for the SQ rotation axes.

Question: do SUPERDEC superquadric orientations cluster onto a single shared
cube-aligned ("Manhattan") frame in ASE scenes? If yes, a direct camera-rotation
prior voted from SQ axes could attack the rotation-dominated pose_auc_5 ceiling.

We fold each SQ rotation by the 24 proper rotational symmetries of a box
(octahedral group) and measure the residual geodesic angle to an estimated
global frame R_m (chordal L2 rotation averaging with symmetry assignment).
A tight distribution near 0 deg => SQs share a Manhattan frame.

Light CPU only (numpy over 10 small npz). Safe on login node.
"""
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

NPZ_DIR = Path("/work/courses/3dv/team39/compose/data/output_npz")


def octahedral_group():
    """24 proper rotation matrices mapping a box onto itself."""
    mats = []
    # all signed permutations of axes with det = +1
    from itertools import permutations, product
    for perm in permutations(range(3)):
        for signs in product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            if abs(np.linalg.det(M) - 1.0) < 1e-9:
                mats.append(M)
    return np.array(mats)  # (24,3,3)


G = octahedral_group()


def angle_between(A, B):
    """Geodesic angle (deg) between rotation matrices A,B (broadcast last 2 dims)."""
    Rrel = np.einsum("...ij,...kj->...ik", A, B)  # A B^T
    tr = np.trace(Rrel, axis1=-2, axis2=-1)
    c = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def nearest_sym_angle(Rs, R_m):
    """For each R in Rs (N,3,3), min angle over the 24 syms of R_m@g to R."""
    cand = np.einsum("ij,gjk->gik", R_m, G)        # (24,3,3)  R_m g
    # angle between each R (N) and each cand (24): build (N,24)
    Rrel = np.einsum("nij,gkj->ngik", Rs, cand)    # Rs cand^T -> (N,24,3,3)
    tr = np.trace(Rrel, axis1=-2, axis2=-1)        # (N,24)
    c = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    ang = np.degrees(np.arccos(c))
    g_idx = np.argmin(ang, axis=1)
    return ang[np.arange(len(Rs)), g_idx], g_idx


def estimate_manhattan(Rs, iters=10):
    """Chordal L2 rotation averaging with octahedral symmetry assignment."""
    R_m = np.eye(3)
    for _ in range(iters):
        _, g_idx = nearest_sym_angle(Rs, R_m)
        # targets: R_i @ g_idx^{-1}  (should approximate R_m)
        targets = np.einsum("nij,njk->nik", Rs, np.transpose(G[g_idx], (0, 2, 1)))
        M = targets.sum(axis=0)
        U, _, Vt = np.linalg.svd(M)
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        R_m = U @ D @ Vt
    res, _ = nearest_sym_angle(Rs, R_m)
    return R_m, res


def load_rot_and_aniso(npz_path, exist_thr=0.5):
    npz = np.load(str(npz_path), allow_pickle=True)
    exist = npz["exist"][..., 0]
    active = exist > exist_thr
    oi, pi = np.where(active)
    R = npz["rotation"][oi, pi].astype(np.float64)     # (K,3,3)
    scale = np.abs(npz["scale"][oi, pi].astype(np.float64))
    smax = scale.max(1); smin = scale.min(1)
    aniso = smax / np.clip(smin, 1e-9, None)           # aspect ratio
    return R, aniso, scale


print(f"{'scene':<14}{'Nsq':>5}{'res:identity-med':>18}{'res:R_m-med':>14}"
      f"{'<10deg%':>10}{'<15deg%':>10}{'aniso-med':>10}")
all_res_id, all_res_m, all_aniso = [], [], []
for k in range(10):
    p = NPZ_DIR / f"ase_scene_{k}.npz"
    if not p.exists():
        continue
    R, aniso, scale = load_rot_and_aniso(p)
    # focus on anisotropic SQs (axis is meaningful); aspect>1.5
    sel = aniso > 1.5
    Rs = R[sel] if sel.sum() >= 5 else R
    res_id, _ = nearest_sym_angle(Rs, np.eye(3))
    R_m, res_m = estimate_manhattan(Rs)
    all_res_id.append(res_id); all_res_m.append(res_m); all_aniso.append(aniso[sel])
    print(f"{'scene_'+str(k):<14}{len(Rs):>5}{np.median(res_id):>18.2f}"
          f"{np.median(res_m):>14.2f}{100*np.mean(res_m<10):>10.1f}"
          f"{100*np.mean(res_m<15):>10.1f}{np.median(aniso[sel]):>10.2f}")

allid = np.concatenate(all_res_id); allm = np.concatenate(all_res_m)
print("\n=== AGGREGATE (anisotropic SQs, per-scene Manhattan frame) ===")
print(f"residual to identity-frame:  median={np.median(allid):.2f}deg  "
      f"p25={np.percentile(allid,25):.2f}  p75={np.percentile(allid,75):.2f}")
print(f"residual to fitted R_m:      median={np.median(allm):.2f}deg  "
      f"p25={np.percentile(allm,25):.2f}  p75={np.percentile(allm,75):.2f}")
print(f"fraction within 10deg of R_m: {100*np.mean(allm<10):.1f}%")
print(f"fraction within 15deg of R_m: {100*np.mean(allm<15):.1f}%")
print(f"fraction within 20deg of R_m: {100*np.mean(allm<20):.1f}%")
print("\nINTERPRETATION: tight concentration (<~10-15deg, high %) => SQ axes")
print("vote a clean Manhattan frame => a camera-rotation prior has real signal.")
print("Spread-out (median>25deg, low %) => no shared frame => lever is dead.")
