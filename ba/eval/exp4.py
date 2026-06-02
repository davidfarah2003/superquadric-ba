"""Exp4: does the NORMAL/tangent-plane residual (mode 5/6, the rotation lever)
beat the one-sided HINGE (mode 1) winner? Offline ranking; live validates top.

Mode 5 d_n = (F-1)/||gradF|| is a clean point-to-plane DISTANCE (no ||q||
amplification), so it likely needs a HIGHER lambda than the hinge's 15. Screen a
wide lambda for the normal modes; mode1 lam15 is the in-run reference (the live
29.6 winner). num_threads=3 (coarse/noisy ~0.5); take the top to live.
"""
import sys
import json
import argparse

sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")
sys.path.insert(0, "/work/courses/3dv/team39/ba/python")

import run_strategy as rs  # noqa: E402

BAR = 29.42


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--num_threads", type=int, default=3)
    args = ap.parse_args(argv)

    common = dict(max_points=None, function_tolerance=1e-6,
                  num_threads=args.num_threads, fix_first_camera=True,
                  huber_threshold=1.0, assoc_max_distance=0.0372,
                  surface_huber=2.749, n_outer=2, inner_iters=41, warmup=True)
    rows = []

    def em(mode, lam):
        p = dict(common); p.update(residual_mode=mode, lambda_surface=lam)
        auc = rs.score_strategy("em_reassoc", args.cache_dir, p,
                                jobs=args.jobs)["pose_auc_5"]
        rows.append((f"mode{mode}_lam{lam}", auc))
        print(f"  mode{mode}_lam{lam:<5} pose_auc_5={auc:.3f}"
              f"{'  >=BAR' if auc >= BAR else ''}", flush=True)

    print("=== reference: mode1 (hinge) lam15 (live=29.6) ===", flush=True)
    em(1, 15)
    print("=== mode5 (normal-outside, rotation lever) lambda screen ===", flush=True)
    for lam in [15, 40, 100, 250]:
        em(5, lam)
    print("=== mode6 (normal two-sided) lambda screen ===", flush=True)
    for lam in [40, 100]:
        em(6, lam)

    rows.sort(key=lambda t: -t[1])
    print(f"\n=== RANKED (offline; bar≈{BAR}, live offset ~+0.9) ===")
    for n, a in rows:
        print(f"  {a:8.3f}  {n}")
    print(json.dumps({"ranked": [{"name": n, "auc": a} for n, a in rows]}, indent=2))


if __name__ == "__main__":
    main()
