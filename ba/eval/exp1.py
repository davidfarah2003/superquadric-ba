"""Exp1: separate the BACKEND-robustness gap from the SURFACE term, full fidelity.

Faithful eval revealed the "regular BA" bar (29.42) is the *mast3r* backend with
HuberLoss(1.0), while all surface work uses the *mast3r_sq* backend whose plain
BA at huber=2.0 is only 28.93. So two questions, answered here at full fidelity
(no subsample, live iters/tolerance), scene-parallel:

  Q1  Plain mast3r_sq (lambda=0) huber sweep -> how high does the backend go?
      Does huber~1.0 recover ~29.42 (i.e. the gap IS just the robust kernel)?
  Q2  Does the one-sided HINGE residual (mode 1/4) on the EM machinery ADD value
      over radial (mode 0) and over the plain ceiling -> beat 29.42 with SQs?

Each row prints pose_auc_5; the bar is 29.42 (mast3r live). Run on a 2080ti node
(36 cores) via --cpus-per-gpu=32.
"""
import sys
import json
import argparse

sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")
sys.path.insert(0, "/work/courses/3dv/team39/ba/python")

import offline_eval as oe  # noqa: E402
import run_strategy as rs  # noqa: E402

BAR = 29.42


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--num_threads", type=int, default=3)
    ap.add_argument("--ftol", type=float, default=1e-6)
    args = ap.parse_args(argv)

    common = dict(max_points=None, max_iterations=200,
                  function_tolerance=args.ftol, num_threads=args.num_threads)
    rows = []

    def plain(huber):
        r = oe.score({"lambda_surface": 0.0, "surface_huber": 0.0,
                      "assoc_max_distance": 0.15, "huber_threshold": huber,
                      **common}, args.cache_dir, jobs=args.jobs)
        return r["pose_auc_5"]

    # Q1 — plain mast3r_sq backend ceiling vs the robust-kernel width.
    print("=== Q1: plain mast3r_sq (lambda=0) huber sweep ===", flush=True)
    for h in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        auc = plain(h)
        rows.append((f"plain_huber{h}", auc))
        print(f"  plain  huber={h:<4} pose_auc_5={auc:.3f}"
              f"{'  >=BAR' if auc >= BAR else ''}", flush=True)

    # Faithfulness anchor: em_reassoc at the EXACT live config (mode0, huber0.738)
    # must reproduce the live EM number (28.93); plain_huber2.0 above must
    # reproduce 28.93 too. If both anchors hold, the offline ranking is trusted.
    def em(mode, lam, huber):
        p = dict(assoc_max_distance=0.0372, surface_huber=2.749,
                 huber_threshold=huber, n_outer=2, inner_iters=41, warmup=True,
                 fix_first_camera=True, residual_mode=mode, lambda_surface=lam,
                 **common)
        return rs.score_strategy("em_reassoc", args.cache_dir, p,
                                 jobs=args.jobs)["pose_auc_5"]

    print("\n=== Faithfulness anchor: em mode0 @ live config (expect ~28.93) ===",
          flush=True)
    a = em(0, 3.347, 0.738)
    rows.append(("ANCHOR_em_live_mode0_h0.738", a))
    print(f"  ANCHOR em mode0 huber0.738 lam3.347 = {a:.3f}", flush=True)

    # Q2 — EM machinery, hinge vs radial, at the mast3r-backend robustness
    # (huber=1.0) and a tight huber (0.75). Does the one-sided hinge ADD value?
    print("\n=== Q2: em_reassoc residual_mode x lambda x huber ===", flush=True)
    for huber in [1.0, 0.75]:
        for mode in [0, 1, 4]:
            for lam in ([3.347] if mode == 0 else [3.347, 15.0]):
                auc = em(mode, lam, huber)
                name = f"em_mode{mode}_lam{lam}_h{huber}"
                rows.append((name, auc))
                print(f"  {name:26s} pose_auc_5={auc:.3f}"
                      f"{'  >=BAR' if auc >= BAR else ''}", flush=True)

    rows.sort(key=lambda t: -t[1])
    print(f"\n=== RANKED (bar = mast3r {BAR}) ===")
    for n, a in rows:
        print(f"  {a:8.3f}  {n}{'  *BEATS*' if a > BAR else ''}")
    print(json.dumps({"bar": BAR,
                      "ranked": [{"name": n, "auc": a} for n, a in rows]},
                     indent=2))


if __name__ == "__main__":
    main()
