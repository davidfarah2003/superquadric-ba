"""Score every structural strategy + references on the proxy, print a ranked table.

References (via offline_eval, same proxy settings):
    regular_ba : lambda_surface=0           (plain reprojection BA)
    surface    : lambda_surface=50, assoc=0.15  (the current one-shot surface BA)

Strategies: every strategies/<name>.py (except baseline is included as a control).
All scored at the same max_points / max_iterations so numbers are comparable.
The bar to beat is the proxy regular_ba number (apples-to-apples at this proxy);
winners get re-validated at full fidelity on the real Slurm benchmark.

Run on a Slurm node (NOT login):
  sbatch --account=3dv -J ba-eval --time=01:30:00 \
    --output=compose/logs/%j.out --wrap \
    "source envs/3dv/bin/activate; cd /work/courses/3dv/team39; \
     python ba/eval/eval_all.py --cache_dir compose/data/ba_cache --jobs 2"
"""
import os
import sys
import glob
import json
import argparse
import importlib

sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")
sys.path.insert(0, "/work/courses/3dv/team39/ba/python")

import numpy as np  # noqa: E402
import offline_eval as oe  # noqa: E402
import run_strategy as rs  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--max_points", type=int, default=5000)
    ap.add_argument("--max_iterations", type=int, default=50)
    ap.add_argument("--lambda_surface", type=float, default=50.0)
    ap.add_argument("--assoc", type=float, default=0.15)
    ap.add_argument("--num_threads", type=int, default=1)
    args = ap.parse_args(argv)

    # common params superset; each strategy/refine reads the keys it needs
    P = dict(lambda_surface=args.lambda_surface, lambda_max=args.lambda_surface,
             assoc_max_distance=args.assoc, surface_huber=0.0, huber_threshold=2.0,
             max_points=args.max_points, max_iterations=args.max_iterations,
             inner_iters=args.max_iterations, function_tolerance=1e-3,
             num_threads=args.num_threads)

    rows = []

    # --- references via offline_eval ---
    for name, lam in [("regular_ba", 0.0), ("surface", args.lambda_surface)]:
        try:
            r = oe.score({"lambda_surface": lam, "surface_huber": 0.0,
                          "assoc_max_distance": args.assoc,
                          "max_points": args.max_points,
                          "max_iterations": args.max_iterations,
                          "function_tolerance": 1e-3,
                          "num_threads": args.num_threads}, args.cache_dir, jobs=args.jobs)
            rows.append((name, r["pose_auc_5"], r.get("n_scenes")))
            print(f"[ref ] {name:16s} pose_auc_5={r['pose_auc_5']:.3f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[ref ] {name:16s} ERROR {e}", flush=True)

    # --- strategies ---
    strat_dir = "/work/courses/3dv/team39/ba/eval/strategies"
    names = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(strat_dir, "*.py"))
                   if not os.path.basename(p).startswith("__"))
    for name in names:
        try:
            importlib.import_module(f"strategies.{name}")  # validate import
            r = rs.score_strategy(name, args.cache_dir, P, jobs=args.jobs)
            rows.append((name, r["pose_auc_5"], r.get("n_scenes")))
            print(f"[strat] {name:16s} pose_auc_5={r['pose_auc_5']:.3f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[strat] {name:16s} ERROR {e}", flush=True)

    rows.sort(key=lambda t: -t[1])
    print("\n=== RANKED (pose_auc_5, higher=better; beat regular_ba) ===")
    for name, auc, n in rows:
        print(f"  {auc:8.3f}   {name}")
    print(json.dumps({"ranked": [{"name": n, "pose_auc_5": a} for n, a, _ in rows],
                      "proxy": {"max_points": args.max_points,
                                "max_iterations": args.max_iterations}}, indent=2))


if __name__ == "__main__":
    main()
