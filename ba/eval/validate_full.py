"""Full-fidelity validation of a tuned strategy vs the references.

The proxy (max_points=5000, max_iter=50, ftol=1e-3) is fast but biased: it
reproduces surface=19.42 exactly yet *understates* regular_ba (proxy 27.56 vs
full 29.56). So a proxy win must be re-checked with NO subsampling, the live
iteration budget, and a tight tolerance before we believe it.

This re-scores, all at full fidelity (max_points=None, max_iter=200, ftol=1e-6):
    regular_ba   (lambda_surface=0)            -- calibration: must land ~29.56
    surface      (one-shot, lambda/assoc)      -- calibration: must land ~19.42
    <strategy>   (tuned params from BO)        -- the candidate

Run on a Slurm node (NOT login). Example:
  sbatch --account=3dv -J ba-valid --time=04:00:00 --output=compose/logs/%j.out \
    --wrap "/work/courses/3dv/team39/envs/3dv/bin/python \
      /work/courses/3dv/team39/ba/eval/validate_full.py \
      --cache_dir /work/courses/3dv/team39/compose/data/ba_cache \
      --strategy em_reassoc --jobs 2 --num_threads 1 \
      --params '{...best_params...}'"
"""
import sys
import json
import argparse

sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")
sys.path.insert(0, "/work/courses/3dv/team39/ba/python")

import offline_eval as oe  # noqa: E402
import run_strategy as rs  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--strategy", default="em_reassoc")
    ap.add_argument("--params", default="{}", help="tuned strategy params (JSON)")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--num_threads", type=int, default=1)
    ap.add_argument("--max_iterations", type=int, default=200)
    ap.add_argument("--function_tolerance", type=float, default=1e-6)
    # surface one-shot reference settings (matches the live surface benchmark)
    ap.add_argument("--ref_lambda", type=float, default=50.0)
    ap.add_argument("--ref_assoc", type=float, default=0.15)
    args = ap.parse_args(argv)

    FULL = dict(max_points=None, max_iterations=args.max_iterations,
                function_tolerance=args.function_tolerance,
                num_threads=args.num_threads)

    rows = []

    # --- references (offline_eval, lambda toggles surface on/off) ------------
    for name, lam, assoc in [("regular_ba", 0.0, args.ref_assoc),
                             ("surface", args.ref_lambda, args.ref_assoc)]:
        r = oe.score({"lambda_surface": lam, "surface_huber": 0.0,
                      "assoc_max_distance": assoc, **FULL},
                     args.cache_dir, jobs=args.jobs)
        rows.append((name, r["pose_auc_5"]))
        print(f"[ref ] {name:14s} pose_auc_5={r['pose_auc_5']:.3f}  "
              f"per_scene={json.dumps(r.get('per_scene', {}))}", flush=True)

    # --- candidate strategy at full fidelity --------------------------------
    p = json.loads(args.params)
    p.pop("max_points", None)                  # no subsampling
    p["function_tolerance"] = args.function_tolerance
    p["num_threads"] = args.num_threads
    p.setdefault("inner_iters", args.max_iterations)
    r = rs.score_strategy(args.strategy, args.cache_dir, p, jobs=args.jobs)
    rows.append((args.strategy, r["pose_auc_5"]))
    print(f"[cand] {args.strategy:14s} pose_auc_5={r['pose_auc_5']:.3f}  "
          f"per_scene={json.dumps(r.get('per_scene', {}))}", flush=True)

    rows.sort(key=lambda t: -t[1])
    print("\n=== FULL-FIDELITY RANKED (pose_auc_5) ===")
    for name, auc in rows:
        print(f"  {auc:8.3f}   {name}")
    reg = dict(rows).get("regular_ba")
    cand = dict(rows).get(args.strategy)
    if reg is not None and cand is not None:
        verdict = "WIN" if cand > reg else "LOSE"
        print(f"\n=== VERDICT: {args.strategy} {cand:.3f} vs regular_ba {reg:.3f} "
              f"-> {verdict} (delta {cand - reg:+.3f}) ===")
    print(json.dumps({"full": {"max_iterations": args.max_iterations,
                               "function_tolerance": args.function_tolerance},
                      "ranked": [{"name": n, "pose_auc_5": a} for n, a in rows]},
                     indent=2))


if __name__ == "__main__":
    main()
