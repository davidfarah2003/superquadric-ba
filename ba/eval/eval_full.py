"""Full-fidelity batch ranking of every strategy with ITS OWN default params.

Unlike eval_all.py (which forces one shared lambda/assoc on every strategy), this
passes ONLY the solver knobs (no subsample, tight tolerance, thread count) and
lets each strategy use the algorithm defaults its author tuned. That is the fair
way to compare structurally different strategies. The proxy is dead (it misranked
em_reassoc 31.7 vs live 28.9); this runs at the live settings so the ranking is
trustworthy, scene-parallel across the 10 cached scenes.

References (offline_eval): regular_ba (lambda=0) must reproduce ~29.42; surface
(lambda=50, assoc=0.15) ~19.42 — built-in faithfulness check.

Run on a Slurm node, e.g. (2080ti has 36 cores -> 32 via --cpus-per-gpu):
  sbatch --account=3dv --gpus=2080ti:1 --cpus-per-gpu=32 --mem=48G \
    --time=02:00:00 --output=logs/%j.out --wrap \
    "<venv>/bin/python ba/eval/eval_full.py --cache_dir compose/data/ba_cache \
       --jobs 10 --num_threads 3"
"""
import os
import sys
import glob
import json
import argparse
import importlib

sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")
sys.path.insert(0, "/work/courses/3dv/team39/ba/python")

import offline_eval as oe  # noqa: E402
import run_strategy as rs  # noqa: E402

BAR = 29.42  # regular BA live pose_auc_5 (the bar to beat)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--num_threads", type=int, default=3)
    ap.add_argument("--function_tolerance", type=float, default=1e-6)
    ap.add_argument("--max_iterations", type=int, default=200,
                    help="iteration cap for the references (strategies use their"
                         " own inner_iters defaults)")
    args = ap.parse_args(argv)

    # Solver knobs ONLY -> every strategy keeps its own algorithm defaults.
    # max_points absent => prepare() does NOT subsample (full fidelity).
    knobs = dict(function_tolerance=args.function_tolerance,
                 num_threads=args.num_threads, fix_first_camera=True)

    rows = []

    # --- references via offline_eval (lambda toggles the surface term) ---
    for name, lam, assoc in [("regular_ba", 0.0, 0.15), ("surface", 50.0, 0.15)]:
        r = oe.score({"lambda_surface": lam, "surface_huber": 0.0,
                      "assoc_max_distance": assoc, "max_points": None,
                      "max_iterations": args.max_iterations,
                      "function_tolerance": args.function_tolerance,
                      "num_threads": args.num_threads}, args.cache_dir,
                     jobs=args.jobs)
        rows.append((name, r["pose_auc_5"]))
        print(f"[ref ] {name:18s} pose_auc_5={r['pose_auc_5']:.3f}", flush=True)

    # --- every strategy with its own defaults (knobs only) ---
    strat_dir = "/work/courses/3dv/team39/ba/eval/strategies"
    names = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(strat_dir, "*.py"))
                   if not os.path.basename(p).startswith("__"))
    for name in names:
        try:
            importlib.import_module(f"strategies.{name}")
            r = rs.score_strategy(name, args.cache_dir, dict(knobs), jobs=args.jobs)
            rows.append((name, r["pose_auc_5"]))
            mark = "  <-- BEATS BAR" if r["pose_auc_5"] > BAR else ""
            print(f"[strat] {name:18s} pose_auc_5={r['pose_auc_5']:.3f}{mark}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[strat] {name:18s} ERROR {e}", flush=True)

    rows.sort(key=lambda t: -t[1])
    print(f"\n=== FULL-FIDELITY RANKED (bar = regular_ba {BAR}) ===")
    for name, auc in rows:
        flag = " *" if auc > BAR and name != "regular_ba" else ""
        print(f"  {auc:8.3f}   {name}{flag}")
    winners = [n for n, a in rows if a > BAR and n != "regular_ba"]
    print(f"\nWINNERS over the bar: {winners or 'NONE'}")
    print(json.dumps({"ranked": [{"name": n, "pose_auc_5": a} for n, a in rows],
                      "bar": BAR, "winners": winners}, indent=2))


if __name__ == "__main__":
    main()
