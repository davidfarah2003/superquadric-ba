"""Bayesian optimisation over a structural strategy's parameters (proxy pose_auc_5).

Maximises proxy pose_auc_5 for one strategy (default em_reassoc, the current
leader) over its tunables, to try to push the surface term above regular BA.
Runs on a Slurm node; uses optuna (already installed) with the scene-parallel
run_strategy scorer.

  python strat_bo.py --strategy em_reassoc --cache_dir DIR --n_trials 50 --jobs 2
"""
import sys
import json
import argparse

sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")
sys.path.insert(0, "/work/courses/3dv/team39/ba/python")

import optuna  # noqa: E402
import run_strategy as rs  # noqa: E402

# Per-strategy search spaces. Fixed proxy knobs (max_points/iters/threads) are
# injected at call time so every trial uses the same fast, faithful proxy.
SPACES = {
    "em_reassoc": lambda t: {
        "lambda_surface": t.suggest_float("lambda_surface", 2.0, 120.0, log=True),
        "assoc_max_distance": t.suggest_float("assoc_max_distance", 0.02, 0.25),
        "surface_huber": t.suggest_float("surface_huber", 0.0, 4.0),
        "huber_threshold": t.suggest_float("huber_threshold", 0.5, 4.0),
        "n_outer": t.suggest_int("n_outer", 2, 6),
        "inner_iters": t.suggest_int("inner_iters", 20, 50),
        "warmup": t.suggest_categorical("warmup", [True, False]),
    },
    "two_stage_em": lambda t: {
        "lambda_surface": t.suggest_float("lambda_surface", 2.0, 120.0, log=True),
        "assoc_max_distance": t.suggest_float("assoc_max_distance", 0.02, 0.25),
        "surface_huber": t.suggest_float("surface_huber", 0.0, 4.0),
        "huber_threshold": t.suggest_float("huber_threshold", 0.5, 4.0),
        "stage1_iters": t.suggest_int("stage1_iters", 20, 60),
        "n_outer": t.suggest_int("n_outer", 1, 4),
        "inner_iters": t.suggest_int("inner_iters", 20, 50),
    },
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="em_reassoc")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--max_points", type=int, default=5000)
    ap.add_argument("--num_threads", type=int, default=1)
    args = ap.parse_args(argv)

    space = SPACES[args.strategy]
    fixed = dict(max_points=args.max_points, function_tolerance=1e-3,
                 num_threads=args.num_threads, fix_first_camera=True)

    best = {"auc": -1.0, "params": None}

    def objective(trial):
        params = dict(space(trial)); params.update(fixed)
        auc = rs.score_strategy(args.strategy, args.cache_dir, params,
                                jobs=args.jobs)["pose_auc_5"]
        if auc > best["auc"]:
            best["auc"] = auc; best["params"] = params
        print(f"[trial {trial.number:3d}] auc={auc:7.3f}  best={best['auc']:7.3f}",
              flush=True)
        return auc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    print("\n=== BEST ===")
    print(json.dumps({"strategy": args.strategy, "best_pose_auc_5": study.best_value,
                      "best_params": best["params"]}, indent=2))


if __name__ == "__main__":
    main()
