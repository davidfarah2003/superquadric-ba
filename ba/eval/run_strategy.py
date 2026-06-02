"""Score a structural surface-BA strategy over the cached BA problems.

A strategy is a module ``strategies/<name>.py`` defining
``refine(cache, params) -> cameras (V,10)``. This runner turns the refined
cameras into pose_auc_5 using the SAME validated path as offline_eval
(cameras_to_pred_poses + pose_auc_5), so strategy numbers are directly
comparable to the offline_eval baseline/surface numbers.

Usage:
    python run_strategy.py --strategy em_reassoc --cache_dir DIR \
        --params '{"lambda_surface":50,"assoc_max_distance":0.15,"max_points":5000}' \
        --jobs 8
"""
import os
import sys
import glob
import json
import argparse
import importlib
import concurrent.futures as cf

sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")
sys.path.insert(0, "/work/courses/3dv/team39/ba/python")

import numpy as np  # noqa: E402
import offline_eval as oe  # noqa: E402


def _label(cache, path):
    lab = cache.get("scene_label")
    if lab is None or (hasattr(lab, "size") and lab.size == 0):
        return os.path.splitext(os.path.basename(path))[0]
    return str(lab.item() if hasattr(lab, "item") else lab)


def _score_one(task):
    path, strat_name, params = task
    strat = importlib.import_module(f"strategies.{strat_name}")
    cache = oe.load_cache(path)
    cameras = np.ascontiguousarray(strat.refine(cache, params), np.float64)
    gt_poses = np.asarray(cache["gt_poses"], np.float64)
    gt_centres = np.asarray(cache["gt_centres"], np.float64)
    pred_poses = oe.cameras_to_pred_poses(cameras, gt_centres)
    return _label(cache, path), float(oe.pose_auc_5(pred_poses, gt_poses))


def score_strategy(strat_name, cache_dir, params, jobs=8):
    paths = sorted(glob.glob(os.path.join(str(cache_dir), "*.npz")))
    if not paths:
        raise FileNotFoundError(f"No .npz caches in {cache_dir!r}")
    tasks = [(p, strat_name, dict(params)) for p in paths]
    if jobs and int(jobs) > 1:
        with cf.ProcessPoolExecutor(max_workers=int(jobs)) as ex:
            results = list(ex.map(_score_one, tasks))
    else:
        results = [_score_one(t) for t in tasks]
    per = {k: v for k, v in results}
    return {
        "strategy": strat_name,
        "pose_auc_5": float(np.mean(list(per.values()))),
        "n_scenes": len(per),
        "per_scene": per,
        "params": params,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--params", default="{}", help="JSON dict of strategy params")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args(argv)
    out = score_strategy(args.strategy, args.cache_dir,
                         json.loads(args.params), jobs=args.jobs)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
