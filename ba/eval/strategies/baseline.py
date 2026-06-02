"""Reference strategy: a single surface-BA solve.

This is the TEMPLATE for new strategies and the control for the comparison: it
reproduces offline_eval's surface number (~19.4) because it associates on the
full points then subsamples in lockstep, exactly like offline_eval.run_ba.

A strategy module must define:  refine(cache, params) -> cameras (V,10) float64
"""
import sys

sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")

import numpy as np  # noqa: E402
import strat_common as sc  # noqa: E402


def refine(cache, params):
    p = dict(params or {})
    a = sc.prepare(cache, max_points=p.get("max_points"))
    cams, pts = a["cameras"], a["points"]
    lam = float(p.get("lambda_surface", 50.0))
    assoc = float(p.get("assoc_max_distance", 0.15))

    sq_params = point_to_sq = None
    if lam > 0.0:
        sqp = sc.surface_pred(cache)
        if sqp is not None:
            # Associate on FULL points then subsample in lockstep (matches
            # offline_eval exactly so this control reproduces ~19.4).
            sq_params, full_p2sq, _ = sc.associate(cache["points"], sqp, assoc)
            point_to_sq = (np.ascontiguousarray(full_p2sq[a["keep"]], np.int32)
                           if a["keep"] is not None else full_p2sq)
        else:
            lam = 0.0

    sc.solve(cams, pts, a["observations"], a["cam_indices"], a["pt_indices"],
             lambda_surface=lam,
             surface_huber=float(p.get("surface_huber", 0.0)),
             huber_threshold=float(p.get("huber_threshold", 2.0)),
             fix_first_camera=bool(p.get("fix_first_camera", True)),
             sq_params=sq_params, point_to_sq=point_to_sq,
             max_iterations=int(p.get("max_iterations", 50)),
             function_tolerance=float(p.get("function_tolerance", 1e-3)),
             num_threads=int(p.get("num_threads", 4)))
    return cams
