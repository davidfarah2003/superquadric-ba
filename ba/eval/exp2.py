"""Exp2: grid-search the HINGE surface config to maximize the margin over the bar.

The live win (hinge mode1, lam15, huber1.0 -> 29.6 > 29.42) used the first config
that cleared the bar. Hinge is one-sided so it tolerates / wants higher surface
weight than the radial term did. This grids the hinge knobs at full fidelity
(offline RANKING; live validates the top) to find a bigger-margin config.

Caveats apply (see EXPERIMENTS.md): offline is for ranking only (~+0.9 vs live),
and num_threads>1 adds ~0.5 noise, so treat sub-0.5 offline gaps as ties and
validate the top 2-3 live.
"""
import sys
import json
import argparse
import itertools

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
                  n_outer=2, inner_iters=41, warmup=True, surface_huber=2.749)

    def em(mode, lam, huber, assoc=0.0372):
        p = dict(common); p.update(residual_mode=mode, lambda_surface=lam,
                                   huber_threshold=huber, assoc_max_distance=assoc)
        return rs.score_strategy("em_reassoc", args.cache_dir, p,
                                 jobs=args.jobs)["pose_auc_5"]

    rows = []
    # Main grid: mode x lambda x huber.
    print("=== hinge grid: mode x lambda x huber ===", flush=True)
    for mode, lam, huber in itertools.product([1, 4], [15, 30, 50], [0.75, 1.0]):
        auc = em(mode, lam, huber)
        name = f"m{mode}_lam{lam}_h{huber}"
        rows.append((name, auc))
        print(f"  {name:18s} {auc:.3f}{'  >=BAR' if auc >= BAR else ''}", flush=True)

    # Assoc sweep on the leading family (mode1): hinge is one-sided, so a looser
    # association (include more outside points) may add beneficial pull.
    print("\n=== mode1 lam30 huber0.75 assoc sweep ===", flush=True)
    for assoc in [0.0372, 0.07, 0.12, 0.2]:
        auc = em(1, 30, 0.75, assoc)
        name = f"m1_lam30_h0.75_a{assoc}"
        rows.append((name, auc))
        print(f"  {name:24s} {auc:.3f}{'  >=BAR' if auc >= BAR else ''}", flush=True)

    rows.sort(key=lambda t: -t[1])
    print(f"\n=== RANKED (offline; bar≈{BAR}, live offset ~+0.9) ===")
    for n, a in rows:
        print(f"  {a:8.3f}  {n}")
    print(json.dumps({"ranked": [{"name": n, "auc": a} for n, a in rows]}, indent=2))


if __name__ == "__main__":
    main()
