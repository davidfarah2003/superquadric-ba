"""Exp6: does SQ pose CO-REFINEMENT (refine_sq) break the lambda ceiling?
The frozen SQs cap lambda at ~15 (higher hurts). If Ceres can move each SQ to fit
(anchored to init), high lambda should finally help. Compare co-refine on/off
across lambda. Full fidelity; 4-CPU cap so jobs<=4 (slow). Live validates top.
"""
import sys, json, argparse
sys.path.insert(0,"/work/courses/3dv/team39/ba/eval"); sys.path.insert(0,"/work/courses/3dv/team39/ba/python")
import run_strategy as rs
BAR=29.42
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cache_dir",required=True)
    ap.add_argument("--jobs",type=int,default=4); ap.add_argument("--num_threads",type=int,default=1)
    a=ap.parse_args()
    common=dict(max_points=None,function_tolerance=1e-6,num_threads=a.num_threads,fix_first_camera=True,
                residual_mode=1,huber_threshold=1.0,assoc_max_distance=0.0372,surface_huber=2.749,
                n_outer=2,inner_iters=41,warmup=True)
    rows=[]
    def run(tag,extra):
        p=dict(common); p.update(extra)
        auc=rs.score_strategy("em_reassoc",a.cache_dir,p,jobs=a.jobs)["pose_auc_5"]
        rows.append((tag,auc)); print(f"  {auc:.3f}  {tag}{'  >=BAR' if auc>=BAR else ''}",flush=True)
    run("ref_norefine_lam15",dict(lambda_surface=15,refine_sq=False))   # = the 29.6 winner
    run("refine_lam15",dict(lambda_surface=15,refine_sq=True))
    run("refine_lam30",dict(lambda_surface=30,refine_sq=True))
    run("refine_lam60",dict(lambda_surface=60,refine_sq=True))
    rows.sort(key=lambda t:-t[1])
    print("\n=== RANKED ==="); [print(f"  {v:.3f}  {n}") for n,v in rows]
    print(json.dumps({"ranked":[{"name":n,"auc":v} for n,v in rows]}))
main()
