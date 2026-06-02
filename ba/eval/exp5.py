"""Exp5: does dropping degenerate SQs (sq_hinge_filtered) help, and does it let
lambda go higher? Offline ranking vs the mode1 hinge reference (live 29.6)."""
import sys, json, argparse
sys.path.insert(0,"/work/courses/3dv/team39/ba/eval"); sys.path.insert(0,"/work/courses/3dv/team39/ba/python")
import run_strategy as rs
BAR=29.42
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cache_dir",required=True)
    ap.add_argument("--jobs",type=int,default=10); ap.add_argument("--num_threads",type=int,default=3)
    a=ap.parse_args()
    common=dict(max_points=None,function_tolerance=1e-6,num_threads=a.num_threads,fix_first_camera=True,
                residual_mode=1,huber_threshold=1.0,assoc_max_distance=0.0372,surface_huber=2.749,
                n_outer=2,inner_iters=41,warmup=True)
    rows=[]
    def run(name,extra):
        p=dict(common); p.update(extra)
        auc=rs.score_strategy(name,a.cache_dir,p,jobs=a.jobs)["pose_auc_5"]
        rows.append((f"{name}:{json.dumps(extra,separators=(',',':'))[:40]}",auc))
        print(f"  {auc:.3f}  {name} {json.dumps(extra,separators=(',',':'))[:50]}{'  >=BAR' if auc>=BAR else ''}",flush=True)
    run("em_reassoc",dict(lambda_surface=15))           # ref: no filter, live 29.6
    run("sq_hinge_filtered",dict(lambda_surface=15))    # default filter
    run("sq_hinge_filtered",dict(lambda_surface=15,max_aspect=10.0))  # stricter
    run("sq_hinge_filtered",dict(lambda_surface=30))    # filter + higher lambda
    run("sq_hinge_filtered",dict(lambda_surface=50,max_aspect=10.0))  # filter + much higher
    rows.sort(key=lambda t:-t[1])
    print("\n=== RANKED ==="); [print(f"  {v:.3f}  {n}") for n,v in rows]
    print(json.dumps({"ranked":[{"name":n,"auc":v} for n,v in rows]}))
main()
