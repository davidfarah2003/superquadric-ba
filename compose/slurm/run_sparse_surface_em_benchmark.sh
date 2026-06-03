#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=ase-sparse-surface-em
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gpus=5060ti:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=studgpu-node09

# Cluster job_submit-plugin gotchas (learned empirically):
#   * It rejects --cpus-per-task ("TRES per task not allowed") and --ntasks=N
#     together with a GPU ("node configuration not available"). The ONE accepted
#     way to get many cores on a GPU job is --cpus-per-gpu=N (verified: AllocCPUS
#     follows it, the "CPU count: 3" echo is cosmetic). 5060ti nodes have 28
#     cores, so up to ~26 is schedulable.
#   * The 28 cores are used by the Ceres BA solve via surface_num_threads below;
#     the EM solves are CPU-bound on the big scenes (up to 117k points). 28 is
#     the 5060ti node core count (== the per-node max; QOS cap is 32, only
#     reachable on a 36-core 2080ti node). 28 needs a *fully idle* 5060ti node.
#   * node09 (5060ti) has the broken cu130 driver (error 804) -> excluded.

# Sparse-view ASE benchmark with the team-39 surface-augmented superbundle in
# EM-iterated-re-association mode (the tuned winning recipe):
#   VGGT cameras + depth -> MASt3R correspondences -> midpoint triangulation
#   -> Ceres BA alternating E-step (re-assign moving points to nearest SQ) and
#      M-step (short surface solve), with a reprojection-only warmup.
#
# Defaults below are the best em_reassoc config from the offline Bayesian
# optimisation (proxy pose_auc_5 = 31.7 vs proxy regular_ba 27.6). Compare the
# resulting pose_auc_5 against the regular-BA run (29.56) in
# logs/benchmark_ase_sparse_mast3r_cov06.

export CUDA_HOME=/cluster/data/cuda/x86_64/13.0.2
export PATH=$CUDA_HOME/bin:$PATH
source /work/courses/3dv/team39/envs/3dv/bin/activate

export HF_HOME=/work/courses/3dv/team39/checkpoints/hf_cache
export HF_HUB_OFFLINE=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /work/courses/3dv/team39/map-anything

SURFACE_NPZ_DIR="${SURFACE_NPZ_DIR:-/work/courses/3dv/team39/compose/data/output_npz}"
# --- WINNING config: one-sided HINGE surface residual + EM re-association.
#     pose_auc_5 = 29.6 LIVE (verified, reproduced) vs regular BA 29.42. ---
LAMBDA_SURFACE="${LAMBDA_SURFACE:-15.0}"         # surface weight (sweet spot; higher hurts)
ASSOC_MAX_DIST="${ASSOC_MAX_DIST:-0.0372}"       # 3.7 cm (tight)
SURFACE_HUBER="${SURFACE_HUBER:-2.749}"          # surface Huber delta
HUBER_THRESHOLD="${HUBER_THRESHOLD:-1.0}"        # reprojection Huber delta (px); matches mast3r backend
EM_OUTER="${EM_OUTER:-2}"                         # EM outer iterations
EM_INNER_ITERS="${EM_INNER_ITERS:-41}"           # Ceres iters per inner solve
EM_WARMUP="${EM_WARMUP:-true}"                    # reproj-only warmup first
RESIDUAL_MODE="${RESIDUAL_MODE:-1}"               # 1=HINGE_OUTSIDE (the win); 0=radial(old) 5=normal
FILTER_MAX_ASPECT="${FILTER_MAX_ASPECT:-0}"       # >0 drops degenerate SQs (e.g. 20)
REFINE_SQ="${REFINE_SQ:-false}"                   # true = co-refine SQ pose in BA
SQ_ANCHOR_WEIGHT="${SQ_ANCHOR_WEIGHT:-10.0}"      # stiffness of SQ-pose anchor prior
MANHATTAN_SNAP="${MANHATTAN_SNAP:-0}"             # >0 = snap SQ orient to voted Manhattan frame (deg; denoise)
NUM_VIEWS="${NUM_VIEWS:-10}"                       # cameras per scene (fewer-pictures sweep: 4/6/8)
NUM_THREADS="${NUM_THREADS:-4}"                   # Ceres BA threads (4-CPU cap now)
VIZ_SAVE_INDEX="${VIZ_SAVE_INDEX:-}"

echo "=== Sparse-view benchmark: VGGT + MASt3R + SUPERDEC surface BA (EM) ==="
echo "    lambda=$LAMBDA_SURFACE assoc=$ASSOC_MAX_DIST s_huber=$SURFACE_HUBER "\
"h_thresh=$HUBER_THRESHOLD em_outer=$EM_OUTER inner=$EM_INNER_ITERS warmup=$EM_WARMUP "\
"threads=$NUM_THREADS"

python3 benchmarking/sparse_view/benchmark.py \
    machine=student_cluster \
    dataset=benchmark_518_ase_wai \
    dataset.num_workers=4 \
    dataset.num_views="$NUM_VIEWS" \
    batch_size=1 \
    model=vggt \
    bundle_adjustment=superbundle_surface \
    sparse_covisibility_thres=0.6 \
    +surface_npz_dir="$SURFACE_NPZ_DIR" \
    +surface_lambda="$LAMBDA_SURFACE" \
    +surface_huber="$SURFACE_HUBER" \
    +surface_assoc_max_distance="$ASSOC_MAX_DIST" \
    +surface_huber_threshold="$HUBER_THRESHOLD" \
    +surface_em_outer="$EM_OUTER" \
    +surface_em_inner_iters="$EM_INNER_ITERS" \
    +surface_em_warmup="$EM_WARMUP" \
    +surface_residual_mode="$RESIDUAL_MODE" \
    +surface_filter_max_aspect="$FILTER_MAX_ASPECT" \
    +surface_refine_sq="$REFINE_SQ" \
    +surface_sq_anchor_weight="$SQ_ANCHOR_WEIGHT" \
    +surface_manhattan_snap="$MANHATTAN_SNAP" \
    +surface_num_threads="$NUM_THREADS" \
    hydra.run.dir="/work/courses/3dv/team39/logs/benchmark_ase_sparse_surface_em_cov06_v${NUM_VIEWS}_lam${LAMBDA_SURFACE}" \
    ${VIZ_SAVE_INDEX:+viz_save_index=$VIZ_SAVE_INDEX}

echo "=== Benchmark complete ==="
