#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=views-sweep
#SBATCH --output=/work/courses/3dv/team39/logs/%j.out
#SBATCH --error=/work/courses/3dv/team39/logs/%j.err
#SBATCH --gpus=5060ti:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00

# Fewer-pictures sweep: does the surface prior's gain grow as views shrink?
# For each num_views, run surface-BA (lam=15, hinge, EM) vs reproj-BA (lam=0),
# SAME views (seed 777) so the only difference is the surface term. One job,
# sequential (1-job QOS). Each config prints pose_auc_5 -> parse from this log.
# (num_views=4 already measured: surface 40.667 vs reproj 39.333 = +1.33.)

export CUDA_HOME=/cluster/data/cuda/x86_64/13.0.2
export PATH=$CUDA_HOME/bin:$PATH
source /work/courses/3dv/team39/envs/3dv/bin/activate
export HF_HOME=/work/courses/3dv/team39/checkpoints/hf_cache
export HF_HUB_OFFLINE=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /work/courses/3dv/team39/map-anything

SURFACE_NPZ_DIR=/work/courses/3dv/team39/compose/data/output_npz

run_one() {
  local NV=$1 LAM=$2
  echo "########## SWEEP num_views=$NV lambda=$LAM ##########"
  python3 benchmarking/sparse_view/benchmark.py \
    machine=student_cluster dataset=benchmark_518_ase_wai dataset.num_workers=4 \
    dataset.num_views="$NV" batch_size=1 model=vggt \
    bundle_adjustment=superbundle_surface sparse_covisibility_thres=0.6 \
    +surface_npz_dir="$SURFACE_NPZ_DIR" +surface_lambda="$LAM" \
    +surface_huber=2.749 +surface_assoc_max_distance=0.0372 \
    +surface_huber_threshold=1.0 +surface_em_outer=2 +surface_em_inner_iters=41 \
    +surface_em_warmup=true +surface_residual_mode=1 +surface_filter_max_aspect=0 \
    +surface_refine_sq=false +surface_sq_anchor_weight=10.0 +surface_manhattan_snap=0 \
    +surface_num_threads=4 \
    hydra.run.dir="/work/courses/3dv/team39/logs/sweep_v${NV}_lam${LAM}"
  echo "########## DONE num_views=$NV lambda=$LAM ##########"
}

for NV in 6 8 10; do
  run_one "$NV" 15.0   # surface
  run_one "$NV" 0      # reproj baseline (same views)
done
echo "=== sweep complete ==="
