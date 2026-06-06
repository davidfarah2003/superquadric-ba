#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=lambda-sweep4
#SBATCH --output=/work/courses/3dv/team39/logs/%j.out
#SBATCH --error=/work/courses/3dv/team39/logs/%j.err
#SBATCH --gpus=5060ti:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00

# Sparse-regime lambda tuning at num_views=4 (where the surface prior has leverage).
# lambda=15 was tuned for DENSE views (high lambda hurt there). At 4 views the
# cameras are starved -> the surface term should take a much higher weight and
# AMPLIFY the +1.33 gain. Baseline already known: reproj@4 (lam=0) = 39.33,
# surface@4 lam=15 = 40.67. This sweeps higher lambda. Each config prints pose_auc_5.

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
  local LAM=$1
  echo "########## LAMSWEEP num_views=4 lambda=$LAM ##########"
  python3 benchmarking/sparse_view/benchmark.py \
    machine=student_cluster dataset=benchmark_518_ase_wai dataset.num_workers=4 \
    dataset.num_views=4 batch_size=1 model=vggt \
    bundle_adjustment=superbundle_surface sparse_covisibility_thres=0.6 \
    +surface_npz_dir="$SURFACE_NPZ_DIR" +surface_lambda="$LAM" \
    +surface_huber=2.749 +surface_assoc_max_distance=0.0372 \
    +surface_huber_threshold=1.0 +surface_em_outer=2 +surface_em_inner_iters=41 \
    +surface_em_warmup=true +surface_residual_mode=1 +surface_filter_max_aspect=0 \
    +surface_refine_sq=false +surface_sq_anchor_weight=10.0 +surface_manhattan_snap=0 \
    +surface_num_threads=4 \
    hydra.run.dir="/work/courses/3dv/team39/logs/lamsweep_v4_lam${LAM}"
  echo "########## DONE lambda=$LAM ##########"
}

for LAM in 30 60 100 200; do
  run_one "$LAM"
done
echo "=== lambda sweep complete ==="
