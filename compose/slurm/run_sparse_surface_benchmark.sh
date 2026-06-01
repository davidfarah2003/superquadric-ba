#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=ase-sparse-surface
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gpus=5060ti:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Sparse-view ASE benchmark with the team-39 surface-augmented superbundle:
#   VGGT cameras + depth -> MASt3R correspondences -> midpoint triangulation
#   -> Ceres BA with reprojection + SUPERDEC point-to-superquadric residual.
#
# The per-scene NPZ is resolved automatically from batch["label"] inside
# benchmark.py. All scenes in SURFACE_NPZ_DIR must be pre-computed.

export CUDA_HOME=/cluster/data/cuda/x86_64/13.0.2
export PATH=$CUDA_HOME/bin:$PATH
source /home/lhecker/envs/3dv/bin/activate

export HF_HOME=/work/courses/3dv/team39/checkpoints/hf_cache
export HF_HUB_OFFLINE=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /work/courses/3dv/team39/map-anything

SURFACE_NPZ_DIR="${SURFACE_NPZ_DIR:-/work/courses/3dv/team39/compose/data/output_npz}"
LAMBDA_SURFACE="${LAMBDA_SURFACE:-50.0}"      # pixels-per-meter
SURFACE_HUBER="${SURFACE_HUBER:-0.0}"          # disabled by default
ASSOC_MAX_DIST="${ASSOC_MAX_DIST:-0.15}"       # 15 cm

echo "=== Sparse-view benchmark: VGGT + MASt3R + SUPERDEC surface BA ==="
echo "    lambda_surface=$LAMBDA_SURFACE  assoc_max_dist=$ASSOC_MAX_DIST  npz_dir=$SURFACE_NPZ_DIR"

python3 benchmarking/sparse_view/benchmark.py \
    machine=student_cluster \
    dataset=benchmark_518_ase_wai \
    dataset.num_workers=4 \
    dataset.num_views=10 \
    batch_size=1 \
    model=vggt \
    bundle_adjustment=superbundle_surface \
    +surface_npz_dir="$SURFACE_NPZ_DIR" \
    +surface_lambda="$LAMBDA_SURFACE" \
    +surface_huber="$SURFACE_HUBER" \
    +surface_assoc_max_distance="$ASSOC_MAX_DIST" \
    hydra.run.dir='/work/courses/3dv/team39/logs/benchmark_ase_sparse_surface'

echo "=== Benchmark complete ==="
