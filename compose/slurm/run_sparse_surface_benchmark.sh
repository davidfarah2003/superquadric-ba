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
# The SUPERDEC scene NPZ is currently fixed to ase_scene_0; once the dataset
# loader exposes the scene id per batch, we will plumb that through instead.

export CUDA_HOME=/cluster/data/cuda/x86_64/13.0.2
export PATH=$CUDA_HOME/bin:$PATH
source /work/courses/3dv/team39/envs/3dv/bin/activate

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /work/courses/3dv/team39/map-anything

SURFACE_NPZ="/work/courses/3dv/team39/compose/data/output_npz/ase_scene_0.npz"
LAMBDA_SURFACE="${LAMBDA_SURFACE:-50.0}"      # pixels-per-meter
SURFACE_HUBER="${SURFACE_HUBER:-0.0}"          # disabled by default
ASSOC_MAX_DIST="${ASSOC_MAX_DIST:-0.15}"       # 15 cm

echo "=== Sparse-view benchmark: VGGT + MASt3R + SUPERDEC surface BA ==="
echo "    lambda_surface=$LAMBDA_SURFACE  assoc_max_dist=$ASSOC_MAX_DIST  npz=$SURFACE_NPZ"

python3 benchmarking/sparse_view/benchmark.py \
    machine=student_cluster \
    dataset=benchmark_518_ase_wai \
    dataset.num_workers=4 \
    dataset.num_views=10 \
    batch_size=1 \
    model=vggt \
    bundle_adjustment=superbundle_surface \
    mast3r_max_frames=20 \
    +surface_npz_path="$SURFACE_NPZ" \
    +surface_lambda="$LAMBDA_SURFACE" \
    +surface_huber="$SURFACE_HUBER" \
    +surface_assoc_max_distance="$ASSOC_MAX_DIST" \
    hydra.run.dir='/work/courses/3dv/team39/logs/benchmark_ase_sparse_surface'

echo "=== Benchmark complete ==="
