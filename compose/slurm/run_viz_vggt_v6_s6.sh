#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=viz-vggt-v6
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gpus=5060ti:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00

# VGGT-only (no BA) at 6 views, dumping scene-6 viz cameras. Same dataset config
# and deterministic seed as the lam0/lam15 surface-BA viz runs -> identical 6
# views, so the saved vggt/cameras.json is the genuine raw-VGGT pose for the same
# problem (the surface-BA runs' vggt/ file is aliased to the BA result and unusable).

export CUDA_HOME=/cluster/data/cuda/x86_64/13.0.2
export PATH=$CUDA_HOME/bin:$PATH
source /work/courses/3dv/team39/envs/3dv/bin/activate

export HF_HOME=/work/courses/3dv/team39/checkpoints/hf_cache
export HF_HUB_OFFLINE=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /work/courses/3dv/team39/map-anything

echo "=== VGGT-only (no BA), 6 views, viz scene 6 ==="
python3 benchmarking/sparse_view/benchmark.py \
    machine=student_cluster \
    dataset=benchmark_518_ase_wai \
    dataset.num_workers=4 \
    dataset.num_views=6 \
    batch_size=1 \
    model=vggt \
    bundle_adjustment=none \
    sparse_covisibility_thres=0.6 \
    hydra.run.dir='/work/courses/3dv/team39/logs/viz_vggt_v6_s6' \
    viz_save_index=6

echo "=== Done ==="
