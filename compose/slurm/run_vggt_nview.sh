#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=vggt-nview
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gpus=5060ti:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00

# VGGT-only (no BA) at NV input views -> dataset-average pose_auc_5. Same dataset
# config / deterministic seed as the surface-BA sweeps, so the VGGT-only numbers
# are directly comparable to Baseline/Ours at the same view count. Set NV via env.

export CUDA_HOME=/cluster/data/cuda/x86_64/13.0.2
export PATH=$CUDA_HOME/bin:$PATH
source /work/courses/3dv/team39/envs/3dv/bin/activate
export HF_HOME=/work/courses/3dv/team39/checkpoints/hf_cache
export HF_HUB_OFFLINE=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /work/courses/3dv/team39/map-anything

NV="${NV:-4}"
echo "=== VGGT-only (no BA), ${NV} views ==="
python3 benchmarking/sparse_view/benchmark.py \
    machine=student_cluster \
    dataset=benchmark_518_ase_wai \
    dataset.num_workers=4 \
    dataset.num_views="$NV" \
    batch_size=1 \
    model=vggt \
    bundle_adjustment=none \
    sparse_covisibility_thres=0.6 \
    hydra.run.dir="/work/courses/3dv/team39/logs/vggt_v${NV}"
echo "=== Done ==="
