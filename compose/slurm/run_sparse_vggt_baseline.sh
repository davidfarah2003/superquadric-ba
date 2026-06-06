#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=ase-sparse-vggt
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gpus=5060ti:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

export CUDA_HOME=/cluster/data/cuda/x86_64/13.0.2
export PATH=$CUDA_HOME/bin:$PATH
source /work/courses/3dv/team39/envs/3dv/bin/activate

export HF_HOME=/work/courses/3dv/team39/checkpoints/hf_cache
export HF_HUB_OFFLINE=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /work/courses/3dv/team39/map-anything

VIZ_SAVE_INDEX="${VIZ_SAVE_INDEX:-}"

echo "=== Sparse-view benchmark: VGGT baseline (no BA) ==="
python3 benchmarking/sparse_view/benchmark.py \
    machine=student_cluster \
    dataset=benchmark_518_ase_wai \
    dataset.num_workers=4 \
    dataset.num_views=10 \
    batch_size=1 \
    model=vggt \
    bundle_adjustment=none \
    sparse_covisibility_thres=0.6 \
    hydra.run.dir='/work/courses/3dv/team39/logs/benchmark_ase_sparse_vggt' \
    ${VIZ_SAVE_INDEX:+viz_save_index=$VIZ_SAVE_INDEX}

echo "=== Benchmark complete ==="
