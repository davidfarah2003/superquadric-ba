#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=ase-sparse-mast3r
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gpus=5060ti:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

export CUDA_HOME=/cluster/data/cuda/x86_64/13.0.2
export PATH=$CUDA_HOME/bin:$PATH
source /home/lhecker/envs/3dv/bin/activate

export HF_HOME=/work/courses/3dv/team39/checkpoints/hf_cache
export HF_HUB_OFFLINE=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /work/courses/3dv/team39/map-anything

echo "=== Running sparse-view benchmark: VGGT + MASt3R BA (no superquadrics) on ASE ==="
python3 benchmarking/sparse_view/benchmark.py \
    machine=student_cluster \
    dataset=benchmark_518_ase_wai \
    dataset.num_workers=4 \
    dataset.num_views=10 \
    batch_size=1 \
    model=vggt \
    bundle_adjustment=mast3r \
    sparse_covisibility_thres=0.6 \
    hydra.run.dir='/work/courses/3dv/team39/logs/benchmark_ase_sparse_mast3r_cov06'

echo "=== Benchmark complete ==="
