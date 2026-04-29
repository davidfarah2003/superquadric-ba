#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=ase-vggt-benchmark
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gpus=5060ti:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

export CUDA_HOME=/cluster/data/cuda/x86_64/13.0.2
export PATH=$CUDA_HOME/bin:$PATH
source /work/courses/3dv/team39/envs/3dv/bin/activate

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/courses/3dv/team39/map-anything

echo "=== Running dense N-view benchmark: VGGT on ASE ==="
python3 benchmarking/dense_n_view/benchmark.py \
    machine=student_cluster \
    dataset=benchmark_518_ase_wai \
    dataset.num_workers=4 \
    dataset.num_views=2 \
    batch_size=1 \
    model=vggt \
    hydra.run.dir='/work/courses/3dv/team39/logs/benchmark_ase_vggt'

echo "=== Benchmark complete ==="
