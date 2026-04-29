#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=ase-benchmark-prep
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00

export CUDA_HOME=/cluster/data/cuda/x86_64/13.0.2
export PATH=$CUDA_HOME/bin:$PATH
source /work/courses/3dv/team39/envs/3dv/bin/activate

cd /work/courses/3dv/team39/compose

# Step 1: Create scene list metadata
echo "=== Step 1: Create scene list metadata ==="
python scripts/prepare_benchmark_data.py \
    --wai_root data/wai \
    --metadata_dir data/dataset_metadata \
    --split test

# Step 2: Compute covisibility matrices (requires GPU)
echo "=== Step 2: Compute covisibility ==="
python scripts/compute_covisibility.py \
    --wai_root data/wai \
    --device cuda

echo "=== Preparation complete ==="
echo "To run the benchmark, use run_benchmark.sh"
