#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=3dv-g39
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=04:00:00

# CUDA toolkit (required for RTX 5060 Ti JIT compilation)
export CUDA_HOME=/cluster/data/cuda/x86_64/13.0.2
export PATH=$CUDA_HOME/bin:$PATH

# Activate shared team venv (lives in team folder)
source /work/courses/3dv/team39/envs/3dv/bin/activate

# Project root
cd /work/courses/3dv/team39

# Run whatever is passed as arguments
"$@"
