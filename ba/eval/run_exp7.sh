#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=exp7-manhattan
#SBATCH --output=/work/courses/3dv/team39/logs/%j.out
#SBATCH --error=/work/courses/3dv/team39/logs/%j.err
#SBATCH --gpus=5060ti:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --exclude=studgpu-node09

# Offline ranking: Manhattan-snap SQ orientations (exp7). The submit plugin
# force-attaches a GPU to every job; requesting --gpus=5060ti:1 explicitly grants
# 4 CPUs (vs 2 for an implicit GPU). The GPU is unused — the offline Ceres eval
# is CPU-only (torch tensors stay on CPU, no CUDA init). node09 excluded (bad cu130).
# num_threads=1 for deterministic Ceres (CAVEAT 2); scene-parallel via --jobs.
source /work/courses/3dv/team39/envs/3dv/bin/activate
cd /work/courses/3dv/team39

NJOBS=$(nproc)
echo "=== exp7 Manhattan-snap offline ranking on $(hostname): nproc=$NJOBS ==="
python ba/eval/exp7.py \
    --cache_dir /work/courses/3dv/team39/compose/data/ba_cache \
    --jobs "$NJOBS" --num_threads 1
echo "=== exp7 complete ==="
