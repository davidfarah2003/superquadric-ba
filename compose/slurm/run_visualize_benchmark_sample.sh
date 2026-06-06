#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=viz-benchmark-sample
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gpus=5060ti:1
#SBATCH --mem=8G
#SBATCH --time=00:15:00

# Usage:
#   SAMPLE_DIR=<path> sbatch slurm/run_visualize_benchmark_sample.sh
#
# Optional:
#   FOCAL_LENGTH=450   (pixels; default 450 for 518x518 ASE images)
#   MAX_POINTS=500000  (point-cloud cap; default 500000)
#
# Example:
#   SAMPLE_DIR="/work/courses/3dv/team39/logs/benchmark_ase_sparse_surface_cov06/viz/10 @ ASEWAI/sample_0" \
#   sbatch slurm/run_visualize_benchmark_sample.sh

source /work/courses/3dv/team39/envs/3dv/bin/activate
cd /work/courses/3dv/team39/compose

SAMPLE_DIR="${SAMPLE_DIR:-/work/courses/3dv/team39/logs/benchmark_ase_sparse_surface_cov06/viz/ 10 @ ASEWAI/sample_0}"
FOCAL_LENGTH="${FOCAL_LENGTH:-450}"
MAX_POINTS="${MAX_POINTS:-500000}"

echo "=== Visualize benchmark sample ==="
echo "    sample_dir:   $SAMPLE_DIR"
echo "    focal_length: $FOCAL_LENGTH"
echo "    max_points:   $MAX_POINTS"

python3 scripts/visualize_benchmark_sample.py \
    "$SAMPLE_DIR" \
    --focal-length "$FOCAL_LENGTH" \
    --max-points "$MAX_POINTS"

echo "=== Done. PLY files written to: $SAMPLE_DIR ==="
