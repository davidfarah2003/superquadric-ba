#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=pose-analyze
#SBATCH --output=/work/courses/3dv/team39/logs/%j.out
#SBATCH --error=/work/courses/3dv/team39/logs/%j.err
#SBATCH --mem=32G
#SBATCH --time=03:00:00

# Refined-camera pose decomposition: run reproj-BA + surface-BA(no-snap) +
# surface-BA(snap) per scene, decompose pose_auc_5 into rotation-only /
# translation-only / combined. CPU-only (forced onto a GPU node @ 2 CPUs by the
# plugin; GPU unused). Saves analysis/pose_decomp.{npz,json} + figures.
source /work/courses/3dv/team39/envs/3dv/bin/activate
cd /work/courses/3dv/team39
echo "=== refined-camera pose decomposition on $(hostname) ==="
python ba/eval/analyze_pose.py --with-ba
echo "=== done ==="
