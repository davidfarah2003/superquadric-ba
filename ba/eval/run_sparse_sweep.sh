#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=sparse-sweep
#SBATCH --output=/work/courses/3dv/team39/logs/%j.out
#SBATCH --error=/work/courses/3dv/team39/logs/%j.err
#SBATCH --mem=32G
#SBATCH --time=03:00:00

# Point-density sweep: does the surface prior gain leverage as points get sparse?
# CPU-only work (forced onto a GPU node @ 2 CPUs; GPU unused). Incremental save.
source /work/courses/3dv/team39/envs/3dv/bin/activate
cd /work/courses/3dv/team39
echo "=== sparse-density sweep on $(hostname) ==="
python ba/eval/exp_sparse.py --jobs 2
echo "=== done ==="
