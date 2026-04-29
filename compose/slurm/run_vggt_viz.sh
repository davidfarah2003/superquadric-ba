#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=vggt-aria-viz
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gpus=5060ti:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00

export CUDA_HOME=/cluster/data/cuda/x86_64/13.0.2
export PATH=$CUDA_HOME/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /work/courses/3dv/team39/envs/3dv/bin/activate
cd /work/courses/3dv/team39

python compose/scripts/viz_vggt_aria.py \
    --scene 0 \
    --num_frames 4 \
    --frame_start 0 \
    --out_dir compose/data/compare/vggt_scene0
