#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=superdec-ase
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gpus=5060ti:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4

# CUDA toolkit
export CUDA_HOME=/cluster/data/cuda/x86_64/13.0.2
export PATH=$CUDA_HOME/bin:$PATH

source ~/envs/3dv/bin/activate

cd /work/courses/3dv/team39/superdec

# Run inference on ASE scene 0
python superdec/evaluate/to_npz.py \
  checkpoints_folder="checkpoints/normalized" \
  output_dir="/work/courses/3dv/team39/superdec_tests/data/output_npz" \
  dataset=scene \
  scene.path="data" \
  scene.name="ase_scene_0" \
  scene.z_up=true \
  scene.gt=true \
  dataloader.batch_size=32 \
  dataloader.num_workers=4 \
  device=cuda
