#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=ase-surface-e2e
#SBATCH --output=/work/courses/3dv/team39/logs/%j.out
#SBATCH --error=/work/courses/3dv/team39/logs/%j.err
#SBATCH --gpus=5060ti:1
#SBATCH --mem=32G
#SBATCH --time=00:45:00

# End-to-end smoke test:
#   Run the same small ASE sample twice — once with vanilla `superbundle`,
#   once with the new `superbundle_surface` — and compare ATE/RPE/AUC.

export CUDA_HOME=/cluster/data/cuda/x86_64/13.0.2
export PATH=$CUDA_HOME/bin:$PATH
# NOTE: the team-shared venv lacks several mapanything deps (uniception,
# jaxtyping, ...). For now we source the working per-user venv that the team
# currently runs from. Once the shared venv is brought to parity (one-time pip
# install of the missing deps), revert this to envs/3dv.
source /home/rbesenfel/envs/3dv/bin/activate
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/courses/3dv/team39/checkpoints/hf_cache

cd /work/courses/3dv/team39/map-anything

NDP=5
NV=5

# ---- A. Baseline: VGGT + MASt3R superbundle (no surface) -----------------
echo "============================================================"
echo "=== A. superbundle (no surface) — $NDP datapoints, $NV views ==="
echo "============================================================"
python3 benchmarking/sparse_view/benchmark.py \
    machine=student_cluster \
    dataset=benchmark_518_ase_wai \
    dataset.num_workers=2 \
    dataset.num_views=$NV \
    dataset.no_of_datapoints=$NDP \
    batch_size=1 \
    model=vggt \
    bundle_adjustment=superbundle \
    +mast3r_max_frames=20 \
    hydra.run.dir=/work/courses/3dv/team39/logs/test_e2e_${SLURM_JOB_ID}_baseline
echo

# ---- B. Surface-augmented: superbundle + L_surface -----------------------
echo "============================================================"
echo "=== B. superbundle_surface (lambda=50, assoc=0.15 m) ========"
echo "============================================================"
python3 benchmarking/sparse_view/benchmark.py \
    machine=student_cluster \
    dataset=benchmark_518_ase_wai \
    dataset.num_workers=2 \
    dataset.num_views=$NV \
    dataset.no_of_datapoints=$NDP \
    batch_size=1 \
    model=vggt \
    bundle_adjustment=superbundle_surface \
    +mast3r_max_frames=20 \
    +surface_npz_path=/work/courses/3dv/team39/compose/data/output_npz/ase_scene_0.npz \
    +surface_lambda=50.0 \
    +surface_huber=0.0 \
    +surface_assoc_max_distance=0.15 \
    hydra.run.dir=/work/courses/3dv/team39/logs/test_e2e_${SLURM_JOB_ID}_surface

echo "=== Done ==="
