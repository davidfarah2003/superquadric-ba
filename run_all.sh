#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=ase-pipeline
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=04:00:00

export CUDA_HOME=/cluster/data/cuda/x86_64/13.0.2
export PATH=$CUDA_HOME/bin:$PATH
source ~/envs/3dv/bin/activate

cd /work/courses/3dv/team39/superdec_tests

# Step 1: Convert raw ASE scenes to WAI format
echo "=== Step 1: ASE -> WAI conversion ==="
for i in $(seq 0 9); do
    if [ -f "data/wai/$i/scene_meta.json" ]; then
        echo "Scene $i already converted, skipping"
        continue
    fi
    echo "Converting scene $i..."
    python scripts/convert_ase_to_wai.py --scene_path data/ase/$i --output_path data/wai/$i
done

# Step 2: Extract per-object point clouds from WAI
echo "=== Step 2: Extract point clouds ==="
for i in $(seq 0 9); do
    if [ -d "data/pointclouds/$i" ] && [ "$(ls data/pointclouds/$i/*.npz 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "Scene $i point clouds exist, skipping"
        continue
    fi
    echo "Extracting point clouds for scene $i..."
    python scripts/extract_pointclouds.py --wai_path data/wai/$i --output_path data/pointclouds/$i --frame_stride 5
done

# Step 3: Copy point clouds into superdec data dir and run inference
echo "=== Step 3: SuperDec inference ==="
SUPERDEC=/work/courses/3dv/team39/superdec
for i in $(seq 0 9); do
    scene_name="ase_scene_$i"
    output_file="data/output_npz/${scene_name}.npz"
    if [ -f "$output_file" ]; then
        echo "Scene $i inference exists, skipping"
        continue
    fi
    echo "Running SuperDec on scene $i..."
    # Copy point clouds to superdec data dir
    mkdir -p "$SUPERDEC/data/$scene_name/pc_gt"
    cp data/pointclouds/$i/*.npz "$SUPERDEC/data/$scene_name/pc_gt/"

    cd "$SUPERDEC"
    python superdec/evaluate/to_npz.py \
        checkpoints_folder="checkpoints/normalized" \
        output_dir="/work/courses/3dv/team39/superdec_tests/data/output_npz" \
        dataset=scene scene.path="data" scene.name="$scene_name" scene.z_up=true scene.gt=true \
        dataloader.batch_size=32 dataloader.num_workers=2 device=cuda
    cd /work/courses/3dv/team39/superdec_tests
done

# Step 4: Export GLB files for visualization
echo "=== Step 4: Export GLB meshes ==="
for i in $(seq 0 9); do
    npz_file="data/output_npz/ase_scene_$i.npz"
    glb_file="data/output_glb/scene_$i.glb"
    if [ -f "$glb_file" ]; then
        echo "Scene $i GLB exists, skipping"
        continue
    fi
    if [ -f "$npz_file" ]; then
        mkdir -p data/output_glb
        echo "Exporting scene $i to GLB..."
        python scripts/export_meshes.py "$npz_file" "$glb_file"
    fi
done

echo "=== All done! ==="
