See [../CLAUDE.md](../CLAUDE.md) for cluster usage guidelines.

# superdec_tests

Test scripts and data pipelines for running SuperDec on Aria Synthetic Environments (ASE) scenes.

## Documentation

- [docs/project_structure.md](docs/project_structure.md) — full project layout, data flow, and format reference

## Quick Pipeline

```bash
# 1. Download ASE scenes
python ase_downloader.py --set train --scene-ids 0-2 \
  --cdn-file aria_synthetic_environments_dataset_download_urls.json \
  --output-dir data/ase --unzip True

# 2. Extract per-object point clouds
python extract_pointclouds.py --wai_path data/ase/0 --frame_stride 5

# 3. Copy to superdec and run inference (needs GPU via Slurm)
cp data/pointclouds/0/*.npz ../superdec/data/ase_scene_0/pc_gt/
srun --account=3dv --gpus=1 --mem=24G --time=00:10:00 bash -c \
  'source ~/envs/3dv/bin/activate && export CUDA_HOME=/cluster/data/cuda/x86_64/13.0.2 && \
   export PATH=$CUDA_HOME/bin:$PATH && cd /work/courses/3dv/team39/superdec && \
   python superdec/evaluate/to_npz.py checkpoints_folder="checkpoints/normalized" \
   output_dir="/work/courses/3dv/team39/superdec_tests/data/output_npz" \
   dataset=scene scene.path="data" scene.name="ase_scene_0" scene.z_up=true scene.gt=true \
   dataloader.batch_size=32 dataloader.num_workers=2 device=cuda'
```

## Notes

- `convert_ase_to_wai.py` is TA-provided for the **map-anything** pipeline (WAI format), not needed for superdec
- `extract_pointclouds.py` works directly on raw ASE data (depth + instance masks)
- SuperDec expects `.npz` files with key `points` (float32, `[N, 3]`, N >= 4096)
