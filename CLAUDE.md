See [../CLAUDE.md](../CLAUDE.md) for cluster usage guidelines.

# compose

Test scripts and data pipelines for running SuperDec on Aria Synthetic Environments (ASE) scenes.

## Documentation

- [docs/project_structure.md](docs/project_structure.md) — full project layout, data flow, and format reference
- **Keep docs up to date** — when changing scripts, data paths, or pipeline steps, update both this file and `docs/project_structure.md`.

## Quick Pipeline

```bash
# 1. Download ASE scenes
python ase_downloader.py --set train --scene-ids 0-9 \
  --cdn-file aria_synthetic_environments_dataset_download_urls.json \
  --output-dir data/ase --unzip True

# 2. Convert to WAI format (shared with map-anything pipeline)
python scripts/convert_ase_to_wai.py --scene_path data/ase/0 --output_path data/wai/0

# 3. Extract per-object point clouds from WAI
python scripts/extract_pointclouds.py --wai_path data/wai/0 --output_path data/pointclouds/0 --frame_stride 5

# 4. Run SuperDec inference (needs GPU via Slurm)
# Copy point clouds, then run to_npz.py (see run_all.sh)

# Or just run the full pipeline on all scenes:
sbatch slurm/run_all.sh
```

## Visualization

```bash
# Export GLB (viewable in VS Code or browser)
python scripts/export_meshes.py data/output_npz/ase_scene_0.npz scene_0.glb

# Interactive viser viewer (needs GPU node + port forwarding)
srun --account=3dv --gpus=1 --mem=24G --time=01:00:00 --pty bash
cd /work/courses/3dv/team39/superdec && python superdec/visualization/object_visualizer.py \
  dataset=scene split="ase_scene_0" npz_folder="/work/courses/3dv/team39/compose/data/output_npz"
```

## Layout

- `scripts/` — Python pipeline + analysis scripts (conversion, extraction, export, covisibility, viz).
- `slurm/` — Sbatch wrappers; submit from `compose/` so log paths resolve to `compose/logs/`.
- `utils/` — Helpers (ASE downloader, plotting).
- `data/` — gitignored. Subfolders: `ase/`, `wai/`, `pointclouds/`, `output_npz/`, `output_glb/`, `dataset_metadata/`, `covisibility_analysis/`, `compare/`.

## Notes

- WAI conversion is the first step — shared intermediate format for both superdec and map-anything
- `extract_pointclouds.py` works on WAI data (depth + instance masks -> per-object point clouds)
- SuperDec expects `.npz` files with key `points` (float32, `[N, 3]`, N >= 4096)
- `slurm/run_all.sh` runs the full pipeline (WAI conversion -> point cloud extraction -> inference -> GLB export)
- All sbatch wrappers source the shared team venv at `/work/courses/3dv/team39/envs/3dv` and set `CUDA_HOME=/cluster/data/cuda/x86_64/13.0.2`. Don't combine `--gpus` with `--cpus-per-task` — Slurm rejects it (CPU count defaults to 2).
