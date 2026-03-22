See [../CLAUDE.md](../CLAUDE.md) for cluster usage guidelines.

# superdec_tests

Test scripts and data pipelines for running SuperDec on Aria Synthetic Environments (ASE) scenes.

## Directory Layout

```
superdec_tests/
  test_superdec.py          # Unit tests (imports, forward pass, normalization)
  convert_ase_to_wai.py     # TA-provided: ASE -> WAI format converter
  ase_downloader.py          # ASE scene downloader (from projectaria_tools repo)
```

## ASE -> SuperDec Pipeline

### 1. Download ASE scenes

Requires `aria_synthetic_environments_dataset_download_urls.json` from
https://www.projectaria.com/datasets/ase/ (registration required).

```bash
python ase_downloader.py \
  --set train --scene-ids 0-2 \
  --cdn-file aria_synthetic_environments_dataset_download_urls.json \
  --output-dir data/ase --unzip True
```

Each scene contains: `rgb/`, `depth/`, `instances/`, `trajectory.csv`,
`object_instances_to_classes.json`, optionally `sq.npz` (superquadric GT).

### 2. Convert ASE to WAI format

```bash
python convert_ase_to_wai.py --scene_path data/ase/0 --output_path data/wai/0
```

Outputs undistorted pinhole images (512x512 portrait), EXR depth maps,
instance segmentation masks, and `scene_meta.json` with camera params.

### 3. Extract per-object point clouds

Back-project depth maps using camera intrinsics, segment by instance ID,
aggregate across frames, downsample to 4096 points, save as `.npz` with
key `points` (shape `[N, 3]`).

Output goes to `superdec/data/<scene_name>/pc_gt/` (one `.npz` per object).

### 4. Run SuperDec inference

```bash
cd ../superdec
python superdec/evaluate/to_npz.py \
  checkpoints_folder="checkpoints/normalized" \
  output_dir="output_npz" \
  dataset=scene \
  scene.name="<scene_name>" \
  scene.z_up=true
```

### 5. Visualize results

```bash
python superdec/visualization/object_visualizer.py \
  dataset=scene split="<scene_name>" npz_folder="output_npz"
```

## SuperDec Input Format

- Per-object point clouds as `.npz` files with key `points` (float32, shape `[N, 3]`)
- N >= 4096 recommended (auto-subsampled to 4096 during loading)
- Stored in `superdec/data/<scene_name>/pc_gt/`

## Pretrained Checkpoints

Downloaded via `superdec/scripts/download_checkpoints.sh` (uses gdown):
- `checkpoints/normalized/` — for scenes (normalized input)
- `checkpoints/shapenet/` — for ShapeNet objects

## Dependencies

- `projectaria-tools` — ASE calibration, undistortion, data reading
- `opencv-python` with OpenEXR support — depth map I/O
- `trimesh` — PLY/mesh processing
- `open3d` — point cloud operations (FPS downsampling)
- Standard: `torch`, `numpy`, `scipy`, `PIL`
