# Project Structure

## Repository Layout

```
team39/
├── CLAUDE.md                  # LLM project instructions
├── CLUSTER.md                 # ETH cluster reference (Slurm, GPUs, storage)
├── run.sh                     # Shared Slurm batch script
│
├── superdec/                  # SuperDec library (ICCV 2025, git submodule)
│   ├── superdec/
│   │   ├── superdec.py        # Main model (encoder + decoder + head)
│   │   ├── models/            # StackedPVConv encoder, transformer decoder
│   │   ├── loss/              # Chamfer distance, surface points, cuboid loss
│   │   ├── data/              # Dataset classes (ShapeNet, Scene, ScenesDataset)
│   │   ├── evaluate/          # to_npz.py — batch inference, saves predictions
│   │   ├── visualization/     # object_visualizer.py — viser-based 3D viewer
│   │   ├── utils/             # predictions_handler, ply_to_npz, transforms
│   │   └── functional/        # CUDA kernels (PVCNN voxelization, JIT-compiled)
│   ├── train/                 # Training scripts (train.py, trainer.py)
│   ├── configs/               # Hydra YAML configs (train, eval, save_npz)
│   ├── checkpoints/           # Pretrained weights
│   │   ├── normalized/        # For scene-level inference (ckpt.pt)
│   │   └── shapenet/          # For single-object inference (ckpt.pt)
│   ├── data/                  # Input data for superdec
│   │   └── ase_scene_0/       # ASE scene (created by our pipeline)
│   │       └── pc_gt/         # Per-object point clouds (.npz)
│   ├── scripts/               # download_checkpoints.sh, run_on_scene.sh
│   ├── demo_viser.py          # Interactive single-object demo
│   └── demo_planning.py       # RRT* path planning with superquadrics
│
└── superdec_tests/            # Our test scripts and data pipelines
    ├── CLAUDE.md              # Local LLM instructions
    ├── docs/                  # Documentation (this folder)
    │   └── project_structure.md
    │
    ├── test_superdec.py       # Unit tests (imports, forward pass, normalization)
    ├── extract_pointclouds.py # ASE → per-object point clouds (.npz)
    ├── convert_ase_to_wai.py  # TA-provided: ASE → WAI format (for map-anything)
    ├── ase_downloader.py      # ASE dataset downloader (from projectaria_tools)
    ├── run_superdec_ase.sh    # Slurm script for superdec inference
    │
    ├── *.json                 # ASE download URL files (CDN, ATEK, mesh)
    │
    └── data/                  # (gitignored)
        ├── ase/               # Raw ASE scenes (rgb/, depth/, instances/, ...)
        ├── wai/               # WAI-converted scenes (for map-anything, not superdec)
        ├── pointclouds/       # Extracted per-object point clouds
        └── output_npz/        # SuperDec inference results
```

## Data Flow

```
ASE scene (rgb, depth, instances, trajectory)
    │
    ├──→ convert_ase_to_wai.py ──→ WAI format (for map-anything pipeline)
    │
    └──→ extract_pointclouds.py ──→ per-object .npz (4096 pts each)
                                        │
                                        └──→ superdec/evaluate/to_npz.py
                                                │
                                                └──→ output .npz with superquadric params
                                                     (scale, rotation, translation,
                                                      exponents, existence, assignment)
```

## Key Formats

| Format | Description | Used by |
|--------|-------------|---------|
| ASE raw | Fisheye RGB + uint16 depth + uint16 instances + trajectory.csv | Source data |
| WAI | Undistorted pinhole images + EXR depth + scene_meta.json | map-anything |
| Point cloud NPZ | `points` key, float32 `[N, 3]`, N≥4096 | SuperDec input |
| Output NPZ | `scale`, `rotation`, `translation`, `exponents`, `exist`, `assign_matrix`, `pc`, `names` | SuperDec output |
