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
│   │   └── ase_scene_N/       # ASE scenes (created by our pipeline)
│   │       └── pc_gt/         # Per-object point clouds (.npz)
│   ├── scripts/               # download_checkpoints.sh, run_on_scene.sh
│   ├── demo_viser.py          # Interactive single-object demo
│   └── demo_planning.py       # RRT* path planning with superquadrics
│
└── compose/                   # Our test scripts and data pipelines
    ├── CLAUDE.md              # Local LLM instructions
    ├── docs/                  # Documentation (this folder)
    │   └── project_structure.md
    ├── scripts/               # Python pipeline / analysis scripts
    │   ├── convert_ase_to_wai.py       # TA-provided: ASE → WAI format
    │   ├── extract_pointclouds.py      # WAI → per-object point clouds (.npz)
    │   ├── export_meshes.py            # SuperDec output → .glb for viewing
    │   ├── compute_covisibility.py     # Pairwise covisibility matrices for WAI scenes
    │   ├── covisibility_evaluation.py  # Plots covisibility-vs-step distributions
    │   ├── covisibility_pair_examples.py  # Sample image pairs at given covisibility
    │   ├── prepare_benchmark_data.py   # Generate metadata for map-anything benchmark
    │   ├── render_scene_vs_superdec.py # Render side-by-side comparison images
    │   └── viz_vggt_aria.py            # VGGT inference + viz on ASE WAI scenes
    │
    ├── slurm/                 # Sbatch wrappers (submit from compose/)
    │   ├── run_all.sh            # Full pipeline (WAI → pointclouds → inference → GLB)
    │   ├── run_superdec_ase.sh   # SuperDec inference on a single scene
    │   ├── run_benchmark_prep.sh # Scene metadata + covisibility (prep step)
    │   ├── run_benchmark.sh      # Dense N-view VGGT benchmark on ASE
    │   └── run_vggt_viz.sh       # VGGT viz job
    │
    ├── utils/                 # Helpers
    │   ├── ase_downloader.py     # ASE dataset downloader (from projectaria_tools)
    │   ├── npy_to_csv.py
    │   └── plot_results.py
    │
    ├── test_superdec.py       # Unit tests (imports, forward pass, normalization)
    │
    ├── *.json                 # ASE download URL files (CDN, ATEK, mesh)
    │
    └── data/                  # (gitignored)
        ├── ase/               # Raw ASE scenes (rgb/, depth/, instances/, ...)
        ├── wai/               # WAI-converted scenes (shared with map-anything)
        ├── pointclouds/       # Extracted per-object point clouds
        ├── dataset_metadata/  # Scene lists + covisibility for benchmark
        ├── covisibility_analysis/  # Covisibility figures and example pairs
        ├── compare/           # VGGT viz outputs
        ├── output_npz/        # SuperDec inference results
        └── output_glb/        # Exported GLB meshes for visualization
```

## Data Flow

```
ASE scene (rgb, depth, instances, trajectory)
    │
    └──→ convert_ase_to_wai.py ──→ WAI format (undistorted pinhole + depth + instances)
                                        │
                                        ├──→ extract_pointclouds.py ──→ per-object .npz (4096 pts each)
                                        │                                    │
                                        │                                    └──→ superdec/evaluate/to_npz.py
                                        │                                            │
                                        │                                            └──→ output .npz with superquadric params
                                        │                                                 │
                                        │                                                 └──→ export_meshes.py ──→ .glb
                                        │
                                        └──→ map-anything (future)
```

## Key Formats

| Format | Description | Used by |
|--------|-------------|---------|
| ASE raw | Fisheye RGB + uint16 depth + uint16 instances + trajectory.csv | Source data |
| WAI | Undistorted pinhole images + EXR depth + scene_meta.json | Shared intermediate (superdec + map-anything) |
| Point cloud NPZ | `points` key, float32 `[N, 3]`, N>=4096 | SuperDec input |
| Output NPZ | `scale`, `rotation`, `translation`, `exponents`, `exist`, `assign_matrix`, `pc`, `names` | SuperDec output |
| GLB | 3D mesh file, viewable in VS Code or browser | Visualization |
