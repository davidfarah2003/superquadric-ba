# Project Guidelines

## Documentation

- `CLUSTER.md` is the team's cluster reference. **Keep it up to date** — when you help change environment setup, GPU config, CUDA versions, Slurm settings, or any cluster-related configuration, update `CLUSTER.md` to reflect the change.
- `run.sh` is the shared Slurm batch script. Keep it consistent with `CLUSTER.md`.

## Repository structure

Single root git repo at `/work/courses/3dv/team39`. Tracked code lives in:

- `ba/` — Ceres-based bundle adjustment package (C++ extensions + Python bindings).
- `compose/` — data pipelines, SUPERDEC orchestration, visualization scripts. Has its own `CLAUDE.md`.

Vendored frameworks live in-tree but are **gitignored** and treated as upstream-as-is:

- `map-anything/` — multi-view reconstruction backbone (hosts the VGGT wrapper used by benchmarks).
- `mast3r/` — pairwise matcher (currently used by `superbundle` mode; expected to be replaced by VGGT tracks).
- `superdec/` — superquadric decomposition library (ICCV 2025).

If a vendored framework needs a code change, prefer writing a thin wrapper in `ba/` or `compose/` over editing the vendored source in place — keeps our diff against upstream empty.

Heavy/regenerable dirs are also gitignored: `envs/` (shared venv), `lib/`, `checkpoints/`, `logs/`, `data/`, `build/`, `*/build/`, `*.so`, `*.npz`, etc. See `.gitignore` for the full list.

## Cluster Operations

When assisting with cluster-related tasks (Slurm jobs, environment setup, storage, GPU usage), reference `CLUSTER.md` in this directory first. If something is unclear or not covered there, consult the official ETH Student Cluster documentation linked at the top of that guide.
