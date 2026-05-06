# D-INFK Student Cluster Guide — 3D Vision (252-0579-00L, 2026S)

**Group 39** | Course: 3D Vision | Semester: Spring 2026

## Official Documentation

- [ETH Student Cluster Computing Guide](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentCluster)
- [Technical Node Information](https://www.isg.inf.ethz.ch/Main/ServicesClusterComputingStudentClusterTechnicalInformation)

## Quick Reference

| Item | Value |
|---|---|
| SSH login | `ssh <eth-username>@student-cluster.inf.ethz.ch` |
| Slurm account | `3dv` (required: `--account=3dv`) |
| Team folder | `/work/courses/3dv/team39` |
| Scratch space | `/work/scratch/<eth-username>` |
| Home directory | `~` (20 GB, persistent until semester end) |
| GPU hours budget | **800 hours total** (extensions possible but discouraged) |
| Team storage quota | **100 GB** (hard limit, shared across team) |
| CUDA toolkit | `/cluster/data/cuda/x86_64/13.0.2` (cluster-provided) |
| PyTorch | `2.10.0+cu130` (shared venv at `/work/courses/3dv/team39/envs/3dv`) |

## Storage

### Persistent (use for important data)

- **Home (`~`)**: 20 GB per user. Persists until access is revoked at semester end.
- **Team folder (`/work/courses/3dv/team39`)**: 100 GB shared across the team. Persists for the course duration. Store datasets, checkpoints, and final results here.

### Temporary (do NOT rely on for anything you want to keep)

- **Scratch (`/work/scratch/<username>`)**: 100 GB / 100k files max. **Auto-deleted** on a daily schedule (23:00):
  - < 10 GB used: files older than **7 days** deleted
  - 10–50 GB used: files older than **2 days** deleted
  - \> 50 GB used: files older than **1 day** deleted
  - Touching files to reset mtime does not prevent cleanup.
- **`/tmp`**: Per-node ephemeral storage. Gone when the job ends.

### Storage Rules

- The entire `/work/courses/3dv` has a **5 TB hard limit** across all 50 teams. If it fills up, **everyone is blocked**.
- Clean up unused files frequently — intermediate outputs, old checkpoints, etc.
- Never store large datasets that can be re-downloaded. Keep a download script instead.

## Running Jobs (Slurm)

The cluster uses **Slurm** for job scheduling. Always use Slurm — do not run GPU workloads on login nodes.

### GPU Types

| GPU | Count | VRAM | Compute Capability | Slurm name | Priority |
|---|---|---|---|---|---|
| RTX 5060 Ti | 32 (4 nodes x 8) | 16 GB | 12.0 | `5060ti` | 1 (used first) |
| RTX 2080 Ti | 32 (4 nodes x 8) | 11 GB | 7.5 | `2080ti` | 2 |
| GTX 1080 Ti | 192 (24 nodes x 8) | 11 GB | 6.1 | `1080ti` | 3 |
| GB10 (ARM) | 6 (6 nodes x 1) | 128 GB shared | 12.1 | `gb10` | Must request explicitly |

Request a specific GPU type with `--gpus=<name>:<count>`, e.g. `--gpus=5060ti:1`.
Without a specific type (`--gpus=1`), Slurm assigns by priority (5060 Ti first).

> **Gotcha**: `--gpus` cannot be combined with `--cpus-per-task` — Slurm rejects it as "Specifying TRES per task is not allowed". Use `--gpus=1` alone (you get 2 CPUs by default).

### Interactive GPU Session

```bash
srun --account=3dv --gpus=5060ti:1 --mem=24G --time=02:00:00 --pty bash
```

### Batch Job Script

Create a file like `run.sh`:

```bash
#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=3dv-g39
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=04:00:00

# Activate shared team venv
source /work/courses/3dv/team39/envs/3dv/bin/activate

# Project root
cd /work/courses/3dv/team39

# Run whatever is passed as arguments
"$@"
```

Submit with:

```bash
mkdir -p logs
sbatch run.sh
```

### Useful Slurm Commands

```bash
squeue -u $USER          # Check your running/pending jobs
scancel <job-id>          # Cancel a job
scancel -u $USER          # Cancel all your jobs
sacct -u $USER --starttime=today  # Today's job history
sinfo                     # Cluster node status
```

## Environment Setup

### CUDA Toolkit

The default system NVCC (12.0) does not support the RTX 5060 Ti (compute 12.0 / Blackwell). The cluster provides newer CUDA toolkits at `/cluster/data/cuda/x86_64/`. Use CUDA 13.0 to match PyTorch cu130.

Available versions: `ls /cluster/modules/cuda/` (9.2 through 13.1).

Add to your `~/.bashrc`:

```bash
export CUDA_HOME=/cluster/data/cuda/x86_64/13.0.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
```

### C++ build deps for `ba/` (Ceres + SuiteSparse + glog + …)

The `ba/` Ceres-based BA package builds against a team-shared CMake prefix at
`/work/courses/3dv/team39/envs/3dv-cmake-prefix/usr` populated from the Ubuntu
24.04 (`noble`) `libceres-dev` / `libeigen3-dev` / `libsuitesparse-dev` /
`libglog-dev` / `libgflags-dev` / `libabsl-dev` / `libmetis-dev` /
`libunwind-dev` apt packages plus their matching runtime `.so` packages. All
shared libs in the prefix have a `RUNPATH` patched in (via `patchelf`) so the
resulting `ba` extensions resolve their transitive deps without
`LD_LIBRARY_PATH`.

The legacy `team39/lib/libceres.so.4` is a conda build with `libglog.so.2`,
incompatible with the `libglog.so.1` in the prefix. The CMakeLists pins the
prefix first in `CMAKE_INSTALL_RPATH` so the apt build wins at runtime.

Rebuild from scratch:

```bash
cd /work/courses/3dv/team39/ba && rm -rf build && mkdir build && cd build
cmake .. && cmake --build . -j$(nproc)
```

If you ever need to refresh the dev prefix (e.g. after a security update on
the noble repos):

```bash
WORK=/work/scratch/$USER/ceres-build
mkdir -p "$WORK" && cd "$WORK"
apt-get download libceres-dev libeigen3-dev libgflags-dev libgoogle-glog-dev \
    libsuitesparse-dev libmetis-dev libabsl-dev libcxsparse4 libunwind-dev \
    libgflags2.2 libgoogle-glog0v6t64 libabsl20220623t64 libceres4t64 \
    libcholmod5 libamd3 libcamd3 libccolamd3 libcolamd3 libspqr4 \
    libsuitesparseconfig7 libunwind8 libmetis5
PREFIX=/work/courses/3dv/team39/envs/3dv-cmake-prefix
for d in *.deb; do dpkg-deb -x "$d" "$PREFIX"; done
RUNPATH="$PREFIX/usr/lib/x86_64-linux-gnu:/work/courses/3dv/team39/lib"
for so in $(find "$PREFIX/usr/lib" -maxdepth 1 -type f -name "*.so*"); do
    /work/courses/3dv/team39/envs/3dv/bin/patchelf --set-rpath "$RUNPATH" "$so" 2>/dev/null
done
```

### Python venv

The team uses a **single shared venv** at `/work/courses/3dv/team39/envs/3dv` (~9.4 GB). Activating it from any team member's session works the same way:

```bash
source /work/courses/3dv/team39/envs/3dv/bin/activate
```

`run.sh` already does this. Don't create per-user venvs in `~/envs/3dv` — home is only 20 GB and a full DL stack fills it.

If you need to install or upgrade a package, do it in the shared venv — but coordinate with the team first, since changes affect everyone:

```bash
source /work/courses/3dv/team39/envs/3dv/bin/activate
pip install <pkg>
```

Initial setup (already done, kept here for reference):

```bash
python3 -m venv /work/courses/3dv/team39/envs/3dv
source /work/courses/3dv/team39/envs/3dv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -e superdec/
```

## GPU Hours Management

- **Budget: 800 GPU hours total** for the entire project.
- Use `sacct` to monitor consumed hours.
- Debug on CPU or with small batches before submitting long GPU jobs.
- Set reasonable `--time` limits — jobs are killed when time runs out, but unused reserved time still counts.
- Checkpoint frequently so you can resume interrupted runs.

## Tips

- **Before a long run**: verify your code works with a quick test (few iterations, small data).
- **Checkpointing**: save model checkpoints to `/work/courses/3dv/team39/checkpoints/` periodically. Name them clearly (e.g., `model_epoch_10.pt`).
- **Logging**: use `logs/` directory with Slurm's `%j` (job ID) substitution to keep logs organized.
- **Disk usage**: run `du -sh /work/courses/3dv/team39/*` regularly to monitor team storage.
- **IT support**: only contact IT for account or cluster infrastructure issues — not for Python/Linux/env debugging.
- **Debugging**: try to resolve issues yourself (docs, search, AI tools) before posting on Moodle.

## LLM Instructions

When assisting with this project on the cluster:

- All GPU work must go through Slurm (`sbatch` or `srun`). Never suggest running training directly on login nodes.
- Prefer `/work/courses/3dv/team39/` for any persistent project data. Use scratch only for throwaway intermediates.
- Be conservative with GPU hours — always suggest the minimum viable `--time` and test locally first.
- When writing training scripts, always include checkpointing logic.
- The cluster uses Linux. Assume bash shell, standard GNU tools. CUDA toolkits are at `/cluster/data/cuda/x86_64/` (versions 9.2–13.1).
- Always set `CUDA_HOME=/cluster/data/cuda/x86_64/13.0.2` in job scripts and srun commands. The default system NVCC (12.0) cannot compile for RTX 5060 Ti.
- Use `--gpus=5060ti:1` by default. Fall back to `--gpus=2080ti:1` or `--gpus=1080ti:1` if needed.
- PyTorch must be the `cu130` build (installed from `https://download.pytorch.org/whl/cu130`).
- The team uses a single shared venv at `/work/courses/3dv/team39/envs/3dv`. Do not create per-user venvs in `~/envs/3dv` — home is too small. Activate with `source /work/courses/3dv/team39/envs/3dv/bin/activate`.
- If storage is tight, suggest cleaning old outputs before creating new ones.
- The user's ETH username for SSH is their standard ETH credentials.

## Benchmarking Process

### General Notes
- For all benchmarks, you must pass the `no_of_datapoints` parameter to the benchmarking scripts to specify the number of used datapoints. The default value is 10. The GPU processes approximately **125 datapoints per minute**—use this to estimate and budget your compute time.
- You can use the `map-anything/bash_scripts/benchmark/run_multiple_benchmarks.sh` to run multiple benchmarks at once.

### Dense View Benchmarking
- To run dense view benchmarks, use the `pi3.sh`, `mapa_24v.sh` and `vggt.sh` scripts in the `map-anything/bash_scripts/benchmark/dense_n_view/` folder. Example:
  ```bash
  sbatch map-anything/bash_scripts/benchmark/dense_n_view/vggt.sh no_of_datapoints=100
  ```
- In the corresponding `*.sh` file, you can specify the different dataloaders to use for your experiments.

### Sparse View Benchmarking
- For sparse view benchmarking, use the `pi3.sh`, `mapa_24v.sh` and `vggt.sh` scripts in the `map-anything/bash_scripts/benchmark/sparse_view/` folder.
- You must specify the number of views per datapoint. If you want to use Bundle Adjustment or SuperBundle specify the `bundle_adjustment` parameter. Example:
  ```bash
  sbatch map-anything/bash_scripts/benchmark/sparse_view/vggt.sh num_views=4 no_of_datapoints=100 bundle_adjustment=superbundle
  ```
- The script will use this threshold both for the experiment and to set the output directory.

### Number of View Pairs per Covisibility Threshold for the first 10 Scenes of the ase Dataset
| Threshold | View Pairs |
|-----------|------------|
| 0.25      | 683523     |
| 0.2       | 619657     |
| 0.15      | 541403     |
| 0.1       | 443376     |
| 0.05      | 307035     |
| 0.01      | 124330     |
| 0.005     | 83393      |


