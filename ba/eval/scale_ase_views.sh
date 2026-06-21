#!/bin/bash
# =============================================================================
# scale_ase_views.sh -- scale SuperBundle ASE evaluation across view counts
# =============================================================================
#
# WHAT IT DOES
#   Runs the live VGGT -> MASt3R -> triangulation -> Ceres BA pipeline on the
#   ASE/WAI benchmark for a configurable grid of sparse VIEW COUNTS and BA modes
#   (none / mast3r / superbundle / superbundle_surface), over as many sampled
#   view-tuples as you ask for (`NO_OF_DATAPOINTS`). It is a thin wrapper around
#   the committed sparse-view benchmark entry point
#   (map-anything/benchmarking/sparse_view/benchmark.py) and reuses the same
#   Hydra config (`benchmark_518_ase_wai`) that the existing vggt.sh uses, so
#   the numbers are directly comparable to the current logs/.
#
#   It ALSO (optionally) dumps the per-scene, parameter-independent BA inputs to
#   a per-view-count BA cache dir (via the BA_DUMP_DIR hook in
#   ba/python/ba/__init__.py:660), so the CPU offline harness
#   (ba/eval/offline_eval.py) can later re-solve / sweep lambda & association on
#   the NEW view counts without re-touching the GPU. Set DUMP_CACHE=1 to enable.
#
# WHY THIS EXISTS
#   The current cache (compose/data/ba_cache/0..9.npz) is a FIXED 10-view setup
#   for the 10 ASE scenes only. Adding view counts (4/6/8/16/...) or more
#   sampled tuples requires re-running the GPU pipeline -- this script automates
#   that grid in one job. (Scene COUNT on disk is fixed at 10; see the README
#   note at the bottom on how to add more ASE scenes.)
#
# HOW TO LAUNCH  (do NOT run on the login node -- always sbatch)
#   cd /work/courses/3dv/team39
#   mkdir -p logs
#   # default grid (views 4,6,8 x modes none,superbundle_surface; 1000 tuples):
#   sbatch ba/eval/scale_ase_views.sh
#   # custom grid, more datapoints, also dump offline caches:
#   sbatch --export=ALL,VIEWS="4 8 16",MODES="none superbundle_surface",\
# NO_OF_DATAPOINTS=2000,DUMP_CACHE=1 ba/eval/scale_ase_views.sh
#
#   Tunable env vars (override via --export=ALL,VAR=...):
#     VIEWS            space-separated view counts      (default "4 6 8")
#     MODES            space-separated BA modes         (default "none superbundle_surface")
#                      one of: none mast3r superbundle superbundle_surface
#     NO_OF_DATAPOINTS sampled view-tuples per run      (default 1000)
#     SPARSE_COVIS     sparse_covisibility_thres        (default unset -> config default)
#     DUMP_CACHE       1 = also write offline BA caches  (default 0)
#     EXTRA            extra raw Hydra overrides appended verbatim
#
# RUNTIME / GPU-HOUR BUDGET (read before launching!)
#   GPU throughput is ~125 datapoints/min (CLUSTER.md). One run of N datapoints
#   ~= N/125 min of model forward + a CPU BA tail (2 CPUs only, see CLUSTER.md).
#   Total runs = |VIEWS| x |MODES|. Example: 3 views x 2 modes x 1000 tuples
#   ~= 6 * 8 min ~= 48 min model time + BA tails. BUDGET IS 800 GPU-h TOTAL --
#   estimate (sum of NO_OF_DATAPOINTS over the grid)/125 min and keep --time
#   tight. Edit --time below if your grid is large.
#
# MONITOR
#   squeue -u $USER                      # is it running / pending?
#   tail -f logs/<jobid>.out             # live stdout (per-run progress)
#   tail -f logs/<jobid>.err             # live stderr (Hydra/torch errors)
#   sacct -u $USER --starttime=today \
#       --format=JobID,JobName,Elapsed,State,MaxRSS   # elapsed + GPU-h used
#   # results land here (one dir per views/mode):
#   ls /work/courses/3dv/team39/logs/benchmark_sparse_ase_vggt/views_*/
#   cat logs/benchmark_sparse_ase_vggt/views_8_superbundle_surface/*per_dataset_results.json
#   # offline caches (if DUMP_CACHE=1):
#   ls /work/courses/3dv/team39/compose/data/ba_cache_views/views_*/
#
# NOTE: 1-job-per-user QOS (CLUSTER.md) -- a 2nd sbatch pends until this ends.
# =============================================================================
#SBATCH --account=3dv
#SBATCH --job-name=scale-ase-views
#SBATCH --output=/work/courses/3dv/team39/logs/%j.out
#SBATCH --error=/work/courses/3dv/team39/logs/%j.err
#SBATCH --gpus=5060ti:1
#SBATCH --mem=32G
#SBATCH --time=1-20:00:00

set -euo pipefail

# --- Environment (matches run.sh / vggt.sh; shared venv per CLUSTER.md) -------
export CUDA_HOME=/cluster/data/cuda/x86_64/13.0.2
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/work/courses/3dv/team39/lib:${LD_LIBRARY_PATH:-}"
source /work/courses/3dv/team39/envs/3dv/bin/activate

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/courses/3dv/team39/checkpoints/hf_cache

# --- Tunables (env-overridable) ----------------------------------------------
VIEWS="${VIEWS:-4 6 8}"
MODES="${MODES:-none superbundle_surface}"
NO_OF_DATAPOINTS="${NO_OF_DATAPOINTS:-1000}"
SPARSE_COVIS="${SPARSE_COVIS:-}"
DUMP_CACHE="${DUMP_CACHE:-0}"
EXTRA="${EXTRA:-}"

ROOT=/work/courses/3dv/team39
DATASET=benchmark_518_ase_wai
cd "$ROOT/map-anything"

echo "=== scale_ase_views on $(hostname) ==="
echo "VIEWS=[$VIEWS]  MODES=[$MODES]  NO_OF_DATAPOINTS=$NO_OF_DATAPOINTS"
echo "SPARSE_COVIS='${SPARSE_COVIS:-<config default>}'  DUMP_CACHE=$DUMP_CACHE"
echo

for nv in $VIEWS; do
    for mode in $MODES; do
        ba_suffix=""
        [[ "$mode" != "none" ]] && ba_suffix="_${mode}"
        run_dir="$ROOT/logs/benchmark_sparse_ase_vggt/views_${nv}${ba_suffix}"

        covis_override=""
        [[ -n "$SPARSE_COVIS" ]] && covis_override="sparse_covisibility_thres=$SPARSE_COVIS"

        # Optional offline-eval cache dump for THIS view count (surface modes only,
        # since the SQ surface_npz_path is only set for superbundle_surface).
        if [[ "$DUMP_CACHE" == "1" && "$mode" == "superbundle_surface" ]]; then
            export BA_DUMP_DIR="$ROOT/compose/data/ba_cache_views/views_${nv}"
            mkdir -p "$BA_DUMP_DIR"
            echo ">>> BA_DUMP_DIR=$BA_DUMP_DIR"
        else
            unset BA_DUMP_DIR || true
        fi

        echo ">>> RUN views=$nv mode=$mode -> $run_dir"
        python3 benchmarking/sparse_view/benchmark.py \
            machine=student_cluster \
            dataset=$DATASET \
            dataset.num_workers=12 \
            dataset.num_views="$nv" \
            dataset.no_of_datapoints="$NO_OF_DATAPOINTS" \
            batch_size=1 \
            model=vggt \
            bundle_adjustment="$mode" \
            hydra.run.dir="$run_dir" \
            $covis_override \
            $EXTRA
        echo ">>> DONE views=$nv mode=$mode"
        echo
    done
done

echo "=== scale_ase_views complete ==="
echo "Summarize with: python $ROOT/ba/eval/summarize_runs.py \\"
echo "    --runs_root $ROOT/logs/benchmark_sparse_ase_vggt"
