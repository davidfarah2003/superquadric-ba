#!/bin/bash
# =============================================================================
# scale_real_dataset.sh -- run SuperBundle on a REAL (non-ASE) dataset
# =============================================================================
#
# PURPOSE
#   Template to evaluate the SuperBundle pipeline (VGGT -> MASt3R -> Ceres BA,
#   with/without the superquadric surface prior) on a REAL-WORLD dataset once it
#   has been downloaded and converted to WAI format. Every result so far is on
#   the SYNTHETIC ASE dataset; real-world generalization is the open blocker.
#   This script parameterizes the dataset root + Hydra config so you only have
#   to point it at data on disk.
#
# >>> RECOMMENDED DATASET: ETH3D <<<
#   ETH3D is the best fit because the wiring ALREADY EXISTS in this repo --
#   no new loader code is needed, only the data on disk:
#     * Loader class      : ETH3DWAI
#         map-anything/mapanything/datasets/wai/eth3d.py
#     * Hydra configs     : map-anything/configs/dataset/eth3d_wai/{,test/}default.yaml
#                           map-anything/configs/dataset/benchmark_518_eth3d_wai.yaml
#                             (NEW: ETH3D-only sparse benchmark wrapper added with
#                              this template; the stock benchmark_518_eth3d_snpp_tav2
#                              mixes in ScanNet++ + TartanAir and is NOT real-only)
#     * Real-world        : indoor + outdoor scans, REAL camera poses + GT depth
#                           (is_synthetic=False, is_metric_scale=True in the loader)
#     * 13 test scenes -> better statistics than the 10 synthetic ASE scenes
#   Alternatives with loaders present (see map-anything/mapanything/datasets/wai/):
#   scannetpp.py, tav2_wb.py, mpsd.py, blendedmvs.py, ... all consume the SAME
#   WAI format below, so swapping DATASET= + DATA_ROOT= is all that changes.
#
#   CAVEAT for the surface prior: ASE ships per-scene SuperDec superquadric
#   files (compose/data/output_npz/ase_scene_*.npz) AND the prior is registered
#   with a GT-camera-centre Sim(3) (journal blocker #1). For a real dataset you
#   must EITHER (a) run only reprojection BA modes (none / mast3r / superbundle),
#   which need NO superquadric file -- start here -- OR (b) first build SuperDec
#   decompositions for the real scenes (compose/slurm/run_all.sh pipeline) and
#   solve the GT-free registration before superbundle_surface is meaningful.
#   This template defaults MODES to reprojection-only for that reason.
#
# -----------------------------------------------------------------------------
# EXPECTED ON-DISK FORMAT (WAI)  -- what you must produce before launching
# -----------------------------------------------------------------------------
#   1) DATA_ROOT/<scene_name>/scene_meta.json  + per-frame rgb/depth (WAI), one
#      dir per scene. Mirror compose/data/wai/0/ (the ASE example already on
#      disk) -- scene_meta.json holds intrinsics + C2W poses + frame paths.
#      Convert raw downloads with the WAI tooling, e.g. the ASE converter
#      compose/scripts/convert_ase_to_wai.py is the reference pattern; ETH3D
#      uses map-anything's own WAI conversion (see the WAI utils referenced by
#      mapanything/utils/wai/core.py: load_data / load_frame).
#   2) A precomputed scene-list metadata file the loader reads:
#         <DATASET_METADATA_DIR>/test/eth3d_scene_list_test.npy
#      (ETH3DWAI._load_data hard-codes this name; other loaders use
#      <name>_scene_list_test.npy). DATASET_METADATA_DIR defaults to
#      /work/courses/3dv/team39/compose/data/dataset_metadata
#      (machine=student_cluster's mapanything_dataset_metadata_dir).
#   3) (surface mode only) DATA_ROOT/../output_npz/<scene>.npz SuperDec files.
#
#   Sanity check before sbatch:
#     ls $DATA_ROOT/*/scene_meta.json
#     ls $DATASET_METADATA_DIR/test/eth3d_scene_list_test.npy
#
# -----------------------------------------------------------------------------
# HOW TO LAUNCH  (always sbatch -- never the login node)
# -----------------------------------------------------------------------------
#   cd /work/courses/3dv/team39 && mkdir -p logs
#   # ETH3D, reprojection-only BA, 8 views, 1000 tuples (defaults):
#   sbatch --export=ALL,DATA_ROOT=/work/courses/3dv/team39/compose/data/eth3d \
#       ba/eval/scale_real_dataset.sh
#   # override the grid / config:
#   sbatch --export=ALL,\
# DATASET=benchmark_518_eth3d_snpp_tav2,\
# DATA_ROOT=/work/courses/3dv/team39/compose/data/eth3d,\
# VIEWS="4 8",MODES="none mast3r",NO_OF_DATAPOINTS=2000 \
#       ba/eval/scale_real_dataset.sh
#
#   Tunable env vars:
#     DATASET            Hydra dataset config name      (default benchmark_518_eth3d_wai)
#                        must define `test_dataset` (the benchmark_* wrappers do;
#                        a bare loader config like eth3d_wai does NOT and will fail)
#     LOADER             loader sub-config namespace for the test.ROOT override
#                        (default eth3d_wai). This is the per-loader group whose
#                        test.ROOT / test.dataset_metadata_dir get redirected to
#                        DATA_ROOT; it is NOT the same as DATASET (the wrapper).
#     DATA_ROOT          REQUIRED: dir of WAI scenes    (no default -> aborts)
#     DATASET_METADATA_DIR  scene-list .npy parent dir  (default compose/data/dataset_metadata)
#     VIEWS              space-separated view counts     (default "8")
#     MODES              BA modes                        (default "none mast3r")
#                        surface mode also needs SuperDec npz + GT-free reg (see above)
#     NO_OF_DATAPOINTS   sampled view-tuples per run     (default 1000)
#     EXTRA              extra raw Hydra overrides appended verbatim
#
# RUNTIME / BUDGET: ~125 datapoints/min (CLUSTER.md). runs = |VIEWS|x|MODES|.
#   Estimate sum(NO_OF_DATAPOINTS)/125 min; keep --time tight (800 GPU-h total).
#
# MONITOR
#   squeue -u $USER
#   tail -f logs/<jobid>.out ; tail -f logs/<jobid>.err
#   sacct -u $USER --starttime=today --format=JobID,Elapsed,State
#   ls /work/courses/3dv/team39/logs/benchmark_real_${DATASET}/views_*/
#   python ba/eval/summarize_runs.py \
#       --runs_root /work/courses/3dv/team39/logs/benchmark_real_${DATASET}
#
# NOTE: 1-job-per-user QOS -- a 2nd sbatch pends until this ends.
# =============================================================================
#SBATCH --account=3dv
#SBATCH --job-name=scale-real-ds
#SBATCH --output=/work/courses/3dv/team39/logs/%j.out
#SBATCH --error=/work/courses/3dv/team39/logs/%j.err
#SBATCH --gpus=5060ti:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

set -euo pipefail

# --- Environment (matches run.sh / CLUSTER.md) -------------------------------
export CUDA_HOME=/cluster/data/cuda/x86_64/13.0.2
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/work/courses/3dv/team39/lib:${LD_LIBRARY_PATH:-}"
source /work/courses/3dv/team39/envs/3dv/bin/activate

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/courses/3dv/team39/checkpoints/hf_cache

# --- Tunables ----------------------------------------------------------------
DATASET="${DATASET:-benchmark_518_eth3d_wai}"
LOADER="${LOADER:-eth3d_wai}"
DATA_ROOT="${DATA_ROOT:-}"
DATASET_METADATA_DIR="${DATASET_METADATA_DIR:-/work/courses/3dv/team39/compose/data/dataset_metadata}"
VIEWS="${VIEWS:-8}"
MODES="${MODES:-none mast3r}"
NO_OF_DATAPOINTS="${NO_OF_DATAPOINTS:-1000}"
EXTRA="${EXTRA:-}"

ROOT=/work/courses/3dv/team39

# --- Preflight: fail fast on the login-side mistakes -------------------------
if [[ -z "$DATA_ROOT" ]]; then
    echo "ERROR: DATA_ROOT is required. Pass --export=ALL,DATA_ROOT=/path/to/wai/scenes"
    echo "       (a dir whose subdirs each contain scene_meta.json). Aborting."
    exit 2
fi
if ! ls "$DATA_ROOT"/*/scene_meta.json >/dev/null 2>&1; then
    echo "ERROR: no '<scene>/scene_meta.json' found under DATA_ROOT=$DATA_ROOT"
    echo "       Convert the dataset to WAI format first (see header)."
    exit 2
fi

cd "$ROOT/map-anything"
echo "=== scale_real_dataset on $(hostname) ==="
echo "DATASET=$DATASET  DATA_ROOT=$DATA_ROOT"
echo "VIEWS=[$VIEWS]  MODES=[$MODES]  NO_OF_DATAPOINTS=$NO_OF_DATAPOINTS"
echo "scenes found: $(ls -d "$DATA_ROOT"/*/scene_meta.json 2>/dev/null | wc -l)"
echo

for nv in $VIEWS; do
    for mode in $MODES; do
        ba_suffix=""
        [[ "$mode" != "none" ]] && ba_suffix="_${mode}"
        run_dir="$ROOT/logs/benchmark_real_${DATASET}/views_${nv}${ba_suffix}"

        echo ">>> RUN dataset=$DATASET views=$nv mode=$mode -> $run_dir"
        # We point the loader's ROOT at DATA_ROOT via a Hydra override so the same
        # config works for data placed anywhere on disk. The override key is the
        # nested test.ROOT for the dataset (mirrors configs/dataset/*/test/default.yaml).
        python3 benchmarking/sparse_view/benchmark.py \
            machine=student_cluster \
            dataset=$DATASET \
            dataset.num_workers=12 \
            dataset.num_views="$nv" \
            dataset.no_of_datapoints="$NO_OF_DATAPOINTS" \
            "dataset.${LOADER}.test.ROOT=$DATA_ROOT" \
            "dataset.${LOADER}.test.dataset_metadata_dir=$DATASET_METADATA_DIR" \
            batch_size=1 \
            model=vggt \
            bundle_adjustment="$mode" \
            hydra.run.dir="$run_dir" \
            $EXTRA
        echo ">>> DONE views=$nv mode=$mode"
        echo
    done
done

echo "=== scale_real_dataset complete ==="
echo "Summarize with: python $ROOT/ba/eval/summarize_runs.py \\"
echo "    --runs_root $ROOT/logs/benchmark_real_${DATASET}"
