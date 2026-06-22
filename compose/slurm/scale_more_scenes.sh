#!/bin/bash
#SBATCH --account=3dv
#SBATCH --job-name=scale-scenes
#SBATCH --output=/work/courses/3dv/team39/logs/%j.out
#SBATCH --error=/work/courses/3dv/team39/logs/%j.err
#SBATCH --gpus=5060ti:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
# =============================================================================
# scale_more_scenes.sh -- prepare MORE ASE scenes and re-test the surface prior
# over them, to tighten the significance of the +1.5 six-view gain (currently
# n=10, not significant: Wilcoxon p=0.11).
#
# Per scene (all idempotent, tolerant -- one bad scene does not kill the batch):
#   download -> WAI -> point clouds -> SuperDec  (compose/slurm/run_all.sh logic)
# Then rebuild the benchmark scene list (only fully-prepared scenes) + covisibility,
# and run reproj-BA (lambda=0) vs surface-BA (lambda=15) at the chosen view counts
# using the EXACT config from run_views_sweep.sh. Per-scene pose_auc_5 lands in
# logs/scale_v<NV>_lam<LAM>/ for a significance re-test (ba/eval/exp_significance.py).
#
# A prep TIME BUDGET guarantees the benchmark still runs (on whatever is ready)
# before --time expires, so we always get a result. Resumable: rerun to continue.
#
# IMPORTANT sizing notes (learned the hard way on job 100594):
#  - benchmark.py writes its per-scene results only at the END of each run, so a
#    run killed mid-way produces NOTHING. Size DATAPOINTS_PER_SCENE so every run
#    finishes well inside --time, and order VIEWS so the HEADLINE count runs FIRST
#    (6-view here -> the +1.5 gain pair lands before any time risk).
#  - observed rate ~150 datapoints/h, so 10/scene over 50 scenes ~= 3.3 h/run.
#
# 1-job-per-user QOS: this hogs the slot until done. sbatch only.
#   sbatch compose/slurm/scale_more_scenes.sh
#   sbatch --export=ALL,SCENES="10-29",VIEWS="6",PREP_BUDGET_SEC=36000 compose/slurm/scale_more_scenes.sh
#
# Env-overridable: SCENES ("10-49"), VIEWS ("6 4"), LAMS ("15.0 0"),
#   DATAPOINTS_PER_SCENE (10), COVIS (0.6), PREP_BUDGET_SEC (64800=18h), DELETE_ASE (1).
# =============================================================================
set -uo pipefail

export CUDA_HOME=/cluster/data/cuda/x86_64/13.0.2
export PATH="$CUDA_HOME/bin:$PATH"
source /work/courses/3dv/team39/envs/3dv/bin/activate
export HF_HOME=/work/courses/3dv/team39/checkpoints/hf_cache
export HF_HUB_OFFLINE=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/work/courses/3dv/team39
COMPOSE=$ROOT/compose
SUPERDEC=$ROOT/superdec
NPZ_DIR=$COMPOSE/data/output_npz
MANIFEST=aria_synthetic_environments_dataset_download_urls.json
SCENE_LIST=$COMPOSE/data/dataset_metadata/test/ase_scene_list_test.npy

SCENES="${SCENES:-10-49}"
VIEWS="${VIEWS:-6 4}"
LAMS="${LAMS:-15.0 0}"
DATAPOINTS_PER_SCENE="${DATAPOINTS_PER_SCENE:-10}"
COVIS="${COVIS:-0.6}"
PREP_BUDGET_SEC="${PREP_BUDGET_SEC:-64800}"
DELETE_ASE="${DELETE_ASE:-1}"

expand_ids() {  # "10-49" or "10,12,15-18" -> "10 11 12 ..."
  local spec="$1" out=() tok a b i
  IFS=',' read -ra toks <<< "$spec"
  for tok in "${toks[@]}"; do
    if [[ "$tok" == *-* ]]; then a=${tok%-*}; b=${tok#*-}; for ((i=a;i<=b;i++)); do out+=("$i"); done
    else out+=("$tok"); fi
  done
  echo "${out[@]}"
}
IDS=$(expand_ids "$SCENES")

echo "=== scale_more_scenes on $(hostname) @ $(date) ==="
echo "SCENES=$SCENES -> [$IDS]"
echo "VIEWS=[$VIEWS] LAMS=[$LAMS] DATAPOINTS_PER_SCENE=$DATAPOINTS_PER_SCENE COVIS=$COVIS"
echo "PREP_BUDGET_SEC=$PREP_BUDGET_SEC DELETE_ASE=$DELETE_ASE"
echo

cd "$COMPOSE"
mkdir -p "$ROOT/logs"
START=$SECONDS
prepared=0; failed=0

for i in $IDS; do
  if (( SECONDS - START > PREP_BUDGET_SEC )); then
    echo ">>> prep time budget reached after $(((SECONDS-START)/60)) min; moving to benchmark"; break
  fi
  if [ -f "$NPZ_DIR/ase_scene_$i.npz" ]; then
    echo "[$i] already prepared, skip"; prepared=$((prepared+1)); continue
  fi
  echo "===== [$i] preparing ($(((SECONDS-START)/60)) min elapsed) ====="
  (
    set -e
    if [ ! -d "data/ase/$i" ] && [ ! -f "data/wai/$i/scene_meta.json" ]; then
      echo "[$i] download"
      python utils/ase_downloader.py --set train --scene-ids "$i" \
        --cdn-file "$MANIFEST" --output-dir data/ase --unzip True
    fi
    if [ ! -f "data/wai/$i/scene_meta.json" ]; then
      echo "[$i] convert -> WAI"
      python scripts/convert_ase_to_wai.py --scene_path "data/ase/$i" --output_path "data/wai/$i"
    fi
    [ "$DELETE_ASE" = "1" ] && rm -rf "data/ase/$i"
    if [ -z "$(ls data/pointclouds/$i/*.npz 2>/dev/null)" ]; then
      echo "[$i] extract point clouds"
      python scripts/extract_pointclouds.py --wai_path "data/wai/$i" --output_path "data/pointclouds/$i" --frame_stride 5
    fi
    echo "[$i] SuperDec"
    mkdir -p "$SUPERDEC/data/ase_scene_$i/pc_gt"
    cp data/pointclouds/$i/*.npz "$SUPERDEC/data/ase_scene_$i/pc_gt/"
    cd "$SUPERDEC"
    python superdec/evaluate/to_npz.py \
      checkpoints_folder="checkpoints/normalized" output_dir="$NPZ_DIR" \
      dataset=scene scene.path="data" scene.name="ase_scene_$i" scene.z_up=true scene.gt=true \
      dataloader.batch_size=32 dataloader.num_workers=2 device=cuda
  )
  if [ -f "$NPZ_DIR/ase_scene_$i.npz" ]; then
    echo "[$i] OK"; prepared=$((prepared+1))
  else
    echo "[$i] FAILED"; failed=$((failed+1))
  fi
done

echo
echo "=== prep done: $prepared prepared, $failed failed, $(((SECONDS-START)/60)) min ==="

echo "=== rebuild scene list from FULLY-prepared scenes (have output_npz) + covisibility ==="
cd "$COMPOSE"
python scripts/prepare_benchmark_data.py --wai_root data/wai --metadata_dir data/dataset_metadata --split test || true
# Keep only scenes that have a SuperDec npz (surface mode needs it, even at lambda=0).
READY=$(python3 - "$NPZ_DIR" "$SCENE_LIST" <<'PY'
import os, sys, glob, numpy as np
npz_dir, list_path = sys.argv[1], sys.argv[2]
ready = sorted({os.path.basename(p)[len("ase_scene_"):-4] for p in glob.glob(os.path.join(npz_dir, "ase_scene_*.npz"))}, key=int)
os.makedirs(os.path.dirname(list_path), exist_ok=True)
np.save(list_path, np.array(ready, dtype=object))
print(" ".join(ready))
PY
)
N_SCENES=$(echo "$READY" | wc -w)
echo "scene list ($N_SCENES): $READY"
python scripts/compute_covisibility.py --wai_root data/wai --device cuda --scenes $READY || \
  python scripts/compute_covisibility.py --wai_root data/wai --device cuda || true

DATAPOINTS=$(( DATAPOINTS_PER_SCENE * N_SCENES ))
echo "=== benchmark over $N_SCENES scenes, no_of_datapoints=$DATAPOINTS ==="
cd "$ROOT/map-anything"
for NV in $VIEWS; do
  for LAM in $LAMS; do
    echo "########## benchmark num_views=$NV lambda=$LAM ($N_SCENES scenes) ##########"
    python3 benchmarking/sparse_view/benchmark.py \
      machine=student_cluster dataset=benchmark_518_ase_wai dataset.num_workers=4 \
      dataset.num_views="$NV" dataset.no_of_datapoints="$DATAPOINTS" batch_size=1 model=vggt \
      bundle_adjustment=superbundle_surface sparse_covisibility_thres="$COVIS" \
      +surface_npz_dir="$NPZ_DIR" +surface_lambda="$LAM" \
      +surface_huber=2.749 +surface_assoc_max_distance=0.0372 \
      +surface_huber_threshold=1.0 +surface_em_outer=2 +surface_em_inner_iters=41 \
      +surface_em_warmup=true +surface_residual_mode=1 +surface_filter_max_aspect=0 \
      +surface_refine_sq=false +surface_sq_anchor_weight=10.0 +surface_manhattan_snap=0 \
      +surface_num_threads=4 \
      hydra.run.dir="$ROOT/logs/scale_v${NV}_lam${LAM}" || echo "!!! benchmark failed: v$NV lam$LAM"
    echo "########## DONE num_views=$NV lambda=$LAM ##########"
  done
done

echo "=== ALL DONE @ $(date): $prepared scenes prepared, $N_SCENES in scene list ==="
echo "Re-test significance: results in logs/scale_v*_lam* (per-scene json), feed to ba/eval/exp_significance.py"
