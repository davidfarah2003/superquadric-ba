# SuperBundle — Journal Experiment Plan & Status

Scope: sparse-view camera-pose estimation. Pipeline = VGGT init -> MASt3R correspondences ->
midpoint triangulation -> Ceres BA with a one-sided superquadric (SQ) surface prior + EM re-association.
Headline metric: `pose_auc_5` (pose AUC@5deg, %, mean of per-scene AUCs). Data on disk: ASE only,
first 10 scenes; 10 cached pre-BA problems in `compose/data/ba_cache/{0..9}.npz`.

All numbers below were computed by running code on CPU from already-saved artifacts (no pipeline re-run).
Numbers not yet computed are explicitly marked BLOCKED / NOT YET AVAILABLE — nothing is invented.

---

## 1. Results computed now

### 1a. Statistical significance of the surface prior (blocker #2)
Ours (lambda=15) vs Baseline (lambda=0) on per-scene `pose_auc_5`, paired by scene.
Source: `logs/sweep_v{6,8}_lam{0,15.0}/10 @ ASEWAI_per_scene_results.json`.
Method: paired gain, 20000-resample bootstrap 95% CI, 20000 sign-flip permutation, scipy Wilcoxon (two-sided).
Script: `ba/eval/exp_significance.py`; outputs in `ba/eval/analysis/significance.{json,csv,md}`, `significance_gain.png`.

| Views | n | Base mean | Ours mean | Mean gain | std | sem | Bootstrap 95% CI | Wilcoxon p | Perm p | W/L/T |
|------:|--:|----------:|----------:|----------:|----:|----:|------------------|-----------:|-------:|:-----:|
| 6 | 10 | 29.600 | 31.067 | +1.467 | 3.044 | 0.963 | [0.000, 3.467] | 0.1088 | 0.2525 | 3/0/7 |
| 8 | 10 | 27.929 | 28.000 | +0.071 | 1.947 | 0.616 | [-1.000, 1.286] | 0.8927 | 1.0000 | 3/2/5 |

- 6 views: gain scene-concentrated (only scenes 2 +4.0, 6 +9.33, 9 +1.33 changed; all non-negative). NOT significant; CI lower bound touches 0.
- 8 views: effectively zero gain, clearly NOT significant (CI straddles 0).
- Conclusion: prior gain is marginal, scene-concentrated (driven by 1-3 scenes, esp. scene 6 @ 6 views), and NOT statistically significant at either available view count -> substantiates blocker #2.
- **PARTIAL**: views=4 has no `sweep_v4_*` dirs; views=10 has no `sweep_v10_lam0/` and `sweep_v10_lam15.0/` holds only `benchmark.log` (crashed). 4 and 10 require GPU sweeps to compute.
- Caveat: numbers are still GT-dependent (prior registered via GT camera centres; blocker #1).

### 1b. Additional pose metrics (VGGT-only vs BA baseline vs Ours)
Source: `ba/eval/analysis/pose_decomp.npz` (10 scenes x 45 pairs = 450). Script: `ba/eval/exp_extra_metrics.py`.
Configs: VGGT-only=`raw`, BA baseline=`reproj` (lam=0), Ours=`surface` / `surface_snap` (surface + Manhattan snap).
Validation: AUC@5 combined reproduces `pose_decomp.json` exactly (raw 9.33, reproj 28.93, surface 28.98).

(1) Combined pose AUC (%) @ 1 / 3 / 5 / 10 deg:
```
VGGT-only:            1.33 /  5.63 /  9.33 / 17.13
BA baseline (reproj):22.44 / 26.30 / 28.93 / 34.31
Ours (surface):      22.67 / 26.52 / 28.98 / 34.36
Ours (surface+snap): 22.89 / 26.59 / 29.11 / 34.51
```
(2) AUC@5 rotation-only / translation-only / combined (%):
```
VGGT-only:            13.73 / 15.42 /  9.33
BA baseline (reproj): 34.31 / 32.44 / 28.93
Ours (surface):       33.82 / 32.18 / 28.98
Ours (surface+snap):  34.31 / 32.58 / 29.11
```
- BA gives a huge jump over raw VGGT (9.33 -> 28.93 @5); surface+snap adds only +0.18 over reproj -> marginal (consistent with blocker #2).
- Combined < both rot-only and trans-only for every config => `pose_auc_5` is jointly bound by rotation AND translation.

(3) ATE = RMSE of camera centres after Sim(3)-with-scale alignment to GT centres (scene/gauge-free units), mean over 10 scenes:
```
VGGT-only (raw, from cache): per-scene [1.27,2.68,0.36,2.31,3.62,0.39,2.25,2.49,0.26,2.44]; mean=1.806, median=2.278
BA baseline (reproj) ATE:    NOT YET AVAILABLE
Ours (surface_snap) ATE:     NOT YET AVAILABLE
```
- **PARTIAL**: reproj + surface_snap ATE require a per-scene BA re-solve (EM, Ceres). The background run did not finish; `ba/eval/analysis/extra_metrics.json` is not on disk yet. Re-run to produce it (command in section 2).
- Note: this ATE is camera-centre alignment only and in scene units; live-harness `pose_ate_rmse` uses pointmap normalisation absent from the cache, so absolute scale differs from `logs/` but relative ordering is meaningful.

---

## 2. Experiments staged & ready to run

### CPU-only (runnable now on the 10 cached scenes; no GPU)
- **Finish ATE in 1b** — `ba/eval/exp_extra_metrics.py`
  - `envs/3dv/bin/python ba/eval/exp_extra_metrics.py`  (full; resolves reproj + surface_snap ATE via Ceres EM)
  - `envs/3dv/bin/python ba/eval/exp_extra_metrics.py --no-ba`  (fast: AUC + raw-ATE only)
  - Cost: full ~tens of min (Ceres per scene; account capped at 2 CPU/GPU); `--no-ba` seconds.
- **Hyperparameter + robustness ablations** — `ba/eval/exp_ablations.py`
  - `envs/3dv/bin/python ba/eval/exp_ablations.py --cache_dir compose/data/ba_cache --only all --max_points 5000 --jobs 10 --out ba/eval/analysis/ablations.json`
  - Sweeps one-at-a-time around baseline (lam=15, assoc=0.15, huber_thr=2.0): `lambda {0,5,15,30,50,100,200}`, `huber {0,1,2,4,8}`, `assoc {0.05..0.50}`, `surface_huber {0,0.5,1,2}`, `em {outer 1,2,3,5; +warmup; +inner50}` (ports EM loop from `ba/python/ba/__init__.py:714-749`), `n_primitives` via SUPERDEC existence threshold {0.1..0.9}, and `perturb_sq` (jitter SQ trans/scale/orient + random primitive drop -> graceful-degradation test).
  - Cost: subsampled suite ~10-25 min; full-fidelity = hours (stage on Slurm).
  - Caveat: ablations vary the prior, NOT the GT registration (blocker #1 untouched).
- **Run aggregation** — `ba/eval/summarize_runs.py` (stdlib only, login-node safe)
  - `python ba/eval/summarize_runs.py <logs_or_views_dir> --csv ba/eval/analysis/runs.csv`

### GPU / sbatch (STAGED, NOT submitted)
- **View-count x BA-mode scaling on ASE/WAI** — `ba/eval/scale_ase_views.sh`
  - `sbatch ba/eval/scale_ase_views.sh`  (env-tunable: `VIEWS="4 6 8"`, `MODES="none superbundle_surface"`, `NO_OF_DATAPOINTS`, `SPARSE_COVIS`, `DUMP_CACHE`, `EXTRA`)
  - `DUMP_CACHE` populates `compose/data/ba_cache_views/views_<N>/` so the CPU harness can later re-solve new view counts. Fills the missing views=4/10 sweeps for blocker #2.
  - Cost: ~125 datapoints/min/GPU; scales with VIEWS x MODES x NO_OF_DATAPOINTS.
- **Real-dataset run (ETH3D recommended)** — `ba/eval/scale_real_dataset.sh` + new config `map-anything/configs/dataset/benchmark_518_eth3d_wai.yaml`
  - `DATA_ROOT=/path/to/eth3d_wai sbatch ba/eval/scale_real_dataset.sh`  (`DATA_ROOT` required; `LOADER`/`DATASET`/`VIEWS`/`MODES`/`NO_OF_DATAPOINTS` tunable)
  - Defaults to reprojection-only modes (surface mode needs SuperDec files + GT-free registration). ETH3D loader `ETH3DWAI` + configs already exist; real-world, 13 scenes, `is_synthetic=False`.
  - Cost: GPU; depends on scene/datapoint count. **Data-blocked** until ETH3D in WAI format is on disk (see section 3).

All sbatch headers follow `run.sh`/`vggt.sh` conventions (account=3dv, 5060ti:1, 32G, shared venv, CUDA_HOME, no `--cpus-*` per 2-CPU/GPU cap). Nothing submitted.

---

## 3. Blocked experiments

| Experiment | Blocker | Needed to unblock |
|---|---|---|
| Significance at views=4 and 10 | Per-scene result files absent (`sweep_v4_*` missing; `sweep_v10_lam15.0` crashed, no `sweep_v10_lam0`) | Run `scale_ase_views.sh` with `VIEWS="4 10"` and both lam configs (GPU) |
| reproj / surface_snap ATE (metric 1b-3) | Background Ceres EM re-solve unfinished; `extra_metrics.json` not written | Re-run `exp_extra_metrics.py` (CPU) to completion |
| Real-world generalization (ETH3D / any non-ASE) | No real dataset on disk (only synthetic ASE x10); only Hydra config stubs exist for eth3d/scannetpp/replica | Download ETH3D, convert to WAI (`<scene>/scene_meta.json` + rgb/depth + scene-list npy), then `scale_real_dataset.sh` (GPU). SuperDec npz needed for surface mode |
| More scenes for statistics | ASE count fixed at 10 on disk | `compose/` download + WAI + SuperDec pipeline, then GPU re-run (out of scope for staging scripts) |
| GT-FREE surface prior (blocker #1) | Code change, not data: prior registered via `umeyama_sim3_pred_to_world(cam_centres, gt_centres)` in 3 places (`ba/python/ba/__init__.py:629-647`, `ba/eval/offline_eval.py:124-139`, `ba/eval/strat_common.py:73-77`) | Implement Variant B (Sim3-ICP of pred cloud to SQ surface; pure-Python drop-in, CPU). Variant C (global Sim3 block in `ba/src/mast3r_sq_ba.cpp`) needs C++ rebuild |

Note: `pose_auc_5` SCORING also uses GT (`offline_eval.py:262-329` Sim3 alignment to GT centres) — that is standard evaluation protocol and must stay; do not conflate with the prior's GT dependence.

---

## 4. Mapping to journal blockers

**GT-free registration (blocker #1)** — Status: still GT-dependent in all paths; every result above inherits it. Variant B (Sim3-ICP pred cloud -> SQ surface) is the lowest-risk CPU-only drop-in (`ba/eval/strat_common.py:73-77` is the swap point); Variant C is the journal-grade C++ follow-up. Not yet implemented or measured.
  - Next action: implement `ba/eval/reg_icp.py` (Variant B) and score it against the GT-registration baseline on the 10 cached scenes.

**Significance of gains (blocker #2)** — Status: computed at views 6 & 8; gain marginal, scene-concentrated, NOT significant (sec 1a). Views 4 & 10 missing.
  - Next action: run `scale_ase_views.sh` for `VIEWS="4 10"` (both lam configs) to complete the view-count significance table.

**Breadth / real data** — Status: synthetic-only (ASE x10); fully data-blocked. ETH3D scaffolding (script + Hydra config) is staged.
  - Next action: obtain ETH3D in WAI format on disk, then `sbatch scale_real_dataset.sh`.

**Baselines** — Status: VGGT-only vs BA-baseline (reproj) vs Ours quantified at AUC@1/3/5/10 + rot/trans split (sec 1b); BA huge over raw VGGT, surface prior marginal over BA.
  - Next action: finish reproj/surface_snap ATE (`exp_extra_metrics.py`) so the baseline comparison includes ATE alongside AUC.

**Ablations / robustness** — Status: harness `exp_ablations.py` built and smoke-tested on scene 0; full sweep not yet executed.
  - Next action: run the full ablation suite (`exp_ablations.py --only all`) and tabulate `pose_auc_5` vs each axis incl. `perturb_sq` robustness.

---

## Artifact index
- Results: `ba/eval/analysis/significance.{json,csv,md}`, `significance_gain.png`; (`extra_metrics.json` pending re-run)
- Scripts (CPU): `exp_significance.py`, `exp_extra_metrics.py`, `exp_ablations.py`, `summarize_runs.py`
- Scripts (GPU, staged): `scale_ase_views.sh`, `scale_real_dataset.sh`; config `map-anything/configs/dataset/benchmark_518_eth3d_wai.yaml`
- Offline harness: `ba/eval/offline_eval.py`, `ba/eval/strat_common.py`
