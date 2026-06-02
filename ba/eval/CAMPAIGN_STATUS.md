# Surface-BA optimization campaign — status

**Goal:** make SUPERDEC surface-augmented BA beat regular (MASt3R) BA on the
sparse-view ASE benchmark. Target metric: `pose_auc_5`. Bar = **full regular_ba
= 29.56** (surface one-shot was 19.42, i.e. the surface term was *hurting*).

## Done
- **Fast offline harness** (`ba/eval/offline_eval.py`): replays the Ceres surface
  BA + recomputes `pose_auc_5` exactly from a per-scene cache, no GPU/VGGT/MASt3R.
  Validated: reproduces live surface = **19.422** exactly.
- **Cache**: `compose/data/ba_cache/*.npz` (10 scenes, param-independent BA inputs),
  produced by the env-guarded dump in `ba/python/ba/__init__.py` (set `BA_DUMP_DIR`,
  run `compose/slurm/run_sparse_surface_benchmark.sh`).
- **Speed**: point subsampling (`max_points`) + exposed Ceres `max_num_iterations`/
  `function_tolerance`/`num_threads` (rebuilt `ba/src/mast3r_sq_ba.cpp`; defaults
  preserved so live behaviour unchanged). Proxy = `mp=5000, max_iter=50, ftol=1e-3`
  (surface reproduces 19.42 exactly under it).
- **Param BO on one-shot surface**: plateaus ~27.4 = proxy baseline → params alone
  don't beat regular BA; the *formulation* must change.
- **Structural strategies** (`ba/eval/strategies/*.py`, scored by `run_strategy.py`,
  built on `strat_common.py`): em_reassoc 26.5, two_stage_em 26.3, two_stage 26.1,
  anneal 24.9, sq_filter 19.3 (proxy regular_ba ref = 27.56). EM re-association
  (re-assign points↔SQs as BA moves them) is the winner.

## DONE (cont.)
- **em_reassoc param BO** (job 94083, COMPLETED 01:55): best **proxy pose_auc_5 =
  31.73** (vs proxy regular_ba 27.56). Best params:
  `lambda_surface=3.347, assoc_max_distance=0.0372, surface_huber=2.749,
  huber_threshold=0.738, n_outer=2, inner_iters=41, warmup=true`.
- **Live EM port STAGED** (backward-compatible, default em_outer=1 = unchanged):
  - `ba/python/ba/__init__.py`: `mast3r_bundle_adjust` gained `em_outer`,
    `em_inner_iters`, `em_warmup`; solve section runs warmup + EM loop
    (re-`assign_points_to_sqs` on moving `points` → short surface solve) when
    `em_outer>1`. Compiles clean.
  - `map-anything/benchmarking/sparse_view/benchmark.py` (vendored/gitignored):
    superbundle_surface call now passes `huber_threshold` + `em_*` from Hydra
    cfg via `args.get(key, default)`.
  - `compose/slurm/run_sparse_surface_em_benchmark.sh`: new script, tuned config
    baked in, writes to `logs/benchmark_ase_sparse_surface_em_cov06`.

## LIVE RESULT — the proxy win did NOT hold (job 94099, cov06, 10 views)
Ran the tuned EM config on the REAL benchmark (`run_sparse_surface_em_benchmark.sh`,
28 cores). Apples-to-apples vs regular BA (`logs/benchmark_ase_sparse_mast3r_cov06`):

| metric                | regular BA | EM surface | winner          |
|-----------------------|-----------:|-----------:|-----------------|
| **pose_auc_5** (↑)    |  **29.42** |  **28.93** | regular (−0.49) |
| pose_ate_rmse (↓)     |     0.3747 |     0.3741 | EM (marginal)   |
| pointmaps_abs_rel (↓) |     1.800  |     1.747  | EM              |
| z_depth_abs_rel (↓)   |     0.8226 |     0.8354 | regular         |
| metric_scale_abs_rel  |     0.6255 |     0.6402 | regular         |

**Conclusion: the EM config LOSES on the target metric (28.93 < 29.42).** The
proxy (5k-pt subsample, 50 iters) scored it 31.73 — it does NOT predict live
ranking. The whole BO was optimising an unfaithful proxy. The EM *machinery*
works (lifts one-shot surface 19.42 → 28.93, near parity), but the surface term
as tuned doesn't cross the bar. The committed params are the proxy optimum and
are kept only as a documented baseline, NOT a win.

## NEXT (corrected approach)
1. Replace the proxy with a many-core FULL-FIDELITY offline evaluator (no
   subsample, live iters/ftol; scene-parallel via `--cpus-per-gpu=32` +
   ProcessPoolExecutor). VERIFY it reproduces the live 28.93 for this config.
2. Re-run the BO against that faithful evaluator (real metric, not proxy).
3. Validate any winner live before claiming anything. Possible outcome: the
   surface term can't beat plain BA on pose for these scenes (be honest if so).

## Cluster notes (verified)
- Plugin forces gpu=1 and defaults to 2 CPUs. To get more cores use
  **`--cpus-per-gpu=N`** (verified AllocCPUS follows it); `--cpus-per-task` and
  `--ntasks=N` with a GPU are rejected. Per-job QOS cap = 32 CPUs; node core
  counts cap it (5060ti=28, 2080ti=36, 1080ti=20). 1-job-per-user QOS limit.
- BA solve thread-scaling measured empirically (see `thread_scaling.py`,
  `logs/<jid>.out`) — do not assume it uses all allocated cores.
- Bad node: studgpu-node09 (CUDA err 804) — `--exclude=studgpu-node09`.
- Do NOT run heavy compute on the login node.
