# Surface-BA experiment ledger — beating regular BA with superquadrics

**Goal:** make a SUPERDEC-superquadric-augmented bundle adjustment beat plain
("regular") BA on the sparse-view ASE benchmark. **Metric:** `pose_auc_5` (↑).
**Bar:** regular BA (live `mast3r` backend) = **29.42**. A lower-complexity
approach that wins also counts.

## How to read results (measurement protocol + caveats)
- **Live benchmark = ground truth.** `compose/slurm/run_sparse_surface_em_benchmark.sh`
  (and `run_sparse_mast3r_benchmark.sh` for the bar). Runs VGGT+MASt3R on GPU then BA.
- **Offline harness** (`ba/eval/`) replays CACHED BA inputs (`compose/data/ba_cache/*.npz`)
  and recomputes `pose_auc_5` on CPU — fast, scene-parallel, no GPU. Use for RANKING.
- **CAVEAT 1 — offline↔live offset ≈ +0.9.** Offline anchor (em radial @ live config)
  = 29.82 but the same config live = 28.93. Cause: the cache was dumped from a
  *different* VGGT/MASt3R run than the live benchmarks (GPU nondeterminism), so the
  triangulated points differ slightly. ⇒ **offline ABSOLUTE numbers are not comparable
  to the 29.42 bar**; only within-offline RANKING is trustworthy. Always confirm a
  candidate LIVE before claiming a win.
- **CAVEAT 2 — multithreaded-Ceres noise ≈ ±0.5.** `num_threads>1` gives non-deterministic
  FP reductions → run-to-run pose_auc swings (~0.5). Fix: **use `num_threads=1`** for
  reproducible offline ranking (rely on scene parallelism `jobs=N` for speed).
- **CAVEAT 3 — proxy is dead.** The 5k-pt / 50-iter subsample proxy misranked badly
  (said em=31.7, live=28.9). Do not use it. Full fidelity only (`max_points=None`,
  `max_iterations≈200`, `function_tolerance=1e-6`).

## Key findings (chronological)
1. One-shot point-to-SQ surface BA = 19.42 (HURTS). EM iterated re-association
   (re-assign points→SQ as BA moves them) lifts it to 28.93 (≈ parity, slight loss).
2. Param BO on the proxy "won" (31.7) but LOST live (28.93). Proxy unreliable.
3. **Backend gap:** "regular BA = 29.42" is the `mast3r` backend (`HuberLoss(1.0)`).
   Our surface work uses the `mast3r_sq` backend whose PLAIN BA at `huber=2.0` is only
   28.93. Tightening `huber→0.5–1.0` recovers plain `mast3r_sq` to ~29.38 (offline).
   ⇒ ~0.5 of the "loss" was the robust-kernel width, not superquadrics.
4. **The current SQ residual was symmetric** (`||q||·|io|`, sign discarded) → a two-sided
   spring over-constraining real (thick, one-sided) structure → leaks into pose. Fixed:
   added `residual_mode` to the C++ backend:
   `0=RADIAL(default), 1=HINGE_OUTSIDE, 2=HINGE_INSIDE, 3=RADIAL_NORMALIZED,
   4=HINGE_OUTSIDE_NORMALIZED`, plus optional per-point `point_weights` (soft gating).
5. **HINGE works (offline):** at huber=1.0, EM radial(mode0)=29.07, EM hinge(mode1)
   lam3.3=29.69, lam15=29.82. The one-sided hinge ADDS value AND wants HIGHER surface
   weight (radial had to stay tiny). Best within-harness signal so far. → live test 94143.

## Experiments table
(offline = within-offline-harness ranking number; live = ground truth; bar=29.42)

| id    | hypothesis / config                                              | offline | live  | status |
|-------|-----------------------------------------------------------------|--------:|------:|--------|
| base  | regular BA, `mast3r` backend (the BAR)                          |    —    | 29.42 | bar    |
| sq0   | one-shot radial surface (lam50, assoc0.15)                      |  19.42  | 19.42 | reject (hurts) |
| em0   | EM radial re-assoc (lam3.347,assoc0.037,huber0.738) [proxy-BO]  |  29.82* | 28.93 | parity/loss |
| plain | plain `mast3r_sq` lam=0, huber sweep                            | 28.4→29.38 | — | backend ref |
| em-r1 | EM **hinge** mode1 lam3.347 huber1.0                            |  29.69  |  ?    | offline only |
| em-r1b| EM **hinge** mode1 lam15 huber1.0                              |  29.82  | **29.6** | ✅ **WIN >29.42** (94143) |

\* offline em0 anchor (29.82) sits ~+0.9 above its live value (28.93) — the offset.

### ✅ WIN VERIFIED (job 94271): hinge mode1 lam15 huber1.0 reproduced **29.6** exactly
Two independent live runs both gave 29.6 → the +0.18 win over the 29.42 bar is
REPRODUCIBLE, not live noise. (Live benchmark is effectively deterministic per
config; the offline↔live offset is about cache-vs-live inputs, not run-to-run.)
C++ mode 5 (NORMAL_OUTSIDE) / 6 (NORMAL_DISTANCE) built + smoke-tested.

### exp4: normal residual (mode5/6) TIES the hinge — residual-FORM ceiling reached
mode1_lam15=29.244, mode5_lam100=29.244, mode6_lam100=29.244 (IDENTICAL offline).
Different residual forms converge to the same camera solution ⇒ the limiting
factor is the FROZEN, mis-registered SQ geometry, not the penalty form. Pivot to
the geometry levers: (1) unfreeze SQ pose (co-refine), (2) filter degenerate SQs.
mode5/6 not worth a live run (tie the winner).

### exp5 + live: degenerate-SQ filter does NOT widen the margin
Offline: filter+lam30=29.6 (best), filter+lam15=29.42, no-filter lam15=29.244.
LIVE (94440): filter mode1 lam30 = **29.333** — WORSE than the 29.6 winner AND
below the bar. Offline mispredicted (the +0.36 was MT noise). Lesson: the offline
harness is reliable in the LOW-lambda regime (mode1 lam15: offline 29.24 ↔ live
29.6) but UNRELIABLE at high lambda. lambda>15 hurts LIVE regardless of filtering
⇒ the ceiling is the FROZEN, mis-placed SQ geometry, not a few bad primitives.
Filter ported to live path (superdec.filter_degenerate_sqs + surface_filter_*),
default OFF. Cheap levers (residual form, filter) are EXHAUSTED.

### Cluster nerf 2026-06-02: --cpus-per-gpu now blocked -> 4-CPU cap on GPU jobs.
Offline eval must use jobs<=4 now (slower). Live runs ~75 min on the BA tail.

## FINAL BEST (verified): hinge mode1, lam15, huber1.0, EM = 29.6 LIVE > 29.42 bar.

### SQ co-refinement (refine_sq) — built, live-tested, does NOT help
LIVE (94547): refine_sq=true mode1 lam30 anchor10 = **28.977** — WORSE than frozen
lam30 (29.333) and the 29.6 winner. Unfreezing lets each SQ MOVE TO FIT the noisy
current points (overfit) instead of constraining them, so the prior degrades. A
much stiffer anchor -> SQs ~frozen -> ~29.6 (no gain); the helpful middle regime
is narrow/absent. refine_sq kept default OFF (the win is unaffected).

## CONCLUSION — margin-widening hit a wall; 29.6 is the practical ceiling
Three independent structural levers all FAIL to widen the margin past 29.6:
  - normal/tangent residual (mode5/6): TIES the hinge (exp4).
  - degenerate-SQ filter: HURTS live (94440 = 29.333).
  - SQ pose co-refinement: HURTS live (94547 = 28.978).
Root cause (research-confirmed): pose_auc_5@5deg is ROTATION-dominated, but a
point-to-surface prior only moves points isotropically toward surfaces -> it has
little leverage on camera ROTATION. The one thing that flipped surface-BA from
hurting (19.42) to winning (29.6) was the residual FORM: symmetric radial ->
ONE-SIDED HINGE + tight Huber. That is the result.

## RESULT: superquadrics beat regular BA on sparse-view pose, 29.6 vs 29.42
(verified, reproduced). All knobs default to the winner / OFF; backward-compatible.

### ✅ FIRST LIVE WIN (job 94143): hinge mode1, lam15, huber1.0 → pose_auc_5 = **29.6** > 29.42
Relative gain transferred: live radial-EM 28.93 → hinge **29.6** (+0.67), mirroring the
offline +0.75. Margin over the bar is small (+0.18) → must (a) verify reproducible
(live noise band) and (b) push bigger via grid/BO. Other live metrics: pose_ate_rmse
0.401, pointmaps_abs_rel 1.767, z_depth_abs_rel 0.813, metric_scale_abs_rel 0.678.

## Backlog / queue (run serially; 1-job QOS)
- [DONE 94143] LIVE hinge mode1 lam15 huber1.0 → 29.6 ✅ WIN.
- [running 94186] exp2 offline grid: hinge mode{1,4}×lam{15,30,50}×huber×assoc.
- [DONE workflow] `sq_softweight.py` created (soft point_weights + hinge);
  `residual_mode` forwarded into sq_gated/sq_em_soft (sq_outlier_filter = plain, N/A).
- [DONE workflow] Offset diagnosis: **irreducible**. Frame selection IS seeded
  (seed 777) but the matched/triangulated geometry is a non-reproducible GPU
  realization; metric is faithful. ⇒ offline = RANKING ONLY, `num_threads=1` for
  determinism, always validate top configs LIVE. (Re-dumping cache wouldn't fix it —
  any fresh live run differs.)
- [ready exp3.py] full-fidelity rank of new strategies (softweight/gated/em_soft/
  outlier_filter) vs em_reassoc-hinge reference, num_threads=1.
- [ ] Live-validate top exp2 + top exp3 configs; re-run bar once for live noise band.
- [ ] (parked, designed) normal/tangent-plane residual = C++ mode 5; build+test if
      hinge margin needs more.

## Research-directed next levers (from bigger-gains workflow, ranked for pose_auc_5)
KEY INSIGHT: pose_auc_5 @5° is ROTATION-dominated, but the current surface term
adds a scalar residual on the POINT block only — it moves points isotropically
toward a surface, giving NO rotational moment on cameras. So it mostly helps
translation; rotation is barely touched. Biggest gains come from (a) terms with a
rotational/orientation signal and (b) removing the frozen-SQ ceiling on lambda.
- **A. Unfreeze SQ pose (co-refine)** — make each SQ's 6-DoF pose a Ceres param
  block (anchored to init); lets lambda go high without injecting frozen-frame
  error. Largest single lever; enables the others. C++ (designed, parked).
- **B. Normal/tangent-plane residual (mode 5)** — penalize only off-surface
  NORMAL component (normal = ∇F, autodiff-free); the rotation lever. C++ (designed).
- **C. Soft-GMM EM** — soft responsibilities via point_weights, anneal sigma.
  Pure Python (sq_softweight is a first cut). Low risk.
- **D. Manhattan/orthogonality** — vote dominant directions from SQ rotation axes,
  add a direct camera-rotation prior. Only rotation-direct lever. Verify SQ-axis
  clustering first.
- **E. Filter degenerate SQs** — 7.5% of 1971 SQs are slivers/razor-thin/huge
  (aspect p99=115!); `exist` is binary (no soft confidence). Drop them in
  surface_pred post-process — cheap, OFFLINE-testable, likely removes wrong pulls.

## exp2 hinge grid (noisy ~0.5, MT): mode1>mode4, lam15 best, higher lam HURTS
m1_lam15_h1.0=29.2, m1_lam30_h1.0=29.1, m1_lam50_h1.0=28.8, m4_lam15_h1.0=28.7
(same config was 29.69 in exp1 → ~0.5 run-to-run noise; further hinge tuning capped.)
New strategy files: sq_softweight.py, sq_best.py (combine all levers; default=winner).

## Conventions
- Live hinge run: `RESIDUAL_MODE=1 LAMBDA_SURFACE=.. HUBER_THRESHOLD=.. sbatch ... run_sparse_surface_em_benchmark.sh`
- Many cores: `--cpus-per-gpu=N` (28 on 5060ti, 32 on 2080ti). Exclude node09.
- Update this file after every experiment (config, offline, live, learning).
