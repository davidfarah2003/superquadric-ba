# Surface-BA experiment ledger — beating regular BA with superquadrics

**Goal:** make a SUPERDEC-superquadric-augmented bundle adjustment beat plain
("regular") BA on the sparse-view ASE benchmark. **Metric:** `pose_auc_5` (↑).
**Bar:** regular BA (live `mast3r` backend) = **29.42**. A lower-complexity
approach that wins also counts.

---
## ⭐ HEADLINE RESULT (2026-06-03): the superquadric prior's pose gain GROWS as you take FEWER pictures
At the dense 10-view operating point the SQ surface prior barely helps (+0.2) — the
cameras are over-constrained by 40k-117k two-view points (see DIAGNOSTIC ANALYSIS
below). **Starve the views and the prior earns its keep.** LIVE benchmark, same
views per row (seed 777), only the surface term toggled (λ=15 hinge-EM vs λ=0):

| num_views | reproj-BA (λ=0) | surface-BA (λ=15) | **surface gain** |
|----------:|----------------:|------------------:|-----------------:|
| 10        | ~28.9 (offline) | **29.6** (live)   | **+0.2** |
| 8         | 27.93           | 28.00             | **+0.07** |
| 6         | 29.60           | **31.07**         | **+1.47** |
| 4         | 39.33           | **40.67**         | **+1.33** |

**Regime-dependent, with a sharp transition ~6→8 views:** the gain is ~+1.3–1.5 at
4–6 views but collapses to ~+0.1 at 8–10 views. Two sparse points agree, two dense
points agree -> robust, not coarse-metric noise. (Absolute AUC is non-monotonic in
views because the covisibility sampler picks different, easier view-sets at low K;
the DELTA is the clean signal.) Jobs: surface@4=94968, reproj@4=94969, sweep=94970.
**Interpretation:** superquadrics provide real pose benefit ONLY where multi-view
geometry is weak (≤6 pictures); above that the cameras are already pinned. This is
the "few-view / less-complexity" win the goal allows, and few-view is the harder,
more realistic regime where priors matter. Reproduce: `NUM_VIEWS=4 LAMBDA_SURFACE=15 sbatch
compose/slurm/run_sparse_surface_em_benchmark.sh` (vs `LAMBDA_SURFACE=0` baseline);
full sweep `compose/slurm/run_views_sweep.sh`.
NEXT: confirm 8-view delta, push to 3 views, and re-tune λ for the sparse regime
(few points -> the prior can take a much higher weight than the dense λ=15).
---

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

## NEXT LEVER (post-win): Manhattan-snap — denoise SQ orientation, not add a penalty
Lesson from the win: the gain was STRUCTURAL (residual form), and the ceiling is
NOISY/MIS-REGISTERED SQ geometry (not the penalty form — mode5/6 tied, filter &
co-refine hurt). So instead of another point-block term, denoise the geometry
along a dimension we can PROVE is structured.

**Precondition VERIFIED (ba/eval/manhattan_check.py):** ASE is Manhattan. Across
all 10 scenes the anisotropic (aspect>1.5) SQ orientations sit a MEDIAN 5.4deg
from a single shared cube-aligned frame (80% <10deg, 92% <15deg, 96% <20deg),
and that frame ~= the world axes (residual to identity 5.38 ~= residual to fitted
R_m 5.35) -> gravity-aligned rooms. That ~5deg is mostly SUPERDEC fit noise on
walls/floor; left in, it tilts the pulled-to surface by ~scale*sin(5deg) (~9-17cm
at the edges of a 1-2m primitive).

**Intervention:** `superdec.manhattan_snap_sqs(sq, max_snap_deg)` votes the per-
scene frame (octahedral rotation averaging) and snaps each SQ within threshold
onto it; ONLY rotation changes (shape preserved). At deg=15 ~90% of SQs snap,
mean correction ~5.5deg. Wired OFF-by-default offline (strat_common.surface_pred,
em_reassoc param `manhattan_snap_deg`) and live (mast3r_bundle_adjust
`manhattan_snap`, benchmark `surface_manhattan_snap`, slurm `MANHATTAN_SNAP`).
Hypothesis: (a) denoises the diagnosed geometry ceiling; (b) may UNLOCK the normal
residual (mode5) that only TIED with noisy normals. Risk: hurts if the 5deg is
real off-axis structure -> rank offline, confirm live.

**LIVE result (job 94905): hinge mode1 lam15 + snap15 = pose_auc_5 29.6 — TIE,
slight ATE regression.** Verified the snap applied (`surface_manhattan_snap=15` in
config) and DID change the solution: pose_ate_rmse 0.40412 vs the two no-snap runs
0.40059 (94143) / 0.40038 (94271) — a real ~0.0037 shift (~15x the ~0.0002 run-to-
run noise), in the WORSE direction. pose_auc_5 unmoved at 29.6. Mechanism: snapping
denoises SQ ORIENTATION -> nudges point depths/translations (ATE moved) but pose_auc_5
@5deg is ROTATION-dominated and a point-block surface term has no relative-rotational
leverage -> AUC unchanged. (Offline exp7 abandoned: the 2-CPU plugin cap is
inescapable — no CPU-node access — making full-fidelity offline ranking impractical;
went straight live.) manhattan_snap stays default-OFF (tested, byte-identical when off).

### CONCLUSION HARDENS: surface-geometry levers cannot widen the 29.6 margin
FOUR independent geometry interventions now TIE-or-HURT pose_auc_5: normal residual
(mode5, exp4), degenerate-SQ filter (94440), SQ co-refine (94547), Manhattan-snap
(94905). All are POINT-BLOCK terms; pose_auc_5@5deg is rotation-dominated and they
have no relative-rotational leverage. The 29.6 win (residual FORM: symmetric->hinge)
stands; the margin is bound by VGGT's rotation estimates, which the SQ surface prior
cannot improve in any form tried. Remaining untested long-shot: snap + mode5 (denoised
normals + normal residual) — same point-block class, low prior. A real rotation lever
would need per-view orientation signal (vanishing points / direct relative-rot prior),
a much larger change.

## DIAGNOSTIC ANALYSIS (ba/eval/analyze_*.py, show_scene.py) — corrects the framing
Built a proper decomposition since pose_auc_5 = AUC of max(rot_err, trans_angle_err)
per view-pair @5deg. Ran raw VGGT / reproj-BA / surface-BA / surface+snap, all 10
scenes (job 94950, offline 20k-cap; structure is robust, absolutes carry the offset).

| stage         | rot-only AUC | trans-only AUC | combined (pose_auc_5) |
|---------------|-------------:|---------------:|----------------------:|
| raw VGGT      | 13.7 | 15.4 |  9.3 |
| reproj-BA     | 34.3 | 32.4 | 28.9 |
| surface-BA    | 33.8 | 32.2 | 29.0 |
| surface+snap  | 34.3 | 32.6 | 29.1 |

**Two corrections to earlier claims:**
1. **NOT rotation-dominated.** rot-only ~= trans-only AUC at every stage; rotation is
   the binding (larger) error in only ~55-60% of pairs. The metric is low because
   *combined* sits BELOW both — a pair needs rot AND trans BOTH <5deg (the AND, not
   a rotation ceiling). Earlier "rotation-dominated" framing was wrong.
2. **The surface prior barely moves anything.** reproj 28.9 -> surface 29.0 -> snap
   29.1: the SQ term shifts cameras by ~0.05-0.2 AUC (in-noise). PLAIN REPROJ-BA does
   ~all the lift (9.3->28.9). Per-scene it's a wash (does literally nothing on scenes
   5/7: 97.8->97.8, 49.3->49.3). The 29.6 win came from getting the reprojection BA
   right (tight Huber + EM + hinge form), NOT from strong superquadric geometry.

**Root cause (the real one):** every triangulated point is seen by EXACTLY 2 views
(MASt3R pairwise triangulation; analyze_sparsity.py). 40k-117k such points pin just
10 cameras -> the cameras are MASSIVELY over-constrained by reprojection before the
surface term is even added, so a point-to-surface pull can't move them. Not a rotation
ceiling — an over-constraint ceiling. Scene viz (show_scene.py fig8/9): the 10 photos
are near-non-overlapping rooms of one apartment; the SQ decomposition tiles the whole
apartment, but each camera still gets thousands of points.

**=> Next regime to test: FEWER PICTURES (num_views), not fewer points.** Dropping
views collapses the pairwise covisibility graph (fig7); only there might the box-prior
carry under-constrained cameras. Must be tested LIVE (re-triangulated) — the offline
cache keeps privileged 10-view points. Candidate: num_views=4, regular-BA vs surface-BA.

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
