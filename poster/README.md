# 3D Vision poster — team39

A1 **portrait** scientific poster for the 3D Vision final poster session
(HG D-Nord, 11 June 2026). Built with `beamerposter`: a **light** theme — light-grey
page, white rounded content cards, **ETH-Blue** (`#215CAF`) accents, numbered
section badges, a full-width white header band with the **ETH zürich** wordmark,
group name, title and authors all centred, and two bottom-aligned columns.

Layout follows a "Better Poster" hybrid (scan-first, evidence-based): a
**full-width plain-language takeaway band** sits directly under the title as the
focal point, methods are visually demoted, and the **Results** card dominates
(a qualitative VGGT→BA→prior pose figure + two side-by-side bar charts: a
3-series **VGGT / Baseline / Ours** absolute-AUC@5 chart and a single-series
**gain-from-the-prior** ΔAUC@5 chart). A **QR code** in the References card links
to the code/report.

The poster leads with the project's headline finding: the super-quadric
environment prior **helps most when the input is sparse** — the realistic
relocalization regime (gain grows from +0.2 AUC@5 at 10 views to +1.5 at 6 and
+1.3 at 4, up to +2.0 with tuning).

**Scientific honesty (important).** Bundle adjustment, not the prior, does most
of the work of fixing raw VGGT poses (at 10 views ≈88% of the rotation-error
reduction is plain BA, only ≈12% the prior). The poster therefore credits the
prior only with its *marginal gain on top of BA*, shown via **Baseline (BA, no
prior) vs Ours (BA + prior)** — never VGGT-vs-Ours, which would let BA's win
masquerade as the prior's.

The Results bar charts are driven directly by the benchmark's own
dataset-average `pose_auc_5` numbers (VGGT / Baseline / Ours at 4/6/8/10 views),
so that part of the figure **is** the metric. (An earlier numbers table was
dropped as redundant with the two charts.)

The qualitative 3-panel pose figure (Fig 2, `poses_v6_s6.png`) needs care:
AUC@5 scores *pairwise relative rotation + translation-angle*, not absolute
camera position, and the three configs live in different gauges — so a naive
position scatter can show a worse config looking better (an earlier version did
exactly this for VGGT-vs-BA). The current figure **Sim(3)-aligns each config's
cameras to ground truth** (standard trajectory-comparison alignment) and badges
the residual *rotation* error, which ranks VGGT (20°) > BA (10°) > Ours (2°)
monotonically — matching AUC@5 (29 < 48 < 57). It is a qualitative illustration;
the bar chart/table carry the exact metric.

There is **no official LaTeX version** of the ETH poster template — ETH only
ships PowerPoint + Illustrator. This is a `beamerposter` layout in ETH colours,
pre-filled with this project's real figures and results.

- Official ETH template (PPTX/AI, for reference / logos):
  <https://ethz.ch/staffnet/en/service/communication/corporate-design/templates.html#WissenschaftlichesPlakat>
- ETH colours: <https://ethz.ch/staffnet/en/service/communication/corporate-design/colours.html>

## Files

| File | What |
|------|------|
| `poster.tex`   | The poster source (self-contained; A1 portrait). |
| `figures/`     | Figures embedded in the poster (copied from `ba/eval/analysis/`). |
| `poster.pdf`   | Last compiled output (this is what you upload to Moodle). |

## Compiling

### Overleaf (recommended)
Upload `poster.tex` + the `figures/` folder, set the compiler to **pdfLaTeX**
(Menu → Compiler). Everything else is in Overleaf's TeX Live by default.

### Locally (cluster)
The cluster's `texlive-base` is missing most packages; they were installed into
the per-user tree `~/texmf` from the archived TeX Live 2023 repo. To compile:

```bash
cd poster
pdflatex poster.tex && pdflatex poster.tex   # two passes
```

Required packages (already installed in `~/texmf`): `beamer`, `beamerposter`,
`pgf`/`tikz`, `pgfplots`, `xcolor`, `xkeyval`, `helvetic`, `type1cm`, `booktabs`,
`colortbl`, `qrcode` (the last installed via `tlmgr install qrcode`).
If you move to a fresh machine and have a full TeX Live, no extra steps needed.

## Editing checklist

The header (title / authors / affiliations / group) is hand-built in the
`headline` template near the top of `poster.tex` — edit it there, not via
`\author`/`\institute`.

- [x] **QR code URL** — `\posterurl` points at
      `https://github.com/davidfarah2003/superquadric-ba` (verified: the compiled
      QR decodes to that URL). **Make sure the repo is public before printing.**
- [x] **Author name** — David Farah listed first; last author is `Linfei Pan`.
- [ ] Section content lives in the `psection`/`block` environments in the body;
      section numbers are automatic. The takeaway band is the `tikzpicture`
      right after `\begin{frame}`. The **Method Overview** is also an inline
      `tikzpicture`: a left-to-right pipeline (Sparse frames -> VGGT + MASt3R ->
      Bundle Adjustment -> Camera poses) with the **super-quadric prior** as the
      one solid-blue "ours" box feeding a `+ surface loss` term into BA from
      below (blue = our contribution, matching the takeaway band and the Ours
      bars). Box styles `io`/`proc`/`ours` are defined in that `tikzpicture`.
- [ ] Figures (all real outputs): `superquadric_family.png` (the "what is a
      super-quadric" explainer) and `superquadrics_3d_clean.png` (Fig 1, the
      scene) in the left column. Results uses two side-by-side `pgfplots` bar
      charts (drawn inline, not images): a 3-series VGGT/Baseline/Ours absolute
      AUC@5 chart and a single-series ΔAUC@5 "gain from the prior" chart. See
      the provenance table below.
- [ ] Layout knobs: `\margin` / `\gutter` (even frame + column gap), the
      takeaway-band font sizes (`\huge`/`\LARGE`), and `scale=` in the
      `\usepackage[...]{beamerposter}` line (scales all text).
- [ ] Vertical fill: each column is wrapped in a stretching minipage of height
      `\colheight` (derived as `\paperheight - 27cm`, i.e. the A1 page minus the
      fixed header+takeaway+margins), and `\flexgap` is `\vfill`, so the cards
      spread to fill the page and both columns bottom-align. If you add/remove a
      card or change the header height, re-check the bottom margin; if a column
      overflows, nudge the `27cm` up (more top matter) — the page stays A1.
- [ ] Avoiding airy gaps: `\flexgap` distributes *leftover* space into the gaps,
      so the way to tight gaps is **more card content**, not smaller `\colheight`
      (shrinking it just dumps the slack into the bottom margin). The levers used:
      body text size (`block body` font, set via an explicit
      `\fontsize{25}{30}` — note `size=\LARGE` silently no-ops here, so use
      `\fontsize`), the two result charts' `height=`, and the Method-Overview
      diagram's box heights / prior offset. The header is full-width and centred.
- [ ] Card padding: rounded beamer blocks have ~no internal padding, so
      `\cardpad` (0.6 cm) insets every card's body (via an `\addtobeamertemplate`
      minipage) and its title, giving uniform breathing room without changing the
      card's outer width.
- [ ] **Print size:** the poster is true **A1 portrait** (`size=a1`,
      594 x 841 mm, one page) — confirm with `pdfinfo poster.pdf` (Page size
      `1683.78 x 2383.94 pts`) before sending to print.

## Where the content comes from (and how to regenerate it)

All figures and numbers are this project's real outputs.

### Figures (`figures/`)
| Poster file | Source | Generated by |
|-------------|--------|--------------|
| `superquadric_family.png` (explainer, **used**) | written directly into `figures/` | `ba/eval/fig_superquadric_family.py` (pure-geometry shape-family morph; no data) |
| `superquadrics_3d_clean.png` (Fig 1, **used**) | written directly into `figures/` | `ba/eval/fig_sq3d_clean.py` (poster-ready: axes/box hidden, large cameras, big flat floor/ceiling slabs dropped so the top-down view isn't washed out; reuses `show_scene.match_views`/`_sq_surface`) |
| `poses_v6_s6.png` (Fig 2, **used**) | benchmark viz dumps (see below) | `ba/eval/fig_poses_v6_s6.py` — Sim(3)-aligned 3-panel VGGT→+BA→+prior, scene-6 6-view, mean rotation error 20°→10°→2° |
| `before_after_poses.png` (**superseded**, unused) | — | `ba/eval/fig_before_after.py` — the older VGGT-vs-Ours figure; dropped as misleading (credited the prior for BA's work) |
| `input_views.png` (unused)      | `ba/eval/analysis/fig8_views_scene6.png` | `ba/eval/show_scene.py` (`fig_views()`) |
| `recon_poses_clean.png` (unused) | `ba/eval/analysis/fig4_recon_scene6.png` | `ba/eval/analyze_recon.py` |
| `auc_decomp.png` (unused) | `ba/eval/analysis/fig1_auc_decomp.png` | `ba/eval/analyze_pose.py` |

**Regenerate the explainer** (instant, no cluster):
```bash
cd ba/eval && ../../envs/3dv/bin/python fig_superquadric_family.py
```

**Regenerate Figure 2 (the 3-panel pose figure)** — three GPU runs dump scene 6's
cameras at the **same** 6 views (deterministic seed), then the plot script
Sim(3)-aligns each to GT:
```bash
# from repo root, on the cluster
sbatch compose/slurm/run_viz_vggt_v6_s6.sh                                          # VGGT-only (no BA)
NUM_VIEWS=6 LAMBDA_SURFACE=0    VIZ_SAVE_INDEX=6 sbatch compose/slurm/run_sparse_surface_em_benchmark.sh  # +BA (Baseline)
NUM_VIEWS=6 LAMBDA_SURFACE=15.0 VIZ_SAVE_INDEX=6 sbatch compose/slurm/run_sparse_surface_em_benchmark.sh  # +prior (Ours)
cd ba/eval && ../../envs/3dv/bin/python fig_poses_v6_s6.py
```
NB: in the *surface-BA* runs the per-run `vggt/cameras.json` is aliased to the BA
result (`preds_vggt = preds` before in-place BA), so genuine VGGT poses come from
the separate no-BA run; the script reads `ba/cameras.json` from the λ=0 / λ=15
runs as Baseline / Ours. The script reports both the rotation residuals and the
benchmark AUC@5 (so you can confirm the picture and the metric still agree).

### Headline numbers (Results chart + table)
The 3-series `pgfplots` chart and the table are the `pose_auc_5` (`Average`)
field of these benchmark runs under `logs/` (VGGT-only / Baseline λ=0 / Ours
λ=15):

| views | VGGT-only | Baseline (λ=0) | Ours (λ=15) | Δ (prior) | log dirs |
|---|---|---|---|---|---|
| 4  | 27.67 | 39.33 | 40.67 | +1.33 | VGGT `vggt_v4`; BA `benchmark_ase_sparse_surface_em_cov06_v4_lam{0,15.0}` (peak 41.33 @ `lamsweep_v4_lam100`) |
| 6  | 20.40 | 29.60 | 31.07 | +1.47 | VGGT `vggt_v6` (or `viz_vggt_v6_s6`); BA `sweep_v6_lam{0,15.0}` |
| 8  | 13.36 | 27.93 | 28.00 | +0.07 | VGGT `vggt_v8`; BA `sweep_v8_lam{0,15.0}` |
| 10 |  9.33 | 29.42 | 29.60 | +0.18 | VGGT `benchmark_ase_sparse_vggt_cov06`; BA `benchmark_ase_sparse_surface_em_cov06` |

VGGT-only rows come from `compose/slurm/run_vggt_nview.sh` (`NV=4/8 sbatch …`,
`bundle_adjustment=none`); the 6-view value also matches `viz_vggt_v6_s6`.
The table-caption "+9.3 / hurts none" per-scene facts are from the v6
`*per_scene_results.json` (scene 6: 48.0→57.3).

To re-extract the numbers:
```bash
python3 - <<'PY'
import json, os
root = "/work/courses/3dv/team39/logs"
for d in ["vggt_v4", "vggt_v6", "vggt_v8",
          "benchmark_ase_sparse_surface_em_cov06_v4_lam0", "sweep_v6_lam0",
          "sweep_v6_lam15.0"]:
    avg = json.load(open(os.path.join(root, d, "per_dataset_results.json")))["Average"]
    print(d, round(avg["pose_auc_5"], 2))
PY
```

To regenerate the source figures, run the corresponding `ba/eval/*.py` script
(they read the BA cache in `compose/data/ba_cache/*.npz` and write into
`ba/eval/analysis/`), then re-copy into `figures/`.

## Note on the course email

The announcement is dated 2026 but lists a "4th of May (Thursday) midnight"
Moodle upload deadline and a session on the 11th — those day/date combos look
copied from a prior year. **Check the actual Moodle deadline.**
