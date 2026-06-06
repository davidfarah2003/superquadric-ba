# Poster reproduction target

We are reproducing the team's existing poster (`poster3dv.pdf`, made in the
official ETH PowerPoint/Illustrator template, **green** colour variant) faithfully
in LaTeX. This file is the source of truth for content + design.

> If you have the original `poster3dv.pdf`, drop it in this folder so the
> design can be matched visually.

## Design (official ETH scientific-poster template, green variant)
- A1 portrait. **ETH Green `#627313`** as the frame/background colour.
- Top: black **ETH zürich** wordmark top-LEFT; "Computer Vision and Geometry
  Group, ETH Zurich" top-RIGHT (small, right-aligned).
- Large **bold black** title, then author line, then affiliation line.
- Body: **white rounded content boxes** on the green background (green shows as a
  frame around everything and in the gaps between boxes).
- Numbered section headers: `1 Introduction`, `2 Method Overview`, … bold black.
- Two columns. Helvetica/Arial. References box at the bottom of the right column.

## Header (verbatim)
- **Title:** Improving relocalization accuracy via environment prior
- **Authors:** Lars Hecker¹, Raman Besenfelder¹, David Farah¹, Fatemeh Sadat
  Daneshmand², Elisabetta Fedele¹, Linfei¹   *(NOTE: last name "Linfei" looks
  truncated in the source — confirm full name.)*
- **Affiliations:** ¹ETH Zurich  ²ZHAW
- **Group (top-right):** Computer Vision and Geometry Group, ETH Zurich

## Sections (verbatim content)

### 1 Introduction
Recent advancements in transformer-based Structure from Motion (SfM) took the
3D-Vision community by storm [1]. Combining fast inference times with strong
accuracy, they outperform traditional feature-matching + bundle adjustment (BA)
pipelines in many settings [3].

In relocalization problems such as the "kidnapped robot" scenario, accurate
camera pose estimation becomes the main objective while a prior on the
environment, such as a point cloud, exists. Current transformer-based SfM
approaches cannot exploit such priors.

This work combines transformer-based SfM with one such prior. More specifically,
a super-quadric cloud is used as this environmental prior. [4] Super-quadrics act
as versatile geometric primitives and provide a denser, more compact description
of the environment's topology than raw point clouds do.

### 2 Method Overview  (flowchart)
`Sparse frames` → `VGGT + MASt3R` → (arrow labelled "Pose Estimates +
Correspondences") → `Bundle Adjustment` → `Camera poses`.
`World map` box feeds into `Bundle Adjustment` via an arrow labelled
"super-quadric loss".

### 3 Materials
- **Dataset:** Aria Synthetic Environments
- **Super-Quadric Environment:** SuperDec for fitting superquadrics given scene's point cloud
- **Models:** VGGT for feed-forward pose estimation (benchmarked via MapAnything [2]), MASt3R for feature matching. [3]
- **Optimization:** Full triangulation + BA in Ceres Solver
- Equation:
  E = Σ_{i,j} ρ(‖π(R_i,t_i,X_j) − x_ij‖²)            [reprojection]
    + Σ_j ( λ‖q_j‖ · max(0, 1 − F(q_j)^{−ε₁}) )²       [one-sided surface prior]
  where X_j is a 3D point, q_j its coordinates of the associated super-quadric
  a(j), F(·) the inside-outside function, ε₁ the shape parameter, and λ the prior
  weight. [5]
- **Metrics:** Pose-estimation Area Under Curve (@5) and Absolute Trajectory Error.

### 4 Results and Discussion
We compare three configurations:
- **VGGT-only:** poses taken directly from the feed-forward model.
- **VGGT & BA (Baseline):** full triangulation and bundle adjustment using VGGT poses initialized with MASt3R correspondences.
- **Prior-aware BA (Ours):** Baseline augmented with the super-quadric prior term.

Chart A "10 Input Views" (bar, AUC@5 %): VGGT 9.3, Baseline 29.4, Ours 29.6.
Chart B "Sparse Sequences" (line, x = number of input views 4/6/8, AUC@5 %):
  Baseline {4:39.3, 6:29.6, 8:27.9}; Ours {4:40.7, 6:31.1, 8:28.0}.

### 5 Conclusion
Transformer-based SfM cannot natively exploit an environment prior; we bridge
this gap for relocalization.

Encoding the prior as a super-quadric cloud and adding a point-to-quadric term to
BA improves pose accuracy both over VGGT-only and standard BA baselines. Gains are
most significant in settings where co-visibility between frames is low.

### Figures
- **Figure 1:** SuperQuadrics fitted to Environment.  (placeholder: figures/superquadrics_3d.png — swap for the exact colourful render if available)
- **Figure 2:** Sample Scenes with Camera Poses.  (placeholder: figures/recon_poses.png / superquadrics_3d.png)

### References
[1] Wang et al., VGGT: Visual Geometry Grounded Transformer. CVPR 2025.
[2] Keetha et al., MapAnything: Universal Feed-Forward Metric 3D Reconstruction. 3DV 2026.
[3] Leroy et al., Grounding Image Matching in 3D with MASt3R. ECCV 2024.
[4] Fedele et al., SuperDec: 3D Scene Decomposition with Superquadric Primitives. ICCV 2025.
[5] Mueller et al., Reconstructing People, Places, and Cameras. CVPR 2025.
