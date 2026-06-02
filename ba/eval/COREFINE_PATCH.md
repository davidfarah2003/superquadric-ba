All callers use keyword arguments for `run_bundle_adjustment_mast3r_sq`, so adding new keyword-only-positioned params at the end of the C++ signature (after `function_tolerance`) is safe. Since the wrapper forwards positionally, I'll append the two new params at the very end of both the C++ signature and the wrapper's positional forwarding. This preserves byte-identical behavior for all existing callers.

I have everything I need to produce the complete patch.

---

# COMPLETE READY-TO-APPLY PATCH: optional `refine_sq` superquadric-pose refinement

All edits are minimal, default-OFF, and byte-identical to current behavior when `refine_sq=false`. Two files change: `ba/src/mast3r_sq_ba.cpp` (do NOT hand-edit — apply via orchestrator) and `ba/python/ba/__init__.py`. One optional convenience edit to `ba/eval/strat_common.py`.

---

## FILE 1 — `ba/src/mast3r_sq_ba.cpp`

### Edit 1.1 — `SurfaceResidual`: add SQ-pose as a 2nd parameter block

Replace the entire struct (current lines 104–244). **OLD block** (the whole struct from `struct SurfaceResidual {` through its closing `};`):

```cpp
struct SurfaceResidual {
    SurfaceResidual(double a1, double a2, double a3,
                    double e1, double e2,
                    const double* rotation_aa,
                    const double* translation,
                    double lambda_surface,
                    int    mode,
                    double weight)
        : a1_(a1), a2_(a2), a3_(a3),
          e1_(e1), e2_(e2),
          lambda_(lambda_surface),
          mode_(mode),
          sqrt_w_(weight > 0.0 ? std::sqrt(weight) : 0.0)
    {
        // Negate angle-axis: R(-aa) is the inverse of R(aa).
        neg_aa_[0] = -rotation_aa[0];
        neg_aa_[1] = -rotation_aa[1];
        neg_aa_[2] = -rotation_aa[2];
        t_[0] = translation[0];
        t_[1] = translation[1];
        t_[2] = translation[2];
    }
```

…through the `Create` and members:

```cpp
    static ceres::CostFunction* Create(double a1, double a2, double a3,
                                        double e1, double e2,
                                        const double* rotation_aa,
                                        const double* translation,
                                        double lambda_surface,
                                        int    mode,
                                        double weight) {
        return new ceres::AutoDiffCostFunction<SurfaceResidual, 1, 3>(
            new SurfaceResidual(a1, a2, a3, e1, e2,
                                rotation_aa, translation, lambda_surface,
                                mode, weight));
    }

    double a1_, a2_, a3_;
    double e1_, e2_;
    double neg_aa_[3];
    double t_[3];
    double lambda_;
    int    mode_;
    double sqrt_w_;
};
```

**NEW block** (full replacement struct). Key changes: shape `a1_..e2_` stay fixed members; the pose (`aa`,`t`) becomes the 2nd parameter block; `operator()` now takes `(point, sq_pose, residual)` and computes `q = AngleAxisRotatePoint(-aa, point - t)` from the **passed** templated pose; `neg_aa_`/`t_` members and their init are removed; both a single-block `Create` (frozen-pose path) and a two-block `CreateRefine` (pose-refining path) are provided so the `refine_sq=false` path uses the **identical** 1-block functor as today.

```cpp
struct SurfaceResidual {
    SurfaceResidual(double a1, double a2, double a3,
                    double e1, double e2,
                    double lambda_surface,
                    int    mode,
                    double weight)
        : a1_(a1), a2_(a2), a3_(a3),
          e1_(e1), e2_(e2),
          lambda_(lambda_surface),
          mode_(mode),
          sqrt_w_(weight > 0.0 ? std::sqrt(weight) : 0.0)
    {}

    template <typename T>
    static T safe_clamp(T x, double lo, double hi) {
        T r = x;
        if (r < T(lo)) r = T(lo);
        if (r > T(hi)) r = T(hi);
        return r;
    }

    template <typename T>
    static T safe_pow_pos(T x, T y) {
        // Clamp |x| to [1e-3, 5e2] to keep gradients finite, mirroring SUPERDEC.
        T cx = safe_clamp(ceres::abs(x), 1e-3, 5e2);
        T r = ceres::pow(cx, y);
        return safe_clamp(r, 1e-3, 5e2);
    }

    // Core surface magnitude given the SQ pose (aa, t) as templated params.
    //   q = R(aa)^T (point - t) = AngleAxisRotatePoint(-aa, point - t)
    // aa is angle-axis for R: canonical -> world; t is the SQ centre (world).
    template <typename T>
    bool operator()(const T* const point,
                    const T* const sq_pose,   // [aa(3), t(3)]
                    T* residual) const {
        const T* aa = sq_pose;          // canonical -> world rotation
        const T* t  = sq_pose + 3;      // SQ centre in world coords

        // diff = point - t
        T diff[3] = {
            point[0] - t[0],
            point[1] - t[1],
            point[2] - t[2],
        };

        // q = R^T diff = AngleAxisRotate(-aa, diff)
        const T neg_aa[3] = {-aa[0], -aa[1], -aa[2]};
        T q[3];
        ceres::AngleAxisRotatePoint(neg_aa, diff, q);

        // ||q|| with epsilon to keep sqrt differentiable at 0.
        T q_norm_sq = q[0]*q[0] + q[1]*q[1] + q[2]*q[2];
        T q_norm = ceres::sqrt(q_norm_sq + T(1e-8));
        q_norm = safe_clamp(q_norm, 1e-4, 1e6);

        // qa_i = clamp(|q_i| / a_i, [1e-3, 5e2])
        T qa0 = safe_clamp(ceres::abs(q[0]) / T(a1_), 1e-3, 5e2);
        T qa1 = safe_clamp(ceres::abs(q[1]) / T(a2_), 1e-3, 5e2);
        T qa2 = safe_clamp(ceres::abs(q[2]) / T(a3_), 1e-3, 5e2);

        const T two_over_e2 = T(2.0 / e2_);
        const T two_over_e1 = T(2.0 / e1_);
        const T ratio       = T(e2_ / e1_);

        T Fxy = safe_pow_pos(qa0, two_over_e2) + safe_pow_pos(qa1, two_over_e2);
        Fxy   = safe_pow_pos(Fxy, ratio);
        T Fz  = safe_pow_pos(qa2, two_over_e1);
        T F   = safe_clamp(Fxy + Fz, 1e-3, 5e2);

        // io > 0 inside, io < 0 outside, 0 on the surface.
        T io = safe_pow_pos(F, T(-e1_ / 2.0)) - T(1.0);

        // Per-mode magnitude (>= 0). Hinges keep only one side via max(0, .).
        T mag;
        switch (mode_) {
            case SR_HINGE_OUTSIDE:            // outside -> -io > 0
                mag = q_norm * ceres::fmax(T(0.0), -io);
                break;
            case SR_HINGE_INSIDE:             // inside  ->  io > 0
                mag = q_norm * ceres::fmax(T(0.0),  io);
                break;
            case SR_RADIAL_NORMALIZED:        // drop ||q|| prefactor
                mag = ceres::abs(io);
                break;
            case SR_HINGE_OUTSIDE_NORMALIZED: // drop ||q|| + outside only
                mag = ceres::fmax(T(0.0), -io);
                break;
            case SR_NORMAL_OUTSIDE:           // one-sided point-to-plane (outside)
            case SR_NORMAL_DISTANCE: {        // two-sided point-to-plane distance
                // grad_q F : analytic SQ normal. Only its norm is used, so the
                // sign(q_i)/a_i factors square to 1/a_i^2 and drop their sign.
                // P = s0^{2/e2} + s1^{2/e2}  (inner sum; Fxy = P^ratio).
                T P = safe_pow_pos(qa0, two_over_e2) + safe_pow_pos(qa1, two_over_e2);
                P   = safe_clamp(P, 1e-3, 5e2);
                T c  = ratio * safe_pow_pos(P, ratio - T(1.0)) * two_over_e2;
                T G0 = c * safe_pow_pos(qa0, two_over_e2 - T(1.0)) / T(a1_);
                T G1 = c * safe_pow_pos(qa1, two_over_e2 - T(1.0)) / T(a2_);
                T G2 = two_over_e1 * safe_pow_pos(qa2, two_over_e1 - T(1.0)) / T(a3_);
                T grad_norm = ceres::sqrt(G0*G0 + G1*G1 + G2*G2 + T(1e-12));
                grad_norm = safe_clamp(grad_norm, 1e-4, 1e6);
                T d_n = (F - T(1.0)) / grad_norm;   // >0 outside, <0 inside
                if (mode_ == SR_NORMAL_OUTSIDE)
                    mag = ceres::fmax(T(0.0), d_n); // penalize only outside
                else
                    mag = ceres::abs(d_n);          // two-sided
                break;
            }
            case SR_RADIAL:                   // current (default)
            default:
                mag = q_norm * ceres::abs(io);
                break;
        }

        residual[0] = T(sqrt_w_) * T(lambda_) * mag;
        return true;
    }

    // Frozen-pose path (refine_sq=false): wrap the SQ pose as a CONSTANT
    // parameter block so the SAME functor serves both paths. Byte-identical
    // residual values to the old 1-block functor (pose just isn't optimised).
    static ceres::CostFunction* CreateRefine(double a1, double a2, double a3,
                                             double e1, double e2,
                                             double lambda_surface,
                                             int    mode,
                                             double weight) {
        return new ceres::AutoDiffCostFunction<SurfaceResidual, 1, 3, 6>(
            new SurfaceResidual(a1, a2, a3, e1, e2,
                                lambda_surface, mode, weight));
    }

    double a1_, a2_, a3_;
    double e1_, e2_;
    double lambda_;
    int    mode_;
    double sqrt_w_;
};
```

> Note: the old `Create(...)`/`safe_clamp`/`safe_pow_pos` are folded into this single struct above. The 1-block path no longer exists as a separate functor — instead the frozen-pose path (Edit 1.3) passes a **constant** 6-D pose block to `CreateRefine`, which yields identical residuals to before because a constant block contributes no Jacobian. This keeps exactly one functor and removes the risk of two copies of the residual drifting apart.

---

### Edit 1.2 — add the `SQPoseAnchor` prior cost (new struct)

Insert this **immediately after** the closing `};` of `SurfaceResidual` and **before** the `// Run bundle adjustment in-place.` comment (i.e. between current lines 244 and 246).

**Anchor before** (context, unchanged — the line you insert after):
```cpp
    double sqrt_w_;
};


// Run bundle adjustment in-place.
```

**Insert this new struct** in the blank gap:
```cpp
// -------------------------------------------------------------------------
// Soft anchor keeping each refined SQ pose near its SUPERDEC init.
// residual = sqrt(anchor_weight) * (pose - pose0)   (6-vector: [aa(3), t(3)]).
// Fixes the gauge for SQs with few/no surface points and prevents drift.
struct SQPoseAnchor {
    SQPoseAnchor(const double* pose0, double anchor_weight)
        : sw_(anchor_weight > 0.0 ? std::sqrt(anchor_weight) : 0.0) {
        for (int i = 0; i < 6; ++i) pose0_[i] = pose0[i];
    }

    template <typename T>
    bool operator()(const T* const pose, T* residual) const {
        for (int i = 0; i < 6; ++i)
            residual[i] = T(sw_) * (pose[i] - T(pose0_[i]));
        return true;
    }

    static ceres::CostFunction* Create(const double* pose0,
                                       double anchor_weight) {
        return new ceres::AutoDiffCostFunction<SQPoseAnchor, 6, 6>(
            new SQPoseAnchor(pose0, anchor_weight));
    }

    double pose0_[6];
    double sw_;
};
```

> The anchor mixes a rotation (radians) and a translation (meters) in one block with one weight. That is acceptable here because the SUPERDEC-init poses are already metric-scaled into the predicted frame and the anchor's job is only to keep the gauge well-posed, not to be a calibrated prior. If you later want separate rot/trans stiffness, split into two `AutoDiffCostFunction<...,3,3>` anchors on `pose` and `pose+3`; not needed for the default.

---

### Edit 1.3 — `run_bundle_adjustment`: signature + per-SQ pose blocks + anchors + ordering

**Edit 1.3a — signature.** Append the two new params **after** `function_tolerance` (current lines 289–291).

**OLD:**
```cpp
        int    max_num_iterations = 200,
        int    num_threads        = 4,
        double function_tolerance = 1e-6)
{
```
**NEW:**
```cpp
        int    max_num_iterations = 200,
        int    num_threads        = 4,
        double function_tolerance = 1e-6,
        bool   refine_sq          = false,   // optimise each SQ rigid pose
        double sq_anchor_weight   = 10.0)    // soft prior keeping pose near init
{
```

**Edit 1.3b — surface block construction.** This replaces the surface-block section. **OLD block** (current lines 329–390, from the comment through the closing brace of the `if`):

```cpp
    // Optional surface residual against frozen SUPERDEC primitives.
    int num_surface_blocks = 0;
    if (lambda_surface > 0.0
        && !sq_params.is_none()
        && !point_to_sq.is_none())
    {
        auto sq_arr = py::cast<py::array_t<double, py::array::c_style>>(sq_params);
        auto p2s_arr = py::cast<py::array_t<int, py::array::c_style>>(point_to_sq);
        auto sq_buf  = sq_arr.request();
        auto p2s_buf = p2s_arr.request();

        if (sq_buf.ndim != 2 || sq_buf.shape[1] != 11)
            throw std::runtime_error("sq_params must have shape (K, 11)");
        if (p2s_buf.ndim != 1 || p2s_buf.shape[0] != num_pts)
            throw std::runtime_error(
                "point_to_sq must have shape (num_points,) matching points");

        const int num_sqs = static_cast<int>(sq_buf.shape[0]);
        const double* sq_data  = static_cast<double*>(sq_buf.ptr);
        const int*    p2s_data = static_cast<int*>(p2s_buf.ptr);

        // Optional per-point surface weight (soft gating). None -> all 1.0.
        py::array_t<double, py::array::c_style> w_arr;
        const double* w_data = nullptr;
        if (!point_weights.is_none()) {
            w_arr = py::cast<py::array_t<double, py::array::c_style>>(point_weights);
            auto w_buf = w_arr.request();
            if (w_buf.ndim != 1 || w_buf.shape[0] != num_pts)
                throw std::runtime_error(
                    "point_weights must have shape (num_points,) matching points");
            w_data = static_cast<double*>(w_buf.ptr);
        }

        for (int i = 0; i < num_pts; ++i) {
            const int s = p2s_data[i];
            if (s < 0) continue;
            if (s >= num_sqs)
                throw std::runtime_error(
                    "point_to_sq value out of range for sq_params");

            const double w = (w_data ? w_data[i] : 1.0);
            if (w <= 0.0) continue;   // zero weight -> skip block entirely

            const double* row = sq_data + 11 * s;
            const double a1 = row[0], a2 = row[1], a3 = row[2];
            const double e1 = row[3], e2 = row[4];
            const double* rot_aa = row + 5;     // (3,)
            const double* trans  = row + 8;     // (3,)

            ceres::LossFunction* loss = nullptr;
            if (surface_huber > 0.0)
                loss = new ceres::HuberLoss(surface_huber);

            problem.AddResidualBlock(
                SurfaceResidual::Create(a1, a2, a3, e1, e2,
                                        rot_aa, trans, lambda_surface,
                                        residual_mode, w),
                loss,
                pt_data + 3 * i);
            ++num_surface_blocks;
        }
    }
```

**NEW block.** A persistent `sq_pose` buffer (one `[aa,t]` row per SQ) holds the optimised pose; it is created up-front from `sq_params` so every surface block of the same SQ `s` shares the **same** pose block pointer. When `refine_sq=false`, the pose blocks are added then immediately `SetParameterBlockConstant` (identical residuals + zero pose Jacobian => byte-identical to the old frozen path). When `refine_sq=true`, used SQs get one `SQPoseAnchor`, stay free, and are collected for the ordering. Note the new declarations `sq_pose`, `sq_pose_used`, and `num_sqs_decl` are hoisted to function scope so Edit 1.3c (ordering) can see them.

```cpp
    // Optional surface residual against SUPERDEC primitives.
    // The SQ pose ([aa(3), t(3)]) is a 6-D parameter block per SQ. With
    // refine_sq=false it is held CONSTANT (byte-identical to the frozen-pose
    // path); with refine_sq=true it is optimised jointly with cameras+points,
    // softly anchored to its SUPERDEC init via SQPoseAnchor.
    int num_surface_blocks = 0;
    int num_anchor_blocks  = 0;
    std::vector<std::array<double, 6>> sq_pose;   // per-SQ [aa, t], persistent
    std::vector<char> sq_pose_used;               // 1 if SQ s has >=1 point
    int num_sqs_decl = 0;
    if (lambda_surface > 0.0
        && !sq_params.is_none()
        && !point_to_sq.is_none())
    {
        auto sq_arr = py::cast<py::array_t<double, py::array::c_style>>(sq_params);
        auto p2s_arr = py::cast<py::array_t<int, py::array::c_style>>(point_to_sq);
        auto sq_buf  = sq_arr.request();
        auto p2s_buf = p2s_arr.request();

        if (sq_buf.ndim != 2 || sq_buf.shape[1] != 11)
            throw std::runtime_error("sq_params must have shape (K, 11)");
        if (p2s_buf.ndim != 1 || p2s_buf.shape[0] != num_pts)
            throw std::runtime_error(
                "point_to_sq must have shape (num_points,) matching points");

        const int num_sqs = static_cast<int>(sq_buf.shape[0]);
        num_sqs_decl = num_sqs;
        const double* sq_data  = static_cast<double*>(sq_buf.ptr);
        const int*    p2s_data = static_cast<int*>(p2s_buf.ptr);

        // Persistent per-SQ pose blocks, initialised from sq_params rows.
        // aa = row[5..7], t = row[8..10]. These doubles are mutated in place
        // by Ceres when refine_sq is true; read back by the wrapper if needed.
        sq_pose.resize(num_sqs);
        sq_pose_used.assign(num_sqs, 0);
        for (int s = 0; s < num_sqs; ++s) {
            const double* row = sq_data + 11 * s;
            for (int k = 0; k < 3; ++k) sq_pose[s][k]     = row[5 + k]; // aa
            for (int k = 0; k < 3; ++k) sq_pose[s][3 + k] = row[8 + k]; // t
        }

        // Optional per-point surface weight (soft gating). None -> all 1.0.
        py::array_t<double, py::array::c_style> w_arr;
        const double* w_data = nullptr;
        if (!point_weights.is_none()) {
            w_arr = py::cast<py::array_t<double, py::array::c_style>>(point_weights);
            auto w_buf = w_arr.request();
            if (w_buf.ndim != 1 || w_buf.shape[0] != num_pts)
                throw std::runtime_error(
                    "point_weights must have shape (num_points,) matching points");
            w_data = static_cast<double*>(w_buf.ptr);
        }

        for (int i = 0; i < num_pts; ++i) {
            const int s = p2s_data[i];
            if (s < 0) continue;
            if (s >= num_sqs)
                throw std::runtime_error(
                    "point_to_sq value out of range for sq_params");

            const double w = (w_data ? w_data[i] : 1.0);
            if (w <= 0.0) continue;   // zero weight -> skip block entirely

            const double* row = sq_data + 11 * s;
            const double a1 = row[0], a2 = row[1], a3 = row[2];
            const double e1 = row[3], e2 = row[4];

            ceres::LossFunction* loss = nullptr;
            if (surface_huber > 0.0)
                loss = new ceres::HuberLoss(surface_huber);

            problem.AddResidualBlock(
                SurfaceResidual::CreateRefine(a1, a2, a3, e1, e2,
                                              lambda_surface,
                                              residual_mode, w),
                loss,
                pt_data + 3 * i,
                sq_pose[s].data());        // shared per-SQ pose block
            sq_pose_used[s] = 1;
            ++num_surface_blocks;
        }

        // Configure the SQ pose blocks now that they are in the problem.
        for (int s = 0; s < num_sqs; ++s) {
            if (!sq_pose_used[s]) continue;
            double* pose = sq_pose[s].data();
            if (!problem.HasParameterBlock(pose)) continue;
            if (!refine_sq) {
                // Frozen: constant pose -> zero Jacobian -> identical residuals
                // and identical solution to the old single-block path.
                problem.SetParameterBlockConstant(pose);
            } else {
                // Refine: keep free + add a soft anchor to its init pose.
                if (sq_anchor_weight > 0.0) {
                    double pose0[6];
                    for (int k = 0; k < 6; ++k) pose0[k] = pose[k];
                    problem.AddResidualBlock(
                        SQPoseAnchor::Create(pose0, sq_anchor_weight),
                        nullptr,
                        pose);
                    ++num_anchor_blocks;
                }
            }
        }
    }
```

**Edit 1.3c — ordering.** Add the free SQ pose blocks to ordering **group 1** (eliminated after points, alongside cameras) so SPARSE_SCHUR stays valid. Insert this loop **immediately after** the camera-ordering loop and **before** `ceres::Solver::Options options;`.

**OLD** (current lines 432–440):
```cpp
    for (int i = 0; i < num_cams; ++i) {
        double* ptr = cam_data + 10 * i;
        if (problem.HasParameterBlock(ptr)) {
            ordering->AddElementToGroup(ptr, 1);
            ++cams_in_problem;
        }
    }

    ceres::Solver::Options options;
```
**NEW:**
```cpp
    for (int i = 0; i < num_cams; ++i) {
        double* ptr = cam_data + 10 * i;
        if (problem.HasParameterBlock(ptr)) {
            ordering->AddElementToGroup(ptr, 1);
            ++cams_in_problem;
        }
    }
    // Free SQ pose blocks (refine_sq) join group 1: non-point unknowns
    // eliminated after the structure block in the Schur complement.
    if (refine_sq) {
        for (int s = 0; s < num_sqs_decl; ++s) {
            if (s >= static_cast<int>(sq_pose.size()) || !sq_pose_used[s])
                continue;
            double* pose = sq_pose[s].data();
            if (problem.HasParameterBlock(pose) &&
                !problem.IsParameterBlockConstant(pose))
                ordering->AddElementToGroup(pose, 1);
        }
    }

    ceres::Solver::Options options;
```

> Requires `<vector>` and `<array>` headers. `<vector>` is pulled in transitively via Ceres/pybind, but to be safe add them explicitly (Edit 1.4).

**Edit 1.3d — verbose line (optional).** Extend the existing surface verbose print to report anchors. **OLD** (current lines 458–462):
```cpp
        if (num_surface_blocks > 0)
            std::cout << "[surface] " << num_surface_blocks
                      << " surface residual blocks active "
                      << "(lambda=" << lambda_surface
                      << ", huber=" << surface_huber << ")\n";
```
**NEW:**
```cpp
        if (num_surface_blocks > 0)
            std::cout << "[surface] " << num_surface_blocks
                      << " surface residual blocks active "
                      << "(lambda=" << lambda_surface
                      << ", huber=" << surface_huber
                      << ", refine_sq=" << (refine_sq ? 1 : 0)
                      << ", anchors=" << num_anchor_blocks
                      << ", anchor_w=" << sq_anchor_weight << ")\n";
```

---

### Edit 1.4 — headers

**OLD** (current lines 1–6):
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cmath>
#include <iostream>
```
**NEW:**
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <array>
```

---

### Edit 1.5 — pybind registration

Append the two new args **after** `function_tolerance` (current lines 488–489).

**OLD:**
```cpp
          py::arg("max_num_iterations") = 200,
          py::arg("num_threads")        = 4,
          py::arg("function_tolerance") = 1e-6,
          R"doc(
```
**NEW:**
```cpp
          py::arg("max_num_iterations") = 200,
          py::arg("num_threads")        = 4,
          py::arg("function_tolerance") = 1e-6,
          py::arg("refine_sq")          = false,
          py::arg("sq_anchor_weight")   = 10.0,
          R"doc(
```

---

## FILE 2 — `ba/python/ba/__init__.py`

### Edit 2.1 — `run_bundle_adjustment_mast3r_sq` wrapper

The wrapper forwards positionally, so the two new args go at the **end** of both the signature and the forwarded call (matching the C++ append order).

**OLD** (lines 62–96):
```python
def run_bundle_adjustment_mast3r_sq(cameras, points, observations,
                                    cam_indices, pt_indices,
                                    fix_first_camera=True,
                                    huber_threshold=2.0,
                                    verbose=False,
                                    fix_points=False,
                                    sq_params=None,
                                    point_to_sq=None,
                                    lambda_surface=0.0,
                                    surface_huber=0.0,
                                    residual_mode=0,
                                    point_weights=None,
                                    max_num_iterations=200,
                                    num_threads=4,
                                    function_tolerance=1e-6):
    """Run MASt3R + superquadric Ceres BA. All arrays are modified **in place**.

    Surface-residual mode is engaged when ``lambda_surface > 0`` and both
    ``sq_params`` (K, 11) and ``point_to_sq`` (num_points,) are supplied.
    See ``ba.superdec.pack_for_ceres`` for the expected sq_params layout.

    ``max_num_iterations`` / ``num_threads`` / ``function_tolerance`` expose the
    Ceres solver budget (defaults match the live benchmark; lower iterations or
    a looser tolerance trade a little accuracy for large speed-ups in sweeps).
    """
    core = _load_core("mast3r_sq_ba_core")
    return core.run_bundle_adjustment(cameras, points, observations,
                                      cam_indices, pt_indices,
                                      fix_first_camera, huber_threshold,
                                      verbose, fix_points,
                                      sq_params, point_to_sq,
                                      lambda_surface, surface_huber,
                                      residual_mode, point_weights,
                                      max_num_iterations, num_threads,
                                      function_tolerance)
```
**NEW:**
```python
def run_bundle_adjustment_mast3r_sq(cameras, points, observations,
                                    cam_indices, pt_indices,
                                    fix_first_camera=True,
                                    huber_threshold=2.0,
                                    verbose=False,
                                    fix_points=False,
                                    sq_params=None,
                                    point_to_sq=None,
                                    lambda_surface=0.0,
                                    surface_huber=0.0,
                                    residual_mode=0,
                                    point_weights=None,
                                    max_num_iterations=200,
                                    num_threads=4,
                                    function_tolerance=1e-6,
                                    refine_sq=False,
                                    sq_anchor_weight=10.0):
    """Run MASt3R + superquadric Ceres BA. All arrays are modified **in place**.

    Surface-residual mode is engaged when ``lambda_surface > 0`` and both
    ``sq_params`` (K, 11) and ``point_to_sq`` (num_points,) are supplied.
    See ``ba.superdec.pack_for_ceres`` for the expected sq_params layout.

    ``max_num_iterations`` / ``num_threads`` / ``function_tolerance`` expose the
    Ceres solver budget (defaults match the live benchmark; lower iterations or
    a looser tolerance trade a little accuracy for large speed-ups in sweeps).

    ``refine_sq`` (default False -> identical to the frozen-pose behaviour)
    lets Ceres optimise each used SQ's rigid pose ([aa(3), t(3)]) jointly with
    cameras+points. ``sq_anchor_weight`` is the stiffness of a soft prior
    pulling each refined SQ pose back to its SUPERDEC init (keeps the gauge
    well-posed; ignored when refine_sq is False).
    """
    core = _load_core("mast3r_sq_ba_core")
    return core.run_bundle_adjustment(cameras, points, observations,
                                      cam_indices, pt_indices,
                                      fix_first_camera, huber_threshold,
                                      verbose, fix_points,
                                      sq_params, point_to_sq,
                                      lambda_surface, surface_huber,
                                      residual_mode, point_weights,
                                      max_num_iterations, num_threads,
                                      function_tolerance,
                                      refine_sq, sq_anchor_weight)
```

---

## FILE 3 (optional convenience) — `ba/eval/strat_common.py`

Lets offline strategies forward the flag. Default-OFF keeps every current strategy byte-identical.

### Edit 3.1 — `solve()` signature + forwarding

**OLD** (lines 88–108):
```python
def solve(cameras, points, observations, cam_indices, pt_indices, *,
          lambda_surface=0.0, surface_huber=0.0, huber_threshold=2.0,
          fix_first_camera=True, sq_params=None, point_to_sq=None,
          residual_mode=0, point_weights=None,
          max_iterations=50, function_tolerance=1e-3, num_threads=4):
    """Run one Ceres mast3r_sq solve IN PLACE on cameras/points.

    Returns (final_cost, num_successful_steps). Pass sq_params/point_to_sq with
    lambda_surface>0 to enable the surface term; omit them for plain reprojection.
    ``residual_mode`` selects the surface-residual form (0=RADIAL default,
    1=HINGE_OUTSIDE, 2=HINGE_INSIDE, 3=RADIAL_NORMALIZED, 4=HINGE_OUTSIDE_NORMALIZED);
    ``point_weights`` (M,) optionally soft-weights each point's surface term.
    """
    return ba.run_bundle_adjustment_mast3r_sq(
        cameras, points, observations, cam_indices, pt_indices,
        fix_first_camera=fix_first_camera, huber_threshold=huber_threshold,
        verbose=False, fix_points=False, sq_params=sq_params,
        point_to_sq=point_to_sq, lambda_surface=lambda_surface,
        surface_huber=surface_huber, residual_mode=residual_mode,
        point_weights=point_weights, max_num_iterations=max_iterations,
        function_tolerance=function_tolerance, num_threads=num_threads)
```
**NEW:**
```python
def solve(cameras, points, observations, cam_indices, pt_indices, *,
          lambda_surface=0.0, surface_huber=0.0, huber_threshold=2.0,
          fix_first_camera=True, sq_params=None, point_to_sq=None,
          residual_mode=0, point_weights=None,
          max_iterations=50, function_tolerance=1e-3, num_threads=4,
          refine_sq=False, sq_anchor_weight=10.0):
    """Run one Ceres mast3r_sq solve IN PLACE on cameras/points.

    Returns (final_cost, num_successful_steps). Pass sq_params/point_to_sq with
    lambda_surface>0 to enable the surface term; omit them for plain reprojection.
    ``residual_mode`` selects the surface-residual form (0=RADIAL default,
    1=HINGE_OUTSIDE, 2=HINGE_INSIDE, 3=RADIAL_NORMALIZED, 4=HINGE_OUTSIDE_NORMALIZED);
    ``point_weights`` (M,) optionally soft-weights each point's surface term.
    ``refine_sq`` (default False) lets Ceres refine each used SQ's rigid pose,
    softly anchored to its SUPERDEC init with stiffness ``sq_anchor_weight``.
    """
    return ba.run_bundle_adjustment_mast3r_sq(
        cameras, points, observations, cam_indices, pt_indices,
        fix_first_camera=fix_first_camera, huber_threshold=huber_threshold,
        verbose=False, fix_points=False, sq_params=sq_params,
        point_to_sq=point_to_sq, lambda_surface=lambda_surface,
        surface_huber=surface_huber, residual_mode=residual_mode,
        point_weights=point_weights, max_num_iterations=max_iterations,
        function_tolerance=function_tolerance, num_threads=num_threads,
        refine_sq=refine_sq, sq_anchor_weight=sq_anchor_weight)
```

> A strategy then just calls `solve(..., refine_sq=True, sq_anchor_weight=W)`. If you also expose it in `mast3r_bundle_adjust` for the live EM loop, add `refine_sq=False`/`sq_anchor_weight=10.0` kwargs there and forward them into the three `run_bundle_adjustment_mast3r_sq(...)` calls — but that is not required for offline tuning and is left out to keep the live path byte-identical.

---

## Numerical / gauge risk note

Refining SQ pose adds a per-SQ rigid 6-DoF block that is fully observable only when an SQ owns enough well-spread points; an SQ with one or two near-coplanar points is rank-deficient in pose and would drift, and globally the surface term alone is gauge-free up to the same Sim3 that maps GT↔predicted (the cameras' fixed first pose pins rotation+translation but **not** the scene scale that the SQ poses also live in). The `SQPoseAnchor` resolves both: it makes every used SQ-pose block locally full-rank (so SPARSE_SCHUR's group-1 elimination never hits a singular reduced camera/pose system) and ties the SQ cloud to its SUPERDEC init, removing the residual gauge freedom. `sq_anchor_weight` is the explicit bias/variance knob: large (≫λ_surface) ⇒ poses barely move, recovering the frozen result and wasting the new DoF; very small (→0) ⇒ poses chase noisy points, the gauge softens, and pose can co-adapt with camera error exactly the way frozen mis-registration currently caps λ — so the anchor must stay stiff enough that the SQ cloud moves as a near-rigid whole rather than per-primitive. Recommended sweep: start `sq_anchor_weight ∈ {3, 10, 30}` with the current best λ_surface, watch that median point→SQ distance drops without `pose_auc_5` regressing, and only then push λ higher (the whole point of refine_sq is that a movable, anchored SQ cloud should now tolerate a larger λ than the frozen λ>15 ceiling). Mixing rotation (rad) and translation (m) under one weight is a minor miscalibration but harmless at the gauge-fixing scale; split the anchor into 3+3 blocks if rotational drift dominates. Default `refine_sq=false` makes the constant pose block contribute a zero Jacobian, so the frozen path is provably byte-identical to today.

### Files referenced
- `/work/courses/3dv/team39/ba/src/mast3r_sq_ba.cpp` (Edits 1.1–1.5; apply via orchestrator, do NOT hand-edit)
- `/work/courses/3dv/team39/ba/python/ba/__init__.py` (Edit 2.1)
- `/work/courses/3dv/team39/ba/eval/strat_common.py` (Edit 3.1, optional)
- `/work/courses/3dv/team39/ba/python/ba/superdec.py` (`pack_for_ceres` layout confirmed: row = `[scale(3), exp(2), aa(3), t(3)]`, so `aa=row+5`, `t=row+8`, matching Edit 1.3b)