#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

// Camera parameterisation (10 params):
//   camera[0..2]  angle-axis rotation  (world-to-camera)
//   camera[3..5]  translation          (world-to-camera)
//   camera[6]     fx
//   camera[7]     fy
//   camera[8]     cx
//   camera[9]     cy
struct ReprojectionError {
    ReprojectionError(double obs_x, double obs_y)
        : obs_x_(obs_x), obs_y_(obs_y) {}

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const {
        // Rotate point into camera frame (world-to-camera)
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);

        // Translate
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Perspective divide
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        // Apply intrinsics: fx, fy, cx, cy
        T predicted_x = camera[6] * xp + camera[8];
        T predicted_y = camera[7] * yp + camera[9];

        residuals[0] = predicted_x - T(obs_x_);
        residuals[1] = predicted_y - T(obs_y_);

        return true;
    }

    static ceres::CostFunction* Create(double obs_x, double obs_y) {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 10, 3>(
            new ReprojectionError(obs_x, obs_y));
    }

    double obs_x_;
    double obs_y_;
};


// -------------------------------------------------------------------------
// Surface residual against a frozen superquadric primitive.
//
// Mirrors SUPERDEC's LM-time radial-distance residual at
//   superdec/superdec/lm_optimization/lm_optimizer.py:99-136
// (inverse-form Solina). The SQ parameters are baked in at construction
// (held constant during BA — Phase 1 of the experiment plan).
//
// Frame convention:
//   - rotation_aa is angle-axis encoding R, where R: canonical -> world.
//     We rotate by (-aa) to get the world -> canonical mapping.
//   - translation is the SQ centre in world coords.
//   - point parameter block (3 floats) is the world-frame BA point.
//
// Residual (single scalar, in pixel-equivalent units after lambda scaling):
//   q = R^T (p - t)
//   F = (|q_x/a_1|^{2/eps_2} + |q_y/a_2|^{2/eps_2})^{eps_2/eps_1}
//     + |q_z/a_3|^{2/eps_1}
//   r = lambda_surface * ||q|| * | F^{-eps_1/2} - 1 |
//
// Numerical safeguards mirror superdec.utils.safe_operations.safe_pow:
// abs() inputs to pow are clamped to [1e-3, 5e2] before and after.
struct SurfaceResidual {
    SurfaceResidual(double a1, double a2, double a3,
                    double e1, double e2,
                    const double* rotation_aa,
                    const double* translation,
                    double lambda_surface)
        : a1_(a1), a2_(a2), a3_(a3),
          e1_(e1), e2_(e2),
          lambda_(lambda_surface)
    {
        // Negate angle-axis: R(-aa) is the inverse of R(aa).
        neg_aa_[0] = -rotation_aa[0];
        neg_aa_[1] = -rotation_aa[1];
        neg_aa_[2] = -rotation_aa[2];
        t_[0] = translation[0];
        t_[1] = translation[1];
        t_[2] = translation[2];
    }

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

    template <typename T>
    bool operator()(const T* const point, T* residual) const {
        // diff = point - t
        T diff[3] = {
            point[0] - T(t_[0]),
            point[1] - T(t_[1]),
            point[2] - T(t_[2]),
        };

        // q = R^T diff = AngleAxisRotate(-aa, diff)
        const T neg_aa[3] = {T(neg_aa_[0]), T(neg_aa_[1]), T(neg_aa_[2])};
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

        T inside_outside = safe_pow_pos(F, T(-e1_ / 2.0)) - T(1.0);

        residual[0] = T(lambda_) * q_norm * ceres::abs(inside_outside);
        return true;
    }

    static ceres::CostFunction* Create(double a1, double a2, double a3,
                                        double e1, double e2,
                                        const double* rotation_aa,
                                        const double* translation,
                                        double lambda_surface) {
        return new ceres::AutoDiffCostFunction<SurfaceResidual, 1, 3>(
            new SurfaceResidual(a1, a2, a3, e1, e2,
                                rotation_aa, translation, lambda_surface));
    }

    double a1_, a2_, a3_;
    double e1_, e2_;
    double neg_aa_[3];
    double t_[3];
    double lambda_;
};


// Run bundle adjustment in-place.
//
// cameras           : (num_cameras, 10) float64  –  modified in place
// points            : (num_points,  3) float64  –  modified in place
// observations      : (num_obs,     2) float64  –  (u, v) pixel observations
// cam_indices       : (num_obs,)       int32
// pt_indices        : (num_obs,)       int32
// fix_first_camera  : bool             keep camera[0] constant (fixes gauge)
// huber_threshold   : double           Huber loss threshold in pixels
// verbose           : bool             print Ceres summary
// fix_points        : bool             freeze 3-D structure (cameras-only refine)
//
// Surface residual (optional, all four kwargs must be supplied together to
// activate; supply None / 0 to disable):
//   sq_params      : (K, 11) float64
//                    [0:3]  scale (a1, a2, a3)
//                    [3:5]  exponents (eps1, eps2)
//                    [5:8]  rotation angle-axis (canonical -> world)
//                    [8:11] translation (world frame)
//   point_to_sq    : (num_points,) int32  index into sq_params, or -1 = unassigned
//   lambda_surface : double  pixels-per-meter weight applied to surface residual.
//                    0 disables the surface term entirely.
//   surface_huber  : double  Huber delta in pixel-equivalent units (post-lambda).
//                    <=0 disables Huber on the surface term.
//
// Returns (final_cost, num_iterations_taken).
py::tuple run_bundle_adjustment(
        py::array_t<double, py::array::c_style> cameras,
        py::array_t<double, py::array::c_style> points,
        py::array_t<double, py::array::c_style> observations,
        py::array_t<int,    py::array::c_style> cam_indices,
        py::array_t<int,    py::array::c_style> pt_indices,
        bool   fix_first_camera = true,
        double huber_threshold  = 2.0,
        bool   verbose          = false,
        bool   fix_points       = false,
        py::object sq_params    = py::none(),
        py::object point_to_sq  = py::none(),
        double lambda_surface   = 0.0,
        double surface_huber    = 0.0)
{
    auto cam_buf  = cameras.request();
    auto pt_buf   = points.request();
    auto obs_buf  = observations.request();
    auto ci_buf   = cam_indices.request();
    auto pi_buf   = pt_indices.request();

    if (cam_buf.ndim != 2 || cam_buf.shape[1] != 10)
        throw std::runtime_error("cameras must have shape (N, 10)");
    if (pt_buf.ndim != 2 || pt_buf.shape[1] != 3)
        throw std::runtime_error("points must have shape (M, 3)");
    if (obs_buf.ndim != 2 || obs_buf.shape[1] != 2)
        throw std::runtime_error("observations must have shape (K, 2)");

    const int num_cams = static_cast<int>(cam_buf.shape[0]);
    const int num_pts  = static_cast<int>(pt_buf.shape[0]);
    const int num_obs  = static_cast<int>(obs_buf.shape[0]);

    double* cam_data  = static_cast<double*>(cam_buf.ptr);
    double* pt_data   = static_cast<double*>(pt_buf.ptr);
    double* obs_data  = static_cast<double*>(obs_buf.ptr);
    int*    ci_data   = static_cast<int*>(ci_buf.ptr);
    int*    pi_data   = static_cast<int*>(pi_buf.ptr);

    ceres::Problem problem;

    for (int i = 0; i < num_obs; ++i) {
        double u = obs_data[2 * i];
        double v = obs_data[2 * i + 1];

        problem.AddResidualBlock(
            ReprojectionError::Create(u, v),
            new ceres::HuberLoss(huber_threshold),
            cam_data + 10 * ci_data[i],
            pt_data  + 3 * pi_data[i]);
    }

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

        for (int i = 0; i < num_pts; ++i) {
            const int s = p2s_data[i];
            if (s < 0) continue;
            if (s >= num_sqs)
                throw std::runtime_error(
                    "point_to_sq value out of range for sq_params");

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
                                        rot_aa, trans, lambda_surface),
                loss,
                pt_data + 3 * i);
            ++num_surface_blocks;
        }
    }

    // Fix camera 0 to remove gauge freedom (world frame = view-0 frame).
    // Guard with HasParameterBlock — a camera that no residual block references
    // (e.g. a surface-only solve) is not in the problem yet.
    if (fix_first_camera && num_cams > 0 && problem.HasParameterBlock(cam_data))
        problem.SetParameterBlockConstant(cam_data);

    // Intrinsics are GT calibration from the dataset — fix them so BA cannot
    // trade off focal length against scene depth (focal-length/depth ambiguity).
    // Skip camera 0 when it is already constant (SetManifold + constant is invalid).
    const int first_free = (fix_first_camera && num_cams > 0) ? 1 : 0;
    for (int i = first_free; i < num_cams; ++i) {
        double* ptr = cam_data + 10 * i;
        if (problem.HasParameterBlock(ptr))
            problem.SetManifold(ptr, new ceres::SubsetManifold(10, {6, 7, 8, 9}));
    }

    // When GT points are provided as structure, freeze them so that only camera
    // poses are optimised.  Without this, reprojection error is scale-invariant:
    // scaling all points by λ and all free camera translations by λ gives
    // identical residuals, causing translation magnitude to drift arbitrarily.
    if (fix_points) {
        for (int i = 0; i < num_pts; ++i) {
            double* ptr = pt_data + 3 * i;
            if (problem.HasParameterBlock(ptr))
                problem.SetParameterBlockConstant(ptr);
        }
    }

    // Schur-complement ordering: group 0 = 3-D points, group 1 = cameras.
    // Skip parameter blocks the problem hasn't seen (e.g. cameras with no
    // observations in a surface-only test).  When the problem contains no
    // cameras at all, fall back to SPARSE_NORMAL_CHOLESKY since Schur
    // elimination is meaningless without a structure/camera split.
    int cams_in_problem = 0;
    auto* ordering = new ceres::ParameterBlockOrdering;
    for (int i = 0; i < num_pts; ++i) {
        double* ptr = pt_data + 3 * i;
        if (problem.HasParameterBlock(ptr))
            ordering->AddElementToGroup(ptr, 0);
    }
    for (int i = 0; i < num_cams; ++i) {
        double* ptr = cam_data + 10 * i;
        if (problem.HasParameterBlock(ptr)) {
            ordering->AddElementToGroup(ptr, 1);
            ++cams_in_problem;
        }
    }

    ceres::Solver::Options options;
    if (cams_in_problem > 0 && num_obs > 0) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.linear_solver_ordering.reset(ordering);
    } else {
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        delete ordering;
    }
    options.minimizer_progress_to_stdout = verbose;
    options.num_threads                  = 4;
    options.max_num_iterations           = 200;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (verbose) {
        std::cout << summary.FullReport() << "\n";
        if (num_surface_blocks > 0)
            std::cout << "[surface] " << num_surface_blocks
                      << " surface residual blocks active "
                      << "(lambda=" << lambda_surface
                      << ", huber=" << surface_huber << ")\n";
    }

    return py::make_tuple(summary.final_cost,
                          summary.num_successful_steps);
}

PYBIND11_MODULE(mast3r_sq_ba_core, m) {
    m.doc() = "MASt3R + superquadric bundle adjustment using Ceres Solver";
    m.def("run_bundle_adjustment", &run_bundle_adjustment,
          py::arg("cameras"),
          py::arg("points"),
          py::arg("observations"),
          py::arg("cam_indices"),
          py::arg("pt_indices"),
          py::arg("fix_first_camera") = true,
          py::arg("huber_threshold")  = 2.0,
          py::arg("verbose")          = false,
          py::arg("fix_points")       = false,
          py::arg("sq_params")        = py::none(),
          py::arg("point_to_sq")      = py::none(),
          py::arg("lambda_surface")   = 0.0,
          py::arg("surface_huber")    = 0.0,
          R"doc(
Run sparse bundle adjustment in place.

Parameters
----------
cameras           : ndarray (N, 10)  angle-axis[3] | translation[3] | fx[1] | fy[1] | cx[1] | cy[1]
points            : ndarray (M, 3)   3-D world points
observations      : ndarray (K, 2)   (u, v) pixel observations
cam_indices       : ndarray (K,)     int32  camera index per observation
pt_indices        : ndarray (K,)     int32  point  index per observation
fix_first_camera  : bool             keep camera[0] constant to fix gauge (default True)
huber_threshold   : float            Huber loss delta in pixels (default 2.0)
verbose           : bool             print Ceres solver output
fix_points        : bool             freeze 3-D structure (cameras-only refine)

Surface-residual mode (optional — all four together):
sq_params         : ndarray (K, 11) float64 or None
                    columns: [scale(3), exponents(2), rotation_aa(3), translation(3)]
                    rotation_aa encodes R(canonical -> world); residual uses R^T.
point_to_sq       : ndarray (M,) int32 or None  point index -> sq row, -1 = skip
lambda_surface    : float            pixels-per-meter weight on surface residual
                                     (0 disables — default)
surface_huber     : float            Huber delta in pixel-equivalent units
                                     (<=0 disables Huber on surface term)

Returns
-------
(final_cost, num_successful_steps)
          )doc");
}
