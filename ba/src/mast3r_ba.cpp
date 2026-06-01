#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
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
        // world-to-camera
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

// Run bundle adjustment in-place.
//
// cameras      : (num_cameras, 10) float64  –  modified in place
// points       : (num_points,  3) float64  –  modified in place
// observations : (num_obs,     2) float64  –  (u, v) pixel observations
// cam_indices  : (num_obs,)       int32
// pt_indices   : (num_obs,)       int32
//
// Returns (final_cost, num_iterations_taken).
py::tuple run_bundle_adjustment(
        py::array_t<double, py::array::c_style> cameras,
        py::array_t<double, py::array::c_style> points,
        py::array_t<double, py::array::c_style> observations,
        py::array_t<int,    py::array::c_style> cam_indices,
        py::array_t<int,    py::array::c_style> pt_indices,
        bool verbose = false)
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

    const int num_obs = static_cast<int>(obs_buf.shape[0]);

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
            new ceres::HuberLoss(1.0),
            cam_data + 10 * ci_data[i],
            pt_data  + 3 * pi_data[i]);
    }

    // Intrinsics are GT calibration from the dataset — fix them so BA cannot
    // trade off focal length against scene depth (focal-length/depth ambiguity).
    const int num_cams = static_cast<int>(cam_buf.shape[0]);
    for (int i = 0; i < num_cams; ++i) {
        double* ptr = cam_data + 10 * i;
        if (problem.HasParameterBlock(ptr))
            problem.SetManifold(ptr, new ceres::SubsetManifold(10, {6, 7, 8, 9}));
    }

    ceres::Solver::Options options;
    options.linear_solver_type           = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = verbose;
    options.num_threads                  = 4;
    options.max_num_iterations           = 200;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (verbose)
        std::cout << summary.FullReport() << "\n";

    return py::make_tuple(summary.final_cost,
                          summary.num_successful_steps);
}

PYBIND11_MODULE(mast3r_ba_core, m) {
    m.doc() = "MASt3R bundle adjustment using Ceres Solver";
    m.def("run_bundle_adjustment", &run_bundle_adjustment,
          py::arg("cameras"),
          py::arg("points"),
          py::arg("observations"),
          py::arg("cam_indices"),
          py::arg("pt_indices"),
          py::arg("verbose") = false,
          R"doc(
Run sparse bundle adjustment in place.

Parameters
----------
cameras      : ndarray (N, 10)  angle-axis[3] | translation[3] | fx[1] | fy[1] | cx[1] | cy[1]
points       : ndarray (M, 3)  3-D world points
observations : ndarray (K, 2)  (u, v) pixel observations
cam_indices  : ndarray (K,)    int32  camera index per observation
pt_indices   : ndarray (K,)    int32  point  index per observation
verbose      : bool            print Ceres solver output

Returns
-------
(final_cost, num_successful_steps)
          )doc");
}
