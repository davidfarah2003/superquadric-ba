#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(vggt_sq_ba_core, m) {
    m.doc() = "VGGT + superquadric bundle adjustment using Ceres Solver";
}
