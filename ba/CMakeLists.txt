cmake_minimum_required(VERSION 3.15)
project(ba_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point CMake to your active Conda environment
list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")

# Find dependencies
find_package(Ceres REQUIRED)
find_package(pybind11 REQUIRED)

# Output .so files directly into the Python package directory so that
# `import ba` works after `pip install -e python/`
set(BA_PYTHON_PKG "${CMAKE_SOURCE_DIR}/python/ba")

# --- baseline BA ---
pybind11_add_module(baseline_ba_core src/baseline_ba.cpp)
target_link_libraries(baseline_ba_core PRIVATE Ceres::ceres)
set_target_properties(baseline_ba_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${BA_PYTHON_PKG}")

# --- custom BA ---
pybind11_add_module(custom_ba_core src/custom_ba.cpp)
target_link_libraries(custom_ba_core PRIVATE Ceres::ceres)
set_target_properties(custom_ba_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${BA_PYTHON_PKG}")