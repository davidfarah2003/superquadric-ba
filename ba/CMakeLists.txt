cmake_minimum_required(VERSION 3.15)
project(ba_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Bake an RPATH into the produced .so files so they pick up the team-shared
# runtime libs (Ceres, glog, gflags, SuiteSparse, …) without requiring users
# to set LD_LIBRARY_PATH. Both the legacy /work/courses/3dv/team39/lib runtime
# stash and the new envs/3dv-cmake-prefix dir are searched.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH    TRUE)
# envs/3dv-cmake-prefix MUST come first: the legacy team39/lib stash holds a
# conda-built libceres.so.4 / libglog.so.2 set, while the prefix holds an
# apt-built libceres.so.4 / libglog.so.1 set. Mixing the two yields
# duplicate-glog-flag registration errors at import time. We link and run
# entirely against the apt set.
set(CMAKE_INSTALL_RPATH
    "/work/courses/3dv/team39/envs/3dv-cmake-prefix/usr/lib/x86_64-linux-gnu"
    "/work/courses/3dv/team39/lib")

# Resolve dependency roots (Ceres, SuiteSparse, glog, gflags, eigen, pybind11).
#
# Search order:
#   1. CMAKE_PREFIX_PATH already passed on the command line (highest priority).
#   2. $TEAM_DEPS_PREFIX from the environment, when set.
#   3. The shared team prefix at envs/3dv-cmake-prefix (populated from apt
#      libceres-dev / libeigen3-dev / libsuitesparse-dev / libglog-dev /
#      libgflags-dev / libabsl-dev / libmetis-dev .deb extracts).
#   4. $CONDA_PREFIX (kept for back-compat with personal conda envs).
#   5. The team venv (for pybind11 installed via pip).
if(DEFINED ENV{TEAM_DEPS_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{TEAM_DEPS_PREFIX}")
endif()
list(APPEND CMAKE_PREFIX_PATH "/work/courses/3dv/team39/envs/3dv-cmake-prefix/usr")
list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
list(APPEND CMAKE_PREFIX_PATH
     "/work/courses/3dv/team39/envs/3dv/lib/python3.12/site-packages/pybind11")

# glog ships its FindUnwind.cmake outside the package config dir; expose it
# via CMAKE_MODULE_PATH so glog-config can find_dependency(Unwind).
list(APPEND CMAKE_MODULE_PATH
     "/work/courses/3dv/team39/envs/3dv-cmake-prefix/usr/share/glog/cmake")

# Find dependencies. glog must be located before Ceres so that the
# glog::glog imported target referenced by CeresTargets.cmake is in scope.
find_package(glog REQUIRED)
find_package(Ceres REQUIRED)
find_package(pybind11 REQUIRED)

# Output .so files directly into the Python package directory so that
# `import ba` works after `pip install -e python/`
set(BA_PYTHON_PKG "${CMAKE_SOURCE_DIR}/python/ba")

# --- MASt3R BA ---
pybind11_add_module(mast3r_ba_core src/mast3r_ba.cpp)
target_link_libraries(mast3r_ba_core PRIVATE Ceres::ceres)
set_target_properties(mast3r_ba_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${BA_PYTHON_PKG}")

# --- MASt3R + superquadric BA ---
pybind11_add_module(mast3r_sq_ba_core src/mast3r_sq_ba.cpp)
target_link_libraries(mast3r_sq_ba_core PRIVATE Ceres::ceres)
set_target_properties(mast3r_sq_ba_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${BA_PYTHON_PKG}")

# --- VGGT + superquadric BA ---
pybind11_add_module(vggt_sq_ba_core src/vggt_sq_ba.cpp)
target_link_libraries(vggt_sq_ba_core PRIVATE Ceres::ceres)
set_target_properties(vggt_sq_ba_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${BA_PYTHON_PKG}")