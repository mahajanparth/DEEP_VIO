#!/usr/bin/env bash
# Builds GTSAM 4.3.0 into third_party/install, against the system Eigen and without TBB.
#
# Why not the apt package? libgtsam 4.3a0 on this machine was built for Ubuntu 20.04
# (Eigen 3.3.7, TBB 2020) while the system headers are Eigen 3.4 / oneTBB 2021, so mangled
# symbols such as KeyVector's tbb_allocator don't match.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"
TAG=4.3.0
JOBS="${JOBS:-8}"

mkdir -p "$TP"
[ -d "$TP/gtsam" ] || git clone --depth 1 --branch "$TAG" https://github.com/borglab/gtsam.git "$TP/gtsam"

cmake -S "$TP/gtsam" -B "$TP/gtsam/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$TP/install" \
  -DCMAKE_IGNORE_PREFIX_PATH="$HOME/anaconda3" \
  -DGTSAM_USE_SYSTEM_EIGEN=ON \
  -DGTSAM_WITH_TBB=OFF \
  -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
  -DGTSAM_BUILD_UNSTABLE=OFF \
  -DGTSAM_BUILD_TESTS=OFF \
  -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_TIMING_ALWAYS=OFF \
  -DGTSAM_BUILD_PYTHON=OFF \
  -DGTSAM_UNSTABLE_BUILD_PYTHON=OFF \
  -DGTSAM_INSTALL_MATLAB_TOOLBOX=OFF
cmake --build "$TP/gtsam/build" -j"$JOBS"
cmake --install "$TP/gtsam/build"
echo "GTSAM $TAG installed to $TP/install"
