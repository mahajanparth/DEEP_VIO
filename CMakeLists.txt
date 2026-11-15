cmake_minimum_required(VERSION 3.16)
project(deep_vio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Keep Anaconda's copies of OpenCV / yaml-cpp / Eigen out of the package search.
list(APPEND CMAKE_IGNORE_PREFIX_PATH "$ENV{HOME}/anaconda3")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs highgui video features2d calib3d)

# GTSAM must be compiled against the same Eigen as this project. The apt libgtsam on this
# machine targets Ubuntu 20.04 (Eigen 3.3.7 / TBB 2020) and does not link against the 22.04
# headers, so only a GTSAM built by scripts/build_gtsam.sh (third_party/install) is accepted.
find_package(GTSAM 4.3 CONFIG QUIET PATHS ${PROJECT_SOURCE_DIR}/third_party/install NO_DEFAULT_PATH)
if(GTSAM_FOUND)
  set(DEEP_VIO_WITH_GTSAM ON)
  message(STATUS "GTSAM ${GTSAM_VERSION} found: building the VIO backend")
  # libgtsam depends on libmetis-gtsam in the same folder but carries no RPATH itself. A DT_RPATH
  # (unlike the default DT_RUNPATH) on our binaries also applies to such transitive dependencies.
  add_link_options(-Wl,--disable-new-dtags)
else()
  set(DEEP_VIO_WITH_GTSAM OFF)
  message(WARNING "Compatible GTSAM not found; backend and run_euroc are skipped. "
                  "Run scripts/build_gtsam.sh to enable them.")
endif()

add_compile_options(-Wall -Wextra)

# ---- core: everything that does not need GTSAM ----
add_library(deep_vio_core
  src/dataset/euroc_dataset.cpp
  src/geometry/so3.cpp
  src/geometry/camera_model.cpp
  src/geometry/triangulation.cpp
  src/imu/imu_segments.cpp
  src/imu/preintegration.cpp
)
target_include_directories(deep_vio_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(deep_vio_core PUBLIC Eigen3::Eigen ${OpenCV_LIBS})

add_executable(play_euroc apps/play_euroc.cpp)
target_link_libraries(play_euroc PRIVATE deep_vio_core)

# ---- backend: GTSAM sliding-window estimator ----

# ---- tests ----
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  include(GoogleTest)

  add_executable(deep_vio_tests
    tests/test_euroc_dataset.cpp
    tests/test_so3.cpp
    tests/test_camera_model.cpp
    tests/test_triangulation.cpp
  )
  target_link_libraries(deep_vio_tests PRIVATE deep_vio_core GTest::gtest_main)
  gtest_discover_tests(deep_vio_tests)

endif()
