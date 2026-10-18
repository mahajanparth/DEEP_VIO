cmake_minimum_required(VERSION 3.16)
project(deep_vio LANGUAGE CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


if(NOT CMAKE_BUILD_TYPE )
    set(CMAKE_BUILD_TYPE Release)
endif()


find_package(Eigen3 3.3 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)


# add_library(deep_vio src/placeholder.cpp)
# target_include_directories(deep_vio ${PROJECT_SOURCE_DIR}/include)
# target_link_libraries(deep_vio PUBLIC Eigen3::Eigen  ${OpenCV_LIBS gtsam yaml-cpp})


# add_executable(play_euroc apps/play_euroc.cpp)
# target_link_libraries(play_euroc PRIVATE deep_vio)

# option(BUILD_TESTS "Build unit tests" ON)
# if(BUILD_TESTS)

#   include(FetchContent)
#   FetchContent_Declare(googletest
#     URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
#   )
#   FetchContent_MakeAvailable(googletest)
#   enable_testing()
#   add_executable(deep_vio_tests tests/test_sanity.cpp)
#   target_link_libraries(deep_vio_tests PRIVATE deep_vio GTest::gtest_main)
  
#   include(GoogleTest)
#   gtest_discover_tests(deep_vio_tests)
# endif()

