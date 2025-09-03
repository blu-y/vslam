cmake_minimum_required(VERSION 4.0)
project(bundleAdjustment)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Sophus CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)
find_package(Ceres CONFIG REQUIRED)

add_library(bal_library common.cpp)
target_link_libraries(bal_library PRIVATE Eigen3::Eigen)

add_executable(bundleAdjustmentCeres bundleAdjustmentCeres.cpp)
target_link_libraries(bundleAdjustmentCeres PRIVATE
        ${OpenCV_LIBS}
        Sophus::Sophus
        Eigen3::Eigen
        Ceres::ceres
        bal_library)
add_executable(bundleAdjustmentG2o bundleAdjustmentG2o.cpp)
target_link_libraries(bundleAdjustmentG2o PRIVATE
        ${OpenCV_LIBS}
        Eigen3::Eigen
        g2o::core
        g2o::solver_csparse
        Sophus::Sophus
        bal_library)