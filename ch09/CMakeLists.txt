cmake_minimum_required(VERSION 4.0)
project(poseGraph)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Sophus CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)
find_package(Ceres CONFIG REQUIRED)

#add_library(bal_library common.cpp)
#target_link_libraries(bal_library PRIVATE Eigen3::Eigen)

add_executable(poseGraphG2oSE3 poseGraphG2oSE3.cpp)
target_link_libraries(poseGraphG2oSE3 PRIVATE
        ${OpenCV_LIBS}
        Sophus::Sophus
        Eigen3::Eigen
        Ceres::ceres
        g2o::core
        g2o::types_slam3d
        g2o::solver_csparse)
add_executable(poseGraphG2oSophus poseGraphG2oSophus.cpp)
target_link_libraries(poseGraphG2oSophus PRIVATE
        ${OpenCV_LIBS}
        Sophus::Sophus
        Eigen3::Eigen
        Ceres::ceres
        g2o::core
        g2o::solver_csparse)
