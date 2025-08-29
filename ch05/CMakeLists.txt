cmake_minimum_required(VERSION 4.0)
project(nonlinearOptimization)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Ceres CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)

find_package(Sophus CONFIG REQUIRED)
find_package(Pangolin REQUIRED)
include_directories( ${Pangolin_INCLUDE_DIRS} )

add_executable(gaussNewton gaussNewton.cpp)
target_link_libraries(gaussNewton PRIVATE
        ${OpenCV_LIBS}
        Eigen3::Eigen)
add_executable(ceresCurveFitting ceresCurveFitting.cpp)
target_link_libraries(ceresCurveFitting PRIVATE
        ${OpenCV_LIBS}
        Ceres::ceres)
add_executable(g2oCurveFitting g2oCurveFitting.cpp)
target_link_libraries(g2oCurveFitting PRIVATE
        ${OpenCV_LIBS}
        g2o::core)
