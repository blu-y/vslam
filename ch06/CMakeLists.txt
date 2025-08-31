cmake_minimum_required(VERSION 4.0)
project(featureMethod)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Sophus CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)
find_package(Pangolin REQUIRED)
include_directories( ${Pangolin_INCLUDE_DIRS} )

add_executable(orb_cv orb_cv.cpp)
target_link_libraries(orb_cv PRIVATE ${OpenCV_LIBS})
add_executable(orb_self orb_self.cpp)
target_link_libraries(orb_self PRIVATE ${OpenCV_LIBS})
add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d PRIVATE ${OpenCV_LIBS})
add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation PRIVATE ${OpenCV_LIBS})
add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d PRIVATE
        ${OpenCV_LIBS}
        g2o::core
        Sophus::Sophus
        Eigen3::Eigen)
