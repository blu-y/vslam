cmake_minimum_required(VERSION 4.0)
project(denseRGBD)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Sophus CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)
find_package(Ceres CONFIG REQUIRED)

#add_executable(pointcloudMapping pointcloudMapping.cpp)
#target_link_libraries(pointcloudMapping PRIVATE
#        ${OpenCV_LIBS}
#        Sophus::Sophus
#        Eigen3::Eigen
#        Ceres::ceres)
#add_executable(surfelMapping surfelMapping.cpp)
#target_link_libraries(surfelMapping PRIVATE
#        ${OpenCV_LIBS}
#        Sophus::Sophus
#        Eigen3::Eigen
#        Ceres::ceres)
#add_executable(octomapMapping octomapMapping.cpp)
#target_link_libraries(octomapMapping PRIVATE
#        ${OpenCV_LIBS}
#        Sophus::Sophus
#        Eigen3::Eigen
#        Ceres::ceres)