cmake_minimum_required(VERSION 4.0)
project(directMethod)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Sophus CONFIG REQUIRED)

add_executable(optical_flow optical_flow.cpp)
target_link_libraries(optical_flow PRIVATE
        ${OpenCV_LIBS}
        Eigen3::Eigen)
add_executable(direct_method direct_method.cpp)
target_link_libraries(direct_method PRIVATE
        ${OpenCV_LIBS}
        Sophus::Sophus
        Eigen3::Eigen)
