cmake_minimum_required(VERSION 4.0)
project(denseMono)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Sophus CONFIG REQUIRED)

add_executable(denseMapping denseMapping.cpp)
target_link_libraries(denseMapping PRIVATE
        ${OpenCV_LIBS}
        Sophus::Sophus
        Eigen3::Eigen)