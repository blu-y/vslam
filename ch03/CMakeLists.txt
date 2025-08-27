cmake_minimum_required(VERSION 4.0)
project(useSophus)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 CONFIG REQUIRED)
find_package(Sophus CONFIG REQUIRED)
find_package(Pangolin REQUIRED)
include_directories( ${Pangolin_INCLUDE_DIRS} )

add_executable(useSophus useSophus.cpp)
target_link_libraries(useSophus PRIVATE
        Eigen3::Eigen
        Sophus::Sophus)
add_executable(trajectoryError trajectoryError.cpp)
target_link_libraries(trajectoryError PRIVATE
        ${Pangolin_LIBRARIES}
        Sophus::Sophus)