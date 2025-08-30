cmake_minimum_required(VERSION 4.0)
project(featureMethod)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Sophus CONFIG REQUIRED)
find_package(Pangolin REQUIRED)
include_directories( ${Pangolin_INCLUDE_DIRS} )

add_executable(orb_cv orb_cv.cpp)
target_link_libraries(orb_cv PRIVATE ${OpenCV_LIBS})
add_executable(orb_self orb_self.cpp)
target_link_libraries(orb_self PRIVATE ${OpenCV_LIBS})
