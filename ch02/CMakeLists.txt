cmake_minimum_required(VERSION 4.0)
project( useEigen )

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 20)

find_package( Pangolin REQUIRED )
include_directories( ${Pangolin_INCLUDE_DIRS} )
include_directories( "/opt/homebrew/include/eigen3" )

add_executable(eigenMatrix eigenMatrix.cpp)
add_executable(useGeometry useGeometry.cpp)
add_executable(coordinateTransform coordinateTransform.cpp)
add_executable(plotTrajectory plotTrajectory.cpp)
target_link_libraries(plotTrajectory ${Pangolin_LIBRARIES})
add_executable(visualizeGeometry visualizeGeometry.cpp)
target_link_libraries(visualizeGeometry ${Pangolin_LIBRARIES})
