cmake_minimum_required(VERSION 4.0)
project(imageVision)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Sophus CONFIG REQUIRED)
find_package(Pangolin REQUIRED)
include_directories( ${Pangolin_INCLUDE_DIRS} )

add_executable(imageBasics imageBasics.cpp)
target_link_libraries(imageBasics PRIVATE ${OpenCV_LIBS})
add_executable(undistortImage undistortImage.cpp)
target_link_libraries(undistortImage PRIVATE ${OpenCV_LIBS})
add_executable(stereoVision stereoVision.cpp)
target_link_libraries(stereoVision PRIVATE
        Eigen3::Eigen
        ${Pangolin_LIBRARIES}
        ${OpenCV_LIBS})
add_executable(rgbdJointMap rgbdJointMap.cpp)
target_link_libraries(rgbdJointMap PRIVATE
        Sophus::Sophus
        Eigen3::Eigen
        ${Pangolin_LIBRARIES}
        ${OpenCV_LIBS})
