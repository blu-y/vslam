cmake_minimum_required(VERSION 4.0)
project(vslam)

set(CMAKE_CXX_STANDARD 20)

add_executable(vslam main.cpp
        ch1/libHelloSLAM.cpp
        ch1/libHelloSLAM.h
        ch1/useHello.cpp)
