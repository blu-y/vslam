cmake_minimum_required(VERSION 4.0)
project( HelloSLAM )

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 20)

add_library(hello libHelloSLAM.cpp) # for static library >> .a file
add_library(hello_shared SHARED libHelloSLAM.cpp) # for shared library >> .so(.dylib on mac) file

add_executable(helloSLAM helloSLAM.cpp)

add_executable(useHello useHello.cpp)
target_link_libraries(useHello hello_shared)
