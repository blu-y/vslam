cmake_minimum_required(VERSION 4.0)
project(loopClosure)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV CONFIG REQUIRED)
find_package(DBoW3 REQUIRED)

add_executable(loopClosure loopClosure.cpp)
target_link_libraries(loopClosure PRIVATE
        DBoW3
        ${OpenCV_LIBS})
add_executable(featureTraining featureTraining.cpp)
target_link_libraries(featureTraining PRIVATE
        DBoW3
        ${OpenCV_LIBS})
add_executable(genVocabLarge genVocabLarge.cpp)
target_link_libraries(genVocabLarge PRIVATE
        DBoW3
        ${OpenCV_LIBS})
