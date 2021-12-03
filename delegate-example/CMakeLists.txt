cmake_minimum_required(VERSION 3.21)

project(delegate-example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Eigen3 REQUIRED)

include_directories(src tensorflow ${EIGEN_INCLUDE_DIR})

add_subdirectory(tensorflow/tensorflow/lite)

add_library(eigen-delegate SHARED
    tensorflow/tensorflow/lite/delegates/utils/simple_delegate.cc
    src/eigen-delegate.cpp
)
target_link_libraries(eigen-delegate PRIVATE
    Eigen3::Eigen
    tensorflow-lite
)
