CMAKE_MINIMUM_REQUIRED(VERSION 3.18)

# this has to be set before project()
IF (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
ENDIF ()

PROJECT(traffic LANGUAGES CXX CUDA)
FIND_PACKAGE(CUDAToolkit)
include_directories( /usr/local/cuda-13.0/targets/x86_64-linux/include )

ADD_SUBDIRECTORY(src)