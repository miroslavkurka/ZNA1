cmake_minimum_required(VERSION 3.27)
project(zna CUDA)

set(CMAKE_CUDA_STANDARD 23)

add_executable(zna main.cu
        object_intersection.cu
        object_intersection.cu
        rice_siff.cu
        object_instersection.cuh
        object_instersection.cuh
        object_intersection.cu
        object_intersection.cu)

set_target_properties(zna PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
