cmake_minimum_required(VERSION 3.10)
project(IzhikevichGPU)

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 11)

add_executable(IzhikevichGPU cuda_sim.cu)
