cmake_minimum_required(VERSION 3.10)
project(IzhikevichGPU)

set(CMAKE_CXX_STANDARD 11)

add_executable(IzhikevichGPU Neuron.h Neuron.cpp main.cpp)
