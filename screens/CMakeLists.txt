cmake_minimum_required(VERSION 2.8)

project(screens)

file(GLOB SOURCES "*.cpp")

add_executable(main ${SOURCES})