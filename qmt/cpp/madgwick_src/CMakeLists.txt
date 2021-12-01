# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 2.8)

project (madgwick)
#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -ansi -std=c++98")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(madgwick SHARED
    MadgwickAHRS.cpp
    MahonyAHRS.cpp
    MadgwickB.cpp
)

set_target_properties(madgwick PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
