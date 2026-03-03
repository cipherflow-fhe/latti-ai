# Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)

option(INFERENCE_SDK_ENABLE_GPU "Enable GPU acceleration support" OFF)
message(STATUS "INFERENCE_SDK_ENABLE_GPU: ${INFERENCE_SDK_ENABLE_GPU}")

# Configure lattisense options before add_subdirectory
set(LATTISENSE_ENABLE_GPU ${INFERENCE_SDK_ENABLE_GPU} CACHE BOOL "Enable GPU in lattisense" FORCE)
set(LATTISENSE_BUILD_MODE "SHARED" CACHE STRING "Lattisense build mode" FORCE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_COMPILER "/usr/bin/gcc-12")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")

project(latti-ai)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -maes")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_compile_options(-fPIC)
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=1)

message(STATUS "The CXX compiler : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "The C   compiler : ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE}")

find_package(Threads REQUIRED)
find_package(HDF5 COMPONENTS CXX HL)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  message("OPENMP FOUND")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(${HDF5_INCLUDE_DIRS})

enable_testing()

add_subdirectory(inference)
add_subdirectory(examples)

add_definitions(-w)
