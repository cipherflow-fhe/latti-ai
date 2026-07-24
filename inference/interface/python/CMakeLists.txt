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
project(latti_inference_bindings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)

set(LATTI_AI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(LATTI_AI_BUILD "${LATTI_AI_ROOT}/build")

include_directories(${LATTI_AI_ROOT}/inference)
include_directories(${LATTI_AI_ROOT}/inference/lib)
include_directories(${LATTI_AI_ROOT}/inference/lattisense)
include_directories(${LATTI_AI_ROOT}/inference/lattisense/fhe_ops_lib)
include_directories(${LATTI_AI_ROOT}/inference/lattisense/lib)
include_directories(${HDF5_INCLUDE_DIRS})

link_directories(${LATTI_AI_BUILD}/inference/interface)
link_directories(${LATTI_AI_BUILD}/inference/inference_task)
link_directories(${LATTI_AI_BUILD}/inference/fhe_layers)
link_directories(${LATTI_AI_BUILD}/inference/lattisense)
link_directories(${LATTI_AI_BUILD}/inference)

pybind11_add_module(latti_inference bindings.cpp)
target_compile_options(latti_inference PRIVATE -maes)
target_link_libraries(latti_inference PRIVATE
    inference_interface
    inference_task
    fhe_layers
    log
    lattisense
    ${HDF5_CXX_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Set RPATH so the .so can find shared libs relative to itself
set_target_properties(latti_inference PROPERTIES
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../../../build/inference/lattisense"
    BUILD_WITH_INSTALL_RPATH ON
)

install(TARGETS latti_inference DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(client)
