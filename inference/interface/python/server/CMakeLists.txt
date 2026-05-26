cmake_minimum_required(VERSION 3.15)
project(latti_server VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Paths ──────────────────────────────────────────────────────────────────
# Standalone build:  cmake -DLATTI_AI_BUILD=/path/to/latti-ai/build ..
set(LATTI_AI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." CACHE PATH "Path to latti-ai root")
set(LATTI_INFERENCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(LATTI_INTERFACE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(LATTI_AI_BUILD "${CMAKE_CURRENT_SOURCE_DIR}/../../../../build" CACHE PATH "Path to latti-ai build directory")

# ── Find dependencies ─────────────────────────────────────────────────────
find_package(pybind11 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(OpenMP)

link_directories(
    ${LATTI_AI_BUILD}/inference/interface
    ${LATTI_AI_BUILD}/inference/inference_task
    ${LATTI_AI_BUILD}/inference/fhe_layers
    ${LATTI_AI_BUILD}/inference/lattisense
    ${LATTI_AI_BUILD}/inference
)

include_directories(
    ${LATTI_INFERENCE_DIR}
    ${LATTI_INFERENCE_DIR}/lib
    ${LATTI_INFERENCE_DIR}/lattisense
    ${LATTI_INFERENCE_DIR}/lattisense/fhe_ops_lib
    ${LATTI_INFERENCE_DIR}/lattisense/lib
    ${HDF5_INCLUDE_DIRS}
)

# ── pybind11 module ───────────────────────────────────────────────────────
pybind11_add_module(latti_server bindings.cpp)

target_link_libraries(latti_server PRIVATE
    inference_interface
    inference_task
    fhe_layers
    log
    lattisense
    ${HDF5_CXX_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(latti_server PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_properties(latti_server PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../../../../build/inference/lattisense:$ORIGIN/../../../../../build/inference/lattisense"
    BUILD_WITH_INSTALL_RPATH ON
)
