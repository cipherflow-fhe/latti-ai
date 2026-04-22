cmake_minimum_required(VERSION 3.15)
project(latti_client VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Paths ──────────────────────────────────────────────────────────────────
# Standalone build:  cmake -DLATTISENSE_DIR=/path/to/lattisense ..
# Integrated build:  built via inference/interface/CMakeLists.txt
set(LATTISENSE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../lattisense" CACHE PATH "Path to lattisense submodule")
set(LATTI_INFERENCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(LATTI_INTERFACE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# ── Find pre-built lattisense ─────────────────────────────────────────────
# Standalone build: pass -DLATTISENSE_LIB=/path/to/liblattisense.so
# Integrated build:  lattisense target already exists, use find_library
if(NOT LATTISENSE_LIB)
    find_library(LATTISENSE_LIB
        NAMES lattisense
        PATHS
            "${LATTISENSE_DIR}/build"
            "${LATTISENSE_DIR}/lib"
            "${CMAKE_BINARY_DIR}/inference/lattisense"
        NO_DEFAULT_PATH
    )
endif()

if(NOT LATTISENSE_LIB)
    message(FATAL_ERROR
        "lattisense library not found. Build lattisense first.\n"
        "  Standalone: cmake -DLATTISENSE_LIB=/path/to/liblattisense.so .."
    )
endif()
message(STATUS "Found lattisense: ${LATTISENSE_LIB}")

# ── inference_client_lib ──────────────────────────────────────────────────
add_library(inference_client_lib STATIC
    ${LATTI_INTERFACE_DIR}/inference_client.cpp
    ${LATTI_INFERENCE_DIR}/data_structs/feature.cpp
    ${LATTI_INFERENCE_DIR}/data_structs/feature0d.cpp
    ${LATTI_INFERENCE_DIR}/data_structs/feature1d.cpp
    ${LATTI_INFERENCE_DIR}/data_structs/feature2d.cpp
    ${LATTI_INFERENCE_DIR}/data_structs/feature_mat.cpp
    ${LATTI_INFERENCE_DIR}/util.cpp
)

target_include_directories(inference_client_lib PUBLIC
    ${LATTI_INFERENCE_DIR}               # data_structs/*.h, util.h
    ${LATTISENSE_DIR}                     # cxx_sdk_v2/, fhe_ops_lib/, mega_ag_runners/
    ${LATTISENSE_DIR}/lib                 # nlohmann/json.hpp, util/
    ${LATTISENSE_DIR}/cxx_sdk_v2          # <cxx_sdk_v2/cxx_fhe_task.h>
    ${LATTISENSE_DIR}/fhe_ops_lib         # <fhe_ops_lib/fhe_lib_v2.h>
    ${LATTISENSE_DIR}/mega_ag_runners     # headers
)

target_link_libraries(inference_client_lib PUBLIC ${LATTISENSE_LIB})

set_target_properties(inference_client_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Windows: copy lattisense.dll to output directory
if(WIN32)
    get_filename_component(LATTISENSE_LIB_DIR "${LATTISENSE_LIB}" DIRECTORY)
    add_custom_command(TARGET inference_client_lib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LATTISENSE_LIB_DIR}/lattisense.dll"
            "$<TARGET_FILE_DIR:inference_client_lib>"
        COMMENT "Copying lattisense.dll to output directory"
    )
endif()

# ── pybind11 module ───────────────────────────────────────────────────────
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    message(STATUS "pybind11 found, building latti_client Python module")
    pybind11_add_module(latti_client bindings.cpp)
    target_link_libraries(latti_client PRIVATE inference_client_lib)

    if(WIN32)
        add_custom_command(TARGET latti_client POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${LATTISENSE_LIB_DIR}/lattisense.dll"
                "$<TARGET_FILE_DIR:latti_client>"
            COMMENT "Copying lattisense.dll next to latti_client.pyd"
        )
    endif()
else()
    message(STATUS "pybind11 not found, skipping latti_client Python module")
endif()
