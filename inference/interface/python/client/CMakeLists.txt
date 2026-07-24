cmake_minimum_required(VERSION 3.15)
project(latti_client VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Paths ──────────────────────────────────────────────────────────────────
set(LATTISENSE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../lattisense" CACHE PATH "Path to lattisense submodule")
set(LATTI_INFERENCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(LATTI_INTERFACE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# ── Find pre-built fhe_ops_lib ────────────────────────────────────────────
# lattisense with LATTISENSE_CLIENT_ONLY=ON produces fhe_ops_lib (not lattisense).
# Standalone: cmake -DFHE_OPS_LIB=/path/to/libfhe_ops_lib.so ..
# Integrated: cmake picks it up from LATTISENSE_DIR/build automatically.
if(NOT FHE_OPS_LIB)
    find_library(FHE_OPS_LIB
        NAMES fhe_ops_lib
        PATHS
            "${LATTISENSE_DIR}/build/fhe_ops_lib"
            "${LATTISENSE_DIR}/build"
            "${LATTISENSE_DIR}/lib"
        NO_DEFAULT_PATH
    )
endif()

if(NOT FHE_OPS_LIB)
    message(FATAL_ERROR
        "fhe_ops_lib not found. Build lattisense with LATTISENSE_CLIENT_ONLY=ON first.\n"
        "  Standalone: cmake -DFHE_OPS_LIB=/path/to/libfhe_ops_lib.so .."
    )
endif()
message(STATUS "Found fhe_ops_lib: ${FHE_OPS_LIB}")

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

target_link_libraries(inference_client_lib PUBLIC ${FHE_OPS_LIB})

set_target_properties(inference_client_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Windows: copy fhe_ops_lib.dll to output directory
if(WIN32)
    get_filename_component(FHE_OPS_LIB_DIR "${FHE_OPS_LIB}" DIRECTORY)
    add_custom_command(TARGET inference_client_lib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FHE_OPS_LIB_DIR}/libfhe_ops_lib.dll"
            "$<TARGET_FILE_DIR:inference_client_lib>"
        COMMENT "Copying libfhe_ops_lib.dll to output directory"
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
                "${FHE_OPS_LIB_DIR}/libfhe_ops_lib.dll"
                "$<TARGET_FILE_DIR:latti_client>"
            COMMENT "Copying libfhe_ops_lib.dll next to latti_client.pyd"
        )
    endif()
else()
    message(STATUS "pybind11 not found, skipping latti_client Python module")
endif()
