cmake_minimum_required(VERSION 3.15)
project(latti_client_node VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Paths ──────────────────────────────────────────────────────────────────
set(LATTISENSE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../lattisense" CACHE PATH "Path to lattisense submodule")
set(LATTI_INFERENCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(LATTI_INTERFACE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

# ── Find pre-built fhe_ops_lib ────────────────────────────────────────────
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
        "fhe_ops_lib not found. Build lattisense first.\n"
        "  cmake -DFHE_OPS_LIB=/path/to/libfhe_ops_lib.so .."
    )
endif()
message(STATUS "Found fhe_ops_lib: ${FHE_OPS_LIB}")

# ── inference_client_lib (static) ─────────────────────────────────────────
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
    ${LATTI_INFERENCE_DIR}
    ${LATTISENSE_DIR}
    ${LATTISENSE_DIR}/lib
    ${LATTISENSE_DIR}/cxx_sdk_v2
    ${LATTISENSE_DIR}/fhe_ops_lib
    ${LATTISENSE_DIR}/mega_ag_runners
)

target_link_libraries(inference_client_lib PUBLIC ${FHE_OPS_LIB})
set_target_properties(inference_client_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ── Node.js N-API addon ──────────────────────────────────────────────────
# cmake-js provides these variables:
#   CMAKE_JS_INC  - Node.js + node-addon-api headers
#   CMAKE_JS_LIB  - Node.js import library (Windows)
#   CMAKE_JS_SRC  - (unused)
if(NOT CMAKE_JS_INC)
    message(FATAL_ERROR "CMAKE_JS_INC not set. Build with cmake-js, not cmake directly.")
endif()

# node-addon-api headers
execute_process(
    COMMAND node -p "require('node-addon-api').include_dir"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE NAPI_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "node-addon-api include: ${NAPI_INCLUDE_DIR}")

add_library(latti_client_node SHARED binding.cpp)
set_target_properties(latti_client_node PROPERTIES
    PREFIX ""
    SUFFIX ".node"
    OUTPUT_NAME "latti_client"
)

target_include_directories(latti_client_node PRIVATE ${CMAKE_JS_INC} ${NAPI_INCLUDE_DIR})
target_link_libraries(latti_client_node PRIVATE inference_client_lib)

if(CMAKE_JS_LIB)
    target_link_libraries(latti_client_node PRIVATE ${CMAKE_JS_LIB})
endif()

# node-addon-api: define NAPI_DISABLE_CPP_EXCEPTIONS for performance
target_compile_definitions(latti_client_node PRIVATE NAPI_DISABLE_CPP_EXCEPTIONS)
