include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_definitions(-DOPENSSL_API_COMPAT=10100)
add_definitions(-DSOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}")

set(GLOBAL_NUM 1)

add_library(log STATIC ${CMAKE_CURRENT_SOURCE_DIR}/lib/log/log.c)

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/fhe-mpc/mpc/fhe-libs)
find_package(SEAL 4.0.0 EXACT REQUIRED CONFIG
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/fhe-mpc/mpc/fhe-libs/lib/cmake/SEAL-4.0
    NO_DEFAULT_PATH
)

add_subdirectory(lattisense)
add_subdirectory(fhe_layers)
add_subdirectory(fhe-mpc)
add_subdirectory(inference_task)
add_subdirectory(interface)
add_subdirectory(unittests)
