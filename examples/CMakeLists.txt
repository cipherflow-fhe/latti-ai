include_directories(${CMAKE_SOURCE_DIR}/inference)
include_directories(${CMAKE_SOURCE_DIR}/inference/lib)

add_definitions(-DSOURCE_PATH="${CMAKE_SOURCE_DIR}/inference")

add_executable(inference inference.cpp)
target_link_libraries(inference PUBLIC
    -Wl,--start-group
    inference_interface
    inference_task
    fhe_layers
    log
    -Wl,--end-group
    lattisense
    ${HDF5_CXX_LIBRARIES}
)
