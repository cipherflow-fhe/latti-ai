include_directories(${CMAKE_SOURCE_DIR}/inference)
include_directories(${CMAKE_SOURCE_DIR}/inference/lib)

add_definitions(-DSOURCE_PATH="${CMAKE_SOURCE_DIR}/inference")

add_executable(provision_encrypted_runner_bundle provision_encrypted_runner_bundle.cpp)
target_link_libraries(provision_encrypted_runner_bundle PUBLIC
    -Wl,--start-group
    inference_interface
    inference_task
    fhe_layers
    log
    -Wl,--end-group
    lattisense
    ${HDF5_CXX_LIBRARIES}
)

add_executable(decrypt_runner_output decrypt_runner_output.cpp)
target_link_libraries(decrypt_runner_output PUBLIC
    -Wl,--start-group
    inference_interface
    inference_task
    fhe_layers
    log
    -Wl,--end-group
    lattisense
    ${HDF5_CXX_LIBRARIES}
)

