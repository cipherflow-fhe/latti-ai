include_directories(${CMAKE_SOURCE_DIR}/inference)
include_directories(${CMAKE_SOURCE_DIR}/inference/lib)

function(add_mpc_inference_example target source)
    add_executable(${target} ${source})
    target_compile_definitions(${target} PRIVATE LATTIAI_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(${target} PUBLIC
        -Wl,--start-group
        inference_interface
        inference_task
        mpc_wrapper
        mpc
        fhe_layers
        log
        -Wl,--end-group
        lattisense
        ${HDF5_CXX_LIBRARIES}
    )
endfunction()

add_mpc_inference_example(mpc_inference_sdk_server mpc_inference_sdk_server.cpp)
add_mpc_inference_example(mpc_inference_sdk_client mpc_inference_sdk_client.cpp)
