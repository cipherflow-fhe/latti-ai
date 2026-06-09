find_package(HDF5 REQUIRED)

add_library(inference_task
    inference_process.cpp
    inference_process_client.cpp
    mpc_task_meta_data.cpp
)

add_executable(client_task client_task.cpp)

target_link_libraries(inference_task PUBLIC lattisense fhe_layers mpc_layers log ${HDF5_CXX_LIBRARIES})
target_link_libraries(client_task PRIVATE
    -Wl,--start-group
    inference_task
    mpc_layers
    fhe_layers
    log
    SCI-Cheetah
    Cheetah-Linear
    gemini
    SCI-Cheetah-BuildingBlocks
    SCI-Math
    SCI-LinearOT
    SCI-BuildingBlocks
    -Wl,--end-group
    lattisense
    ${HDF5_CXX_LIBRARIES}
)

if(INFERENCE_SDK_ENABLE_GPU)
    target_compile_definitions(inference_task PRIVATE INFERENCE_SDK_ENABLE_GPU)
endif()
