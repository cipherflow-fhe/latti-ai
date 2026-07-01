set(MPC_WRAPPER_FHE_UT_LIBS
    -Wl,--start-group
    inference_task
    mpc_wrapper
    mpc
    fhe_layers
    log
    -Wl,--end-group
    lattisense
)

add_executable(test_mpc_layers
    test_mpc_layers.cpp
    ${CMAKE_SOURCE_DIR}/inference/unittests/ut_util.cpp
)
target_include_directories(test_mpc_layers PRIVATE
    ${CMAKE_SOURCE_DIR}/inference/unittests
)
target_link_libraries(test_mpc_layers PUBLIC ${MPC_WRAPPER_FHE_UT_LIBS})

add_executable(test_mpc_layers_client test_mpc_layers_client.cpp)
target_include_directories(test_mpc_layers_client PRIVATE
    ${CMAKE_SOURCE_DIR}/inference/unittests
)
target_link_libraries(test_mpc_layers_client PUBLIC ${MPC_WRAPPER_FHE_UT_LIBS})

add_executable(client_task mpc_client_task.cpp)
target_link_libraries(client_task PUBLIC ${MPC_WRAPPER_FHE_UT_LIBS})

add_dependencies(test_mpc_layers client_task)
