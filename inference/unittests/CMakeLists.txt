include(CTest)
enable_testing()

add_executable(test_data_structs
  test_data_structs.cpp
  ut_util.cpp
)
target_link_libraries(test_data_structs inference_task)
add_test(NAME test_data_structs COMMAND test_data_structs)

# Copy test_gen_layers.py to build directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_gen_layers.py
    ${CMAKE_CURRENT_BINARY_DIR}/test_gen_layers.py
    COPYONLY
)

# 总是编译test_fhe_layers_hetero
add_executable(test_fhe_layers_hetero test_fhe_layers_hetero.cpp ut_util.cpp)
target_link_libraries(test_fhe_layers_hetero PUBLIC fhe_layers lattisense)

# 只在GPU启用时定义INFERENCE_SDK_ENABLE_GPU宏
if(INFERENCE_SDK_ENABLE_GPU)
    target_compile_definitions(test_fhe_layers_hetero PRIVATE INFERENCE_SDK_ENABLE_GPU)
endif()

add_test(NAME test_fhe_layers_hetero COMMAND test_fhe_layers_hetero)

