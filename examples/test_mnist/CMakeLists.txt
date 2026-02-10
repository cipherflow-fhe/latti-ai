add_executable(test_mnist main.cpp)
target_link_libraries(test_mnist PUBLIC ${EXAMPLE_LINK_LIBRARIES})
copy_task_data(test_mnist)
