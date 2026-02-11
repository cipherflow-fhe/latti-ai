add_executable(test_cifar10 inference.cpp)
target_link_libraries(test_cifar10 PUBLIC ${EXAMPLE_LINK_LIBRARIES})
copy_task_data(test_cifar10)
