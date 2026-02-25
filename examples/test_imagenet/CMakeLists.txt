add_executable(test_imagenet inference.cpp)
target_link_libraries(test_imagenet PUBLIC ${EXAMPLE_LINK_LIBRARIES})
copy_task_data(test_imagenet)
