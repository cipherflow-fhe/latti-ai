include_directories(${CMAKE_SOURCE_DIR}/inference)
include_directories(${CMAKE_SOURCE_DIR}/inference/lib)

add_definitions(-DSOURCE_PATH="${CMAKE_SOURCE_DIR}/inference")

# Common link libraries (used by subdirectories)
set(EXAMPLE_LINK_LIBRARIES
    -Wl,--start-group
    inference_task
    fhe_layers
    log
    -Wl,--end-group
    lattisense
    ${HDF5_CXX_LIBRARIES}
)

# Copy the task/ directory from source to the build output directory for a target.
# Must be called from the subdirectory CMakeLists.txt where CMAKE_CURRENT_SOURCE_DIR
# already points to the example directory (e.g. test_mnist/).
function(copy_task_data TARGET_NAME)
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/task
            ${CMAKE_CURRENT_BINARY_DIR}/task
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/gen_mega_ag.py
            ${CMAKE_CURRENT_BINARY_DIR}/gen_mega_ag.py
        COMMENT "Copying task data and gen_mega_ag.py for ${TARGET_NAME}"
    )
endfunction()

add_subdirectory(test_mnist)
add_subdirectory(test_cifar10)
add_subdirectory(test_imagenet)
