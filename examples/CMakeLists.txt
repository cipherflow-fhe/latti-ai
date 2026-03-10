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

# --- Example verification tests ---
set(EXAMPLES_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# CPU tests
add_test(NAME example-mnist-cpu
    COMMAND inference --task-dir ${EXAMPLES_DIR}/test_mnist/task --input ${EXAMPLES_DIR}/test_mnist/task/client/img.csv --verify)
add_test(NAME example-cifar10-cpu
    COMMAND inference --task-dir ${EXAMPLES_DIR}/test_cifar10/task --input ${EXAMPLES_DIR}/test_cifar10/task/client/img.csv --verify)
add_test(NAME example-imagenet-cpu
    COMMAND inference --task-dir ${EXAMPLES_DIR}/test_imagenet/task --input ${EXAMPLES_DIR}/test_imagenet/task/client/img.csv --verify)

set_tests_properties(example-mnist-cpu PROPERTIES LABELS "example;example-cpu;mnist")
set_tests_properties(example-cifar10-cpu PROPERTIES LABELS "example;example-cpu;cifar10")
set_tests_properties(example-imagenet-cpu PROPERTIES LABELS "example;example-cpu;imagenet")

# GPU tests
add_test(NAME example-mnist-gpu
    COMMAND inference --task-dir ${EXAMPLES_DIR}/test_mnist/task --input ${EXAMPLES_DIR}/test_mnist/task/client/img.csv --verify --gpu)
add_test(NAME example-cifar10-gpu
    COMMAND inference --task-dir ${EXAMPLES_DIR}/test_cifar10/task --input ${EXAMPLES_DIR}/test_cifar10/task/client/img.csv --verify --gpu)
add_test(NAME example-imagenet-gpu
    COMMAND inference --task-dir ${EXAMPLES_DIR}/test_imagenet/task --input ${EXAMPLES_DIR}/test_imagenet/task/client/img.csv --verify --gpu)

set_tests_properties(example-mnist-gpu PROPERTIES LABELS "example;example-gpu;mnist")
set_tests_properties(example-cifar10-gpu PROPERTIES LABELS "example;example-gpu;cifar10")
set_tests_properties(example-imagenet-gpu PROPERTIES LABELS "example;example-gpu;imagenet")

# Timeouts (seconds)
set_tests_properties(example-mnist-cpu example-mnist-gpu PROPERTIES TIMEOUT 300)
set_tests_properties(example-cifar10-cpu example-cifar10-gpu PROPERTIES TIMEOUT 1800)
set_tests_properties(example-imagenet-cpu example-imagenet-gpu PROPERTIES TIMEOUT 3600)
