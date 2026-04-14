# CKKS primitive ops GPU benchmark
add_executable(benchmark_ckks_ops_gpu benchmark_ckks_ops_gpu.cpp)
target_link_libraries(benchmark_ckks_ops_gpu PRIVATE lattisense)
