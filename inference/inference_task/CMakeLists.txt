find_package(HDF5 REQUIRED)

add_library(inference_task inference_process.cpp)

target_link_libraries(inference_task PUBLIC lattisense fhe_layers log ${HDF5_CXX_LIBRARIES})

if(INFERENCE_SDK_ENABLE_GPU)
    target_compile_definitions(inference_task PRIVATE INFERENCE_SDK_ENABLE_GPU)
endif()
