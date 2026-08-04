add_library(inference_interface
    encrypted_parameter_store.cpp
    inference_client.cpp
    inference_provisioner.cpp
    inference_runner.cpp
    inference_server.cpp
    runner_bundle_io.cpp
)
target_link_libraries(inference_interface PUBLIC inference_task fhe_layers lattisense log ${HDF5_CXX_LIBRARIES})
