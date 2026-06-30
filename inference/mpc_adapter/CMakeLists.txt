find_package(HDF5 REQUIRED)

add_library(mpc_adapter
    enc_share_conversion.cpp
    inference_process_client.cpp
    mpc_context_utils.cpp
    mpc_data_transmission.cpp
    inference_process_mpc_server.cpp
    mpc_adapter.cpp
)

target_link_libraries(mpc_adapter PUBLIC lattisense fhe_layers mpc log ${HDF5_CXX_LIBRARIES})
