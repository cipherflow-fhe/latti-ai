add_library(inference_interface inference_client.cpp inference_server.cpp)
target_link_libraries(inference_interface PUBLIC inference_task fhe_layers lattisense log ${HDF5_CXX_LIBRARIES})
