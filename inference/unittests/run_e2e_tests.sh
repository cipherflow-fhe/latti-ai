#!/bin/bash
# E2E test runner commands
# Usage: Run individual commands below, or ./run_e2e_tests.sh to run all GPU tests
#
# From build/inference/unittests/ directory

# --- Run all ---
# ./test_e2e "[cpu]"              # All CPU tests
# ./test_e2e "[gpu]"              # All GPU tests
# ./test_e2e "[e2e]"              # All (CPU + GPU)

# --- Single layer (poly_n=8192) ---
./test_e2e "e2e_gpu" -c "single_conv"
./test_e2e "e2e_gpu" -c "single_act"
./test_e2e "e2e_gpu" -c "single_avgpool"
./test_e2e "e2e_gpu" -c "single_dense"
./test_e2e "e2e_gpu" -c "single_add"

# --- Layer interaction (poly_n=8192) ---
./test_e2e "e2e_gpu" -c "conv_batchnorm"
./test_e2e "e2e_gpu" -c "conv_avgpool_reshape_dense"

# --- No-BTP (poly_n=8192, <=5 levels) ---
./test_e2e "e2e_gpu" -c "poly_n_8192"

# --- No-BTP (poly_n=16384, 6-9 levels) ---
./test_e2e "e2e_gpu" -c "conv_act"
./test_e2e "e2e_gpu" -c "resnet_basic_block"

# --- No-BTP (poly_n=32768, 10-17 levels) ---
./test_e2e "e2e_gpu" -c "poly_n_32768"

# --- No-BTP (poly_n=65536, 18-33 levels) ---
./test_e2e "e2e_gpu" -c "poly_n_65536_no_btp"

# --- BTP (poly_n=65536, >33 levels) ---
./test_e2e "e2e_gpu" -c "btp"
./test_e2e "e2e_gpu" -c "conv_series"
./test_e2e "e2e_gpu" -c "act_series"
./test_e2e "e2e_gpu" -c "intertwined"

# --- Big-size (256x256 input) ---
./test_e2e "e2e_gpu" -c "single_avgpool_big_size"
./test_e2e "e2e_gpu" -c "single_conv_with_stride_big_size"

# --- Operator migration from test_fhe_layers_hetero ---
./test_e2e "e2e_gpu" -c "conv_mch_s1"
./test_e2e "e2e_gpu" -c "conv_mch_s2"
./test_e2e "e2e_gpu" -c "depthwise_conv_s1"
./test_e2e "e2e_gpu" -c "depthwise_conv_s2"
./test_e2e "e2e_gpu" -c "mux_conv_large_channel"
