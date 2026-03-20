/*
 * Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <stdlib.h>
#include <filesystem>
#include <hdf5.h>

#include "fhe_layers/reshape_layer.h"
#include "util.h"
#include "fhe_layers/poly_relu2d.h"
#include "fhe_layers/poly_relu_base.h"
#include "fhe_layers/activation_layer.h"
#include "fhe_layers/add_layer.h"
#include "fhe_layers/avgpool2d_layer.h"
#include "fhe_layers/conv2d_depthwise.h"
#include "fhe_layers/conv2d_packed_layer.h"
#include "fhe_layers/dense_packed_layer.h"
#include "fhe_layers/mult_scaler.h"
#include "fhe_layers/inverse_multiplexed_conv2d_layer.h"
#include "fhe_layers/concat_layer.h"
#include "fhe_layers/upsample_layer.h"
#include "fhe_layers/upsample_nearest_layer.h"
#include "fhe_layers/multiplexed_conv2d_pack_layer.h"
#include "fhe_layers/multiplexed_conv2d_pack_layer_depthwise.h"

enum class ComputeDevice { CPU, GPU, FPGA };

class Node {
public:
    Node();
};

class FeatureNode : public Node {
public:
    FeatureNode(string node_id_in,
                int dim_in,
                int channel_in,
                double scale_in,
                uint32_t shape_in[],
                uint32_t skip_in[],
                string ckks_parameter_id_in,
                int pack_channel_per_ciphertext_in);

    string node_id;
    int dim;
    uint32_t channel;
    double scale;
    Duo shape = {0, 0};
    Duo skip = {1, 1};
    Duo virtual_shape = {0, 0};
    Duo virtual_skip = {0, 0};
    string ckks_parameter_id;
    int pack_channel_per_ciphertext;
    int level;
    double ckks_scale;

    FeatureNode(const json& json_data)
        : dim(json_data["dim"]), channel(json_data["channel"]), scale(json_data["scale"]),
          ckks_parameter_id(json_data["ckks_parameter_id"]), pack_channel_per_ciphertext(json_data["pack_num"]),
          level(json_data["level"]), ckks_scale(0.0) {
        if (dim == 2) {
            shape[0] = json_data["shape"][0];
            shape[1] = json_data["shape"][1];
            skip[0] = json_data["skip"][0];
            skip[1] = json_data["skip"][1];
        }
        if (dim == 0) {
            skip[0] = json_data["skip"];
            virtual_shape[0] = json_data["virtual_shape"][0];
            virtual_shape[1] = json_data["virtual_shape"][1];
            virtual_skip[0] = json_data["virtual_skip"][0];
            virtual_skip[1] = json_data["virtual_skip"][1];
        }
    }
};

template <int dim>
Array<double, dim> h5_to_array(const hid_t& h5_file,
                               const std::string& dataset_id,
                               const std::array<uint64_t, dim>& shape,
                               double factor = 1.0) {
    Array<double, dim> result(shape);
    hid_t h5_dataset = H5Dopen(h5_file, dataset_id.c_str(), H5P_DEFAULT);
    H5Dread(h5_dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, result.get_data());
    for (int i = 0; i < result.get_size(); i++) {
        result.set(i, result.get(i) * factor);
    }
    H5Dclose(h5_dataset);
    return result;
}

class InitInferenceProcess {
public:
    InitInferenceProcess() {}
    InitInferenceProcess(const string& project_path_in, bool is_fpga = true);
    virtual ~InitInferenceProcess();

    filesystem::path project_path;
    string task_type;
    string pack_style;
    int n_task;
    string output_id;
    string input_id;
    json task_input_param;
    json task_output_param;
    int start_task_id;
    int end_task_id;
    bool enable_fpga;
    json server_task;
    string current_json_path;
    json json_data;
    json json_features;
    json json_layers;
    Duo block_shape;
    bool is_absorb_polyrelu;
    std::map<std::string, std::unique_ptr<CkksParameter>> ckks_parameters;
    bool fpga_loaded = false;

    bool is_lazy = false;

    // Time statistics
    double total_fhe_time = 0.0;
    double total_fpga_time = 0.0;

    virtual void init_parameters(bool is_bootstrapping = false);
    virtual void load_model_prepare();

    string get_abs_filename(const string& json_filename);
    virtual void init_conv_layer(const string& key, const json& layer, const hid_t& h5_file);
    virtual void init_square_layer(const string& key, const json& layer, const hid_t& h5_file);
    virtual void init_dense_layer(const string& key, const json& layer, const hid_t& h5_file);
    virtual void init_add_layer(const string& key, const json& layer, const string& block_input_feature);
    virtual void init_reshape_layer(const string& key, const json& layer);
    virtual void init_mult_scalar_layer(const string& key,
                                        const json& layer,
                                        const hid_t& h5_file,
                                        const Duo& block_shape = {128, 256});
    virtual void init_drop_level_layer(const string& key, const json& layer);
    virtual void init_fhe_avgpool_layer(const string& key,
                                        const json& layer,
                                        const bool& is_adaptive = true,
                                        const Duo& block_shape = {128, 256});
    void init_multiplexed_conv_layer(const string& key,
                                     const json& layer,
                                     const hid_t& h5_file,
                                     const Duo& block_shape_in = {128, 256});

    void init_poly_relu2d_layer(const string& key,
                                const json& layer,
                                const hid_t& h5_file,
                                bool is_absorb = true,
                                const Duo& block_shape_in = {128, 256});
    void init_concat_layer(const string& key, const json& layer);
    void init_upsample_layer(const string& key, const json& layer, const Duo& block_shape = {128, 256});
    void init_upsample_nearest_layer(const string& key, const json& layer);

    std::map<std::string, std::unique_ptr<Conv2DPackedLayer>> ckks_conv2ds;
    std::map<std::string, std::unique_ptr<Conv2DPackedDepthwiseLayer>> ckks_dw_conv2ds;
    std::map<std::string, std::unique_ptr<SquareLayer>> ckks_squares;
    std::map<std::string, std::unique_ptr<DensePackedLayer>> ckks_denses;
    std::map<std::string, std::unique_ptr<AddLayer>> ckks_adds;
    std::map<std::string, std::unique_ptr<MultScalarLayer>> ckks_mult_scalar;
    std::map<std::string, std::unique_ptr<DropLevelLayer>> ckks_drop_level;
    std::map<std::string, std::unique_ptr<ReshapeLayer>> ckks_reshape;
    std::map<std::string, std::unique_ptr<Avgpool2DLayer>> ckks_avgpool;
    std::map<std::string, std::unique_ptr<PolyRelu>> ckks_poly_relu;
    std::map<std::string, std::unique_ptr<PolyRelu0D>> ckks_poly_relu_0d;
    std::map<std::string, std::unique_ptr<ParMultiplexedConv2DPackedLayer>> ckks_multiplexed_conv2ds;
    std::map<std::string, std::unique_ptr<InverseMultiplexedConv2DLayer>> ckks_big_conv2ds;
    std::map<std::string, std::unique_ptr<ParMultiplexedConv2DPackedLayerDepthwise>> ckks_multiplexed_dw_conv2ds;
    std::map<std::string, std::unique_ptr<ConcatLayer>> ckks_concat;
    std::map<std::string, std::unique_ptr<UpsampleLayer>> ckks_upsample;
    std::map<std::string, std::unique_ptr<UpsampleNearestLayer>> ckks_upsample_nearest;
};

class InferenceProcess {
public:
    InferenceProcess() {}
    InferenceProcess(InitInferenceProcess* fp_in, bool is_fpga_in);
    virtual ~InferenceProcess();
    InitInferenceProcess* fp;
    int task_num;
    bool is_fpga;
    bool debug_all = true;
    ;  // Debug switch: when enabled, outputs all ciphertexts according to task_signature.json
    ComputeDevice compute_device = ComputeDevice::CPU;  // Default to CPU mode
    Array1D template_vec;
    json json_data;
    json json_features;
    json json_layers;

    std::map<std::string, std::unique_ptr<FeatureEncrypted>> intermediate_result;
    std::map<std::string, std::unique_ptr<CkksContext>> ckks_contexts;

    std::map<std::string, Array<double, 3>> p_feature2d_x;
    std::vector<std::string> available_keys;
    std::map<std::string, Array1D> p_feature0d_x;

    void run_task(bool is_mpc = false);

    void run_task_sdk(bool is_mpc = false);

    void run_task_plaintext(bool is_mpc = false);

    void set_feature(const std::string& feature_id, std::unique_ptr<FeatureEncrypted> feature);

    const FeatureEncrypted& get_feature(const std::string& feature_id);

    Feature0DEncrypted get_ciphertext_output_feature0D(const std::string& feature_id);

    Feature2DEncrypted get_ciphertext_output_feature2D(const std::string& feature_id);
};
