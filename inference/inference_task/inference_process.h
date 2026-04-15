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

#include <hdf5.h>

#include <stdexcept>

#include "util.h"
#include "fhe_layers/fhe_layers.h"

namespace ls = cxx_sdk_v2;

enum class ComputeDevice { CPU, GPU, FPGA };

class Node {
public:
    Node();
};

class FeatureNode : public Node {
public:
    FeatureNode(const std::string& node_id_in,
                int dim_in,
                int channel_in,
                double scale_in,
                uint32_t shape_in[],
                uint32_t skip_in[],
                const std::string& ckks_parameter_id_in,
                int pack_channel_per_ciphertext_in);

    std::string node_id;
    int dim;
    uint32_t channel;
    double scale;
    Duo shape = {0, 0};
    Duo skip = {1, 1};
    Duo special_skip = {1, 1};  // 0D from special_info.skip
    Duo invalid_fill = {0, 0};  // 0D from special_info，2D
    int special_info_dim = 0;   // 0: no special_info, 1: from 1D, 2: from 2D
    std::string ckks_parameter_id;
    int pack_channel_per_ciphertext;
    int level = 0;
    double ckks_scale = 0.0;

    FeatureNode(const json& json_data);

    int get_n_ciphertexts(const Duo& block_shape) const;
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
    InitInferenceProcess(const std::string& project_path_in, bool is_fpga = true);
    virtual ~InitInferenceProcess();

    std::filesystem::path project_path;
    std::string task_type;
    std::string pack_style;
    int n_task;
    std::string output_id;
    std::string input_id;
    json task_input_param;
    json task_output_param;
    int start_task_id;
    int end_task_id;
    bool enable_fpga = false;
    json server_task;
    std::filesystem::path current_json_path;
    json json_data;
    json json_features;
    json json_layers;
    Duo block_shape;
    bool is_absorb_polyrelu;
    std::map<std::string, UPtr<ls::CkksParameter>> ckks_parameters;
    bool fpga_loaded = false;

    bool is_lazy = false;

    // Time statistics
    double total_fhe_time = 0.0;
    double total_fpga_time = 0.0;

    virtual void init_parameters(bool is_bootstrapping = false);
    virtual void load_model_prepare();

    std::string get_abs_filename(const std::string& json_filename);
    virtual void init_conv_layer(const std::string& key, const json& layer, const hid_t& h5_file);
    virtual void init_square_layer(const std::string& key, const json& layer, const hid_t& h5_file);
    virtual void init_dense_layer(const std::string& key, const json& layer, const hid_t& h5_file);
    virtual void init_add_layer(const std::string& key, const json& layer, const std::string& block_input_feature);
    virtual void init_reshape_layer(const std::string& key, const json& layer);
    virtual void init_mult_scalar_layer(const std::string& key,
                                        const json& layer,
                                        const hid_t& h5_file,
                                        const Duo& block_shape = {128, 256});
    virtual void init_drop_level_layer(const std::string& key, const json& layer);
    virtual void init_fhe_avgpool_layer(const std::string& key,
                                        const json& layer,
                                        const bool& is_adaptive = true,
                                        const Duo& block_shape = {128, 256});
    virtual void init_fhe_avgpool1d_layer(const std::string& key, const json& layer, const bool& is_adaptive = true);
    void init_multiplexed_conv_layer(const std::string& key,
                                     const json& layer,
                                     const hid_t& h5_file,
                                     const Duo& block_shape_in = {128, 256});

    void init_poly_relu_layer(const std::string& key,
                              const json& layer,
                              const hid_t& h5_file,
                              bool is_absorb = true,
                              const Duo& block_shape_in = {128, 256});
    void init_concat_layer(const std::string& key, const json& layer);
    void init_upsample_layer(const std::string& key, const json& layer, const Duo& block_shape = {128, 256});
    void init_upsample_nearest_layer(const std::string& key, const json& layer);
    void init_conv1d_layer(const std::string& key, const json& layer, const hid_t& h5_file);

private:
    template <typename T> void _prepare_layer(const std::string& key, UPtr<T> layer) {
        if (is_lazy) {
            layer->prepare_weight_lazy();
        } else {
            layer->prepare_weight();
        }
        set_layer(key, std::move(layer));
    }

    template <typename T, typename PrepareFn>
    void _prepare_layer(const std::string& key, UPtr<T> layer, const PrepareFn& prepare) {
        prepare(*layer);
        set_layer(key, std::move(layer));
    }

    template <typename T, typename LazyFn, typename EagerFn>
    void
    _prepare_layer(const std::string& key, UPtr<T> layer, const LazyFn& prepare_lazy, const EagerFn& prepare_eager) {
        if (is_lazy) {
            prepare_lazy(*layer);
        } else {
            prepare_eager(*layer);
        }
        set_layer(key, std::move(layer));
    }

    template <int dim>
    Array<double, dim> _load_h5_tensor(const json& layer,
                                       const hid_t& h5_file,
                                       const std::string& tensor_name,
                                       const std::array<uint64_t, dim>& shape) const {
        const std::string path_key = tensor_name + "_path";
        const std::string scale_key = tensor_name + "_scale";
        const double scale = layer.contains(scale_key) ? layer[scale_key].get<double>() : 1.0;
        return h5_to_array<dim>(h5_file, layer.at(path_key).get<std::string>(), shape, scale);
    }

    std::map<std::string, UPtr<Layer>> ckks_layers_;

public:
    template <typename T> T& get_layer(const std::string& key) {
        auto it = ckks_layers_.find(key);
        if (it == ckks_layers_.end()) {
            throw std::runtime_error("layer not found: " + key);
        }
        auto* layer = dynamic_cast<T*>(it->second.get());
        if (layer == nullptr) {
            throw std::runtime_error("layer type mismatch: " + key);
        }
        return *layer;
    }

    template <typename T> const T& get_layer(const std::string& key) const {
        auto it = ckks_layers_.find(key);
        if (it == ckks_layers_.end()) {
            throw std::runtime_error("layer not found: " + key);
        }
        auto* layer = dynamic_cast<const T*>(it->second.get());
        if (layer == nullptr) {
            throw std::runtime_error("layer type mismatch: " + key);
        }
        return *layer;
    }

    template <typename T> void set_layer(const std::string& key, UPtr<T> layer) {
        ckks_layers_[key] = std::move(layer);
    }
};

class InferenceProcess {
public:
    InferenceProcess() {}
    InferenceProcess(InitInferenceProcess* fp_in, bool is_fpga_in);
    virtual ~InferenceProcess();
    InitInferenceProcess* fp;
    int task_num;
    bool is_fpga;
    ComputeDevice compute_device = ComputeDevice::CPU;  // Default to CPU mode
    Array1D template_vec;
    json json_data;
    json json_features;
    json json_layers;

    std::map<std::string, UPtr<FeatureEncrypted>> intermediate_result;
    std::map<std::string, UPtr<ls::CkksContext>> ckks_contexts;

    std::map<std::string, Array<double, 3>> p_feature2d_x;
    std::map<std::string, Array<double, 2>> p_feature1d_x;
    std::vector<std::string> available_keys;
    std::map<std::string, Array1D> p_feature0d_x;

    void run_task(bool is_mpc = false);
    void run_task_sdk(bool is_mpc = false);
    void run_task_plaintext(bool is_mpc = false);

    void set_feature(const std::string& feature_id, UPtr<FeatureEncrypted> feature);
    const FeatureEncrypted& get_feature(const std::string& feature_id);
    Feature0DEncrypted get_ciphertext_output_feature0D(const std::string& feature_id);
    Feature1DEncrypted get_ciphertext_output_feature1D(const std::string& feature_id);
    Feature2DEncrypted get_ciphertext_output_feature2D(const std::string& feature_id);
};
