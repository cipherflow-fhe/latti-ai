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

#include "inference_process.h"
#include "lattisense/cxx_sdk_v2/cxx_fhe_task.h"
#include <iostream>

using namespace std;
using namespace cxx_sdk_v2;
uint64_t fhe_time = 0;
bool normal_output = false;

Node::Node() {}

InferenceProcess::InferenceProcess(InitInferenceProcess* fp_in, bool is_fpga_in) : task_num(0), is_fpga(is_fpga_in) {
    fp = fp_in;
}

InferenceProcess::~InferenceProcess() {}

FeatureNode::FeatureNode(string node_id_in,
                         int dim_in,
                         int channel_in,
                         double scale_in,
                         uint32_t shape_in[],
                         uint32_t skip_in[],
                         string ckks_parameter_id_in,
                         int pack_channel_per_ciphertext_in) {
    node_id = node_id_in;
    dim = dim_in;
    channel = channel_in;
    scale = scale_in;
    shape[0] = shape_in[0];
    shape[1] = shape_in[1];
    skip[0] = skip_in[0];
    skip[1] = skip_in[1];
    ckks_parameter_id = ckks_parameter_id_in;
    pack_channel_per_ciphertext = pack_channel_per_ciphertext_in;
}

InitInferenceProcess::InitInferenceProcess(const string& project_path_in, bool is_fpga) {
    project_path = project_path_in;
    const json& config = read_json(project_path / "task_config.json");
    task_type = config["task_type"].get<string>();
    pack_style = config["pack_style"].get<string>();
    n_task = (int)config["task_num"];
    start_task_id = (int)config["server_start_id"];
    end_task_id = (int)config["server_end_id"];
    task_input_param = config["task_input_param"];
    task_output_param = config["task_output_param"];
    server_task = config["server_task"];
    block_shape = config["block_shape"];
    is_absorb_polyrelu = config["is_absorb_polyrelu"];
    Timer timer(true);
}

InitInferenceProcess::~InitInferenceProcess() {}

void InitInferenceProcess::init_parameters(bool is_bootstrapping) {
    auto json_params = read_json(project_path / "ckks_parameter.json");
    if (is_bootstrapping) {
        for (auto& param : json_params.items()) {
            string key = param.key();
            auto btp_param = CkksBtpParameter::create_parameter();
            ckks_parameters[key] = make_unique<CkksParameter>(move(btp_param.get_ckks_parameter()));
        }
    } else {
        for (auto& param : json_params.items()) {
            string key = param.key();
            int n = param.value()["poly_modulus_degree"];
            ckks_parameters[key] = make_unique<CkksParameter>(CkksParameter::create_parameter(n));
        }
    }
}

void InitInferenceProcess::init_conv_layer(const string& key, const json& layer, const hid_t& h5_file) {
    FeatureNode feature_input(json_features[layer["feature_input"][0].get<string>()]);
    FeatureNode feature_output(json_features[layer["feature_output"][0].get<string>()]);
    int out_level = feature_output.level;
    int groups = layer["groups"];
    uint32_t channel_input = feature_input.channel / groups;
    Array<double, 4> weight;

    double weight_scale = layer["weight_scale"];
    double bias_scale = layer["bias_scale"];

    weight = h5_to_array<4>(h5_file, layer["weight_path"],
                            {feature_output.channel, channel_input, layer["kernel_shape"][0], layer["kernel_shape"][1]},
                            weight_scale);
    Array<double, 1> bias = h5_to_array<1>(h5_file, layer["bias_path"], {feature_output.channel}, bias_scale);
    CkksParameter& param = *ckks_parameters.at(feature_input.ckks_parameter_id);
    double default_scale = DEFAULT_SCALE;
    double residual_scale = 1.0;

    Duo stride = {layer["stride"][0], layer["stride"][1]};

    if (layer["groups"] == 1) {
        auto conv_layer =
            make_unique<Conv2DPackedLayer>(param, feature_input.shape, weight, bias, stride, feature_input.skip,
                                           feature_input.pack_channel_per_ciphertext, out_level + 1, residual_scale);
        if (is_lazy) {
            conv_layer->prepare_weight_lazy();
        } else {
            conv_layer->prepare_weight();
        }
        ckks_conv2ds[key] = move(conv_layer);
    } else {
        auto dw_layer = make_unique<Conv2DPackedDepthwiseLayer>(
            param, feature_input.shape, weight, bias, stride, feature_input.skip,
            feature_input.pack_channel_per_ciphertext, out_level + 1, residual_scale);
        if (is_lazy) {
            dw_layer->prepare_weight_lazy();
        } else {
            dw_layer->prepare_weight();
        }
        ckks_dw_conv2ds[key] = move(dw_layer);
    }
}

void InitInferenceProcess::init_square_layer(const string& key, const json& layer, const hid_t& h5_file) {
    FeatureNode feature_input(json_features[layer["feature_input"][0].get<string>()]);
    auto squar2d = make_unique<SquareLayer>(*ckks_parameters.at(feature_input.ckks_parameter_id));
    ckks_squares[key] = move(squar2d);
}

void InitInferenceProcess::init_dense_layer(const string& key, const json& layer, const hid_t& h5_file) {
    FeatureNode feature_input(json_features[layer["feature_input"][0].get<string>()]);
    FeatureNode feature_output(json_features[layer["feature_output"][0].get<string>()]);
    int out_level = feature_output.level;
    double weight_scale = layer["weight_scale"];
    double bias_scale = layer["bias_scale"];
    Array<double, 2> weight;
    weight =
        h5_to_array<2>(h5_file, layer["weight_path"], {feature_output.channel, feature_input.channel}, weight_scale);

    auto bias = h5_to_array<1>(h5_file, layer["bias_path"], {feature_output.channel}, bias_scale);
    CkksParameter& param = *ckks_parameters.at(feature_input.ckks_parameter_id);
    double residual_scale = 1.0;

    auto dense = make_unique<DensePackedLayer>(
        *ckks_parameters.at(feature_input.ckks_parameter_id), feature_input.virtual_shape, feature_input.virtual_skip,
        weight, bias, feature_input.pack_channel_per_ciphertext, feature_input.level, 0, residual_scale);
    if (pack_style == "multiplexed") {
        if (is_lazy) {
            dense->prepare_weight_for_mult_pack_lazy();
        } else {
            dense->prepare_weight_for_mult_pack();
        }
    } else {
        if (is_lazy) {
            dense->prepare_weight1_lazy();
        } else {
            dense->prepare_weight1();
        }
    }
    ckks_denses[key] = move(dense);
}

void InitInferenceProcess::init_add_layer(const string& key, const json& layer, const string& block_input_feature) {
    FeatureNode feature_input0(json_features[layer["feature_input"][0].get<string>()]);
    FeatureNode feature_input1(json_features[layer["feature_input"][1].get<string>()]);
    FeatureNode feature_output(json_features[layer["feature_output"][0].get<string>()]);
    CkksParameter& param = *ckks_parameters.at(feature_input0.ckks_parameter_id);
    auto add2d = make_unique<AddLayer>(*ckks_parameters.at(feature_input0.ckks_parameter_id));
    add2d->target_ckks_scale = feature_output.ckks_scale;
    ckks_adds[key] = move(add2d);
}

void InitInferenceProcess::init_mult_scalar_layer(const string& key,
                                                  const json& layer,
                                                  const hid_t& h5_file,
                                                  const Duo& block_shape) {
    FeatureNode feature_input0(json_features[layer["feature_input"][0].get<string>()]);
    FeatureNode feature_output0(json_features[layer["feature_output"][0].get<string>()]);

    Duo block_expansion;
    if (feature_input0.shape[0] > block_shape[0] || feature_input0.shape[1] > block_shape[1]) {
        block_expansion = {feature_input0.shape[0] / block_shape[0], feature_input0.shape[1] / block_shape[1]};
    } else {
        block_expansion = {1, 1};
    }
    Duo upsample_factor = {1, 1};
    CkksParameter& param = *ckks_parameters.at(feature_input0.ckks_parameter_id);

    double scale = layer["weight_scale"];
    auto weight = gen_random_array<1>({feature_input0.channel}, 1.0);
    for (int i = 0; i < feature_input0.channel; i++) {
        weight.set(i, scale);
    }
    auto mult_scalar = make_unique<MultScalarLayer>(param, feature_input0.shape, weight, feature_input0.skip,
                                                    feature_input0.pack_channel_per_ciphertext, feature_input0.level,
                                                    upsample_factor, block_expansion);
    mult_scalar->prepare_weight();
    ckks_mult_scalar[key] = move(mult_scalar);
}

void InitInferenceProcess::init_drop_level_layer(const string& key, const json& layer) {
    FeatureNode feature_input0(json_features[layer["feature_input"][0].get<string>()]);
    CkksParameter& param = *ckks_parameters.at(feature_input0.ckks_parameter_id);
    auto drop_level = make_unique<DropLevelLayer>();

    ckks_drop_level[key] = move(drop_level);
}

void InitInferenceProcess::init_reshape_layer(const string& key, const json& layer) {
    FeatureNode feature_input0(json_features[layer["feature_input"][0].get<string>()]);

    auto reshape = make_unique<ReshapeLayer>(*ckks_parameters.at(feature_input0.ckks_parameter_id));
    ckks_reshape[key] = move(reshape);
}

void InitInferenceProcess::init_concat_layer(const string& key, const json& layer) {
    auto concat = make_unique<ConcatLayer>();
    ckks_concat[key] = move(concat);
}

void InitInferenceProcess::init_upsample_layer(const string& key, const json& layer, const Duo& block_shape) {
    FeatureNode feature_input(json_features[layer["feature_input"][0].get<string>()]);
    FeatureNode feature_output(json_features[layer["feature_output"][0].get<string>()]);

    CkksParameter& param = *ckks_parameters.at(feature_input.ckks_parameter_id);
    Duo block_expansion = {feature_input.shape[0] / block_shape[0], feature_input.shape[1] / block_shape[1]};
    Duo upsample_factor_in = {layer["upsample_factor_in"][0], layer["upsample_factor_in"][1]};

    auto upsample = make_unique<UpsampleLayer>(param, block_expansion, upsample_factor_in, feature_input.level,
                                               feature_input.channel, feature_input.pack_channel_per_ciphertext);
    upsample->prepare_data();
    ckks_upsample[key] = move(upsample);
}

void InitInferenceProcess::init_upsample_nearest_layer(const string& key, const json& layer) {
    FeatureNode feature_input(json_features[layer["feature_input"][0].get<string>()]);
    FeatureNode feature_output(json_features[layer["feature_output"][0].get<string>()]);

    CkksParameter& param = *ckks_parameters.at(feature_input.ckks_parameter_id);
    Duo upsample_factor_in = {layer["upsample_factor_in"][0], layer["upsample_factor_in"][1]};

    auto upsample_nearest =
        make_unique<UpsampleNearestLayer>(param, feature_input.shape, feature_input.skip, upsample_factor_in,
                                          feature_input.pack_channel_per_ciphertext, feature_input.level);
    if (is_lazy) {
        upsample_nearest->prepare_weight_lazy();
    } else {
        upsample_nearest->prepare_weight();
    }
    ckks_upsample_nearest[key] = move(upsample_nearest);
}

void InitInferenceProcess::init_multiplexed_conv_layer(const string& key,
                                                       const json& layer,
                                                       const hid_t& h5_file,
                                                       const Duo& block_shape_in) {
    FeatureNode feature_input(json_features[layer["feature_input"][0].get<string>()]);
    FeatureNode feature_output(json_features[layer["feature_output"][0].get<string>()]);
    int groups = layer["groups"];
    bool is_big_size = layer["is_big_size"];
    Duo block_expansion = {feature_input.shape[0] / block_shape_in[0], feature_input.shape[1] / block_shape_in[1]};
    uint32_t channel_input = feature_input.channel / groups;
    Array<double, 4> weight;
    double weight_scale = layer["weight_scale"];
    double bias_scale = layer["bias_scale"];

    if (key.find("ConvTranspose") != std::string::npos) {
        weight = h5_to_array<4>(
            h5_file, layer["weight_path"],
            {channel_input, feature_output.channel, layer["kernel_shape"][0], layer["kernel_shape"][1]}, weight_scale);
        weight = transpose_weight(weight);
    } else {
        weight = h5_to_array<4>(
            h5_file, layer["weight_path"],
            {feature_output.channel, channel_input, layer["kernel_shape"][0], layer["kernel_shape"][1]}, weight_scale);
    }
    Array<double, 1> bias = h5_to_array<1>(h5_file, layer["bias_path"], {feature_output.channel}, bias_scale);

    CkksParameter& param = *ckks_parameters.at(feature_input.ckks_parameter_id);
    double residual_scale = 1.0;
    Duo stride = {layer["stride"][0], layer["stride"][1]};
    Duo upsample_factor_in = {layer["upsample_factor_in"][0], layer["upsample_factor_in"][1]};

    if (layer["groups"] == 1) {
        if (is_big_size) {
            Duo next_stride = {block_expansion[0] / stride[0], block_expansion[1] / stride[1]};
            Array<int, 1> padding({2});
            if (key.find("ConvTranspose") != std::string::npos && layer["kernel_shape"][0] == 2) {
                padding.set(0, 1);
                padding.set(1, 1);
            } else {
                padding.set(0, -1);
                padding.set(1, -1);
            }
            auto inv_conv_layer = make_unique<InverseMultiplexedConv2DLayer>(
                param, feature_input.shape, weight, bias, padding, stride, next_stride, feature_input.skip,
                block_shape_in, feature_input.level, residual_scale);
            if (is_lazy) {
                inv_conv_layer->prepare_weight_lazy();
            } else {
                inv_conv_layer->prepare_weight();
            }
            ckks_big_conv2ds[key] = move(inv_conv_layer);
        } else {
            auto mux_conv_layer = make_unique<ParMultiplexedConv2DPackedLayer>(
                param, feature_input.shape, weight, bias, stride, feature_input.skip,
                feature_input.pack_channel_per_ciphertext, feature_input.level, residual_scale, upsample_factor_in);
            if (is_lazy) {
                mux_conv_layer->prepare_weight_for_post_skip_rotation_lazy();
            } else {
                mux_conv_layer->prepare_weight_for_post_skip_rotation();
            }

            ckks_multiplexed_conv2ds[key] = move(mux_conv_layer);
        }
    } else {
        if (is_big_size) {
            Duo next_stride = {block_expansion[0] / stride[0], block_expansion[1] / stride[1]};
            Array<int, 1> padding({2});
            if (key.find("ConvTranspose") != std::string::npos && layer["kernel_shape"][0] == 2) {
                padding.set(0, 1);
                padding.set(1, 1);
            } else {
                padding.set(0, -1);
                padding.set(1, -1);
            }
            auto inv_conv_layer = make_unique<InverseMultiplexedConv2DLayer>(
                param, feature_input.shape, weight, bias, padding, stride, next_stride, feature_input.skip,
                block_shape_in, feature_input.level, residual_scale);
            if (is_lazy) {
                inv_conv_layer->prepare_weight_lazy();
            } else {
                inv_conv_layer->prepare_weight();
            }
            ckks_big_conv2ds[key] = move(inv_conv_layer);
        } else {
            auto mux_dw_layer = make_unique<ParMultiplexedConv2DPackedLayerDepthwise>(
                param, feature_input.shape, weight, bias, stride, feature_input.skip,
                feature_input.pack_channel_per_ciphertext, feature_input.level, residual_scale);
            if (is_lazy) {
                mux_dw_layer->prepare_weight_lazy();
            } else {
                mux_dw_layer->prepare_weight();
            }
            ckks_multiplexed_dw_conv2ds[key] = move(mux_dw_layer);
        }
    }
}

void InitInferenceProcess::init_poly_relu2d_layer(const string& key,
                                                  const json& layer,
                                                  const hid_t& h5_file,
                                                  bool is_absorb,
                                                  const Duo& block_shape_in) {
    FeatureNode feature_input(json_features[layer["feature_input"][0].get<string>()]);
    FeatureNode feature_output(json_features[layer["feature_output"][0].get<string>()]);
    Duo block_expansion = {div_ceil(feature_input.shape[0], block_shape_in[0]),
                           div_ceil(feature_input.shape[1], block_shape_in[1])};
    Duo upsample_factor_in = {layer["upsample_factor_in"][0], layer["upsample_factor_in"][1]};
    uint32_t order = layer["order"];
    Array<double, 2> weight;
    double weight_scale = layer["weight_scale"];
    if (is_absorb) {
        weight = h5_to_array<2>(h5_file, layer["weight_path"], {order, feature_input.channel}, weight_scale);
    } else {
        weight = h5_to_array<2>(h5_file, layer["weight_path"], {order + 1, feature_input.channel}, weight_scale);
    }

    CkksParameter& param = *ckks_parameters.at(feature_input.ckks_parameter_id);
    auto layer_poly_relu = make_unique<PolyRelu>(param, feature_input.shape, order, weight, feature_input.skip,
                                                 feature_input.pack_channel_per_ciphertext, feature_input.level,
                                                 upsample_factor_in, block_expansion, pack_style != "multiplexed");
    if (is_absorb) {
        if (is_lazy) {
            layer_poly_relu->prepare_weight_lazy();
        } else {
            layer_poly_relu->prepare_weight();
        }
    } else {
        if (is_lazy) {
            layer_poly_relu->prepare_weight_for_non_absorb_case_lazy();
        } else {
            layer_poly_relu->prepare_weight_for_non_absorb_case();
        }
    }
    ckks_poly_relu[key] = move(layer_poly_relu);
}

void InitInferenceProcess::init_fhe_avgpool_layer(const string& key,
                                                  const json& layer,
                                                  const bool& is_adaptive,
                                                  const Duo& block_shape) {
    FeatureNode feature_input(json_features[layer["feature_input"][0].get<string>()]);
    FeatureNode feature_output(json_features[layer["feature_output"][0].get<string>()]);
    CkksParameter& param = *ckks_parameters.at(feature_input.ckks_parameter_id);
    Duo block_expansion = {feature_input.shape[0] / block_shape[0], feature_input.shape[1] / block_shape[1]};
    Duo stride = {layer["stride"][0], layer["stride"][1]};
    bool is_big_size = layer["is_big_size"];
    if (is_big_size) {
        auto avgpool = make_unique<Avgpool2DLayer>(feature_input.shape, stride);
        ckks_avgpool[key] = move(avgpool);
    } else {
        if (is_adaptive) {
            auto avgpool = make_unique<Avgpool2DLayer>(feature_input.shape, stride);
            ckks_avgpool[key] = move(avgpool);
        } else {
            auto avgpool = make_unique<Avgpool2DLayer>(feature_input.shape, stride);
            avgpool->prepare_weight(param, feature_input.pack_channel_per_ciphertext, feature_input.level,
                                    feature_input.skip, feature_input.shape);
            ckks_avgpool[key] = move(avgpool);
        }
    }
}

void InitInferenceProcess::load_model_prepare() {
    current_json_path = project_path;
    json_data = read_json(current_json_path + "/ergs/erg0.json");
    json_features = json_data.at("feature");
    json_layers = json_data.at("layer");
    string block_input_feature = json_data["input_feature"][0];
    auto block_shape = this->block_shape;

    string h5_filename = project_path / "model_parameters.h5";
    hid_t h5_file = H5Fopen(h5_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    json config_json = read_json(project_path / "task_config.json");
    for (auto& layer : json_layers.items()) {
        const string& key = layer.key();
        const json& value = layer.value();
        const string& layer_type = value["type"].get<string>();
        if (layer_type == "conv2d") {
            if (pack_style == "multiplexed") {
                init_multiplexed_conv_layer(key, value, h5_file, block_shape);
            } else {
                init_conv_layer(key, value, h5_file);
            }
        } else if (layer_type == "square2d") {
            init_square_layer(key, value, h5_file);
        } else if (layer_type == "fc0") {
            init_dense_layer(key, value, h5_file);
        } else if (layer_type == "add2d") {
            init_add_layer(key, value, block_input_feature);
        } else if (layer_type == "reshape") {
            init_reshape_layer(key, value);
        } else if (layer_type == "drop_level") {
            init_drop_level_layer(key, value);
        } else if (layer_type == "concat2d") {
            init_concat_layer(key, value);
        } else if (layer_type == "upsample") {
            init_upsample_layer(key, value, block_shape);
        } else if (layer_type == "upsample_nearest") {
            init_upsample_nearest_layer(key, value);
        } else if (layer_type == "mult_scalar") {
            init_mult_scalar_layer(key, value, h5_file, block_shape);
        } else if (layer_type == "poly_relu2d" || layer_type == "simple_polyrelu") {
            init_poly_relu2d_layer(key, value, h5_file, is_absorb_polyrelu, block_shape);
        } else if (layer_type == "avgpool2d") {
            bool is_adaptive_avgpool = value["is_adaptive_avgpool"];
            init_fhe_avgpool_layer(key, value, is_adaptive_avgpool, block_shape);
        }
    }
    H5Fclose(h5_file);
}

void InferenceProcess::run_task_sdk(bool enable_mpc) {
    // Reset time statistics for each request
    fp->total_fhe_time = 0.0;
    fp->total_fpga_time = 0.0;

    json_data = read_json(fp->project_path / "ergs/erg0.json");
    string block_input_feature = json_data["input_feature"][0];
    json_features = json_data.at("feature");
    json_layers = json_data.at("layer");
    auto block_shape = fp->block_shape;

    // Time statistics for FHE and MPC operations
    Timer fhe_timer;
    Timer mpc_timer;
    while (json_layers.size() > 0) {
        for (const auto& layer : json_layers.items()) {
            const string& key = layer.key();
            const string& layer_type = layer.value()["type"].get<string>();
            auto feature_input = layer.value()["feature_input"].get<vector<string>>();
            auto feature_output = layer.value()["feature_output"].get<vector<string>>();
            bool tag = false;
            for (const auto& fi : feature_input) {
                if (intermediate_result.find(fi) == intermediate_result.end()) {
                    tag = true;
                    break;
                }
            }
            if (tag == true) {
                continue;
            }

            const string& feature_output_id = feature_output[0];
            FeatureNode feature_input_node(json_features[feature_input[0]]);
            unique_ptr<FeatureEncrypted> result;
            auto& context = *ckks_contexts.at(feature_input_node.ckks_parameter_id);
            if (layer_type == "conv2d") {
                fhe_timer.start();
                const FeatureEncrypted& feature_node = get_feature(feature_input[0]);
                if (feature_node.dim == 2) {
                    const Feature2DEncrypted& input2D = dynamic_cast<const Feature2DEncrypted&>(feature_node);
                    if (fp->pack_style == "multiplexed") {
                        if (layer.value()["groups"] == 1) {
                            bool is_big_size = layer.value()["is_big_size"];
                            if (is_big_size) {
                                result =
                                    make_unique<Feature2DEncrypted>(fp->ckks_big_conv2ds[key]->run(context, input2D));
                            } else {
                                result = make_unique<Feature2DEncrypted>(
                                    fp->ckks_multiplexed_conv2ds[key]->run_for_post_skip_rotation(context, input2D));
                            }
                        } else {
                            bool is_big_size = layer.value()["is_big_size"];
                            if (is_big_size) {
                                result =
                                    make_unique<Feature2DEncrypted>(fp->ckks_big_conv2ds[key]->run(context, input2D));
                            } else {
                                result = make_unique<Feature2DEncrypted>(
                                    fp->ckks_multiplexed_dw_conv2ds[key]->run(context, input2D));
                            }
                        }
                    } else {
                        if (layer.value()["groups"] == 1) {
                            result = make_unique<Feature2DEncrypted>(fp->ckks_conv2ds[key]->run(context, input2D));
                        } else {
                            result = make_unique<Feature2DEncrypted>(fp->ckks_dw_conv2ds[key]->run(context, input2D));
                        }
                        const Feature2DEncrypted& res = dynamic_cast<const Feature2DEncrypted&>(*result);
                    }
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted");
                }
                fhe_timer.stop();
            } else if (layer_type == "bootstrapping") {
                const int maximum_refreshed_level = 9;
                const FeatureEncrypted& feature_node = get_feature(feature_input[0]);
                FeatureNode output_feature_node(json_features[feature_output_id]);
                if (feature_node.dim == 2) {
                    const Feature2DEncrypted& input2D = dynamic_cast<const Feature2DEncrypted&>(feature_node);
                    Feature2DEncrypted refresh_result = input2D.refresh_ciphertext();
                    if (maximum_refreshed_level > output_feature_node.level) {
                        result = make_unique<Feature2DEncrypted>(
                            refresh_result.drop_level(maximum_refreshed_level - output_feature_node.level));
                    } else {
                        result = make_unique<Feature2DEncrypted>(move(refresh_result));
                    }
                } else if (feature_node.dim == 0) {
                    const Feature0DEncrypted& input0D = dynamic_cast<const Feature0DEncrypted&>(feature_node);
                    Feature0DEncrypted refresh_result = input0D.refresh_ciphertext();
                    if (maximum_refreshed_level > output_feature_node.level) {
                        result = make_unique<Feature0DEncrypted>(
                            refresh_result.drop_level(maximum_refreshed_level - output_feature_node.level));
                    } else {
                        result = make_unique<Feature0DEncrypted>(move(refresh_result));
                    }
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted or Feature0DEncrypted");
                }
            } else if (layer_type == "batchnorm" || layer_type == "batchnorm2d" || layer_type == "dropout" ||
                       layer_type == "mul" || layer_type == "identity") {
                const FeatureEncrypted& feature_node = get_feature(feature_input[0]);
                if (feature_node.dim == 2) {
                    const Feature2DEncrypted& input2D = dynamic_cast<const Feature2DEncrypted&>(feature_node);
                    result = make_unique<Feature2DEncrypted>(input2D.copy());
                } else if (feature_node.dim == 0) {
                    const Feature0DEncrypted& input0D = dynamic_cast<const Feature0DEncrypted&>(feature_node);
                    result = make_unique<Feature0DEncrypted>(input0D.copy());
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted or Feature0DEncrypted");
                }
            } else if (layer_type == "square2d") {
                fhe_timer.start();
                const FeatureEncrypted& feature_node = get_feature(feature_input[0]);

                if (feature_node.dim == 2) {
                    const Feature2DEncrypted& input2D = dynamic_cast<const Feature2DEncrypted&>(feature_node);
                    result = make_unique<Feature2DEncrypted>(fp->ckks_squares[key]->call(context, input2D));
                } else if (feature_node.dim == 0) {
                    const Feature0DEncrypted& input0D = dynamic_cast<const Feature0DEncrypted&>(feature_node);
                    result = make_unique<Feature0DEncrypted>(fp->ckks_squares[key]->call(context, input0D));
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted ");
                }
                fhe_timer.stop();
            } else if (layer_type == "add2d") {
                fhe_timer.start();
                double target_ckks_scale = json_features[feature_output[0]]["ckks_scale"];
                const Feature2DEncrypted& input0 =
                    dynamic_cast<const Feature2DEncrypted&>(get_feature(feature_input[0]));
                const Feature2DEncrypted& input1 =
                    dynamic_cast<const Feature2DEncrypted&>(get_feature(feature_input[1]));
                if (input0.dim == 2 && input1.dim == 2) {
                    result = make_unique<Feature2DEncrypted>(fp->ckks_adds[key]->run(context, input0, input1));
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted");
                }
                fhe_timer.stop();
            } else if (layer_type == "mult_scalar") {
                fhe_timer.start();
                const Feature2DEncrypted& input0 =
                    dynamic_cast<const Feature2DEncrypted&>(get_feature(feature_input[0]));

                if (input0.dim == 2) {
                    auto res = fp->ckks_mult_scalar[key]->run(context, input0);
                    result = make_unique<Feature2DEncrypted>(move(res));
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted");
                }
                fhe_timer.stop();
            } else if (layer_type == "drop_level") {
                FeatureNode d_input_node(json_features[feature_input[0]]);
                FeatureNode d_output_node(json_features[feature_output[0]]);
                int n_level_to_drop = d_input_node.level - d_output_node.level;
                const FeatureEncrypted& feature_node = get_feature(feature_input[0]);
                if (feature_node.dim == 2) {
                    const Feature2DEncrypted& input2D = dynamic_cast<const Feature2DEncrypted&>(feature_node);
                    result = make_unique<Feature2DEncrypted>(input2D.drop_level(n_level_to_drop));
                } else if (feature_node.dim == 0) {
                    const Feature0DEncrypted& input0D = dynamic_cast<const Feature0DEncrypted&>(feature_node);
                    result = make_unique<Feature0DEncrypted>(input0D.drop_level(n_level_to_drop));
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted or Feature0DEncrypted");
                }
            } else if (layer_type == "fc0" || layer_type == "fc1") {
                fhe_timer.start();
                const FeatureEncrypted& feature_node = get_feature(feature_input[0]);
                if (feature_node.dim == 0) {
                    const Feature0DEncrypted& input0D = dynamic_cast<const Feature0DEncrypted&>(feature_node);
                    if (fp->pack_style == "multiplexed") {
                        result = make_unique<Feature0DEncrypted>(fp->ckks_denses[key]->run_mult_park(context, input0D));
                    } else {
                        result = make_unique<Feature0DEncrypted>(fp->ckks_denses[key]->call(context, input0D));
                    }
                } else {
                    throw runtime_error("input is not available, expect Feature0DEncrypted");
                }
                fhe_timer.stop();
            } else if (layer_type == "reshape") {
                fhe_timer.start();
                const FeatureEncrypted& feature_node = get_feature(feature_input[0]);
                if (feature_node.dim == 2) {
                    const Feature2DEncrypted& input2D = dynamic_cast<const Feature2DEncrypted&>(feature_node);
                    result = make_unique<Feature0DEncrypted>(fp->ckks_reshape[key]->call(context, input2D));
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted");
                }
                fhe_timer.stop();
            } else if (layer_type == "avgpool2d") {
                fhe_timer.start();
                const FeatureEncrypted& feature_node = get_feature(feature_input[0]);
                if (feature_node.dim == 2) {
                    const Feature2DEncrypted& input2D = dynamic_cast<const Feature2DEncrypted&>(feature_node);

                    if (fp->pack_style == "multiplexed") {
                        bool is_adaptive_avgpool = layer.value()["is_adaptive_avgpool"];
                        bool is_big_size = layer.value()["is_big_size"];
                        if (is_adaptive_avgpool) {
                            result = make_unique<Feature2DEncrypted>(
                                fp->ckks_avgpool[key]->run_adaptive_avgpool(context, input2D));
                        } else {
                            if (is_big_size) {
                                Duo block_expansion = {feature_input_node.shape[0] / block_shape[0],
                                                       feature_input_node.shape[1] / block_shape[1]};
                                result = make_unique<Feature2DEncrypted>(
                                    fp->ckks_avgpool[key]->run_split_avgpool(context, input2D, block_expansion));
                            } else {
                                result = make_unique<Feature2DEncrypted>(
                                    fp->ckks_avgpool[key]->run_multiplexed_avgpool(context, input2D));
                            }
                        }
                    } else {
                        result = make_unique<Feature2DEncrypted>(fp->ckks_avgpool[key]->run(context, input2D));
                    }
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted");
                }
                fhe_timer.stop();
            } else if (layer_type == "concat2d") {
                fhe_timer.start();
                vector<Feature2DEncrypted> inputs;
                for (const auto& input_name : feature_input) {
                    const FeatureEncrypted& input_feature_node = get_feature(input_name);
                    if (input_feature_node.dim == 2) {
                        const Feature2DEncrypted& input2D = dynamic_cast<const Feature2DEncrypted&>(input_feature_node);
                        inputs.emplace_back(input2D.copy());
                    } else {
                        throw runtime_error("input is not available, expect Feature2DEncrypted");
                    }
                }
                result = make_unique<Feature2DEncrypted>(fp->ckks_concat[key]->run_multiple_inputs(context, inputs));
                fhe_timer.stop();
            } else if (layer_type == "upsample") {
                fhe_timer.start();
                const FeatureEncrypted& feature_node = get_feature(feature_input[0]);
                if (feature_node.dim == 2) {
                    const Feature2DEncrypted& input2D = dynamic_cast<const Feature2DEncrypted&>(feature_node);
                    result = make_unique<Feature2DEncrypted>(fp->ckks_upsample[key]->run(context, input2D));
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted");
                }
                fhe_timer.stop();
            } else if (layer_type == "upsample_nearest") {
                fhe_timer.start();
                const FeatureEncrypted& feature_node = get_feature(feature_input[0]);
                if (feature_node.dim == 2) {
                    const Feature2DEncrypted& input2D = dynamic_cast<const Feature2DEncrypted&>(feature_node);
                    result = make_unique<Feature2DEncrypted>(fp->ckks_upsample_nearest[key]->run(context, input2D));
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted");
                }
                fhe_timer.stop();
            } else if (layer_type == "poly_relu2d" || layer_type == "simple_polyrelu") {
                fhe_timer.start();
                const FeatureEncrypted& feature_node = get_feature(feature_input[0]);
                if (feature_node.dim == 2) {
                    const Feature2DEncrypted& input2D = dynamic_cast<const Feature2DEncrypted&>(feature_node);
                    if (fp->is_absorb_polyrelu) {
                        result = make_unique<Feature2DEncrypted>(fp->ckks_poly_relu[key]->run(context, input2D));
                    } else {
                        result = make_unique<Feature2DEncrypted>(
                            fp->ckks_poly_relu[key]->run_for_non_absorb_case(context, input2D));
                    }
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted");
                }
                fhe_timer.stop();
            }
            set_feature(feature_output_id, move(result));
            json_layers.erase(key);
            break;
        }
    }
    fp->total_fhe_time += fhe_timer.get_duration().count();
}

void InferenceProcess::run_task(bool is_mpc) {
    // Reset time statistics for each request
    fp->total_fhe_time = 0.0;
    fp->total_fpga_time = 0.0;

    json_data = read_json(fp->project_path / "ergs/erg0.json");
    string block_input_feature = json_data["input_feature"][0];
    json_features = json_data.at("feature");
    json_layers = json_data.at("layer");
    Duo block_shape = fp->block_shape;

    vector<CxxVectorArgument> cxx_args;
    unique_ptr<FeatureEncrypted> result;

    vector<vector<CkksCiphertext>> ct_data(json_data["input_feature"].size());
    for (int i = 0; i < json_data["input_feature"].size(); i++) {
        auto ki = json_data["input_feature"][i];
        FeatureNode feature_input(json_features[ki.get<string>()]);
        if (feature_input.dim == 2) {
            const Feature2DEncrypted& input = dynamic_cast<const Feature2DEncrypted&>(get_feature(ki));
            auto _size = input.data.size();
            for (int j = 0; j < _size; j++) {
                ct_data[i].push_back(input.data[j].copy());
            }
            cxx_args.push_back(CxxVectorArgument{ki, &ct_data});
        }
        if (feature_input.dim == 0) {
            const Feature0DEncrypted& input = dynamic_cast<const Feature0DEncrypted&>(get_feature(ki));
            for (int j = 0; j < input.data.size(); j++) {
                ct_data[i].push_back(input.data[j].copy());
            }
            cxx_args.push_back(CxxVectorArgument{ki, &ct_data});
        }
    }

    for (const auto& layer : json_layers.items()) {
        Timer aa;
        aa.start();
        const string& key = layer.key();

        const string& layer_type = layer.value()["type"].get<string>();
        auto feature_input = layer.value()["feature_input"].get<vector<string>>();
        auto feature_output = layer.value()["feature_output"].get<vector<string>>();

        const string& feature_output_id = feature_output[0];
        FeatureNode feature_input_node(json_features[feature_input[0]]);
        unique_ptr<FeatureEncrypted> result;
        auto& context = *ckks_contexts.at(feature_input_node.ckks_parameter_id);
        if (layer_type == "conv2d") {
            FeatureNode d_input_node(json_features[feature_input[0]]);
            if (d_input_node.dim == 2) {
                if (layer.value()["groups"] == 1) {
                    bool is_big_size = layer.value()["is_big_size"];
                    if (is_big_size) {
                        cxx_args.push_back(
                            CxxVectorArgument{"convw_" + key, &(fp->ckks_big_conv2ds.at(key)->weight_pt)});
                        cxx_args.push_back(CxxVectorArgument{"convb_" + key, &(fp->ckks_big_conv2ds.at(key)->bias_pt)});
                    } else {
                        if (fp->pack_style == "multiplexed") {
                            if (layer.value()["stride"][0] == 1 and d_input_node.skip[0] == 1) {
                            } else {
                                cxx_args.push_back(CxxVectorArgument{"convm_" + key,
                                                                     &(fp->ckks_multiplexed_conv2ds.at(key)->mask_pt)});
                            }
                            cxx_args.push_back(
                                CxxVectorArgument{"convw_" + key, &(fp->ckks_multiplexed_conv2ds.at(key)->weight_pt)});
                            cxx_args.push_back(
                                CxxVectorArgument{"convb_" + key, &(fp->ckks_multiplexed_conv2ds.at(key)->bias_pt)});
                        } else {
                            cxx_args.push_back(
                                CxxVectorArgument{"convw_" + key, &(fp->ckks_conv2ds.at(key)->weight_pt_)});
                            cxx_args.push_back(
                                CxxVectorArgument{"convb_" + key, &(fp->ckks_conv2ds.at(key)->bias_pt_)});
                        }
                    }
                } else {
                    if (fp->pack_style == "multiplexed") {
                        if (layer.value()["stride"][0] == 1) {
                        } else {
                            cxx_args.push_back(
                                CxxVectorArgument{"convm_" + key, &(fp->ckks_multiplexed_dw_conv2ds.at(key)->mask_pt)});
                        }
                        cxx_args.push_back(
                            CxxVectorArgument{"convw_" + key, &(fp->ckks_multiplexed_dw_conv2ds.at(key)->weight_pt)});
                        cxx_args.push_back(
                            CxxVectorArgument{"convb_" + key, &(fp->ckks_multiplexed_dw_conv2ds.at(key)->bias_pt)});
                    } else {
                        cxx_args.push_back(
                            CxxVectorArgument{"convw_" + key, &(fp->ckks_dw_conv2ds.at(key)->weight_pt_)});
                        cxx_args.push_back(CxxVectorArgument{"convb_" + key, &(fp->ckks_dw_conv2ds.at(key)->bias_pt_)});
                    }
                }
            } else {
                throw runtime_error("input is not available, expect Feature2DEncrypted");
            }
        } else if (layer_type == "add2d") {
            continue;
        } else if (layer_type == "fc0" || layer_type == "fc1") {
            cxx_args.push_back(CxxVectorArgument{"densew_" + key, &(fp->ckks_denses.at(key)->weight_pt)});
            cxx_args.push_back(CxxVectorArgument{"denseb_" + key, &(fp->ckks_denses.at(key)->bias_pt)});
        } else if (layer_type == "avgpool2d") {
            FeatureNode d_input_node(json_features[feature_input[0]]);
            if (d_input_node.dim == 2) {
                bool is_adaptive_avgpool = layer.value()["is_adaptive_avgpool"];
                bool is_big_size = layer.value()["is_big_size"];
                if (is_adaptive_avgpool) {
                    continue;
                } else {
                    if (is_big_size) {
                        continue;
                    } else {
                        continue;
                    }
                }
            } else {
                throw runtime_error("input is not available, expect Feature2DEncrypted");
            }
        } else if (layer_type == "poly_relu2d" || layer_type == "simple_polyrelu") {
            for (int i = 0; i < fp->ckks_poly_relu.at(key)->weight_pt.size(); i++) {
                cxx_args.push_back(CxxVectorArgument{"poly_reluw_" + key + "_" + to_string(i),
                                                     &(fp->ckks_poly_relu.at(key)->weight_pt[i])});
            }
        } else if (layer_type == "mult_scalar") {
            cxx_args.push_back(CxxVectorArgument{"mult_scalar_" + key, &(fp->ckks_mult_scalar.at(key)->weight_pt)});
        }
    }

    string context_id;
    int level;
    vector<CkksCiphertext> z_list;
    for (auto& ki : json_data["output_feature"]) {
        FeatureNode feature_output(json_features[ki.get<string>()]);
        context_id = feature_output.ckks_parameter_id;
        level = feature_output.level;
        int n_out_num = div_ceil(feature_output.channel, feature_output.pack_channel_per_ciphertext);
        double encode_scale =
            ckks_contexts.at(feature_output.ckks_parameter_id).get()->get_parameter().get_default_scale();
        for (int i = 0; i < n_out_num; i++) {
            z_list.push_back((*ckks_contexts.at(feature_output.ckks_parameter_id))
                                 .new_ciphertext(feature_output.level, encode_scale));
        }
        cxx_args.push_back(CxxVectorArgument{ki, &z_list});
    }

    // Dynamically create and run task executors based on the compute_device configuration
    switch (compute_device) {
        case ComputeDevice::CPU: {
            auto task = make_unique<FheTaskCpu>(fp->project_path / "ergs");
            fhe_time = fhe_time + task->run(ckks_contexts.at(context_id).get(), cxx_args);
            break;
        }
#ifdef INFERENCE_SDK_ENABLE_GPU
        case ComputeDevice::GPU: {
            auto task = make_unique<FheTaskGpu>(fp->project_path / "ergs");
            fhe_time = fhe_time + task->run(ckks_contexts.at(context_id).get(), cxx_args);
            break;
        }
#else
        case ComputeDevice::GPU:
            throw runtime_error(
                "GPU support is disabled. Reconfigure with -DINFERENCE_SDK_ENABLE_GPU=ON to enable it.");
#endif
        case ComputeDevice::FPGA: throw runtime_error("FPGA mode should use run_task_fpga() instead of run_task_cpu()");
        default: throw runtime_error("Unknown compute device type");
    }
    for (auto& ki : json_data["output_feature"]) {
        FeatureNode feature_output(json_features[ki.get<string>()]);
        if (feature_output.dim == 2) {
            Feature2DEncrypted f2d(ckks_contexts.at(feature_output.ckks_parameter_id).get(), feature_output.level);
            f2d.data = move(z_list);
            f2d.shape = feature_output.shape;
            f2d.skip = feature_output.skip;
            f2d.n_channel_per_ct = feature_output.pack_channel_per_ciphertext;
            f2d.n_channel = feature_output.channel;
            result = make_unique<Feature2DEncrypted>(move(f2d));
        }
        if (feature_output.dim == 0) {
            Feature0DEncrypted f0d(ckks_contexts.at(feature_output.ckks_parameter_id).get(), feature_output.level);
            f0d.data = move(z_list);
            f0d.skip = feature_output.skip[0];
            f0d.n_channel_per_ct = feature_output.pack_channel_per_ciphertext;
            f0d.n_channel = feature_output.channel;
            result = make_unique<Feature0DEncrypted>(move(f0d));
        }
        set_feature(ki, move(result));
    }
}

Array<double, 3> mult_const(Array<double, 3>& x, double const_scale) {
    Array3D res = x.to_array_3d();

    for (int i = 0; i < res.size(); i++) {
        for (int j = 0; j < res[0].size(); j++) {
            for (int k = 0; k < res[0][0].size(); k++) {
                res[i][j][k] = res[i][j][k] * const_scale;
            }
        }
    }
    auto res_array = Array<double, 3>::from_array_3d(res);
    return res_array;
}

void InferenceProcess::run_task_plaintext(bool is_mpc) {
    const json& json_data = read_json(fp->project_path / "ergs/erg0.json");
    json json_features = json_data.at("feature");
    json json_layers = json_data.at("layer");

    while (json_layers.size() > 0) {
        for (auto& layer : json_layers.items()) {
            string key = layer.key();
            string layer_type = layer.value()["type"].get<string>();

            auto feature_input = layer.value()["feature_input"].get<vector<string>>();
            bool tag = false;
            for (auto& fi : feature_input) {
                if (find(available_keys.begin(), available_keys.end(), fi) == available_keys.end()) {
                    tag = true;
                    break;
                }
            }
            if (tag == true) {
                continue;
            }
            string feature_output_id = layer.value()["feature_output"][0];
            Array<double, 3> result;
            vector<double> result0d;
            if (layer_type == "conv2d") {
                auto& input0 = p_feature2d_x[feature_input[0]];
                if (fp->pack_style == "multiplexed") {
                    Duo upsample_factor = {layer.value()["upsample_factor_in"][0],
                                           layer.value()["upsample_factor_in"][1]};
                    if (layer.value()["groups"] == 1) {
                        bool is_big_size = layer.value()["is_big_size"];
                        if (is_big_size) {
                            result = fp->ckks_big_conv2ds[key]->run_plaintext(input0, 1.0);
                        } else {
                            FeatureNode feature_input0(json_features[feature_input[0]]);
                            result = fp->ckks_multiplexed_conv2ds[key]->run_plaintext(input0, 1.0);
                        }
                    } else {
                        FeatureNode feature_input0(json_features[feature_input[0]]);
                        result = fp->ckks_multiplexed_dw_conv2ds[key]->run_plaintext(input0, 1.0);
                    }
                    if (upsample_factor[0] > 1 || upsample_factor[1] > 1) {
                        result = upsample_with_zero(result, upsample_factor);
                    }
                } else {
                    if (layer.value()["groups"] == 1) {
                        FeatureNode feature_input0(json_features[feature_input[0]]);
                        result = fp->ckks_conv2ds[key]->run_plaintext(input0, feature_input0.scale);
                    } else {
                        FeatureNode feature_input0(json_features[feature_input[0]]);
                        result = fp->ckks_dw_conv2ds[key]->run_plaintext(input0, feature_input0.scale);
                    }
                }
            }
            if (layer_type == "bootstrapping" or layer_type == "drop_level" or layer_type == "batchnorm" or
                layer_type == "batchnorm2d" or layer_type == "identity") {
                FeatureNode feature_input0(json_features[feature_input[0]]);
                if (feature_input0.dim == 2) {
                    auto& input0 = p_feature2d_x[feature_input[0]];
                    result = input0.copy();
                } else {
                    auto& input0 = p_feature0d_x[feature_input[0]];
                    result0d = input0;
                }
            }
            if (layer_type == "mult_scalar") {
                const Array<double, 3>& input0 = p_feature2d_x[feature_input[0]];
                result = fp->ckks_mult_scalar[key]->run_plaintext(input0);
            }
            if (layer_type == "concat2d") {
                vector<Array<double, 3>> inputs;
                for (const auto& input_name : feature_input) {
                    inputs.emplace_back(p_feature2d_x[input_name].copy());
                }
                result = fp->ckks_concat[key]->concatenate_channels_multiple_inputs(inputs);
            }
            if (layer_type == "upsample") {
                auto& input0 = p_feature2d_x[feature_input[0]];
                result = fp->ckks_upsample[key]->upsample_with_zero(input0);
            }
            if (layer_type == "upsample_nearest") {
                auto& input0 = p_feature2d_x[feature_input[0]];
                result = fp->ckks_upsample_nearest[key]->run_plaintext(input0);
            }
            if (layer_type == "square2d") {
                FeatureNode feature_input0(json_features[feature_input[0]]);
                if (feature_input0.dim == 2) {
                    auto& input0 = p_feature2d_x[feature_input[0]];
                    result = fp->ckks_squares[key]->run_plaintext(input0);
                } else if (feature_input0.dim == 0) {
                    auto& input0 = p_feature0d_x[feature_input[0]];
                    result0d =
                        fp->ckks_squares[key]->run_plaintext(Array<double, 1>::from_array_1d(input0)).to_array_1d();
                }
            }
            if (layer_type == "add2d") {
                FeatureNode feature_input0(json_features[feature_input[0]]);
                FeatureNode feature_input1(json_features[feature_input[1]]);
                auto& input0 = p_feature2d_x[feature_input[0]];
                auto& input1 = p_feature2d_x[feature_input[1]];
                result = fp->ckks_adds[key]->run_plaintext(input0, input1);
            }
            if (layer_type == "poly_relu2d" || layer_type == "simple_polyrelu") {
                const Array<double, 3>& input0 = p_feature2d_x[feature_input[0]];
                if (fp->is_absorb_polyrelu) {
                    result = fp->ckks_poly_relu[key]->run_plaintext(input0);
                } else {
                    result = fp->ckks_poly_relu[key]->run_plaintext_for_non_absorb_case(input0);
                }
            }
            if (layer_type == "fc0" || layer_type == "fc1") {
                FeatureNode feature_input0(json_features[feature_input[0]]);
                auto input0 = p_feature0d_x[feature_input[0]];
                result0d = fp->ckks_denses[key]
                               ->plaintext_call(Array<double, 1>::from_array_1d(input0), feature_input0.scale)
                               .to_array_1d();
            }
            if (layer_type == "reshape") {
                auto& input0 = p_feature2d_x[feature_input[0]];
                result0d = input0.reshape<1>({0}).to_array_1d();
            }
            if (layer_type == "avgpool2d") {
                auto& input0 = p_feature2d_x[feature_input[0]];
                bool is_adaptive_avgpool = layer.value()["is_adaptive_avgpool"];
                bool is_big_size = layer.value()["is_big_size"];
                if (is_adaptive_avgpool) {
                    result =
                        Array<double, 3>::from_array_3d(fp->ckks_avgpool[key]->plaintext_call(input0).to_array_3d());
                } else {
                    if (is_big_size) {
                        result = fp->ckks_avgpool[key]->plaintext_call(input0);
                    } else {
                        result = fp->ckks_avgpool[key]->plaintext_call_multiplexed(input0);
                    }
                }
            }
            if (result.get_size() != 0) {
                p_feature2d_x[feature_output_id] = move(result);
            }
            if (result0d.size() != 0) {
                p_feature0d_x[feature_output_id] = move(result0d);
            }
            available_keys.push_back(feature_output_id);
            json_layers.erase(key);
            break;
        }
    }
}

void InferenceProcess::set_feature(const string& feature_id, unique_ptr<FeatureEncrypted> feature) {
    intermediate_result[feature_id] = move(feature);
}

const FeatureEncrypted& InferenceProcess::get_feature(const std::string& feature_id) {
    return *intermediate_result[feature_id];
}

Feature0DEncrypted InferenceProcess::get_ciphertext_output_feature0D(const std::string& feature_id) {
    const Feature0DEncrypted& output = dynamic_cast<const Feature0DEncrypted&>(get_feature(feature_id));
    return output.copy();
}

Feature2DEncrypted InferenceProcess::get_ciphertext_output_feature2D(const std::string& feature_id) {
    const Feature2DEncrypted& output = dynamic_cast<const Feature2DEncrypted&>(get_feature(feature_id));
    return output.copy();
}
