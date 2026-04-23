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
#include "../lattisense/cxx_sdk_v2/cxx_fhe_task.h"
#include <cmath>
#include <iostream>

using namespace std;
using namespace lattisense;
uint64_t fhe_time = 0;
bool normal_output = false;

Node::Node() {}

InferenceProcess::InferenceProcess(InitInferenceProcess* fp_in) {
    fp = fp_in;
}

InferenceProcess::~InferenceProcess() {}

FeatureNode::FeatureNode(const string& node_id_in,
                         int dim_in,
                         int channel_in,
                         double scale_in,
                         uint32_t shape_in[],
                         uint32_t skip_in[],
                         const string& ckks_parameter_id_in,
                         int pack_channel_per_ciphertext_in)
    : node_id(node_id_in), dim(dim_in), channel(channel_in), scale(scale_in), ckks_parameter_id(ckks_parameter_id_in),
      pack_channel_per_ciphertext(pack_channel_per_ciphertext_in) {
    shape[0] = shape_in[0];
    shape[1] = shape_in[1];
    skip[0] = skip_in[0];
    skip[1] = skip_in[1];
}

FeatureNode::FeatureNode(const json& json_data)
    : dim(json_data["dim"]), channel(json_data["channel"]), scale(json_data["scale"]),
      ckks_parameter_id(json_data["ckks_parameter_id"]), pack_channel_per_ciphertext(json_data["pack_num"]),
      level(json_data["level"]), ckks_scale(0.0) {
    if (dim == 2) {
        shape = {json_data["shape"][0], json_data["shape"][1]};
        skip = {json_data["skip"][0], json_data["skip"][1]};
        if (json_data.contains("invalid_fill")) {
            invalid_fill = {json_data["invalid_fill"][0], json_data["invalid_fill"][1]};
        }
    }
    if (dim == 1) {
        shape[0] = json_data["shape"][0];
        skip[0] = json_data["skip"][0];
    }
    if (dim == 0) {
        skip[0] = json_data["skip"];
        if (json_data.contains("special_info")) {
            auto& si = json_data["special_info"];
            skip[0] = json_data["skip"];
            special_info_dim = si["invalid_fill"].size();
            if (special_info_dim == 2) {
                shape = {si["shape"][0], si["shape"][1]};
                special_skip = {si["skip"][0], si["skip"][1]};
                invalid_fill = {si["invalid_fill"][0], si["invalid_fill"][1]};
            } else {
                // special_info_dim == 1: from 1D feature
                shape[0] = si["shape"][0];
                special_skip[0] = si["skip"][0];
                invalid_fill[0] = si["invalid_fill"][0];
            }
        }
    }
}

int FeatureNode::get_n_ciphertexts(const Duo& block_shape) const {
    int n_ciphertexts = div_ceil(channel, (uint32_t)pack_channel_per_ciphertext);
    if (dim == 2) {
        if (shape[0] > block_shape[0] || shape[1] > block_shape[1]) {
            Duo out_block_expansion = {shape[0] / block_shape[0], shape[1] / block_shape[1]};
            n_ciphertexts *= out_block_expansion[0] * out_block_expansion[1];
        }
    }
    return n_ciphertexts;
}

InitInferenceProcess::InitInferenceProcess(const string& project_path_in, bool is_fpga)
    : project_path(project_path_in) {
    const json& config = read_json(project_path / "task_config.json");
    pack_style = config["pack_style"].get<string>();
    if (config["block_shape"].size() == 1) {
        block_shape = {config["block_shape"][0], config["block_shape"][0]};
    } else {
        block_shape = {config["block_shape"][0], config["block_shape"][1]};
    }
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
            ckks_parameters_[key] = MakeU<CkksParameter>(move(btp_param.get_ckks_parameter()));
        }
    } else {
        for (auto& param : json_params.items()) {
            string key = param.key();
            int n = param.value()["poly_modulus_degree"];
            ckks_parameters_[key] = MakeU<CkksParameter>(CkksParameter::create_parameter(n));
        }
    }
}

void InitInferenceProcess::_init_conv_layer(const string& key, const json& layer, const hid_t& h5_file) {
    FeatureNode feature_input(json_features[layer["feature_input"][0].get<string>()]);
    FeatureNode feature_output(json_features[layer["feature_output"][0].get<string>()]);
    int out_level = feature_output.level;
    int groups = layer["groups"];
    uint32_t channel_input = feature_input.channel / groups;
    auto weight =
        _load_h5_tensor<4>(layer, h5_file, "weight",
                           {feature_output.channel, channel_input, layer["kernel_shape"][0], layer["kernel_shape"][1]});
    auto bias = _load_h5_tensor<1>(layer, h5_file, "bias", {feature_output.channel});
    CkksParameter& param = *ckks_parameters_.at(feature_input.ckks_parameter_id);
    double residual_scale = 1.0;

    Duo stride = {layer["stride"][0], layer["stride"][1]};

    if (layer["groups"] == 1) {
        auto conv_layer =
            MakeU<Conv2DPackedLayer>(param, feature_input.shape, move(weight), move(bias), stride, feature_input.skip,
                                     feature_input.pack_channel_per_ciphertext, out_level + 1, residual_scale);
        _prepare_layer(key, move(conv_layer));
    } else {
        auto dw_layer = MakeU<Conv2DPackedDepthwiseLayer>(param, feature_input.shape, move(weight), move(bias), stride,
                                                          feature_input.skip, feature_input.pack_channel_per_ciphertext,
                                                          out_level + 1, residual_scale);
        _prepare_layer(key, move(dw_layer));
    }
}

void InitInferenceProcess::_init_conv1d_layer(const string& key, const json& layer, const hid_t& h5_file) {
    FeatureNode feature_input(json_features[layer["feature_input"][0].get<string>()]);
    FeatureNode feature_output(json_features[layer["feature_output"][0].get<string>()]);
    int out_level = feature_output.level;
    uint32_t input_shape = feature_input.shape[0];
    uint32_t kernel_shape = layer["kernel_shape"][0];
    uint32_t stride = layer["stride"][0];
    uint32_t skip = feature_input.skip[0];
    uint32_t n_channel_per_ct = feature_input.pack_channel_per_ciphertext;
    CkksParameter& param = *ckks_parameters_.at(feature_input.ckks_parameter_id);

    auto weight =
        _load_h5_tensor<3>(layer, h5_file, "weight",
                           {(uint64_t)feature_output.channel, (uint64_t)feature_input.channel, (uint64_t)kernel_shape});
    auto bias = _load_h5_tensor<1>(layer, h5_file, "bias", {(uint64_t)feature_output.channel});

    string style = layer.value("style", string("ordinary"));
    if (style == "multiplexed") {
        int groups = layer["groups"];
        if (groups == (int)feature_output.channel && groups != 1) {
            // Depthwise conv1d: weight shape [n_channel, 1, kernel_shape]
            auto dw_weight = _load_h5_tensor<3>(layer, h5_file, "weight",
                                                {(uint64_t)feature_output.channel, 1, (uint64_t)kernel_shape});
            auto conv_layer = MakeU<MultiplexedDWConv1DPackedLayer>(param, input_shape, move(dw_weight), move(bias),
                                                                    stride, skip, n_channel_per_ct, out_level + 1);
            _prepare_layer(key, move(conv_layer));
        } else {
            auto conv_layer = MakeU<MultiplexedConv1DPackedLayer>(param, input_shape, move(weight), move(bias), stride,
                                                                  skip, n_channel_per_ct, out_level + 1);
            _prepare_layer(key, move(conv_layer));
        }
    } else {
        auto conv_layer = MakeU<Conv1DPackedLayer>(param, input_shape, move(weight), move(bias), stride, skip,
                                                   n_channel_per_ct, out_level + 1);
        _prepare_layer(key, move(conv_layer));
    }
}

void InitInferenceProcess::_init_square_layer(const string& key, const json& layer, const hid_t& h5_file) {
    FeatureNode feature_input(json_features[layer["feature_input"][0].get<string>()]);
    auto squar2d = MakeU<SquareLayer>(*ckks_parameters_.at(feature_input.ckks_parameter_id));
    _prepare_layer(key, move(squar2d));
}

void InitInferenceProcess::_init_dense_layer(const string& key, const json& layer, const hid_t& h5_file) {
    FeatureNode feature_input(json_features[layer["feature_input"][0].get<string>()]);
    FeatureNode feature_output(json_features[layer["feature_output"][0].get<string>()]);
    int out_level = feature_output.level;

    auto weight = _load_h5_tensor<2>(layer, h5_file, "weight", {feature_output.channel, feature_input.channel});
    auto bias = _load_h5_tensor<1>(layer, h5_file, "bias", {feature_output.channel});

    CkksParameter& param = *ckks_parameters_.at(feature_input.ckks_parameter_id);
    double residual_scale = 1.0;

    auto dense =
        MakeU<DensePackedLayer>(*ckks_parameters_.at(feature_input.ckks_parameter_id), move(weight), move(bias),
                                feature_input.pack_channel_per_ciphertext, feature_input.level, 0, residual_scale);
    if (feature_input.special_info_dim == 1) {
        uint32_t shape_1d = feature_input.shape[0];
        uint32_t skip_1d = feature_input.special_skip[0];
        uint32_t invalid_fill_1d = feature_input.invalid_fill[0] > 0 ? feature_input.invalid_fill[0] : 1;
        _prepare_layer(
            key, move(dense),
            [&](DensePackedLayer& layer) {
                layer.prepare_weight_for_1d_multiplexed_lazy(shape_1d, skip_1d, invalid_fill_1d);
            },
            [&](DensePackedLayer& layer) {
                layer.prepare_weight_for_1d_multiplexed(shape_1d, skip_1d, invalid_fill_1d);
            });
    } else if (feature_input.special_info_dim == 2) {
        Duo input_shape = feature_input.shape;
        Duo invalid_fill = feature_input.invalid_fill;
        _prepare_layer(
            key, move(dense),
            [&](DensePackedLayer& layer) {
                layer.prepare_weight_for_2d_multiplexed_lazy(input_shape, feature_input.special_skip, invalid_fill);
            },
            [&](DensePackedLayer& layer) {
                layer.prepare_weight_for_2d_multiplexed(input_shape, feature_input.special_skip, invalid_fill);
            });
    } else {
        _prepare_layer(
            key, move(dense),
            [&](DensePackedLayer& layer) { layer.prepare_weight_0d_skip_lazy(feature_input.skip[0]); },
            [&](DensePackedLayer& layer) { layer.prepare_weight_0d_skip(feature_input.skip[0]); });
    }
}

void InitInferenceProcess::_init_add_layer(const string& key, const json& layer, const string& block_input_feature) {
    FeatureNode feature_input0(json_features[layer["feature_input"][0].get<string>()]);
    FeatureNode feature_input1(json_features[layer["feature_input"][1].get<string>()]);
    FeatureNode feature_output(json_features[layer["feature_output"][0].get<string>()]);
    CkksParameter& param = *ckks_parameters_.at(feature_input0.ckks_parameter_id);
    auto add2d = MakeU<AddLayer>(*ckks_parameters_.at(feature_input0.ckks_parameter_id));
    add2d->target_ckks_scale = feature_output.ckks_scale;
    _prepare_layer(key, move(add2d));
}

void InitInferenceProcess::_init_mult_scalar_layer(const string& key,
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
    CkksParameter& param = *ckks_parameters_.at(feature_input0.ckks_parameter_id);

    double scale = layer["weight_scale"];
    auto weight = gen_random_array<1>({feature_input0.channel}, 1.0);
    for (int i = 0; i < feature_input0.channel; i++) {
        weight.set(i, scale);
    }
    auto mult_scalar = MakeU<MultScalarLayer>(param, feature_input0.shape, move(weight), feature_input0.skip,
                                              feature_input0.pack_channel_per_ciphertext, feature_input0.level,
                                              upsample_factor, block_expansion);
    _prepare_layer(
        key, move(mult_scalar), [](MultScalarLayer& layer) { layer.prepare_weight_lazy(); },
        [](MultScalarLayer& layer) { layer.prepare_weight(); });
}

void InitInferenceProcess::_init_drop_level_layer(const string& key, const json& layer) {
    FeatureNode feature_input0(json_features[layer["feature_input"][0].get<string>()]);
    CkksParameter& param = *ckks_parameters_.at(feature_input0.ckks_parameter_id);
    auto drop_level = MakeU<DropLevelLayer>();

    _prepare_layer(key, move(drop_level));
}

void InitInferenceProcess::_init_reshape_layer(const string& key, const json& layer) {
    FeatureNode feature_input0(json_features[layer["feature_input"][0].get<string>()]);

    auto reshape = MakeU<ReshapeLayer>(*ckks_parameters_.at(feature_input0.ckks_parameter_id));
    _prepare_layer(key, move(reshape));
}

void InitInferenceProcess::_init_concat_layer(const string& key, const json& layer) {
    auto concat = MakeU<ConcatLayer>();

    auto feature_inputs = layer["feature_input"].get<vector<string>>();
    vector<uint32_t> input_n_channels;
    bool has_uneven = false;
    string ckks_param_id;
    Duo shape = {0, 0};
    Duo skip = {1, 1};
    int level = 0;
    uint32_t pack = 0;

    for (const auto& fid : feature_inputs) {
        FeatureNode feat(json_features[fid]);
        input_n_channels.push_back(feat.channel);
        if (feat.channel % feat.pack_channel_per_ciphertext != 0) {
            has_uneven = true;
        }
        ckks_param_id = feat.ckks_parameter_id;
        shape = feat.shape;
        skip = feat.skip;
        level = feat.level;
        pack = feat.pack_channel_per_ciphertext;
    }

    if (has_uneven) {
        _prepare_layer(key, move(concat), [&](ConcatLayer& layer) {
            layer.prepare_mask_data(*ckks_parameters_.at(ckks_param_id), input_n_channels, pack, shape, skip, level);
        });
    } else {
        _prepare_layer(key, move(concat));
    }
}

void InitInferenceProcess::_init_upsample_layer(const string& key, const json& layer, const Duo& block_shape) {
    FeatureNode feature_input(json_features[layer["feature_input"][0].get<string>()]);
    FeatureNode feature_output(json_features[layer["feature_output"][0].get<string>()]);

    CkksParameter& param = *ckks_parameters_.at(feature_input.ckks_parameter_id);
    Duo block_expansion = {feature_input.shape[0] / block_shape[0], feature_input.shape[1] / block_shape[1]};
    Duo upsample_factor_in = {layer["upsample_factor_in"][0], layer["upsample_factor_in"][1]};

    auto upsample = MakeU<UpsampleLayer>(param, block_expansion, upsample_factor_in, feature_input.level,
                                         feature_input.channel, feature_input.pack_channel_per_ciphertext);
    _prepare_layer(key, move(upsample), [](UpsampleLayer& layer) { layer.prepare_data(); });
}

void InitInferenceProcess::_init_upsample_nearest_layer(const string& key, const json& layer) {
    FeatureNode feature_input(json_features[layer["feature_input"][0].get<string>()]);
    FeatureNode feature_output(json_features[layer["feature_output"][0].get<string>()]);

    CkksParameter& param = *ckks_parameters_.at(feature_input.ckks_parameter_id);
    Duo upsample_factor_in = {layer["upsample_factor"][0], layer["upsample_factor"][1]};

    auto upsample_nearest =
        MakeU<UpsampleNearestLayer>(param, feature_input.shape, feature_input.skip, upsample_factor_in,
                                    feature_input.pack_channel_per_ciphertext, feature_input.level);
    _prepare_layer(key, move(upsample_nearest));
}

void InitInferenceProcess::_init_multiplexed_conv_layer(const string& key,
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
    if (key.find("ConvTranspose") != std::string::npos) {
        weight = _load_h5_tensor<4>(
            layer, h5_file, "weight",
            {channel_input, feature_output.channel, layer["kernel_shape"][0], layer["kernel_shape"][1]});
        weight = transpose_weight(weight);
    } else {
        weight = _load_h5_tensor<4>(
            layer, h5_file, "weight",
            {feature_output.channel, channel_input, layer["kernel_shape"][0], layer["kernel_shape"][1]});
    }
    auto bias = _load_h5_tensor<1>(layer, h5_file, "bias", {feature_output.channel});

    CkksParameter& param = *ckks_parameters_.at(feature_input.ckks_parameter_id);
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
            auto inv_conv_layer =
                MakeU<InverseMultiplexedConv2DLayer>(param, feature_input.shape, move(weight), move(bias), padding,
                                                     stride, block_shape_in, feature_input.level, residual_scale);
            _prepare_layer(key, move(inv_conv_layer));
        } else {
            auto mux_conv_layer = MakeU<MultiplexedConv2DPackedLayer>(
                param, feature_input.shape, move(weight), move(bias), stride, feature_input.skip,
                feature_input.pack_channel_per_ciphertext, feature_input.level, residual_scale, upsample_factor_in);
            _prepare_layer(key, move(mux_conv_layer));
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
            auto inv_dw_conv_layer = MakeU<InverseMultiplexedConv2DLayerDepthwise>(
                param, feature_input.shape, move(weight), move(bias), padding, stride, block_shape_in,
                feature_input.level, residual_scale);
            _prepare_layer(key, move(inv_dw_conv_layer));
        } else {
            auto mux_dw_layer = MakeU<MultiplexedConv2DPackedLayerDepthwise>(
                param, feature_input.shape, move(weight), move(bias), stride, feature_input.skip,
                feature_input.pack_channel_per_ciphertext, feature_input.level, residual_scale);
            _prepare_layer(key, move(mux_dw_layer));
        }
    }
}

void InitInferenceProcess::_init_poly_relu_layer(const string& key,
                                                 const json& layer,
                                                 const hid_t& h5_file,
                                                 bool is_absorb,
                                                 const Duo& block_shape_in) {
    FeatureNode feature_input(json_features[layer["feature_input"][0].get<string>()]);
    FeatureNode feature_output(json_features[layer["feature_output"][0].get<string>()]);
    uint32_t order = layer["order"];
    auto weight = _load_h5_tensor<2>(layer, h5_file, "weight", {order + 1, feature_input.channel});

    CkksParameter& param = *ckks_parameters_.at(feature_input.ckks_parameter_id);

    if (feature_input.dim == 1) {
        int skip_val = feature_input.skip[0];
        int shape_val = feature_input.shape[0];
        string style = layer.value("style", string("ordinary"));
        auto layer_poly_relu = MakeU<PolyRelu1D>(param, move(weight), feature_input.level, order, skip_val, shape_val);
        if (style == "multiplexed") {
            _prepare_layer(
                key, move(layer_poly_relu), [&](PolyRelu1D& layer) { layer.prepare_weight_bsgs_mux_lazy(); },
                [&](PolyRelu1D& layer) { layer.prepare_weight_bsgs_mux(); });
        } else {
            _prepare_layer(key, move(layer_poly_relu));
        }
    } else if (feature_input.dim == 0) {
        if (is_absorb) {
            weight = _load_h5_tensor<2>(layer, h5_file, "weight", {order, feature_input.channel});
        }
        int ciphertext_skip = feature_input.skip[0];
        auto layer_poly_relu = MakeU<PolyRelu0D>(param, move(weight), feature_input.level, order, ciphertext_skip);
        if (feature_input.invalid_fill[0] == 0 || feature_input.invalid_fill[1] == 0) {
            // Mode 1: direct 0D pack — channel ch at slot ch * ciphertext_skip
            _prepare_layer(key, move(layer_poly_relu));
        } else {
            // Mode 2: from reshape of 2D with shape>1 — mirrors DensePackedLayer multiplexed path
            Duo input_shape = feature_input.shape;
            Duo input_skip;
            input_skip[0] = feature_input.special_skip[0] / input_shape[0];
            input_skip[1] = feature_input.special_skip[1] / input_shape[1];
            _prepare_layer(
                key, move(layer_poly_relu),
                [&](PolyRelu0D& layer) { layer.prepare_weight_2d_multiplexed_lazy(input_shape, input_skip); },
                [&](PolyRelu0D& layer) { layer.prepare_weight_2d_multiplexed(input_shape, input_skip); });
        }
    } else {
        Duo zero_skip_in = {layer["zero_skip"][0], layer["zero_skip"][1]};
        if (is_absorb) {
            weight = _load_h5_tensor<2>(layer, h5_file, "weight", {order, feature_input.channel});
        }
        Duo block_expansion = {div_ceil(feature_input.shape[0], block_shape_in[0]),
                               div_ceil(feature_input.shape[1], block_shape_in[1])};
        auto layer_poly_relu = MakeU<PolyRelu2D>(param, feature_input.shape, order, move(weight), feature_input.skip,
                                                 feature_input.pack_channel_per_ciphertext, feature_input.level,
                                                 zero_skip_in, block_expansion, pack_style != "multiplexed");
        if (is_absorb) {
            _prepare_layer(key, move(layer_poly_relu));
        } else {
            _prepare_layer(
                key, move(layer_poly_relu), [&](PolyRelu2D& layer) { layer.prepare_weight_bsgs_lazy(); },
                [&](PolyRelu2D& layer) { layer.prepare_weight_bsgs(); });
        }
    }
}

void InitInferenceProcess::_init_fhe_avgpool_layer(const string& key,
                                                   const json& layer,
                                                   const bool& is_adaptive,
                                                   const Duo& block_shape) {
    FeatureNode feature_input(json_features[layer["feature_input"][0].get<string>()]);
    FeatureNode feature_output(json_features[layer["feature_output"][0].get<string>()]);
    CkksParameter& param = *ckks_parameters_.at(feature_input.ckks_parameter_id);
    Duo block_expansion = {feature_input.shape[0] / block_shape[0], feature_input.shape[1] / block_shape[1]};
    Duo stride = {layer["stride"][0], layer["stride"][1]};
    bool is_big_size = layer["is_big_size"];
    if (is_big_size) {
        auto avgpool = MakeU<Avgpool2DLayer>(feature_input.shape, stride);
        // Check if output < block_shape (repack needed)
        Duo output_shape = {feature_input.shape[0] / stride[0], feature_input.shape[1] / stride[1]};
        if (output_shape[0] < block_shape[0] || output_shape[1] < block_shape[1]) {
            Duo second_stage_stride = {block_shape[0] / output_shape[0], block_shape[1] / output_shape[1]};
            _prepare_layer(key, move(avgpool), [&](Avgpool2DLayer& layer) {
                layer.prepare_weight_repack(param, feature_input.channel, feature_input.level, second_stage_stride,
                                            block_shape);
            });
        } else {
            _prepare_layer(key, move(avgpool));
        }
    } else {
        if (is_adaptive) {
            auto avgpool = MakeU<Avgpool2DLayer>(feature_input.shape, stride);
            _prepare_layer(key, move(avgpool));
        } else {
            auto avgpool = MakeU<Avgpool2DLayer>(feature_input.shape, stride);
            _prepare_layer(
                key, move(avgpool),
                [&](Avgpool2DLayer& layer) {
                    layer.prepare_weight_lazy(param, feature_input.pack_channel_per_ciphertext, feature_input.channel,
                                              feature_input.level, feature_input.skip, feature_input.shape);
                },
                [&](Avgpool2DLayer& layer) {
                    layer.prepare_weight(param, feature_input.pack_channel_per_ciphertext, feature_input.channel,
                                         feature_input.level, feature_input.skip, feature_input.shape);
                });
        }
    }
}

void InitInferenceProcess::_init_fhe_avgpool1d_layer(const string& key, const json& layer, const bool& is_adaptive) {
    FeatureNode feature_input(json_features[layer["feature_input"][0].get<string>()]);
    CkksParameter& param = *ckks_parameters_.at(feature_input.ckks_parameter_id);
    uint32_t stride = layer["stride"][0];
    bool is_big_size = layer["is_big_size"];
    if (is_big_size) {
        auto avgpool = MakeU<Avgpool1DLayer>(feature_input.shape[0], stride);
        _prepare_layer(key, move(avgpool));
    } else {
        if (is_adaptive) {
            auto avgpool = MakeU<Avgpool1DLayer>(feature_input.shape[0], stride);
            _prepare_layer(key, move(avgpool));
        } else {
            auto avgpool = MakeU<Avgpool1DLayer>(feature_input.shape[0], stride);
            _prepare_layer(
                key, move(avgpool),
                [&](Avgpool1DLayer& layer) {
                    layer.prepare_weight_lazy(param, feature_input.pack_channel_per_ciphertext, feature_input.channel,
                                              feature_input.level, feature_input.skip[0], feature_input.shape[0]);
                },
                [&](Avgpool1DLayer& layer) {
                    layer.prepare_weight(param, feature_input.pack_channel_per_ciphertext, feature_input.channel,
                                         feature_input.level, feature_input.skip[0], feature_input.shape[0]);
                });
        }
    }
}

void InitInferenceProcess::load_model_prepare() {
    json_data = read_json(project_path / "nn_layers_ct_0.json");
    json_features = json_data.at("feature");
    json_layers = json_data.at("layer");
    string block_input_feature = json_data["input_feature"][0];

    string h5_filename = project_path / "model_parameters.h5";
    hid_t h5_file = H5Fopen(h5_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    for (auto& layer : json_layers.items()) {
        const string& key = layer.key();
        const json& value = layer.value();
        const string& layer_type = value["type"].get<string>();
        if (layer_type == "conv2d") {
            if (pack_style == "multiplexed") {
                _init_multiplexed_conv_layer(key, value, h5_file, block_shape);
            } else {
                _init_conv_layer(key, value, h5_file);
            }
        } else if (layer_type == "square2d") {
            _init_square_layer(key, value, h5_file);
        } else if (layer_type == "fc0") {
            _init_dense_layer(key, value, h5_file);
        } else if (layer_type == "add2d") {
            _init_add_layer(key, value, block_input_feature);
        } else if (layer_type == "reshape") {
            _init_reshape_layer(key, value);
        } else if (layer_type == "drop_level") {
            _init_drop_level_layer(key, value);
        } else if (layer_type == "concat2d") {
            _init_concat_layer(key, value);
        } else if (layer_type == "upsample") {
            _init_upsample_layer(key, value, block_shape);
        } else if (layer_type == "upsample_nearest") {
            _init_upsample_nearest_layer(key, value);
        } else if (layer_type == "mult_scalar") {
            _init_mult_scalar_layer(key, value, h5_file, block_shape);
        } else if (layer_type == "poly_relu2d" || layer_type == "polyact") {
            _init_poly_relu_layer(key, value, h5_file, is_absorb_polyrelu, block_shape);
        } else if (layer_type == "avgpool2d") {
            bool is_adaptive_avgpool = value["is_adaptive_avgpool"];
            _init_fhe_avgpool_layer(key, value, is_adaptive_avgpool, block_shape);
        } else if (layer_type == "avgpool1d") {
            bool is_adaptive_avgpool = value["is_adaptive_avgpool"];
            _init_fhe_avgpool1d_layer(key, value, is_adaptive_avgpool);
        } else if (layer_type == "conv1d") {
            _init_conv1d_layer(key, value, h5_file);
        }
    }
    H5Fclose(h5_file);
}

void InferenceProcess::run_task_sdk(bool enable_mpc) {
    // Reset time statistics for each request
    fp->total_fhe_time = 0.0;
    fp->total_fpga_time = 0.0;

    const json& json_features = fp->json_features;
    json json_layers = fp->json_layers;
    string block_input_feature = fp->json_data["input_feature"][0];
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
                if (intermediate_result_.find(fi) == intermediate_result_.end()) {
                    tag = true;
                    break;
                }
            }
            if (tag == true) {
                continue;
            }

            const string& feature_output_id = feature_output[0];
            FeatureNode feature_input_node(json_features[feature_input[0]]);
            UPtr<FeatureEncrypted> result;
            auto& context = *ckks_contexts.at(feature_input_node.ckks_parameter_id);
            cout << ">>> LAYER: " << key << " type=" << layer_type << endl;
            if (layer_type == "conv2d") {
                fhe_timer.start();
                const FeatureEncrypted& feature_node = _get_feature(feature_input[0]);
                if (feature_node.dim == 2) {
                    const Feature2DEncrypted& input2D = dynamic_cast<const Feature2DEncrypted&>(feature_node);
                    if (fp->pack_style == "multiplexed") {
                        if (layer.value()["groups"] == 1) {
                            bool is_big_size = layer.value()["is_big_size"];
                            if (is_big_size) {
                                result = MakeU<Feature2DEncrypted>(
                                    fp->get_layer<InverseMultiplexedConv2DLayer>(key).run(context, input2D));
                            } else {
                                result = MakeU<Feature2DEncrypted>(
                                    fp->get_layer<MultiplexedConv2DPackedLayer>(key).run_for_post_skip_rotation(
                                        context, input2D));
                            }
                        } else {
                            bool is_big_size = layer.value()["is_big_size"];
                            if (is_big_size) {
                                result = MakeU<Feature2DEncrypted>(
                                    fp->get_layer<InverseMultiplexedConv2DLayerDepthwise>(key).run(context, input2D));
                            } else {
                                result = MakeU<Feature2DEncrypted>(
                                    fp->get_layer<MultiplexedConv2DPackedLayerDepthwise>(key).run(context, input2D));
                            }
                        }
                    } else {
                        if (layer.value()["groups"] == 1) {
                            result =
                                MakeU<Feature2DEncrypted>(fp->get_layer<Conv2DPackedLayer>(key).run(context, input2D));
                        } else {
                            result = MakeU<Feature2DEncrypted>(
                                fp->get_layer<Conv2DPackedDepthwiseLayer>(key).run(context, input2D));
                        }
                        const Feature2DEncrypted& res = dynamic_cast<const Feature2DEncrypted&>(*result);
                    }
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted");
                }
                fhe_timer.stop();
            } else if (layer_type == "bootstrapping") {
                const int maximum_refreshed_level = 9;
                const FeatureEncrypted& feature_node = _get_feature(feature_input[0]);
                FeatureNode output_feature_node(json_features[feature_output_id]);
                if (feature_node.dim == 2) {
                    const Feature2DEncrypted& input2D = dynamic_cast<const Feature2DEncrypted&>(feature_node);
                    Feature2DEncrypted refresh_result = input2D.refresh_ciphertext();
                    if (maximum_refreshed_level > output_feature_node.level) {
                        result = MakeU<Feature2DEncrypted>(
                            refresh_result.drop_level(maximum_refreshed_level - output_feature_node.level));
                    } else {
                        result = MakeU<Feature2DEncrypted>(move(refresh_result));
                    }
                } else if (feature_node.dim == 0) {
                    const Feature0DEncrypted& input0D = dynamic_cast<const Feature0DEncrypted&>(feature_node);
                    Feature0DEncrypted refresh_result = input0D.refresh_ciphertext();
                    if (maximum_refreshed_level > output_feature_node.level) {
                        result = MakeU<Feature0DEncrypted>(
                            refresh_result.drop_level(maximum_refreshed_level - output_feature_node.level));
                    } else {
                        result = MakeU<Feature0DEncrypted>(move(refresh_result));
                    }
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted or Feature0DEncrypted");
                }
            } else if (layer_type == "batchnorm" || layer_type == "batchnorm2d" || layer_type == "dropout" ||
                       layer_type == "mul" || layer_type == "identity") {
                const FeatureEncrypted& feature_node = _get_feature(feature_input[0]);
                if (feature_node.dim == 2) {
                    const Feature2DEncrypted& input2D = dynamic_cast<const Feature2DEncrypted&>(feature_node);
                    result = MakeU<Feature2DEncrypted>(input2D.copy());
                } else if (feature_node.dim == 0) {
                    const Feature0DEncrypted& input0D = dynamic_cast<const Feature0DEncrypted&>(feature_node);
                    result = MakeU<Feature0DEncrypted>(input0D.copy());
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted or Feature0DEncrypted");
                }
            } else if (layer_type == "square2d") {
                fhe_timer.start();
                const FeatureEncrypted& feature_node = _get_feature(feature_input[0]);

                if (feature_node.dim == 2) {
                    const Feature2DEncrypted& input2D = dynamic_cast<const Feature2DEncrypted&>(feature_node);
                    result = MakeU<Feature2DEncrypted>(fp->get_layer<SquareLayer>(key).call(context, input2D));
                } else if (feature_node.dim == 0) {
                    const Feature0DEncrypted& input0D = dynamic_cast<const Feature0DEncrypted&>(feature_node);
                    result = MakeU<Feature0DEncrypted>(fp->get_layer<SquareLayer>(key).call(context, input0D));
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted ");
                }
                fhe_timer.stop();
            } else if (layer_type == "add2d") {
                fhe_timer.start();
                double target_ckks_scale = json_features[feature_output[0]]["ckks_scale"];
                const Feature2DEncrypted& input0 =
                    dynamic_cast<const Feature2DEncrypted&>(_get_feature(feature_input[0]));
                const Feature2DEncrypted& input1 =
                    dynamic_cast<const Feature2DEncrypted&>(_get_feature(feature_input[1]));
                if (input0.dim == 2 && input1.dim == 2) {
                    result = MakeU<Feature2DEncrypted>(fp->get_layer<AddLayer>(key).run(context, input0, input1));
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted");
                }
                fhe_timer.stop();
            } else if (layer_type == "mult_scalar") {
                fhe_timer.start();
                const Feature2DEncrypted& input0 =
                    dynamic_cast<const Feature2DEncrypted&>(_get_feature(feature_input[0]));

                if (input0.dim == 2) {
                    auto res = fp->get_layer<MultScalarLayer>(key).run(context, input0);
                    result = MakeU<Feature2DEncrypted>(move(res));
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted");
                }
                fhe_timer.stop();
            } else if (layer_type == "drop_level") {
                FeatureNode d_input_node(json_features[feature_input[0]]);
                FeatureNode d_output_node(json_features[feature_output[0]]);
                int n_level_to_drop = d_input_node.level - d_output_node.level;
                const FeatureEncrypted& feature_node = _get_feature(feature_input[0]);
                if (feature_node.dim == 2) {
                    const Feature2DEncrypted& input2D = dynamic_cast<const Feature2DEncrypted&>(feature_node);
                    result = MakeU<Feature2DEncrypted>(input2D.drop_level(n_level_to_drop));
                } else if (feature_node.dim == 0) {
                    const Feature0DEncrypted& input0D = dynamic_cast<const Feature0DEncrypted&>(feature_node);
                    result = MakeU<Feature0DEncrypted>(input0D.drop_level(n_level_to_drop));
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted or Feature0DEncrypted");
                }
            } else if (layer_type == "fc0" || layer_type == "fc1") {
                fhe_timer.start();
                FeatureNode d_input_node(json_features[feature_input[0]]);
                const FeatureEncrypted& feature_node = _get_feature(feature_input[0]);
                if (feature_node.dim == 0) {
                    const Feature0DEncrypted& input0D = dynamic_cast<const Feature0DEncrypted&>(feature_node);
                    if (d_input_node.special_info_dim == 1) {
                        result = MakeU<Feature0DEncrypted>(
                            fp->get_layer<DensePackedLayer>(key).run_1d_multiplexed(context, input0D));
                    } else if (d_input_node.special_info_dim == 2) {
                        result = MakeU<Feature0DEncrypted>(
                            fp->get_layer<DensePackedLayer>(key).run_2d_multiplexed(context, input0D));
                    } else {
                        result = MakeU<Feature0DEncrypted>(
                            fp->get_layer<DensePackedLayer>(key).run_0d_skip(context, input0D));
                    }
                } else {
                    throw runtime_error("input is not available, expect Feature0DEncrypted");
                }
                fhe_timer.stop();
            } else if (layer_type == "reshape") {
                fhe_timer.start();
                const FeatureEncrypted& feature_node = _get_feature(feature_input[0]);
                if (feature_node.dim == 2) {
                    const Feature2DEncrypted& input2D = dynamic_cast<const Feature2DEncrypted&>(feature_node);
                    result = MakeU<Feature0DEncrypted>(fp->get_layer<ReshapeLayer>(key).call(context, input2D));
                } else if (feature_node.dim == 1) {
                    const Feature1DEncrypted& input1D = dynamic_cast<const Feature1DEncrypted&>(feature_node);
                    Feature0DEncrypted out(&context, input1D.level);
                    for (int i = 0; i < (int)input1D.data.size(); i++) {
                        out.data.push_back(input1D.data[i].copy());
                    }
                    out.dim = 0;
                    out.skip = input1D.shape * input1D.skip;
                    out.level = input1D.level;
                    out.n_channel = input1D.n_channel;
                    out.n_channel_per_ct = input1D.n_channel_per_ct;
                    result = MakeU<Feature0DEncrypted>(move(out));
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted or Feature1DEncrypted");
                }
                fhe_timer.stop();
            } else if (layer_type == "avgpool2d") {
                fhe_timer.start();
                const FeatureEncrypted& feature_node = _get_feature(feature_input[0]);
                if (feature_node.dim == 2) {
                    const Feature2DEncrypted& input2D = dynamic_cast<const Feature2DEncrypted&>(feature_node);

                    if (fp->pack_style == "multiplexed") {
                        bool is_adaptive_avgpool = layer.value()["is_adaptive_avgpool"];
                        bool is_big_size = layer.value()["is_big_size"];
                        if (is_adaptive_avgpool) {
                            result = MakeU<Feature2DEncrypted>(
                                fp->get_layer<Avgpool2DLayer>(key).run_adaptive_avgpool(context, input2D));
                        } else {
                            if (is_big_size) {
                                Duo block_expansion = {feature_input_node.shape[0] / block_shape[0],
                                                       feature_input_node.shape[1] / block_shape[1]};
                                result = MakeU<Feature2DEncrypted>(fp->get_layer<Avgpool2DLayer>(key).run_split_avgpool(
                                    context, input2D, block_expansion));
                            } else {
                                result = MakeU<Feature2DEncrypted>(
                                    fp->get_layer<Avgpool2DLayer>(key).run_multiplexed_avgpool(context, input2D));
                            }
                        }
                    } else {
                        result = MakeU<Feature2DEncrypted>(fp->get_layer<Avgpool2DLayer>(key).run(context, input2D));
                    }
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted");
                }
                fhe_timer.stop();
            } else if (layer_type == "avgpool1d") {
                fhe_timer.start();
                const FeatureEncrypted& feature_node = _get_feature(feature_input[0]);
                if (feature_node.dim == 1) {
                    const Feature1DEncrypted& input1D = dynamic_cast<const Feature1DEncrypted&>(feature_node);

                    bool is_adaptive_avgpool = layer.value()["is_adaptive_avgpool"];
                    bool is_big_size = layer.value()["is_big_size"];
                    if (is_adaptive_avgpool) {
                        result = MakeU<Feature1DEncrypted>(
                            fp->get_layer<Avgpool1DLayer>(key).run_adaptive_avgpool(context, input1D));
                    } else {
                        if (is_big_size) {
                            throw runtime_error("avgpool1d does not support big_size mode");
                        } else {
                            result = MakeU<Feature1DEncrypted>(
                                fp->get_layer<Avgpool1DLayer>(key).run_multiplexed_avgpool(context, input1D));
                        }
                    }
                } else {
                    throw runtime_error("input is not available, expect Feature1DEncrypted");
                }
                fhe_timer.stop();
            } else if (layer_type == "concat2d") {
                fhe_timer.start();
                vector<Feature2DEncrypted> inputs;
                for (const auto& input_name : feature_input) {
                    const FeatureEncrypted& input_feature_node = _get_feature(input_name);
                    if (input_feature_node.dim == 2) {
                        const Feature2DEncrypted& input2D = dynamic_cast<const Feature2DEncrypted&>(input_feature_node);
                        inputs.emplace_back(input2D.copy());
                    } else {
                        throw runtime_error("input is not available, expect Feature2DEncrypted");
                    }
                }
                result =
                    MakeU<Feature2DEncrypted>(fp->get_layer<ConcatLayer>(key).run_multiple_inputs(context, inputs));
                fhe_timer.stop();
            } else if (layer_type == "upsample") {
                fhe_timer.start();
                const FeatureEncrypted& feature_node = _get_feature(feature_input[0]);
                if (feature_node.dim == 2) {
                    const Feature2DEncrypted& input2D = dynamic_cast<const Feature2DEncrypted&>(feature_node);
                    result = MakeU<Feature2DEncrypted>(fp->get_layer<UpsampleLayer>(key).run(context, input2D));
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted");
                }
                fhe_timer.stop();
            } else if (layer_type == "upsample_nearest") {
                fhe_timer.start();
                const FeatureEncrypted& feature_node = _get_feature(feature_input[0]);
                if (feature_node.dim == 2) {
                    const Feature2DEncrypted& input2D = dynamic_cast<const Feature2DEncrypted&>(feature_node);
                    result = MakeU<Feature2DEncrypted>(fp->get_layer<UpsampleNearestLayer>(key).run(context, input2D));
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted");
                }
                fhe_timer.stop();
            } else if (layer_type == "poly_relu2d" || layer_type == "polyact") {
                fhe_timer.start();
                const FeatureEncrypted& feature_node = _get_feature(feature_input[0]);
                if (feature_node.dim == 2) {
                    const Feature2DEncrypted& input2D = dynamic_cast<const Feature2DEncrypted&>(feature_node);
                    if (fp->is_absorb_polyrelu) {
                        result = MakeU<Feature2DEncrypted>(fp->get_layer<PolyRelu2D>(key).run(context, input2D));
                    } else {
                        result = MakeU<Feature2DEncrypted>(fp->get_layer<PolyRelu2D>(key).run_bsgs(context, input2D));
                    }
                } else if (feature_node.dim == 0) {
                    const Feature0DEncrypted& input0D = dynamic_cast<const Feature0DEncrypted&>(feature_node);
                    result = MakeU<Feature0DEncrypted>(fp->get_layer<PolyRelu0D>(key).run(context, input0D));
                } else if (feature_node.dim == 1) {
                    const Feature1DEncrypted& input1D = dynamic_cast<const Feature1DEncrypted&>(feature_node);
                    result = MakeU<Feature1DEncrypted>(fp->get_layer<PolyRelu1D>(key).run(context, input1D));
                } else {
                    throw runtime_error("input is not available, expect Feature2DEncrypted or Feature0DEncrypted");
                }
                fhe_timer.stop();
            } else if (layer_type == "conv1d") {
                fhe_timer.start();
                const FeatureEncrypted& feature_node = _get_feature(feature_input[0]);
                if (feature_node.dim == 1) {
                    const Feature1DEncrypted& input1D = dynamic_cast<const Feature1DEncrypted&>(feature_node);
                    string style = layer.value().value("style", string("ordinary"));
                    if (style == "multiplexed") {
                        int groups = layer.value().value("groups", 1);
                        if (groups == (int)input1D.n_channel && groups != 1) {
                            result = MakeU<Feature1DEncrypted>(fp->get_layer<MultiplexedDWConv1DPackedLayer>(key).run(
                                context, const_cast<Feature1DEncrypted&>(input1D)));
                        } else {
                            result = MakeU<Feature1DEncrypted>(fp->get_layer<MultiplexedConv1DPackedLayer>(key).run(
                                context, const_cast<Feature1DEncrypted&>(input1D)));
                        }
                    } else {
                        result = MakeU<Feature1DEncrypted>(fp->get_layer<Conv1DPackedLayer>(key).run(
                            context, const_cast<Feature1DEncrypted&>(input1D)));
                    }
                } else {
                    throw runtime_error("input is not available, expect Feature1DEncrypted");
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

    const json& json_features = fp->json_features;
    const json& json_layers = fp->json_layers;
    string block_input_feature = fp->json_data["input_feature"][0];
    Duo block_shape = fp->block_shape;

    vector<CxxVectorArgument> cxx_args;
    UPtr<FeatureEncrypted> result;

    vector<vector<CkksCiphertext>> ct_data(fp->json_data["input_feature"].size());
    for (int i = 0; i < fp->json_data["input_feature"].size(); i++) {
        auto ki = fp->json_data["input_feature"][i];
        FeatureNode feature_input(json_features[ki.get<string>()]);
        if (feature_input.dim == 2) {
            const Feature2DEncrypted& input = dynamic_cast<const Feature2DEncrypted&>(_get_feature(ki));
            auto _size = input.data.size();
            for (int j = 0; j < _size; j++) {
                ct_data[i].push_back(input.data[j].copy());
            }
            cxx_args.push_back(CxxVectorArgument{ki, &ct_data[i]});
        }
        if (feature_input.dim == 0) {
            const Feature0DEncrypted& input = dynamic_cast<const Feature0DEncrypted&>(_get_feature(ki));
            for (int j = 0; j < input.data.size(); j++) {
                ct_data[i].push_back(input.data[j].copy());
            }
            cxx_args.push_back(CxxVectorArgument{ki, &ct_data[i]});
        }
        if (feature_input.dim == 1) {
            const Feature1DEncrypted& input = dynamic_cast<const Feature1DEncrypted&>(_get_feature(ki));
            for (int j = 0; j < input.data.size(); j++) {
                ct_data[i].push_back(input.data[j].copy());
            }
            cxx_args.push_back(CxxVectorArgument{ki, &ct_data});
        }
    }

    for (const auto& layer : json_layers.items()) {
        const string& key = layer.key();

        const string& layer_type = layer.value()["type"].get<string>();
        auto feature_input = layer.value()["feature_input"].get<vector<string>>();
        auto feature_output = layer.value()["feature_output"].get<vector<string>>();

        const string& feature_output_id = feature_output[0];
        FeatureNode feature_input_node(json_features[feature_input[0]]);
        UPtr<FeatureEncrypted> result;
        auto& context = *ckks_contexts.at(feature_input_node.ckks_parameter_id);
        if (layer_type == "conv2d") {
            FeatureNode d_input_node(json_features[feature_input[0]]);
            if (d_input_node.dim == 2) {
                if (layer.value()["groups"] == 1) {
                    bool is_big_size = layer.value()["is_big_size"];
                    if (is_big_size) {
                        cxx_args.push_back(CxxVectorArgument{
                            "convw_" + key, &(fp->get_layer<InverseMultiplexedConv2DLayer>(key).weight_pt)});
                        cxx_args.push_back(CxxVectorArgument{
                            "convb_" + key, &(fp->get_layer<InverseMultiplexedConv2DLayer>(key).bias_pt)});
                        if (fp->get_layer<InverseMultiplexedConv2DLayer>(key).need_repack) {
                            cxx_args.push_back(
                                CxxVectorArgument{"repack_mask_" + key,
                                                  &(fp->get_layer<InverseMultiplexedConv2DLayer>(key).repack_mask_pt)});
                        }
                    } else {
                        if (fp->pack_style == "multiplexed") {
                            if (layer.value()["stride"][0] == 1 and d_input_node.skip[0] == 1) {
                            } else {
                                cxx_args.push_back(CxxVectorArgument{
                                    "convm_" + key, &(fp->get_layer<MultiplexedConv2DPackedLayer>(key).mask_pt)});
                            }
                            cxx_args.push_back(CxxVectorArgument{
                                "convw_" + key, &(fp->get_layer<MultiplexedConv2DPackedLayer>(key).weight_pt)});
                            cxx_args.push_back(CxxVectorArgument{
                                "convb_" + key, &(fp->get_layer<MultiplexedConv2DPackedLayer>(key).bias_pt)});
                        } else {
                            cxx_args.push_back(
                                CxxVectorArgument{"convw_" + key, &(fp->get_layer<Conv2DPackedLayer>(key).weight_pt_)});
                            cxx_args.push_back(
                                CxxVectorArgument{"convb_" + key, &(fp->get_layer<Conv2DPackedLayer>(key).bias_pt_)});
                        }
                    }
                } else {
                    bool is_big_size = layer.value()["is_big_size"];
                    if (is_big_size) {
                        cxx_args.push_back(CxxVectorArgument{
                            "convw_" + key, &(fp->get_layer<InverseMultiplexedConv2DLayerDepthwise>(key).weight_pt)});
                        cxx_args.push_back(CxxVectorArgument{
                            "convb_" + key, &(fp->get_layer<InverseMultiplexedConv2DLayerDepthwise>(key).bias_pt)});
                        if (fp->get_layer<InverseMultiplexedConv2DLayerDepthwise>(key).need_repack) {
                            cxx_args.push_back(CxxVectorArgument{
                                "repack_mask_" + key,
                                &(fp->get_layer<InverseMultiplexedConv2DLayerDepthwise>(key).repack_mask_pt)});
                        }
                    } else if (fp->pack_style == "multiplexed") {
                        if (layer.value()["stride"][0] == 1) {
                        } else {
                            cxx_args.push_back(CxxVectorArgument{
                                "convm_" + key, &(fp->get_layer<MultiplexedConv2DPackedLayerDepthwise>(key).mask_pt)});
                        }
                        cxx_args.push_back(CxxVectorArgument{
                            "convw_" + key, &(fp->get_layer<MultiplexedConv2DPackedLayerDepthwise>(key).weight_pt)});
                        cxx_args.push_back(CxxVectorArgument{
                            "convb_" + key, &(fp->get_layer<MultiplexedConv2DPackedLayerDepthwise>(key).bias_pt)});
                    } else {
                        cxx_args.push_back(CxxVectorArgument{
                            "convw_" + key, &(fp->get_layer<Conv2DPackedDepthwiseLayer>(key).weight_pt_)});
                        cxx_args.push_back(CxxVectorArgument{
                            "convb_" + key, &(fp->get_layer<Conv2DPackedDepthwiseLayer>(key).bias_pt_)});
                    }
                }
            } else {
                throw runtime_error("input is not available, expect Feature2DEncrypted");
            }
        } else if (layer_type == "add2d") {
            continue;
        } else if (layer_type == "concat2d") {
            if (!fp->get_layer<ConcatLayer>(key).mask_pt.empty()) {
                cxx_args.push_back(CxxVectorArgument{"concat_mask_" + key, &(fp->get_layer<ConcatLayer>(key).mask_pt)});
            }
        } else if (layer_type == "fc0" || layer_type == "fc1") {
            cxx_args.push_back(CxxVectorArgument{"densew_" + key, &(fp->get_layer<DensePackedLayer>(key).weight_pt)});
            cxx_args.push_back(CxxVectorArgument{"denseb_" + key, &(fp->get_layer<DensePackedLayer>(key).bias_pt)});
        } else if (layer_type == "avgpool2d") {
            FeatureNode d_input_node(json_features[feature_input[0]]);
            if (d_input_node.dim == 2) {
                bool is_adaptive_avgpool = layer.value()["is_adaptive_avgpool"];
                bool is_big_size = layer.value()["is_big_size"];
                if (is_adaptive_avgpool) {
                    continue;
                } else {
                    if (is_big_size) {
                        if (fp->get_layer<Avgpool2DLayer>(key).need_repack) {
                            cxx_args.push_back(CxxVectorArgument{"repack_mask_" + key,
                                                                 &(fp->get_layer<Avgpool2DLayer>(key).repack_mask_pt)});
                        } else {
                            continue;
                        }
                    } else {
                        cxx_args.push_back(CxxVectorArgument{"select_tensor_pt_" + key,
                                                             &(fp->get_layer<Avgpool2DLayer>(key).select_tensor_pt)});
                    }
                }
            } else {
                throw runtime_error("input is not available, expect Feature2DEncrypted");
            }
        } else if (layer_type == "avgpool1d") {
            bool is_adaptive_avgpool = layer.value()["is_adaptive_avgpool"];
            bool is_big_size = layer.value()["is_big_size"];
            if (is_adaptive_avgpool || is_big_size) {
                continue;
            } else {
                cxx_args.push_back(CxxVectorArgument{"select_tensor_pt_" + key,
                                                     &(fp->get_layer<Avgpool1DLayer>(key).select_tensor_pt)});
            }
        } else if (layer_type == "poly_relu2d" || layer_type == "polyact") {
            FeatureNode d_input_node(json_features[feature_input[0]]);
            if (d_input_node.dim == 0) {
                for (int i = 0; i < fp->get_layer<PolyRelu0D>(key).weight_pt.size(); i++) {
                    cxx_args.push_back(CxxVectorArgument{"poly_reluw_" + key + "_" + to_string(i),
                                                         &(fp->get_layer<PolyRelu0D>(key).weight_pt[i])});
                }
            } else if (d_input_node.dim == 1) {
                for (int i = 0; i < fp->get_layer<PolyRelu1D>(key).weight_pt.size(); i++) {
                    cxx_args.push_back(CxxVectorArgument{"poly_reluw_" + key + "_" + to_string(i),
                                                         &(fp->get_layer<PolyRelu1D>(key).weight_pt[i])});
                }
            } else {
                for (int i = 0; i < fp->get_layer<PolyRelu2D>(key).weight_pt.size(); i++) {
                    cxx_args.push_back(CxxVectorArgument{"poly_reluw_" + key + "_" + to_string(i),
                                                         &(fp->get_layer<PolyRelu2D>(key).weight_pt[i])});
                }
            }
        } else if (layer_type == "mult_scalar") {
            cxx_args.push_back(
                CxxVectorArgument{"mult_scalar_" + key, &(fp->get_layer<MultScalarLayer>(key).weight_pt)});
        } else if (layer_type == "conv1d") {
            string style = layer.value().value("style", string("ordinary"));
            if (style == "multiplexed") {
                int groups = layer.value().value("groups", 1);
                int n_out_channel = layer.value().value("channel_output", 1);
                if (groups == n_out_channel && groups != 1) {
                    cxx_args.push_back(CxxVectorArgument{
                        "convw_" + key, &(fp->get_layer<MultiplexedDWConv1DPackedLayer>(key).weight_pt)});
                    cxx_args.push_back(CxxVectorArgument{
                        "convb_" + key, &(fp->get_layer<MultiplexedDWConv1DPackedLayer>(key).bias_pt)});
                    if (!fp->get_layer<MultiplexedDWConv1DPackedLayer>(key).block_select_pt.empty()) {
                        cxx_args.push_back(CxxVectorArgument{
                            "convm_" + key, &(fp->get_layer<MultiplexedDWConv1DPackedLayer>(key).block_select_pt)});
                    }
                } else {
                    cxx_args.push_back(CxxVectorArgument{
                        "convw_" + key, &(fp->get_layer<MultiplexedConv1DPackedLayer>(key).weight_pt)});
                    cxx_args.push_back(
                        CxxVectorArgument{"convb_" + key, &(fp->get_layer<MultiplexedConv1DPackedLayer>(key).bias_pt)});
                    if (!fp->get_layer<MultiplexedConv1DPackedLayer>(key).block_select_pt.empty()) {
                        cxx_args.push_back(CxxVectorArgument{
                            "convm_" + key, &(fp->get_layer<MultiplexedConv1DPackedLayer>(key).block_select_pt)});
                    }
                }
            } else {
                cxx_args.push_back(
                    CxxVectorArgument{"convw_" + key, &(fp->get_layer<Conv1DPackedLayer>(key).weight_pt)});
                cxx_args.push_back(CxxVectorArgument{"convb_" + key, &(fp->get_layer<Conv1DPackedLayer>(key).bias_pt)});
            }
        } else if (layer_type == "upsample_nearest") {
            cxx_args.push_back(CxxVectorArgument{"upsample_select_pt_" + key,
                                                 &(fp->get_layer<UpsampleNearestLayer>(key).select_tensor_pt)});
        }
    }

    string context_id;
    int level;
    vector<UPtr<FeatureEncrypted>> output_features(fp->json_data["output_feature"].size());
    for (int out_idx = 0; out_idx < (int)fp->json_data["output_feature"].size(); out_idx++) {
        auto ki = fp->json_data["output_feature"][out_idx];
        FeatureNode feature_output(json_features[ki.get<string>()]);
        context_id = feature_output.ckks_parameter_id;
        level = feature_output.level;
        int n_out_num = feature_output.get_n_ciphertexts(block_shape);
        auto* output_context = ckks_contexts.at(feature_output.ckks_parameter_id).get();
        double encode_scale = output_context->get_parameter().get_default_scale();

        if (feature_output.dim == 2) {
            auto output = MakeU<Feature2DEncrypted>(output_context, feature_output.level);
            output->shape = feature_output.shape;
            output->skip = feature_output.skip;
            output->n_channel_per_ct = feature_output.pack_channel_per_ciphertext;
            output->n_channel = feature_output.channel;
            for (int i = 0; i < n_out_num; i++) {
                output->data.push_back(output_context->new_ciphertext(feature_output.level, encode_scale));
            }
            cxx_args.push_back(CxxVectorArgument{ki, &output->data});
            output_features[out_idx] = move(output);
        } else if (feature_output.dim == 0) {
            auto output = MakeU<Feature0DEncrypted>(output_context, feature_output.level);
            output->skip = feature_output.skip[0];
            output->n_channel_per_ct = feature_output.pack_channel_per_ciphertext;
            output->n_channel = feature_output.channel;
            for (int i = 0; i < n_out_num; i++) {
                output->data.push_back(output_context->new_ciphertext(feature_output.level, encode_scale));
            }
            cxx_args.push_back(CxxVectorArgument{ki, &output->data});
            output_features[out_idx] = move(output);
        } else if (feature_output.dim == 1) {
            auto output = MakeU<Feature1DEncrypted>(output_context, feature_output.level, feature_output.skip[0]);
            output->shape = feature_output.shape[0];
            output->n_channel_per_ct = feature_output.pack_channel_per_ciphertext;
            output->n_channel = feature_output.channel;
            for (int i = 0; i < n_out_num; i++) {
                output->data.push_back(output_context->new_ciphertext(feature_output.level, encode_scale));
            }
            cxx_args.push_back(CxxVectorArgument{ki, &output->data});
            output_features[out_idx] = move(output);
        } else {
            throw runtime_error("Unsupported output feature dimension");
        }
    }

    // Dynamically create and run task executors based on the compute_device configuration
    switch (compute_device) {
        case ComputeDevice::CPU: {
            if (!fhe_task_cpu_) {
                prepare_task();
            }
            fhe_time = fhe_time + fhe_task_cpu_->run(ckks_contexts.at(context_id).get(), cxx_args);
            break;
        }
#ifdef INFERENCE_SDK_ENABLE_GPU
        case ComputeDevice::GPU: {
            if (!fhe_task_gpu_) {
                prepare_task();
            }
            fhe_time = fhe_time + fhe_task_gpu_->run(ckks_contexts.at(context_id).get(), cxx_args);
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
    for (int out_idx = 0; out_idx < (int)fp->json_data["output_feature"].size(); out_idx++) {
        auto ki = fp->json_data["output_feature"][out_idx];
        set_feature(ki, move(output_features[out_idx]));
    }
}

void InferenceProcess::run_task_plaintext(bool is_mpc) {
    const json& json_features = fp->json_features;
    json json_layers = fp->json_layers;

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
            Array<double, 2> result1d;
            vector<double> result0d;
            if (layer_type == "conv2d") {
                auto& input0 = p_feature2d_x[feature_input[0]];
                if (fp->pack_style == "multiplexed") {
                    Duo upsample_factor = {layer.value()["upsample_factor_in"][0],
                                           layer.value()["upsample_factor_in"][1]};
                    if (layer.value()["groups"] == 1) {
                        bool is_big_size = layer.value()["is_big_size"];
                        if (is_big_size) {
                            result = fp->get_layer<InverseMultiplexedConv2DLayer>(key).run_plaintext(input0, 1.0);
                        } else {
                            FeatureNode feature_input0(json_features[feature_input[0]]);
                            result = fp->get_layer<MultiplexedConv2DPackedLayer>(key).run_plaintext(input0, 1.0);
                        }
                    } else {
                        bool is_big_size = layer.value()["is_big_size"];
                        if (is_big_size) {
                            result =
                                fp->get_layer<InverseMultiplexedConv2DLayerDepthwise>(key).run_plaintext(input0, 1.0);
                        } else {
                            FeatureNode feature_input0(json_features[feature_input[0]]);
                            result =
                                fp->get_layer<MultiplexedConv2DPackedLayerDepthwise>(key).run_plaintext(input0, 1.0);
                        }
                    }
                    if (upsample_factor[0] > 1 || upsample_factor[1] > 1) {
                        result = upsample_with_zero(result, upsample_factor);
                    }
                } else {
                    if (layer.value()["groups"] == 1) {
                        FeatureNode feature_input0(json_features[feature_input[0]]);
                        result = fp->get_layer<Conv2DPackedLayer>(key).run_plaintext(input0, feature_input0.scale);
                    } else {
                        FeatureNode feature_input0(json_features[feature_input[0]]);
                        result =
                            fp->get_layer<Conv2DPackedDepthwiseLayer>(key).run_plaintext(input0, feature_input0.scale);
                    }
                }
            }
            if (layer_type == "bootstrapping" or layer_type == "drop_level" or layer_type == "batchnorm" or
                layer_type == "batchnorm2d" or layer_type == "identity") {
                FeatureNode feature_input0(json_features[feature_input[0]]);
                if (feature_input0.dim == 2) {
                    auto& input0 = p_feature2d_x[feature_input[0]];
                    result = input0.copy();
                } else if (feature_input0.dim == 1) {
                    auto& input0 = p_feature1d_x[feature_input[0]];
                    result1d = input0.copy();
                } else {
                    auto& input0 = p_feature0d_x[feature_input[0]];
                    result0d = input0;
                }
            }
            if (layer_type == "mult_scalar") {
                const Array<double, 3>& input0 = p_feature2d_x[feature_input[0]];
                result = fp->get_layer<MultScalarLayer>(key).run_plaintext(input0);
            }
            if (layer_type == "concat2d") {
                vector<Array<double, 3>> inputs;
                for (const auto& input_name : feature_input) {
                    inputs.emplace_back(p_feature2d_x[input_name].copy());
                }
                result = fp->get_layer<ConcatLayer>(key).concatenate_channels_multiple_inputs(inputs);
            }
            if (layer_type == "upsample") {
                auto& input0 = p_feature2d_x[feature_input[0]];
                result = fp->get_layer<UpsampleLayer>(key).upsample_with_zero(input0);
            }
            if (layer_type == "upsample_nearest") {
                auto& input0 = p_feature2d_x[feature_input[0]];
                result = fp->get_layer<UpsampleNearestLayer>(key).run_plaintext(input0);
            }
            if (layer_type == "square2d") {
                FeatureNode feature_input0(json_features[feature_input[0]]);
                if (feature_input0.dim == 2) {
                    auto& input0 = p_feature2d_x[feature_input[0]];
                    result = fp->get_layer<SquareLayer>(key).run_plaintext(input0);
                } else if (feature_input0.dim == 0) {
                    auto& input0 = p_feature0d_x[feature_input[0]];
                    result0d = fp->get_layer<SquareLayer>(key)
                                   .run_plaintext(Array<double, 1>::from_array_1d(input0))
                                   .to_array_1d();
                }
            }
            if (layer_type == "add2d") {
                FeatureNode feature_input0(json_features[feature_input[0]]);
                FeatureNode feature_input1(json_features[feature_input[1]]);
                auto& input0 = p_feature2d_x[feature_input[0]];
                auto& input1 = p_feature2d_x[feature_input[1]];
                result = fp->get_layer<AddLayer>(key).run_plaintext(input0, input1);
            }
            if (layer_type == "poly_relu2d" || layer_type == "polyact") {
                FeatureNode feature_input0(json_features[feature_input[0]]);
                if (feature_input0.dim == 0) {
                    auto& input0 = p_feature0d_x[feature_input[0]];
                    result0d = fp->get_layer<PolyRelu0D>(key)
                                   .run_plaintext(Array<double, 1>::from_array_1d(input0))
                                   .to_array_1d();
                } else if (feature_input0.dim == 1) {
                    auto& input0 = p_feature1d_x[feature_input[0]];
                    result1d = fp->get_layer<PolyRelu1D>(key).run_plaintext(input0);
                } else {
                    const Array<double, 3>& input0 = p_feature2d_x[feature_input[0]];
                    if (fp->is_absorb_polyrelu) {
                        result = fp->get_layer<PolyRelu2D>(key).run_plaintext_absorb_case(input0);
                    } else {
                        result = fp->get_layer<PolyRelu2D>(key).run_plaintext_for_non_absorb_case(input0);
                    }
                }
            }
            if (layer_type == "fc0" || layer_type == "fc1") {
                FeatureNode feature_input0(json_features[feature_input[0]]);
                auto input0 = p_feature0d_x[feature_input[0]];
                result0d = fp->get_layer<DensePackedLayer>(key)
                               .run_plaintext(Array<double, 1>::from_array_1d(input0), feature_input0.scale)
                               .to_array_1d();
            }
            if (layer_type == "reshape") {
                FeatureNode feature_input0(json_features[feature_input[0]]);
                if (feature_input0.dim == 1) {
                    auto& input0 = p_feature1d_x[feature_input[0]];
                    result0d = input0.reshape<1>({0}).to_array_1d();
                } else {
                    auto& input0 = p_feature2d_x[feature_input[0]];
                    result0d = input0.reshape<1>({0}).to_array_1d();
                }
            }
            if (layer_type == "avgpool2d") {
                auto& input0 = p_feature2d_x[feature_input[0]];
                bool is_adaptive_avgpool = layer.value()["is_adaptive_avgpool"];
                bool is_big_size = layer.value()["is_big_size"];
                if (is_adaptive_avgpool) {
                    result = Array<double, 3>::from_array_3d(
                        fp->get_layer<Avgpool2DLayer>(key).run_plaintext(input0).to_array_3d());
                } else {
                    if (is_big_size) {
                        result = fp->get_layer<Avgpool2DLayer>(key).run_plaintext(input0);
                    } else {
                        result = fp->get_layer<Avgpool2DLayer>(key).run_plaintext_multiplexed(input0);
                    }
                }
            }
            if (layer_type == "avgpool1d") {
                auto& input0 = p_feature1d_x[feature_input[0]];
                bool is_adaptive_avgpool = layer.value()["is_adaptive_avgpool"];
                bool is_big_size = layer.value()["is_big_size"];
                if (is_adaptive_avgpool) {
                    result1d = fp->get_layer<Avgpool1DLayer>(key).run_plaintext(input0);
                } else {
                    if (is_big_size) {
                        result1d = fp->get_layer<Avgpool1DLayer>(key).run_plaintext(input0);
                    } else {
                        result1d = fp->get_layer<Avgpool1DLayer>(key).run_plaintext_multiplexed(input0);
                    }
                }
            }
            if (layer_type == "conv1d") {
                auto& input0 = p_feature1d_x[feature_input[0]];
                string style = layer.value().value("style", string("ordinary"));
                if (style == "multiplexed") {
                    int groups = layer.value().value("groups", 1);
                    if (groups == (int)input0.get_shape()[0] && groups != 1) {
                        result1d = fp->get_layer<MultiplexedDWConv1DPackedLayer>(key).run_plaintext(input0);
                    } else {
                        result1d = fp->get_layer<MultiplexedConv1DPackedLayer>(key).run_plaintext(input0);
                    }
                } else {
                    result1d = fp->get_layer<Conv1DPackedLayer>(key).run_plaintext(input0);
                }
            }
            if (result.get_size() != 0) {
                p_feature2d_x[feature_output_id] = move(result);
            }
            if (result1d.get_size() != 0) {
                p_feature1d_x[feature_output_id] = move(result1d);
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

void InferenceProcess::run_task_lazy(bool is_mpc) {
    fp->total_fhe_time = 0.0;
    fp->total_fpga_time = 0.0;

    const json& json_data = fp->json_data;
    const json& json_features = fp->json_features;
    const json& json_layers = fp->json_layers;
    Duo block_shape = fp->block_shape;

    vector<CxxVectorArgument> cxx_args;
    unique_ptr<FeatureEncrypted> result;

    vector<vector<CkksCiphertext>> ct_data(json_data["input_feature"].size());
    for (int i = 0; i < (int)json_data["input_feature"].size(); i++) {
        auto ki = json_data["input_feature"][i];
        FeatureNode feature_input(json_features[ki.get<string>()]);
        if (feature_input.dim == 2) {
            const Feature2DEncrypted& input = dynamic_cast<const Feature2DEncrypted&>(_get_feature(ki));
            for (int j = 0; j < (int)input.data.size(); j++)
                ct_data[i].push_back(input.data[j].copy());
            cxx_args.push_back(CxxVectorArgument{ki, &ct_data[i]});
        } else if (feature_input.dim == 0) {
            const Feature0DEncrypted& input = dynamic_cast<const Feature0DEncrypted&>(_get_feature(ki));
            for (int j = 0; j < (int)input.data.size(); j++)
                ct_data[i].push_back(input.data[j].copy());
            cxx_args.push_back(CxxVectorArgument{ki, &ct_data[i]});
        } else if (feature_input.dim == 1) {
            const Feature1DEncrypted& input = dynamic_cast<const Feature1DEncrypted&>(_get_feature(ki));
            for (int j = 0; j < (int)input.data.size(); j++)
                ct_data[i].push_back(input.data[j].copy());
            cxx_args.push_back(CxxVectorArgument{ki, &ct_data[i]});
        }
    }

    // 2. 按层顺序注册所有权重参数（eager pt_ringt + CustomData）
    auto layer_data_sources = prepare_layer_data_sources();
    // 用 map 方便按名查找
    unordered_map<string, fhe_ops_lib::CustomData*> data_source_map;
    for (auto& [k, v] : layer_data_sources)
        data_source_map[k] = &v;

    for (const auto& layer : json_layers.items()) {
        const string& key = layer.key();
        const string& layer_type = layer.value()["type"].get<string>();
        if (layer_type == "avgpool2d") {
            FeatureNode d_input_node(json_features[layer.value()["feature_input"][0].get<string>()]);
            if (d_input_node.dim == 2) {
                bool is_big_size = layer.value()["is_big_size"];
                bool is_adaptive = layer.value()["is_adaptive_avgpool"];
                if (is_big_size && fp->get_layer<Avgpool2DLayer>(key).need_repack) {
                    cxx_args.push_back(
                        CxxVectorArgument{"repack_mask_" + key, &(fp->get_layer<Avgpool2DLayer>(key).repack_mask_pt)});
                } else if (!is_big_size && !is_adaptive && data_source_map.count(key)) {
                    cxx_args.push_back(CxxVectorArgument{key, data_source_map[key]});
                }
            }
        } else if (layer_type == "avgpool1d") {
            bool is_big_size = layer.value()["is_big_size"];
            bool is_adaptive = layer.value()["is_adaptive_avgpool"];
            if (!is_big_size && !is_adaptive && data_source_map.count(key)) {
                cxx_args.push_back(CxxVectorArgument{key, data_source_map[key]});
            }
        } else if (layer_type == "concat2d") {
            if (!fp->get_layer<ConcatLayer>(key).mask_pt.empty()) {
                cxx_args.push_back(CxxVectorArgument{"concat_mask_" + key, &(fp->get_layer<ConcatLayer>(key).mask_pt)});
            }
        } else if (data_source_map.count(key)) {
            // MultiplexedConv2DPackedLayer's mask_pt is populated offline in
            // prepare_weight_lazy and referenced as a static Argument; it must
            // be pushed BEFORE the conv_data_source to match the Python task
            // graph's Argument order.
            if (layer_type == "conv2d" && layer.value()["groups"] == 1 && !layer.value()["is_big_size"] &&
                fp->pack_style == "multiplexed") {
                auto& mux_layer = fp->get_layer<MultiplexedConv2DPackedLayer>(key);
                if (!mux_layer.mask_pt.empty()) {
                    cxx_args.push_back(CxxVectorArgument{"convm_" + key, &mux_layer.mask_pt});
                }
            } else if (layer_type == "conv2d" && layer.value()["groups"] != 1 && !layer.value()["is_big_size"] &&
                       fp->pack_style == "multiplexed") {
                auto& mux_dw_layer = fp->get_layer<MultiplexedConv2DPackedLayerDepthwise>(key);
                if (!mux_dw_layer.mask_pt.empty()) {
                    cxx_args.push_back(CxxVectorArgument{"convm_" + key, &mux_dw_layer.mask_pt});
                }
            }
            cxx_args.push_back(CxxVectorArgument{key, data_source_map[key]});
        }
    }

    // 3. 准备输出密文
    string context_id;
    int level;
    vector<vector<CkksCiphertext>> z_lists(json_data["output_feature"].size());
    for (int out_idx = 0; out_idx < (int)json_data["output_feature"].size(); out_idx++) {
        auto ki = json_data["output_feature"][out_idx];
        FeatureNode feature_output(json_features[ki.get<string>()]);
        context_id = feature_output.ckks_parameter_id;
        level = feature_output.level;
        int n_out_num = div_ceil(feature_output.channel, feature_output.pack_channel_per_ciphertext);
        if (feature_output.shape[0] > block_shape[0] || feature_output.shape[1] > block_shape[1]) {
            Duo out_block_expansion = {feature_output.shape[0] / block_shape[0],
                                       feature_output.shape[1] / block_shape[1]};
            n_out_num *= out_block_expansion[0] * out_block_expansion[1];
        }
        double encode_scale =
            ckks_contexts.at(feature_output.ckks_parameter_id).get()->get_parameter().get_default_scale();
        for (int i = 0; i < n_out_num; i++) {
            z_lists[out_idx].push_back((*ckks_contexts.at(feature_output.ckks_parameter_id))
                                           .new_ciphertext(feature_output.level, encode_scale));
        }
        cxx_args.push_back(CxxVectorArgument{ki, &z_lists[out_idx]});
    }

    // 4. 注册执行器并执行
    switch (compute_device) {
        case ComputeDevice::CPU: {
            if (!fhe_task_cpu_) {
                prepare_task();
            }
            fhe_time = fhe_time + fhe_task_cpu_->run(ckks_contexts.at(context_id).get(), cxx_args);
            break;
        }
#ifdef INFERENCE_SDK_ENABLE_GPU
        case ComputeDevice::GPU: {
            if (!fhe_task_gpu_) {
                prepare_task();
            }
            fhe_time = fhe_time + fhe_task_gpu_->run(ckks_contexts.at(context_id).get(), cxx_args);
            break;
        }
#else
        case ComputeDevice::GPU:
            throw runtime_error(
                "GPU support is disabled. Reconfigure with -DINFERENCE_SDK_ENABLE_GPU=ON to enable it.");
#endif
        case ComputeDevice::FPGA:
            throw runtime_error("FPGA mode should use run_task_fpga() instead of run_task_lazy()");
        default: throw runtime_error("Unknown compute device type");
    }

    // 5. 保存输出结果
    for (int out_idx = 0; out_idx < (int)json_data["output_feature"].size(); out_idx++) {
        auto ki = json_data["output_feature"][out_idx];
        FeatureNode feature_output(json_features[ki.get<string>()]);
        if (feature_output.dim == 2) {
            Feature2DEncrypted f2d(ckks_contexts.at(feature_output.ckks_parameter_id).get(), feature_output.level);
            f2d.data = move(z_lists[out_idx]);
            f2d.shape = feature_output.shape;
            f2d.skip = feature_output.skip;
            f2d.n_channel_per_ct = feature_output.pack_channel_per_ciphertext;
            f2d.n_channel = feature_output.channel;
            result = make_unique<Feature2DEncrypted>(move(f2d));
        } else if (feature_output.dim == 0) {
            Feature0DEncrypted f0d(ckks_contexts.at(feature_output.ckks_parameter_id).get(), feature_output.level);
            f0d.data = move(z_lists[out_idx]);
            f0d.skip = feature_output.skip[0];
            f0d.n_channel_per_ct = feature_output.pack_channel_per_ciphertext;
            f0d.n_channel = feature_output.channel;
            result = make_unique<Feature0DEncrypted>(move(f0d));
        } else if (feature_output.dim == 1) {
            Feature1DEncrypted f1d(ckks_contexts.at(feature_output.ckks_parameter_id).get(), feature_output.level,
                                   feature_output.skip[0]);
            f1d.data = move(z_lists[out_idx]);
            f1d.shape = feature_output.shape[0];
            f1d.n_channel_per_ct = feature_output.pack_channel_per_ciphertext;
            f1d.n_channel = feature_output.channel;
            result = make_unique<Feature1DEncrypted>(move(f1d));
        }
        set_feature(ki, move(result));
    }
}

// ==================== CustomData 模式辅助实现 ====================

vector<pair<string, fhe_ops_lib::CustomData>> InferenceProcess::prepare_layer_data_sources() {
    vector<pair<string, fhe_ops_lib::CustomData>> data_sources;

    for (const auto& layer : fp->json_layers.items()) {
        const string& key = layer.key();
        const string& layer_type = layer.value()["type"].get<string>();

        if (layer_type == "conv2d") {
            int groups = layer.value()["groups"];
            bool is_big_size = layer.value()["is_big_size"];
            if (groups == 1) {
                if (is_big_size) {
                    data_sources.emplace_back(key, fhe_ops_lib::CustomData(static_cast<void*>(
                                                       &fp->get_layer<InverseMultiplexedConv2DLayer>(key))));
                } else if (fp->pack_style == "multiplexed") {
                    data_sources.emplace_back(key, fhe_ops_lib::CustomData(static_cast<void*>(
                                                       &fp->get_layer<MultiplexedConv2DPackedLayer>(key))));
                } else {
                    data_sources.emplace_back(
                        key, fhe_ops_lib::CustomData(static_cast<void*>(&fp->get_layer<Conv2DPackedLayer>(key))));
                }
            } else {
                if (is_big_size) {
                    data_sources.emplace_back(key, fhe_ops_lib::CustomData(static_cast<void*>(
                                                       &fp->get_layer<InverseMultiplexedConv2DLayerDepthwise>(key))));
                } else if (fp->pack_style == "multiplexed") {
                    data_sources.emplace_back(key, fhe_ops_lib::CustomData(static_cast<void*>(
                                                       &fp->get_layer<MultiplexedConv2DPackedLayerDepthwise>(key))));
                } else {
                    data_sources.emplace_back(key, fhe_ops_lib::CustomData(static_cast<void*>(
                                                       &fp->get_layer<Conv2DPackedDepthwiseLayer>(key))));
                }
            }
        } else if (layer_type == "fc0" || layer_type == "fc1") {
            data_sources.emplace_back(
                key, fhe_ops_lib::CustomData(static_cast<void*>(&fp->get_layer<DensePackedLayer>(key))));
        } else if (layer_type == "poly_relu2d" || layer_type == "polyact") {
            auto feature_input = layer.value()["feature_input"].get<vector<string>>();
            FeatureNode d_input_node(fp->json_features[feature_input[0]]);
            if (d_input_node.dim == 0) {
                data_sources.emplace_back(key,
                                          fhe_ops_lib::CustomData(static_cast<void*>(&fp->get_layer<PolyRelu0D>(key))));
            } else if (d_input_node.dim == 1) {
                data_sources.emplace_back(key,
                                          fhe_ops_lib::CustomData(static_cast<void*>(&fp->get_layer<PolyRelu1D>(key))));
            } else {
                data_sources.emplace_back(key,
                                          fhe_ops_lib::CustomData(static_cast<void*>(&fp->get_layer<PolyRelu2D>(key))));
            }
        } else if (layer_type == "upsample_nearest") {
            data_sources.emplace_back(
                key, fhe_ops_lib::CustomData(static_cast<void*>(&fp->get_layer<UpsampleNearestLayer>(key))));
        } else if (layer_type == "conv1d") {
            string style = layer.value().value("style", string("ordinary"));
            if (style == "multiplexed") {
                int groups = layer.value().value("groups", 1);
                int n_out_channel = layer.value().value("channel_output", 1);
                if (groups == n_out_channel && groups != 1) {
                    data_sources.emplace_back(key, fhe_ops_lib::CustomData(static_cast<void*>(
                                                       &fp->get_layer<MultiplexedDWConv1DPackedLayer>(key))));
                } else {
                    data_sources.emplace_back(key, fhe_ops_lib::CustomData(static_cast<void*>(
                                                       &fp->get_layer<MultiplexedConv1DPackedLayer>(key))));
                }
            } else {
                data_sources.emplace_back(
                    key, fhe_ops_lib::CustomData(static_cast<void*>(&fp->get_layer<Conv1DPackedLayer>(key))));
            }
        } else if (layer_type == "mult_scalar") {
            data_sources.emplace_back(
                key, fhe_ops_lib::CustomData(static_cast<void*>(&fp->get_layer<MultScalarLayer>(key))));
        } else if (layer_type == "avgpool2d") {
            FeatureNode d_input_node(fp->json_features[layer.value()["feature_input"][0].get<string>()]);
            if (d_input_node.dim == 2) {
                bool is_big_size = layer.value()["is_big_size"];
                bool is_adaptive = layer.value()["is_adaptive_avgpool"];
                if (!is_big_size && !is_adaptive) {
                    data_sources.emplace_back(
                        key, fhe_ops_lib::CustomData(static_cast<void*>(&fp->get_layer<Avgpool2DLayer>(key))));
                }
            }
        } else if (layer_type == "avgpool1d") {
            bool is_big_size = layer.value()["is_big_size"];
            bool is_adaptive = layer.value()["is_adaptive_avgpool"];
            if (!is_big_size && !is_adaptive) {
                data_sources.emplace_back(
                    key, fhe_ops_lib::CustomData(static_cast<void*>(&fp->get_layer<Avgpool1DLayer>(key))));
            }
        }
        // avgpool, concat 等无 lazy 权重生成，跳过
    }
    return data_sources;
}

#ifdef INFERENCE_SDK_ENABLE_GPU
void InferenceProcess::prepare_task() {
    register_custom_executors(task_custom_executors_);
    if (compute_device == ComputeDevice::GPU) {
        fhe_task_gpu_ = make_unique<FheTaskGpu>(fp->project_path);
        fhe_task_gpu_->bind_custom_executors(task_custom_executors_);
    } else {
        fhe_task_cpu_ = make_unique<FheTaskCpu>(fp->project_path);
        fhe_task_cpu_->bind_custom_executors(task_custom_executors_);
    }
}
#else
void InferenceProcess::prepare_task() {
    register_custom_executors(task_custom_executors_);
    fhe_task_cpu_ = make_unique<FheTaskCpu>(fp->project_path);
    fhe_task_cpu_->bind_custom_executors(task_custom_executors_);
}
#endif

void InferenceProcess::register_custom_executors(unordered_map<string, ExecutorFunc>& executors) {
    auto* fp_ptr = this->fp;

    executors["encode_pt"] = [fp_ptr](ExecutionContext& exec_ctx, const unordered_map<NodeIndex, any>& inputs,
                                      any& output, const ComputeNode& self) -> void {
        CkksContext* ckks_ctx_ptr = exec_ctx.get_arithmetic_context<CkksContext>();
        if (!ckks_ctx_ptr) {
            ckks_ctx_ptr = exec_ctx.get_arithmetic_context<CkksBtpContext>();
        }
        if (!ckks_ctx_ptr) {
            throw runtime_error("encode_pt: Cannot get CKKS context");
        }
        auto& ckks_ctx = *ckks_ctx_ptr;

        if (!self.custom_prop.has_value())
            throw runtime_error("encode_pt: missing custom_prop");

        const string op_class = self.custom_prop->attributes["op_class"].get<string>();
        const string type = self.custom_prop->attributes["type"].get<string>();
        int i = self.custom_prop->attributes.value("i", 0);
        int j = self.custom_prop->attributes.value("j", 0);
        int k = self.custom_prop->attributes.value("k", 0);

        NodeIndex input_node_idx = self.input_nodes[0]->index;
        auto raw_ptr = any_cast<shared_ptr<fhe_ops_lib::CustomData>>(inputs.at(input_node_idx));
        auto* custom_data = raw_ptr.get();
        void* layer_ptr = custom_data->get_typed_data<void>();

        CkksPlaintextRingt pt;

        if (op_class == "Conv2DPackedLayer") {
            auto* layer = static_cast<Conv2DPackedLayer*>(layer_ptr);
            if (type == "weight_pt")
                pt = layer->generate_weight_pt_for_indices(ckks_ctx, i, j, k);
            else
                pt = layer->generate_bias_pt_for_index(ckks_ctx, i);
        } else if (op_class == "Conv2DPackedDepthwiseLayer") {
            auto* layer = static_cast<Conv2DPackedDepthwiseLayer*>(layer_ptr);
            if (type == "weight_pt")
                pt = layer->generate_weight_pt_for_indices(ckks_ctx, i, k);
            else
                pt = layer->generate_bias_pt_for_index(ckks_ctx, i);
        } else if (op_class == "InverseMultiplexedConv2DLayer") {
            auto* layer = static_cast<InverseMultiplexedConv2DLayer*>(layer_ptr);
            if (type == "weight_pt")
                pt = layer->generate_weight_pt_for_indices(ckks_ctx, i, j, k);
            else if (type == "bias_pt")
                pt = layer->generate_bias_pt_for_index(ckks_ctx, i);
            else
                pt = layer->generate_repack_mask_pt(ckks_ctx);
        } else if (op_class == "InverseMultiplexedConv2DLayerDepthwise") {
            auto* layer = static_cast<InverseMultiplexedConv2DLayerDepthwise*>(layer_ptr);
            if (type == "weight_pt")
                pt = layer->generate_weight_pt_for_indices(ckks_ctx, i, k);
            else if (type == "bias_pt")
                pt = layer->generate_bias_pt_for_index(ckks_ctx, i);
            else
                pt = layer->generate_repack_mask_pt(ckks_ctx);
        } else if (op_class == "MultiplexedConv2DPackedLayer") {
            auto* layer = static_cast<MultiplexedConv2DPackedLayer*>(layer_ptr);
            if (type == "weight_pt")
                pt = layer->generate_weight_pt_for_indices(ckks_ctx, i, j, k);
            else if (type == "bias_pt")
                pt = layer->generate_bias_pt_for_index(ckks_ctx, i);
            else
                // mask_pt is now offline (prepare_weight_lazy populates mask_pt
                // and run_task_lazy binds it as a static Argument); this branch
                // is retained only as a fallback for task graphs that still
                // emit mask encode_pt nodes.
                pt = layer->generate_mask_pt_for_indices(ckks_ctx, i);
        } else if (op_class == "MultiplexedConv2DPackedLayerDepthwise") {
            auto* layer = static_cast<MultiplexedConv2DPackedLayerDepthwise*>(layer_ptr);
            if (type == "weight_pt")
                pt = layer->generate_weight_pt_for_indices(ckks_ctx, i, j);
            else if (type == "bias_pt")
                pt = layer->generate_bias_pt_for_index(ckks_ctx, i);
            else
                // mask_pt is now offline (prepare_weight_lazy populates mask_pt
                // and run_task_lazy binds it as a static Argument); this branch
                // is retained only as a fallback for task graphs that still
                // emit mask encode_pt nodes with (ct_idx, channel_in_ct) attrs.
                pt = layer->generate_mask_pt_for_indices(ckks_ctx, i, j);
        } else if (op_class == "Conv1DPackedLayer") {
            auto* layer = static_cast<Conv1DPackedLayer*>(layer_ptr);
            if (type == "weight_pt")
                pt = layer->generate_weight_pt_for_indices(ckks_ctx, i, j, k);
            else
                pt = layer->generate_bias_pt_for_index(ckks_ctx, i);
        } else if (op_class == "MultiplexedConv1DPackedLayer") {
            auto* layer = static_cast<MultiplexedConv1DPackedLayer*>(layer_ptr);
            if (type == "weight_pt")
                pt = layer->generate_weight_pt_for_indices(ckks_ctx, i, j, k);
            else if (type == "select_pt")
                pt = layer->generate_select_tensor_pt_for_index(ckks_ctx, i);
            else
                pt = layer->generate_bias_pt_for_index(ckks_ctx, i);
        } else if (op_class == "MultiplexedDWConv1DPackedLayer") {
            auto* layer = static_cast<MultiplexedDWConv1DPackedLayer*>(layer_ptr);
            if (type == "weight_pt")
                pt = layer->generate_weight_pt_for_indices(ckks_ctx, i, j);
            else if (type == "select_pt")
                pt = layer->generate_select_tensor_pt_for_index(ckks_ctx, i);
            else
                pt = layer->generate_bias_pt_for_index(ckks_ctx, i);
        } else if (op_class == "DensePackedLayer") {
            auto* layer = static_cast<DensePackedLayer*>(layer_ptr);
            if (layer->is_1d_multiplexed) {
                if (type == "weight_pt")
                    pt = layer->generate_weight_pt_1d_mult_for_indices(ckks_ctx, i, j);
                else
                    pt = layer->generate_bias_pt_1d_mult_for_index(ckks_ctx, i);
            } else if (layer->normal_dense) {
                if (type == "weight_pt")
                    pt = layer->generate_weight_0d_pt_for_indices(ckks_ctx, i, j);
                else
                    pt = layer->generate_bias_0d_pt_for_index(ckks_ctx, i);
            } else {
                if (type == "weight_pt")
                    pt = layer->generate_weight_pt_mult_pack_for_indices(ckks_ctx, i, j);
                else
                    pt = layer->generate_bias_pt_mult_pack_for_index(ckks_ctx, i);
            }
        } else if (op_class == "PolyRelu0D") {
            auto* layer = static_cast<PolyRelu0D*>(layer_ptr);
            pt = layer->generate_weight_pt_for_bsgs(ckks_ctx, i, j);
        } else if (op_class == "PolyRelu1D") {
            auto* layer = static_cast<PolyRelu1D*>(layer_ptr);
            pt = layer->generate_weight_pt_for_bsgs(ckks_ctx, i, j);
        } else if (op_class == "PolyRelu2D") {
            auto* layer = static_cast<PolyRelu2D*>(layer_ptr);
            pt = layer->generate_weight_pt_for_non_absorb_indices(ckks_ctx, i, j);
        } else if (op_class == "UpsampleNearestLayer") {
            auto* layer = static_cast<UpsampleNearestLayer*>(layer_ptr);
            pt = layer->generate_select_tensor_pt_for_index(ckks_ctx, i);
        } else if (op_class == "MultScalarLayer") {
            auto* layer = static_cast<MultScalarLayer*>(layer_ptr);
            pt = layer->generate_weight_pt_for_index(ckks_ctx, i);
        } else if (op_class == "Avgpool2DLayer") {
            auto* layer = static_cast<Avgpool2DLayer*>(layer_ptr);
            pt = layer->generate_select_tensor_pt_for_index(ckks_ctx, i);
        } else if (op_class == "Avgpool1DLayer") {
            auto* layer = static_cast<Avgpool1DLayer*>(layer_ptr);
            pt = layer->generate_select_tensor_pt_for_index(ckks_ctx, i);
        } else {
            throw runtime_error("encode_pt: unknown op_class: " + op_class);
        }

        output = make_shared<CkksPlaintextRingt>(move(pt));
    };
}

void InferenceProcess::set_feature(const string& feature_id, unique_ptr<FeatureEncrypted> feature) {
    intermediate_result_[feature_id] = move(feature);
}

const FeatureEncrypted& InferenceProcess::_get_feature(const std::string& feature_id) {
    return *intermediate_result_[feature_id];
}
