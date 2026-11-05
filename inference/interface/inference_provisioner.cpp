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

#include "interface/inference_provisioner.h"
#include <set>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <exception>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "fhe_layers/dense_packed_layer.h"
#include "inference_task/inference_process.h"
#include "interface/runner_bundle_io.h"
#include "util/serial.h"

namespace {

uint64_t shape_size(const json& sig) {
    const auto shape = sig.at("size").get<std::vector<uint64_t>>();
    return std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<uint64_t>());
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

bool is_unsigned_integer(const std::string& value) {
    if (value.empty()) {
        return false;
    }
    for (char ch : value) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            return false;
        }
    }
    return true;
}

std::string strip_use_site_suffix(const std::string& arg_id) {
    const auto suffix_pos = arg_id.find("__L");
    if (suffix_pos == std::string::npos) {
        return arg_id;
    }
    const auto level_start = suffix_pos + std::string("__L").size();
    if (level_start >= arg_id.size() || !std::isdigit(static_cast<unsigned char>(arg_id[level_start]))) {
        return arg_id;
    }
    return arg_id.substr(0, suffix_pos);
}

void write_json_file(const std::filesystem::path& path, const json& value) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open JSON file for writing: " + path.string());
    }
    out << value.dump(4);
}

void copy_if_exists(const std::filesystem::path& src, const std::filesystem::path& dst) {
    if (std::filesystem::exists(src)) {
        std::filesystem::create_directories(dst.parent_path());
        std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);
    }
}

}  // namespace

InferenceProvisioner::InferenceProvisioner(const std::string& task_dir) : task_dir_(task_dir) {
    resolve_directories();
}

InferenceProvisioner::~InferenceProvisioner() = default;

void InferenceProvisioner::resolve_directories() {
    if (std::filesystem::exists(task_dir_ / "server" / "task_config.json")) {
        server_dir_ = task_dir_ / "server";
        provisioner_dir_ = task_dir_ / "provisioner";
    } else if (std::filesystem::exists(task_dir_ / "task_config.json") &&
               std::filesystem::exists(task_dir_ / "nn_layers_ct_0.json")) {
        server_dir_ = task_dir_;
        provisioner_dir_ = task_dir_.parent_path() / "provisioner";
    } else {
        throw std::runtime_error("[Provisioner] --task-dir must be a task root containing server/ or a server dir");
    }
    std::filesystem::create_directories(provisioner_dir_);
}

fhe::CkksParameter InferenceProvisioner::make_parameter() const {
    const auto& first_input = task_config_.at("task_input_param").begin().value();
    const std::string ckks_param_id = first_input.at("ckks_parameter_id").get<std::string>();
    const auto& entry = ckks_config_.at(ckks_param_id);
    const int log_n = static_cast<int>(std::log2(entry.at("poly_modulus_degree").get<uint32_t>()));
    const auto q = entry.at("q").get<std::vector<uint64_t>>();
    const auto p = entry.at("p").get<std::vector<uint64_t>>();
    return fhe::CkksParameter::create_custom_parameter(
        log_n, entry.at("log_default_scale").get<int>(), q, p,
        static_cast<fhe::RingType>(entry.value("ring_type", 1)));
}

void InferenceProvisioner::setup_or_load_keys() {
    task_config_ = read_json((server_dir_ / "task_config.json").string());
    ckks_config_ = read_json((server_dir_ / "ckks_parameter.json").string());
    if (task_config_.value("mat_pack_style", std::string()) == "par_diagonal_pack") {
        throw std::runtime_error("[Provisioner] PDM encrypted parameters are unsupported; use PCM");
    }
    const bool use_btp = task_config_.value("use_btp", false);

    const auto eval_context_path = provisioner_dir_ / "eval_context.bin";
    const auto secret_key_path = provisioner_dir_ / "secret_key.bin";
    if (std::filesystem::exists(eval_context_path) && std::filesystem::exists(secret_key_path)) {
        std::cout << "[Provisioner] Loading existing server-owned CKKS keys..." << std::endl;
        context_ = std::make_unique<fhe::CkksContext>(fhe::CkksContext::deserialize(read_binary_file(eval_context_path)));
        context_->set_secret_key(fhe::SecretKey::deserialize(read_binary_file(secret_key_path)));
        return;
    }

    std::cout << "[Provisioner] Generating server-owned CKKS keys..." << std::endl;
    auto param = make_parameter();
    context_ = std::make_unique<fhe::CkksContext>(
        fhe::CkksContext::create_random_context(param, fhe::EncryptorType::PUBLIC_KEY, use_btp));
    context_->gen_rotation_keys();
    write_binary_file(eval_context_path, context_->serialize());
    write_binary_file(secret_key_path, context_->secret_key().serialize());
}

void InferenceProvisioner::copy_private_and_runner_configs(const std::filesystem::path& runner_dir) const {
    json provisioner_config = task_config_;
    provisioner_config["deployment_mode"] = "server_provisioned_runner";
    provisioner_config["input_mode"] = "plaintext";
    provisioner_config["parameter_mode"] = "encrypted_offline";
    provisioner_config["decryptor"] = "provisioner";
    provisioner_config["deployment_role"] = "provisioner";
    write_json_file(provisioner_dir_ / "task_config.json", provisioner_config);

    copy_if_exists(server_dir_ / "ckks_parameter.json", provisioner_dir_ / "ckks_parameter.json");
    copy_if_exists(server_dir_ / "model_parameters.h5", provisioner_dir_ / "model_parameters.h5");

    json runner_config = std::filesystem::exists(runner_dir / "task_config.json") ?
                             read_json((runner_dir / "task_config.json").string()) :
                             task_config_;
    runner_config["deployment_mode"] = "server_provisioned_runner";
    runner_config["input_mode"] = "plaintext";
    runner_config["parameter_mode"] = "encrypted_offline";
    runner_config["decryptor"] = "provisioner";
    runner_config["deployment_role"] = "runner";
    if (runner_config.value("mat_pack_style", std::string()) == "par_block_col_major" &&
        runner_config.contains("layernorm_param")) {
        for (const auto* key : {"inv_var", "inv_std", "var_std_bound", "inv_var_scale", "inv_std_scale"})
            runner_config["layernorm_param"].erase(key);
    }
    write_json_file(runner_dir / "task_config.json", runner_config);

    copy_if_exists(server_dir_ / "ckks_parameter.json", runner_dir / "ckks_parameter.json");
    auto graph = read_json((server_dir_ / "nn_layers_ct_0.json").string());
    if (runner_config.value("mat_pack_style", std::string()) == "par_block_col_major") {
        // Keep geometry and explicit public recipes. Private coefficients and
        // source paths are only needed by the Provisioner encoders.
        const std::set<std::string> public_fields = {
            "type", "feature_input", "feature_output", "K", "k", "order", "degree",
            "weight_shape", "head_shape", "to_expand", "level_cost", "pcm_seed_layout",
            "parameter_sources", "epsilon", "eps", "level", "n_levels", "is_transposed"};
        for (auto& item : graph["layer"].items()) {
            auto& layer = item.value();
            const auto type = layer.at("type").get<std::string>();
            if (type.rfind("pcm", 0) != 0 && type.rfind("par", 0) != 0) continue;
            for (auto it = layer.begin(); it != layer.end();) {
                if (!public_fields.count(it.key())) it = layer.erase(it);
                else ++it;
            }
            if (layer.contains("parameter_sources"))
                for (auto& source : layer["parameter_sources"].items()) source.value().erase("source_refs");
        }
    }
    write_json_file(runner_dir / "nn_layers_ct_0.json", graph);

    if (!std::filesystem::exists(runner_dir / "mega_ag.json") ||
        !std::filesystem::exists(runner_dir / "task_signature.json")) {
        throw std::runtime_error("[Provisioner] runner directory must already contain mega_ag.json and "
                                 "task_signature.json; run gen_mega_ag.py in server_provisioned_runner mode first");
    }
}

InferenceProvisioner::ParameterArgumentInfo
InferenceProvisioner::parse_parameter_argument(const std::string& arg_id, const InitInferenceProcess& init) const {
    ParameterArgumentInfo info;
    const std::string source_id = strip_use_site_suffix(arg_id);
    info.source_id = source_id;
    if (starts_with(source_id, "matrix_")) {
        for (const auto& item : init.json_layers.items()) {
            if (starts_with(source_id, "matrix_" + item.key() + "_") && item.key().size() > info.layer_id.size())
                info.layer_id = item.key();
        }
        if (!info.layer_id.empty()) {
            info.param_kind = "matrix_parameter";
            return info;
        }
    }
    if (starts_with(source_id, "convw_")) {
        info.layer_id = source_id.substr(std::string("convw_").size());
        info.param_kind = "weight";
        if (init.json_layers.contains(info.layer_id) &&
            init.json_layers.at(info.layer_id).contains("seed_weight_subgrid_stride")) {
            info.param_kind = "seed_weight";
        }
        return info;
    }
    if (starts_with(source_id, "convb_")) {
        info.layer_id = source_id.substr(std::string("convb_").size());
        info.param_kind = "bias";
        return info;
    }
    if (starts_with(source_id, "convm_")) {
        info.layer_id = source_id.substr(std::string("convm_").size());
        info.param_kind = "mask";
        return info;
    }
    if (starts_with(source_id, "densew_")) {
        info.layer_id = source_id.substr(std::string("densew_").size());
        info.param_kind = "weight";
        return info;
    }
    if (starts_with(source_id, "denseb_")) {
        info.layer_id = source_id.substr(std::string("denseb_").size());
        info.param_kind = "bias";
        return info;
    }
    if (starts_with(source_id, "mult_scalar_")) {
        info.layer_id = source_id.substr(std::string("mult_scalar_").size());
        info.param_kind = "mult_scalar_weight";
        return info;
    }
    if (starts_with(source_id, "poly_reluw_")) {
        const std::string prefix = "poly_reluw_";
        size_t best_len = 0;
        std::string best_layer_id;
        int best_coeff = -1;
        for (const auto& item : init.json_layers.items()) {
            const std::string& layer_id = item.key();
            const std::string layer_prefix = prefix + layer_id + "_";
            if (!starts_with(source_id, layer_prefix)) {
                continue;
            }
            const std::string coeff_text = source_id.substr(layer_prefix.size());
            if (!is_unsigned_integer(coeff_text)) {
                continue;
            }
            if (layer_id.size() > best_len) {
                best_len = layer_id.size();
                best_layer_id = layer_id;
                best_coeff = std::stoi(coeff_text);
            }
        }
        if (!best_layer_id.empty()) {
            info.layer_id = best_layer_id;
            info.param_kind = "poly_coeff";
            info.coeff_idx = best_coeff;
            return info;
        }
    }
    throw std::runtime_error("[Provisioner] Unsupported encrypted parameter argument id: " + arg_id);
}

std::vector<fhe::CkksCiphertext> InferenceProvisioner::encrypt_parameter_argument(const std::string& arg_id,
                                                                                  const json& sig,
                                                                                  fhe::CkksContext& context,
                                                                                  InitInferenceProcess& init) const {
    const auto info = parse_parameter_argument(arg_id, init);
    const int encrypt_level = sig.at("level").get<int>();
    const auto shape = sig.at("size").get<std::vector<uint64_t>>();
    const uint64_t count = shape_size(sig);

    const auto& layer_cfg = init.json_layers.at(info.layer_id);
    const std::string layer_type = layer_cfg.at("type").get<std::string>();
    const bool is_weight = info.param_kind == "weight" || info.param_kind == "seed_weight";
    const bool is_bias = info.param_kind == "bias";
    const bool is_mask = info.param_kind == "mask";

    std::vector<fhe::CkksCiphertext> encrypted;
    encrypted.reserve(count);

    for (uint64_t flat_idx = 0; flat_idx < count; ++flat_idx) {
        fhe::CkksPlaintext ringt;

        if (info.param_kind == "matrix_parameter") {
            const auto& generator = sig.at("generator");
            if (generator.at("op_class") == "InputZero") {
                ringt = fhe_api::encode_ringt(context, std::vector<double>(context.parameter().max_slots(), 0.0),
                                             context.parameter().default_scale());
            } else {
                ringt = generate_matrix_parameter(context, &init.get_layer<Layer>(info.layer_id), generator);
            }
        } else if (starts_with(arg_id, "dense")) {
            auto& layer = init.get_layer<DensePackedLayer>(info.layer_id);
            const bool tile_seed = layer_cfg.contains("dense_seed_tiles") &&
                layer_cfg.at("dense_seed_tiles").at(is_weight ? "weight" : "bias").get<bool>();
            if (tile_seed) {
                const auto& layout = layer_cfg.at("dense_seed_tiles");
                const auto duo = [&layout](const char* key) {
                    return Duo{layout.at(key).at(0).get<uint32_t>(), layout.at(key).at(1).get<uint32_t>()};
                };
                layer.validate_tile_geometry(duo("input_shape"), duo("skip"), duo("invalid_fill"),
                    layout.at("slots").get<uint32_t>(), duo("weight_shape"), layout.at("use_site_level").get<int>());
                const uint64_t group = layout.at("group_size").get<uint64_t>();
                if (!group || group != prod(duo("tile_grid")) || duo("tile_grid") != duo("invalid_fill"))
                    throw std::runtime_error("[Provisioner] invalid dense tile group");
                const auto original = duo("weight_shape");
                const uint64_t expected = div_ceil(uint64_t(original[0]) * (is_weight ? original[1] : 1), group);
                const auto expected_shape = is_weight ? std::vector<uint64_t>{1, expected} : std::vector<uint64_t>{expected};
                if (shape != expected_shape || encrypt_level != layout.at("use_site_level").get<int>() + (is_weight ? 1 : 0))
                    throw std::runtime_error("[Provisioner] dense tile shape or level mismatch: " + arg_id);
                ringt = layer.generate_tile_seed(context, flat_idx, is_bias);
            } else if (is_weight) {
                if (shape.size() != 2) {
                    throw std::runtime_error("[Provisioner] dense weight signature must be rank 2: " + arg_id);
                }
                const int out_idx = static_cast<int>(flat_idx / shape[1]);
                const int weight_idx = static_cast<int>(flat_idx % shape[1]);
                if (layer.is_1d_multiplexed) {
                    ringt = layer.generate_weight_pt_1d_mult_for_indices(context, out_idx, weight_idx);
                } else if (layer.normal_dense) {
                    ringt = layer.generate_weight_0d_pt_for_indices(context, out_idx, weight_idx);
                } else {
                    ringt = layer.generate_weight_pt_mult_pack_for_indices(context, out_idx, weight_idx);
                }
            } else {
                if (!is_bias || shape.size() != 1) {
                    throw std::runtime_error("[Provisioner] dense bias signature must be rank 1: " + arg_id);
                }
                const int out_idx = static_cast<int>(flat_idx);
                if (layer.is_1d_multiplexed) {
                    ringt = layer.generate_bias_pt_1d_mult_for_index(context, out_idx);
                } else if (layer.normal_dense) {
                    ringt = layer.generate_bias_0d_pt_for_index(context, out_idx);
                } else {
                    ringt = layer.generate_bias_pt_mult_pack_for_index(context, out_idx);
                }
            }
        } else if (starts_with(arg_id, "conv")) {
            if (layer_type != "conv2d") {
                throw std::runtime_error("[Provisioner] encrypted conv argument is only supported for conv2d: " +
                                         arg_id);
            }
            const bool is_big_size = layer_cfg.value("is_big_size", false);
            const int groups = layer_cfg.at("groups").get<int>();
            const int n_out_channel =
                init.json_features.at(layer_cfg.at("feature_output")[0].get<std::string>()).at("channel").get<int>();

            if (init.pack_style == "multiplexed") {
                if (is_big_size) {
                    if (groups == 1) {
                        auto& layer = init.get_layer<InverseMultiplexedConv2DLayer>(info.layer_id);
                        if (is_weight) {
                            if (shape.size() != 3) {
                                throw std::runtime_error(
                                    "[Provisioner] inverse conv weight signature must be rank 3: " + arg_id);
                            }
                            if (shape[0] != n_out_channel || shape[1] != layer.weight_.get_shape()[1] ||
                                shape[2] != layer.seed_weight_kernel_count())
                                throw std::runtime_error("[Provisioner] inverse seed shape mismatch: " + arg_id);
                            if (encrypt_level != init.json_features.at(layer_cfg.at("feature_input").at(0).get<std::string>()).at("level").get<int>() + layer.seed_weight_level_cost())
                                throw std::runtime_error("[Provisioner] inverse seed level mismatch: " + arg_id);
                            const int out_idx = static_cast<int>(flat_idx / (shape[1] * shape[2]));
                            const int in_idx = static_cast<int>((flat_idx / shape[2]) % shape[1]);
                            const int weight_idx = static_cast<int>(flat_idx % shape[2]);
                            ringt = layer.use_seed_weight_pack()
                                        ? layer.generate_weight_pt_for_seed_indices(context, out_idx, in_idx, weight_idx)
                                        : layer.generate_weight_pt_for_indices(context, out_idx, in_idx, weight_idx);
                        } else {
                            if (!is_bias || shape.size() != 1) {
                                throw std::runtime_error("[Provisioner] inverse conv bias signature must be rank 1: " +
                                                         arg_id);
                            }
                            ringt = layer.generate_bias_pt_for_index(context, static_cast<int>(flat_idx));
                        }
                    } else {
                        auto& layer = init.get_layer<InverseMultiplexedConv2DLayerDepthwise>(info.layer_id);
                        if (is_weight) {
                            if (shape.size() != 2) {
                                throw std::runtime_error(
                                    "[Provisioner] inverse depthwise conv weight signature must be rank 2: " + arg_id);
                            }
                            if (shape[0] != n_out_channel || shape[1] != layer.seed_weight_kernel_count())
                                throw std::runtime_error("[Provisioner] inverse depthwise seed shape mismatch: " + arg_id);
                            if (encrypt_level != init.json_features.at(layer_cfg.at("feature_input").at(0).get<std::string>()).at("level").get<int>() + layer.seed_weight_level_cost())
                                throw std::runtime_error("[Provisioner] inverse depthwise seed level mismatch: " + arg_id);
                            const int out_idx = static_cast<int>(flat_idx / shape[1]);
                            const int weight_idx = static_cast<int>(flat_idx % shape[1]);
                            ringt = layer.use_seed_weight_pack()
                                        ? layer.generate_weight_pt_for_seed_indices(context, out_idx, weight_idx)
                                        : layer.generate_weight_pt_for_indices(context, out_idx, weight_idx);
                        } else {
                            if (!is_bias || shape.size() != 1) {
                                throw std::runtime_error(
                                    "[Provisioner] inverse depthwise conv bias signature must be rank 1: " + arg_id);
                            }
                            ringt = layer.generate_bias_pt_for_index(context, static_cast<int>(flat_idx));
                        }
                    }
                } else if (groups == 1) {
                    auto& layer = init.get_layer<MultiplexedConv2DPackedLayer>(info.layer_id);
                    if (is_weight) {
                        if (shape.size() != 3) {
                            throw std::runtime_error(
                                "[Provisioner] multiplexed conv weight signature must be rank 3: " + arg_id);
                        }
                        if (layer.joint_seed_weights() &&
                            (shape[0] != 1 || shape[1] != layer.seed_weight_count() || shape[2] != 1))
                            throw std::runtime_error("[Provisioner] multiplexed joint seed shape mismatch: " + arg_id);
                        if (encrypt_level != init.json_features.at(layer_cfg.at("feature_input").at(0).get<std::string>()).at("level").get<int>() + layer.seed_weight_level_cost())
                            throw std::runtime_error("[Provisioner] multiplexed seed level mismatch: " + arg_id);
                        const int out_idx = static_cast<int>(flat_idx / (shape[1] * shape[2]));
                        const int in_idx = static_cast<int>((flat_idx / shape[2]) % shape[1]);
                        const int kernel_idx = static_cast<int>(flat_idx % shape[2]);
                        ringt = layer.use_seed_weight_pack()
                                    ? layer.generate_weight_pt_for_seed_indices(context, out_idx, in_idx, kernel_idx)
                                    : layer.generate_weight_pt_for_indices(context, out_idx, in_idx, kernel_idx);
                    } else if (is_bias) {
                        if (!is_bias || shape.size() != 1) {
                            throw std::runtime_error("[Provisioner] multiplexed conv bias signature must be rank 1: " +
                                                     arg_id);
                        }
                        ringt = layer.generate_bias_pt_for_index(context, static_cast<int>(flat_idx));
                    } else if (is_mask) {
                        if (shape.size() != 1) {
                            throw std::runtime_error("[Provisioner] multiplexed conv mask signature must be rank 1: " +
                                                     arg_id);
                        }
                        ringt = layer.generate_mask_pt_for_indices(context, static_cast<int>(flat_idx));
                    } else {
                        throw std::runtime_error("[Provisioner] unsupported multiplexed conv parameter: " + arg_id);
                    }
                } else {
                    auto& layer = init.get_layer<MultiplexedConv2DPackedLayerDepthwise>(info.layer_id);
                    if (is_weight) {
                        if (shape.size() != 2) {
                            throw std::runtime_error(
                                "[Provisioner] multiplexed depthwise conv weight signature must be rank 2: " + arg_id);
                        }
                        if (layer.joint_seed_weights() && (shape[0] != 1 || shape[1] != layer.seed_weight_count())) {
                            throw std::runtime_error("[Provisioner] depthwise joint seed shape mismatch: " + arg_id);
                        }
                        if (encrypt_level != init.json_features.at(layer_cfg.at("feature_input").at(0).get<std::string>()).at("level").get<int>() + layer.seed_weight_level_cost())
                            throw std::runtime_error("[Provisioner] depthwise seed level mismatch: " + arg_id);
                        const int ct_idx = static_cast<int>(flat_idx / shape[1]);
                        const int kernel_idx = static_cast<int>(flat_idx % shape[1]);
                        ringt = layer.use_seed_weight_pack()
                                    ? layer.generate_weight_pt_for_seed_indices(context, ct_idx, kernel_idx)
                                    : layer.generate_weight_pt_for_indices(context, ct_idx, kernel_idx);
                    } else if (is_bias) {
                        if (!is_bias || shape.size() != 1) {
                            throw std::runtime_error(
                                "[Provisioner] multiplexed depthwise conv bias signature must be rank 1: " + arg_id);
                        }
                        ringt = layer.generate_bias_pt_for_index(context, static_cast<int>(flat_idx));
                    } else if (is_mask) {
                        if (shape.size() != 1) {
                            throw std::runtime_error(
                                "[Provisioner] multiplexed depthwise conv mask signature must be rank 1: " + arg_id);
                        }
                        const int ct_idx =
                            static_cast<int>(flat_idx / std::max<uint32_t>(1, layer.get_n_channel_per_ct()));
                        const int local_idx =
                            static_cast<int>(flat_idx % std::max<uint32_t>(1, layer.get_n_channel_per_ct()));
                        ringt = layer.generate_mask_pt_for_indices(context, ct_idx, local_idx);
                    } else {
                        throw std::runtime_error("[Provisioner] unsupported multiplexed depthwise conv parameter: " +
                                                 arg_id);
                    }
                }
            } else {
                const bool is_depthwise = groups == n_out_channel && groups != 1;
                if (is_depthwise) {
                    auto& layer = init.get_layer<Conv2DPackedDepthwiseLayer>(info.layer_id);
                    if (is_weight) {
                        if (shape.size() != 2) {
                            throw std::runtime_error(
                                "[Provisioner] ordinary depthwise conv weight signature must be rank 2: " + arg_id);
                        }
                        const int ct_idx = static_cast<int>(flat_idx / shape[1]);
                        const int kernel_idx = static_cast<int>(flat_idx % shape[1]);
                        ringt = layer.generate_weight_pt_for_indices(context, ct_idx, kernel_idx);
                    } else {
                        if (!is_bias || shape.size() != 1) {
                            throw std::runtime_error(
                                "[Provisioner] ordinary depthwise conv bias signature must be rank 1: " + arg_id);
                        }
                        ringt = layer.generate_bias_pt_for_index(context, static_cast<int>(flat_idx));
                    }
                } else {
                    auto& layer = init.get_layer<Conv2DPackedLayer>(info.layer_id);
                    if (is_weight) {
                        if (shape.size() != 3) {
                            throw std::runtime_error("[Provisioner] ordinary conv weight signature must be rank 3: " +
                                                     arg_id);
                        }
                        const int out_idx = static_cast<int>(flat_idx / (shape[1] * shape[2]));
                        const int in_idx = static_cast<int>((flat_idx / shape[2]) % shape[1]);
                        const int kernel_idx = static_cast<int>(flat_idx % shape[2]);
                        ringt = layer.generate_weight_pt_for_indices(context, out_idx, in_idx, kernel_idx);
                    } else {
                        if (!is_bias || shape.size() != 1) {
                            throw std::runtime_error("[Provisioner] ordinary conv bias signature must be rank 1: " +
                                                     arg_id);
                        }
                        ringt = layer.generate_bias_pt_for_index(context, static_cast<int>(flat_idx));
                    }
                }
            }
        } else if (info.param_kind == "poly_coeff") {
            if (shape.size() != 1) {
                throw std::runtime_error("[Provisioner] polyrelu coeff signature must be rank 1: " + arg_id);
            }
            const std::string feature_id = layer_cfg.at("feature_input")[0].get<std::string>();
            const int dim = init.json_features.at(feature_id).at("dim").get<int>();
            const int ct_idx = static_cast<int>(flat_idx);
            if (dim == 0) {
                auto& layer = init.get_layer<PolyRelu0D>(info.layer_id);
                ringt = layer.generate_weight_pt_for_bsgs(context, info.coeff_idx, ct_idx);
            } else if (dim == 1) {
                auto& layer = init.get_layer<PolyRelu1D>(info.layer_id);
                ringt = layer.generate_weight_pt_for_bsgs(context, info.coeff_idx, ct_idx);
            } else {
                auto& layer = init.get_layer<PolyRelu2D>(info.layer_id);
                if (layer_cfg.contains("poly_seed_layout")) {
                    const auto& layout = layer_cfg.at("poly_seed_layout");
                    const auto duo = [&layout](const char* key) {
                        return Duo{layout.at(key).at(0).get<uint32_t>(), layout.at(key).at(1).get<uint32_t>()};
                    };
                    const uint32_t group = layout.at("group_size").get<uint32_t>();
                    const int groups = layout.at("channel_groups").get<int>();
                    const bool packed = layout.at("packed_degrees").at(info.coeff_idx).get<bool>();
                    const int target = layout.at("use_site_levels").at(info.coeff_idx).get<int>();
                    layer.validate_seed_geometry(duo("input_shape"), duo("skip"), layout.at("blocks").get<uint32_t>(),
                                                 layout.at("slots").get<uint32_t>(), groups, info.coeff_idx, target);
                    if (!group || group != prod(duo("stride")) || encrypt_level != target + packed ||
                        shape[0] != (packed ? div_ceil(groups, group) : groups))
                        throw std::runtime_error("[Provisioner] polynomial seed shape or level mismatch: " + arg_id);
                    ringt = packed ? layer.generate_channel_seed(context, info.coeff_idx, ct_idx, duo("stride")) :
                                     layer.generate_weight_pt_for_bsgs(context, info.coeff_idx, ct_idx);
                } else {
                    ringt = layer.generate_weight_pt_for_bsgs(context, info.coeff_idx, ct_idx);
                }
            }
        } else if (info.param_kind == "mult_scalar_weight") {
            if (shape.size() != 1) {
                throw std::runtime_error("[Provisioner] mult_scalar weight signature must be rank 1: " + arg_id);
            }
            auto& layer = init.get_layer<MultScalarLayer>(info.layer_id);
            ringt = layer.generate_weight_pt_for_index(context, static_cast<int>(flat_idx));
        } else {
            throw std::runtime_error("[Provisioner] Unsupported encrypted parameter argument: " + arg_id);
        }

        std::vector<double> values;
        context.decode(ringt, values);
        auto pt = fhe_api::encode(context, values, encrypt_level, ringt.scale());
        encrypted.push_back(context.encrypt(pt));
        if (sig.contains("scale")) {
            const double expected = sig.at("scale").get<double>();
            if (!std::isfinite(expected) || expected <= 0 ||
                std::abs(encrypted.back().scale() / expected - 1.0) > 1e-10)
                throw std::runtime_error("[Provisioner] parameter scale mismatch: " + arg_id);
        }
    }
    return encrypted;
}

void InferenceProvisioner::write_encrypted_argument(const std::filesystem::path& parameter_root,
                                                    const std::string& arg_id,
                                                    const std::vector<fhe::CkksCiphertext>& values,
                                                    json& manifest,
                                                    const json& sig,
                                                    const ParameterArgumentInfo& info,
                                                    bool write_element_index) const {
    const std::string filename = arg_id + ".bin";
    std::stringstream ss;
    uint64_t count = values.size();
    ss_write(ss, count);
    json elements = json::array();
    for (const auto& ct : values) {
        Bytes ct_bytes = ct.serialize();
        if (write_element_index) {
            const auto payload_offset = static_cast<uint64_t>(ss.tellp()) + sizeof(size_t);
            elements.push_back({{"offset", payload_offset}, {"length", static_cast<uint64_t>(ct_bytes.size())}});
        }
        ss_write_vector(ss, ct_bytes);
    }
    write_binary_file(parameter_root / filename, ss_to_bytes(ss));

    manifest["arguments"][arg_id] = {
        {"id", arg_id},
        {"source_id", info.source_id.empty() ? arg_id : info.source_id},
        {"type", "ct"},
        {"level", sig.at("level").get<int>()},
        {"use_site_level", sig.value("use_site_level", sig.at("level").get<int>())},
        {"size", sig.at("size")},
        {"file", filename},
        {"layer_id", info.layer_id},
        {"param_kind", info.param_kind},
    };
    if (write_element_index) {
        manifest["arguments"][arg_id]["elements"] = std::move(elements);
    }
    if (info.coeff_idx >= 0) {
        manifest["arguments"][arg_id]["coeff_idx"] = info.coeff_idx;
    }
}

void InferenceProvisioner::export_runner_bundle(const std::string& runner_dir_arg) {
    if (!std::filesystem::is_regular_file(server_dir_ / "model_parameters.h5")) {
        throw std::runtime_error("[Provisioner] model_parameters.h5 is missing in " + server_dir_.string());
    }
    setup_or_load_keys();

    const std::filesystem::path runner_dir(runner_dir_arg);
    std::filesystem::create_directories(runner_dir);
    copy_private_and_runner_configs(runner_dir);

    task_signature_ = read_json((runner_dir / "task_signature.json").string());
    if (task_signature_.value("algorithm", std::string("")) != "CKKS") {
        throw std::runtime_error("[Provisioner] only CKKS task signatures are supported");
    }
    const json runner_task_config = read_json((runner_dir / "task_config.json").string());

    const Bytes eval_bytes = context_->serialize();
    write_binary_file(provisioner_dir_ / "eval_context.bin", eval_bytes);
    write_binary_file(runner_dir / "eval_context.bin", eval_bytes);

    const auto parameter_root = runner_dir / "encrypted_model_parameters";
    std::filesystem::create_directories(parameter_root);

    const bool runtime_lazy =
        runner_task_config.value("parameter_loading_mode",
                                 task_config_.value("parameter_loading_mode", std::string("eager"))) == "runtime_lazy";
    json manifest = {
        {"format", runtime_lazy ? "latti-ai.encrypted-parameter-store.v2" : "latti-ai.encrypted-parameter-store.v1"},
        {"arguments", json::object()},
    };

    InitInferenceProcess init(server_dir_.string(), false);
    init.init_parameters(task_config_.value("use_btp", false));
    init.is_lazy = true;
    init.load_model_prepare();
    const auto graph = read_json((server_dir_ / "nn_layers_ct_0.json").string());

    if (runtime_lazy && !task_signature_.contains("encrypted_parameters")) {
        throw std::runtime_error("[Provisioner] runtime_lazy requires task_signature.encrypted_parameters");
    }
    const auto& parameter_sigs =
        runtime_lazy ? task_signature_.at("encrypted_parameters") : task_signature_.at("offline");
    for (size_t first = 0; first < parameter_sigs.size();) {
        // Matrix descriptors each contain one ciphertext. Bound parallel
        // provisioning to eight of these small, independent arguments.
        size_t batch_size = 1;
        if (starts_with(parameter_sigs[first].at("id").get<std::string>(), "matrix_")) {
            while (batch_size < 8 && first + batch_size < parameter_sigs.size() &&
                   starts_with(parameter_sigs[first + batch_size].at("id").get<std::string>(), "matrix_"))
                ++batch_size;
        }
        std::vector<std::vector<fhe::CkksCiphertext>> batch(batch_size);
        std::vector<std::exception_ptr> errors(batch_size);
        parallel_for(batch_size, batch_size, *context_, [&](fhe::CkksContext& ctx, int i) {
            try {
                const auto& sig = parameter_sigs[first + i];
                batch[i] = encrypt_parameter_argument(sig.at("id").get<std::string>(), sig, ctx, init);
            } catch (...) {
                errors[i] = std::current_exception();
            }
        });
        for (size_t i = 0; i < batch_size; ++i) {
            if (errors[i]) std::rethrow_exception(errors[i]);
            const auto& sig = parameter_sigs[first + i];
            const std::string arg_id = sig.at("id").get<std::string>();
            if (sig.at("type").get<std::string>() != "ct") {
                throw std::runtime_error("[Provisioner] encrypted_offline only supports ciphertext offline args: " +
                                         arg_id);
            }
            std::cout << "[Provisioner] Encrypting offline parameter " << arg_id << "..." << std::endl;
            const auto info = parse_parameter_argument(arg_id, init);
            auto& encrypted = batch[i];
            write_encrypted_argument(parameter_root, arg_id, encrypted, manifest, sig, info, runtime_lazy);
            manifest["arguments"][arg_id]["scale"] = encrypted.front().scale();
            const auto layout = parameter_layout_metadata(graph, info.layer_id, sig);
            if (sig.contains("layout") && nlohmann::json(sig.at("layout")) != nlohmann::json(layout))
                throw std::runtime_error("[Provisioner] parameter layout mismatch: " + arg_id);
            if (!layout.is_null()) manifest["arguments"][arg_id]["layout"] = layout;
            const auto& layer_cfg = init.json_layers.at(info.layer_id);
            if (starts_with(arg_id, "dense") && layer_cfg.contains("dense_seed_tiles")) {
                const auto& layout = layer_cfg.at("dense_seed_tiles");
                if (layout.at(info.param_kind).get<bool>()) {
                    auto& entry = manifest["arguments"][arg_id];
                    entry["use_site_level"] = layout.at("use_site_level").get<int>() - (info.param_kind == "bias" ? 1 : 0);
                }
            }
            if (info.param_kind == "poly_coeff" && layer_cfg.contains("poly_seed_layout")) {
                auto& entry = manifest["arguments"][arg_id];
                entry["degree"] = info.coeff_idx;
                entry["use_site_level"] = layer_cfg.at("poly_seed_layout").at("use_site_levels").at(info.coeff_idx);
            }
        }
        first += batch_size;
    }

    write_json_file(parameter_root / "manifest.json", manifest);
    write_json_file(provisioner_dir_ / "encrypted_parameter_manifest.json", manifest);

    if (std::filesystem::exists(runner_dir / "secret_key.bin") ||
        std::filesystem::exists(runner_dir / "secret_context.bin") ||
        std::filesystem::exists(runner_dir / "model_parameters.h5")) {
        throw std::runtime_error("[Provisioner] runner bundle contains secret/model files; remove them before use: " +
                                 runner_dir.string());
    }

    std::cout << "[Provisioner] Runner bundle exported to " << runner_dir << std::endl;
}
