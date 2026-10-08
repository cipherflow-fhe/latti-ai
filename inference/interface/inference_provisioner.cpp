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

#include <algorithm>
#include <cctype>
#include <fstream>
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

Bytes serialize_context(const fhe::CkksContext& context) {
    if (const auto* btp_context = dynamic_cast<const fhe::CkksContext*>(&context)) {
        return btp_context->serialize();
    }
    return context.serialize();
}

std::unique_ptr<fhe::CkksContext> deserialize_context(const Bytes& bytes, bool use_btp) {
    if (use_btp) {
        return std::make_unique<fhe::CkksContext>(fhe::CkksContext::deserialize(bytes));
    }
    return std::make_unique<fhe::CkksContext>(fhe::CkksContext::deserialize(bytes));
}

Bytes serialize_public_context(fhe::CkksContext& context, bool /*use_btp*/) {
    // v6：serialize() 天然不含 SK；make_public_context / serialize_advanced 已移除。
    return context.serialize();
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


namespace {

// v3 兼容：galois element -> rotation step 的离散对数（mod 2^(log_n+1)）
// 5 是 galois 生成元；galois_element = 5^r mod 2N（或 -5^r 形式）
int32_t galois_element_to_rotation(uint64_t gal_elem, int log_n) {
    const uint64_t modulus = 1ULL << (log_n + 1);
    const uint64_t mask = modulus - 1;
    uint64_t p = 1;
    for (uint64_t r = 0; r < modulus; ++r) {
        if (p == gal_elem) return static_cast<int32_t>(r);
        p = (p * 5) & mask;
    }
    return -1;
}

}  // namespace

fhe::CkksParameter InferenceProvisioner::make_parameter() const {
    // v3 兼容：优先读 gen_mega_ag 生成的 runner/fhe_parameter.json，
    // 因为它记录了 Python 侧模型编译时真正用到的 q/p。
    // 只读 ckks_parameter.json 会走 Lattigo 内置默认表，Q[0] 和 Python 不一致，
    // 触发 check_parameter 里的 "CKKS parameter Q[0] mismatch"。
    std::filesystem::path fhe_param_path = task_dir_ / "runner" / "fhe_parameter.json";
    if (!std::filesystem::exists(fhe_param_path)) {
        fhe_param_path = provisioner_dir_ / "fhe_parameter.json";
    }
    if (!std::filesystem::exists(fhe_param_path)) {
        fhe_param_path = server_dir_ / "fhe_parameter.json";
    }
    if (std::filesystem::exists(fhe_param_path)) {
        json pj = read_json(fhe_param_path.string());
        if (pj.contains("q") && pj.contains("p") && pj.contains("log_n")) {
            int log_n = pj.at("log_n").get<int>();
            int log_default_scale = pj.value("log_default_scale", 40);
            auto q = pj.at("q").get<std::vector<uint64_t>>();
            auto p = pj.at("p").get<std::vector<uint64_t>>();
            return fhe::CkksParameter::create_custom_parameter(log_n, log_default_scale, q, p);
        }
    }

    // 回退：读 ckks_parameter.json（通常无 q/p，会走 Lattigo 默认表）
    const auto& first_input = task_config_.at("task_input_param").begin().value();
    const std::string ckks_param_id = first_input.at("ckks_parameter_id").get<std::string>();
    const auto& entry = ckks_config_.at(ckks_param_id);
    const int poly_modulus_degree = entry.at("poly_modulus_degree").get<int>();
    int log_n = 0;
    while ((1LL << log_n) < poly_modulus_degree) log_n++;

    int log_default_scale = 40;
    if (entry.contains("log_default_scale")) {
        log_default_scale = entry.at("log_default_scale").get<int>();
    } else if (entry.contains("default_scale")) {
        int64_t ds = static_cast<int64_t>(entry.at("default_scale").get<double>());
        log_default_scale = 0;
        while ((1LL << log_default_scale) < ds) log_default_scale++;
    }

    if (entry.contains("q") && entry.contains("p")) {
        const auto q = entry.at("q").get<std::vector<uint64_t>>();
        const auto p = entry.at("p").get<std::vector<uint64_t>>();
        return fhe::CkksParameter::create_custom_parameter(log_n, log_default_scale, q, p);
    }
    return fhe::CkksParameter::create_parameter(log_n);
}

void InferenceProvisioner::setup_or_load_keys() {
    task_config_ = read_json((server_dir_ / "task_config.json").string());
    ckks_config_ = read_json((server_dir_ / "ckks_parameter.json").string());
    const bool use_btp = task_config_.value("use_btp", false);

    auto secret_context_path = provisioner_dir_ / "secret_context.bin";
    if (!std::filesystem::exists(secret_context_path) && std::filesystem::exists(provisioner_dir_ / "secret_key.bin")) {
        secret_context_path = provisioner_dir_ / "secret_key.bin";
    }
    if (std::filesystem::exists(secret_context_path)) {
        std::cout << "[Provisioner] Loading existing server-owned secret context..." << std::endl;
        context_ = deserialize_context(read_binary_file(secret_context_path), use_btp);
        if (secret_context_path.filename() != "secret_context.bin") {
            write_binary_file(provisioner_dir_ / "secret_context.bin", read_binary_file(secret_context_path));
        }
        return;
    }

    std::cout << "[Provisioner] Generating server-owned CKKS keys..." << std::endl;
    // v6：BTP 参数由 CkksContext 从 residual 参数内部派生（见
    //   fhe_ops_lib/schemes/ckks/bootstrapping.cpp:CreateCkksBtpParameterFromResidualParameter），
    // 不能再手动 create_parameter(btp_log_n)，那样会超出 Lattigo 支持的 logN 范围（12~16）。
    auto param = make_parameter();
    if (use_btp) {
        context_ = std::make_unique<fhe::CkksContext>(
            fhe::CkksContext::create_random_context(param, fhe::EncryptorType::PUBLIC_KEY,
                                                    /*enable_bootstrapping=*/true));
    } else {
        context_ = std::make_unique<fhe::CkksContext>(fhe::CkksContext::create_random_context(param));
    }

    // v3 兼容：按 task_signature.json 里的 glk_order 生成具体 rotations，
    // 而不是只生成默认的 [1,2,4,...]。
    std::vector<int32_t> rotations;
    std::filesystem::path sig_path = task_dir_ / "runner" / "task_signature.json";
    if (!std::filesystem::exists(sig_path)) {
        sig_path = task_dir_ / "task_signature.json";
    }
    if (std::filesystem::exists(sig_path)) {
        try {
            json sig = read_json(sig_path.string());
            if (sig.contains("key") && sig["key"].contains("glk_order")) {
                const int log_n = param.log_n();
                for (const auto& gal_elem_json : sig["key"]["glk_order"]) {
                    uint64_t gal_elem = gal_elem_json.get<uint64_t>();
                    int32_t r = galois_element_to_rotation(gal_elem, log_n);
                    if (r >= 0 && std::find(rotations.begin(), rotations.end(), r) == rotations.end()) {
                        rotations.push_back(r);
                    }
                }
                std::cout << "[Provisioner] Resolved " << rotations.size()
                          << " rotations from task_signature.json glk_order" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[Provisioner] Failed to parse task_signature.json: " << e.what() << std::endl;
        }
    }

    if (rotations.empty()) {
        std::cout << "[Provisioner] Falling back to default rotation keys" << std::endl;
        context_->gen_rotation_keys();
    } else {
        context_->gen_rotation_keys(rotations);
    }

    const Bytes full_context = serialize_context(*context_);
    write_binary_file(secret_context_path, full_context);
    write_binary_file(provisioner_dir_ / "secret_key.bin", full_context);
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
    write_json_file(runner_dir / "task_config.json", runner_config);

    copy_if_exists(server_dir_ / "ckks_parameter.json", runner_dir / "ckks_parameter.json");
    copy_if_exists(server_dir_ / "nn_layers_ct_0.json", runner_dir / "nn_layers_ct_0.json");

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
    if (starts_with(source_id, "convw_")) {
        info.layer_id = source_id.substr(std::string("convw_").size());
        info.param_kind = "weight";
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
    const bool is_weight = info.param_kind == "weight";
    const bool is_bias = info.param_kind == "bias";
    const bool is_mask = info.param_kind == "mask";

    std::vector<fhe::CkksCiphertext> encrypted;
    encrypted.reserve(count);

    for (uint64_t flat_idx = 0; flat_idx < count; ++flat_idx) {
        fhe::CkksPlaintext ringt;

        if (starts_with(arg_id, "dense")) {
            auto& layer = init.get_layer<DensePackedLayer>(info.layer_id);
            if (is_weight) {
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
                            const int out_idx = static_cast<int>(flat_idx / (shape[1] * shape[2]));
                            const int in_idx = static_cast<int>((flat_idx / shape[2]) % shape[1]);
                            const int weight_idx = static_cast<int>(flat_idx % shape[2]);
                            ringt = layer.generate_weight_pt_for_indices(context, out_idx, in_idx, weight_idx);
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
                            const int out_idx = static_cast<int>(flat_idx / shape[1]);
                            const int weight_idx = static_cast<int>(flat_idx % shape[1]);
                            ringt = layer.generate_weight_pt_for_indices(context, out_idx, weight_idx);
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
                        const int out_idx = static_cast<int>(flat_idx / (shape[1] * shape[2]));
                        const int in_idx = static_cast<int>((flat_idx / shape[2]) % shape[1]);
                        const int kernel_idx = static_cast<int>(flat_idx % shape[2]);
                        ringt = layer.generate_weight_pt_for_indices(context, out_idx, in_idx, kernel_idx);
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
                        const int ct_idx = static_cast<int>(flat_idx / shape[1]);
                        const int kernel_idx = static_cast<int>(flat_idx % shape[1]);
                        ringt = layer.generate_weight_pt_for_indices(context, ct_idx, kernel_idx);
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
                ringt = layer.generate_weight_pt_for_bsgs(context, info.coeff_idx, ct_idx);
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

        encrypted.push_back(context.encrypt(ringt));
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
        {"use_site_level", sig.at("level").get<int>()},
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
    setup_or_load_keys();

    const std::filesystem::path runner_dir(runner_dir_arg);
    std::filesystem::create_directories(runner_dir);
    copy_private_and_runner_configs(runner_dir);

    task_signature_ = read_json((runner_dir / "task_signature.json").string());
    if (task_signature_.value("algorithm", std::string("")) != "CKKS") {
        throw std::runtime_error("[Provisioner] only CKKS task signatures are supported");
    }
    const json runner_task_config = read_json((runner_dir / "task_config.json").string());

    const Bytes eval_bytes = serialize_public_context(*context_, task_config_.value("use_btp", false));
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

    if (runtime_lazy && !task_signature_.contains("encrypted_parameters")) {
        throw std::runtime_error("[Provisioner] runtime_lazy requires task_signature.encrypted_parameters");
    }
    const auto& parameter_sigs =
        runtime_lazy ? task_signature_.at("encrypted_parameters") : task_signature_.at("offline");
    for (const auto& sig : parameter_sigs) {
        const std::string arg_id = sig.at("id").get<std::string>();
        if (sig.at("type").get<std::string>() != "ct") {
            throw std::runtime_error("[Provisioner] encrypted_offline only supports ciphertext offline args: " +
                                     arg_id);
        }
        std::cout << "[Provisioner] Encrypting offline parameter " << arg_id << "..." << std::endl;
        const auto info = parse_parameter_argument(arg_id, init);
        auto encrypted = encrypt_parameter_argument(arg_id, sig, *context_, init);
        write_encrypted_argument(parameter_root, arg_id, encrypted, manifest, sig, info, runtime_lazy);
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
