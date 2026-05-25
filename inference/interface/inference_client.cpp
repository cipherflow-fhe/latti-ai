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

#include <iostream>

#include "interface/inference_client.h"
#include "data_structs/feature_mat.h"

using namespace lattisense;

using namespace fhe_ops_lib;

InferenceClient::InferenceClient(const std::string& client_dir) : client_dir_(client_dir) {}

InferenceClient::~InferenceClient() = default;

void InferenceClient::read_configuration() {
    task_config_ = read_json((client_dir_ / "task_config.json").string());
    pack_style_ = task_config_["pack_style"];
    needs_btp_ = task_config_["use_btp"];

    auto& output_param = task_config_["task_output_param"].begin().value();
    int first_output_dim = output_param["dim"];
    if (first_output_dim == 0) {
        output_skip_ = output_param["skip"];
    }

    // Read per-output parameters
    uint32_t global_n_heads = task_config_.value("n_heads", 0u);
    uint32_t global_matmul_block_size = task_config_.value("matmul_block_size", 0u);
    for (auto& [name, param] : task_config_["task_output_param"].items()) {
        OutputParam op;
        op.dim = param["dim"];
        op.channel = param.value("channel", 1);
        op.is_mat = param.value("data_type", std::string("")) == "feature_mat";
        if (op.is_mat) {
            op.height = param["shape"][0];
            op.width = param["shape"][1];
            if (param.contains("head_shape")) {
                op.head_shape = {param["head_shape"][0], param["head_shape"][1]};
            }
            op.matmul_block_size = param.value("matmul_block_size", global_matmul_block_size);
            op.n_heads = param.value("n_heads", global_n_heads);
        } else if (op.dim == 0) {
            op.skip = param["skip"];
        } else if (op.dim == 1) {
            op.length = param["shape"][0];
            if (param.contains("invalid_fill")) {
                op.invalid_fill[0] = param["invalid_fill"][0];
                if (param["invalid_fill"].size() > 1) {
                    op.invalid_fill[1] = param["invalid_fill"][1];
                }
            }
        } else if (op.dim == 2) {
            op.height = param["shape"][0];
            op.width = param["shape"][1];
            if (param.contains("invalid_fill")) {
                op.invalid_fill = {param["invalid_fill"][0], param["invalid_fill"][1]};
            }
        }
        output_params_[name] = op;
    }

    // Read per-input parameters
    for (auto& [name, param] : task_config_["task_input_param"].items()) {
        InputParam ip;
        ip.dim = param["dim"];
        ip.level = param["level"];
        ip.channel = param.value("channel", 1);
        ip.is_mat = param.value("data_type", std::string("")) == "feature_mat";
        if (ip.is_mat) {
            ip.height = param["shape"][0];
            ip.width = param["shape"][1];
            if (param.contains("head_shape")) {
                ip.head_shape = {param["head_shape"][0], param["head_shape"][1]};
            }
            ip.matmul_block_size = param.value("matmul_block_size", global_matmul_block_size);
            ip.n_heads = param.value("n_heads", global_n_heads);
        } else if (ip.dim == 2) {
            ip.height = param["shape"][0];
            ip.width = param["shape"][1];
        } else if (ip.dim == 1) {
            ip.length = param["shape"][0];
        } else if (ip.dim == 0) {
            ip.skip = param.value("skip", 1);
        }
        if (!ip.is_mat) {
            ip.pack_num = param.value("pack_num", 0);
        }
        input_params_[name] = ip;
    }

    // Compute par_block_size from the first input feature (if par format)
    if (global_n_heads > 1 && !input_params_.empty()) {
        auto& first_ip = input_params_.begin()->second;
        if (first_ip.is_mat) {
            par_block_size_ =
                first_ip.matmul_block_size != 0 ? first_ip.matmul_block_size : first_ip.width / global_n_heads;
        }
    }

    // Use first input's ckks params for context setup
    auto& first_param = task_config_["task_input_param"].begin().value();
    auto ckks_config = read_json((client_dir_ / "ckks_parameter.json").string());
    std::string ckks_param_id = first_param["ckks_parameter_id"];
    auto& ckks_entry = ckks_config[ckks_param_id];
    poly_modulus_degree_ = ckks_entry["poly_modulus_degree"].get<int>();
    n_slots_ = poly_modulus_degree_ / 2;
    if (ckks_entry.contains("q") && ckks_entry.contains("p")) {
        q_ = ckks_entry["q"].get<std::vector<uint64_t>>();
        p_ = ckks_entry["p"].get<std::vector<uint64_t>>();
    }
}

void InferenceClient::create_crypto_context() {
    std::cout << "[Client] Generating CKKS context and keys..." << std::endl;
    std::cout << "[Client] Bootstrapping: " << (needs_btp_ ? "Yes" : "No") << std::endl;
    std::cout << "[Client] Poly degree: N=" << poly_modulus_degree_ << std::endl;

    if (needs_btp_) {
        btp_param_ = std::make_unique<CkksBtpParameter>(CkksBtpParameter::create_parameter());
        btp_context_ = std::make_unique<CkksBtpContext>(CkksBtpContext::create_random_context(*btp_param_));
        btp_context_->gen_rotation_keys();
        context_ptr_ = btp_context_.get();
    } else {
        if (!q_.empty() && !p_.empty()) {
            ckks_param_ =
                std::make_unique<CkksParameter>(CkksParameter::create_custom_parameter(poly_modulus_degree_, q_, p_));
        } else {
            ckks_param_ = std::make_unique<CkksParameter>(CkksParameter::create_parameter(poly_modulus_degree_));
        }
        ckks_context_ = std::make_unique<CkksContext>(CkksContext::create_random_context(*ckks_param_));
        ckks_context_->gen_rotation_keys();
        context_ptr_ = ckks_context_.get();
    }

    std::cout << "[Client] Done." << std::endl;
}

double InferenceClient::get_default_scale() const {
    return context_ptr_->get_parameter().get_default_scale();
}

void InferenceClient::setup() {
    read_configuration();
    create_crypto_context();
}

Bytes InferenceClient::export_eval_context() const {
    std::cout << "[Client] Exporting evaluation context..." << std::endl;
    Bytes result;
    if (needs_btp_) {
        auto pub_ctx = btp_context_->make_public_context();
        std::cout << "[Client] Serializing BTP context..." << std::endl;
        result = pub_ctx.serialize();
    } else {
        auto pub_ctx = ckks_context_->make_public_context();
        std::cout << "[Client] Serializing CKKS context..." << std::endl;
        result = pub_ctx.serialize_advanced();
    }
    std::cout << "[Client] Done." << std::endl;
    return result;
}

std::map<std::string, Bytes> InferenceClient::encrypt(const std::map<std::string, std::string>& input_csvs) const {
    std::map<std::string, Bytes> result;
    double scale = get_default_scale();

    for (auto& [name, csv_path] : input_csvs) {
        auto it = input_params_.find(name);
        if (it == input_params_.end()) {
            throw std::runtime_error("[Client] Unknown input name: " + name);
        }
        const auto& param = it->second;

        std::cout << "[Client] Encrypting input '" << name << "' (dim=" << param.dim << ", is_mat=" << param.is_mat
                  << ")..." << std::endl;

        if (param.is_mat) {
            if (param.n_heads <= 1) {
                throw std::runtime_error("[Client] feature_mat input only supports par matrix ops with n_heads > 1: " +
                                         name);
            }
            if (param.width % param.n_heads != 0) {
                throw std::runtime_error("[Client] feature_mat width must be divisible by n_heads: " + name);
            }
            auto input_array = csv_to_array<2>(csv_path, {(uint64_t)param.height, (uint64_t)param.width});
            FeatureMatEncrypted input_ct(context_ptr_, param.level);
            uint32_t head_dim = param.head_shape[1] != 0 ? param.head_shape[1] : param.width / param.n_heads;
            uint32_t d = param.matmul_block_size != 0 ? param.matmul_block_size : head_dim;
            input_ct.par_block_col_major_pack(input_array, d, param.n_heads, head_dim, false, scale);
            result[name] = input_ct.serialize();
        } else if (param.dim == 0) {
            auto input_array = csv_to_array<1>(csv_path);
            Feature0DEncrypted input_ct(context_ptr_, param.level);
            uint32_t input_skip = n_slots_ / param.pack_num;
            input_ct.pack(input_array, false, scale, input_skip);
            result[name] = input_ct.serialize();
        } else if (param.dim == 1) {
            auto input_array = csv_to_array<2>(csv_path, {(uint64_t)param.channel, (uint64_t)param.length});
            uint32_t skip = param.pack_num > 0 ? (uint32_t)(n_slots_ / (param.length * param.pack_num)) : 1;
            Feature1DEncrypted input_ct(context_ptr_, param.level, skip);
            if (pack_style_ == "ordinary") {
                input_ct.pack(input_array, false, scale);
            } else {
                input_ct.pack_multiplexed(input_array, false, scale);
            }
            result[name] = input_ct.serialize();
        } else {
            auto input_array =
                csv_to_array<3>(csv_path, {(uint64_t)param.channel, (uint64_t)param.height, (uint64_t)param.width});
            Feature2DEncrypted input_ct(context_ptr_, param.level, Duo{1, 1});

            if (pack_style_ == "ordinary") {
                input_ct.pack_multiple_channel(input_array, false, scale);
            } else if (param.height * param.width > n_slots_) {
                Duo block_shape = {task_config_["block_shape"][0], task_config_["block_shape"][1]};
                Duo channel_packing_factor = {(uint32_t)(param.height / block_shape[0]),
                                              (uint32_t)(param.width / block_shape[1])};
                input_ct.pack_interleaved(input_array, block_shape, channel_packing_factor, false, scale);
            } else {
                input_ct.pack_multiplexed(input_array, false, scale);
            }
            result[name] = input_ct.serialize();
        }
        std::cout << "[Client] Done." << std::endl;
    }

    return result;
}

std::map<std::string, DecryptedOutput>
InferenceClient::decrypt(const std::map<std::string, Bytes>& encrypted_outputs) const {
    std::map<std::string, DecryptedOutput> results;

    for (auto& [name, bytes] : encrypted_outputs) {
        auto it = output_params_.find(name);
        if (it == output_params_.end()) {
            throw std::runtime_error("[Client] Unknown output name: " + name);
        }
        const auto& param = it->second;
        std::cout << "[Client] Decrypting output '" << name << "' (dim=" << param.dim << ")..." << std::endl;

        DecryptedOutput result;
        if (param.is_mat) {
            if (param.n_heads <= 1) {
                throw std::runtime_error("[Client] feature_mat output only supports par matrix ops with n_heads > 1: " +
                                         name);
            }
            FeatureMatEncrypted output_ct(context_ptr_, 0);
            output_ct.deserialize(bytes);
            uint32_t d = output_ct.matmul_block_size != 0 ? output_ct.matmul_block_size : par_block_size_;
            if (d == 0) {
                throw std::runtime_error("[Client] feature_mat output is missing par block size: " + name);
            }
            Array<double, 2> decrypted;
            uint32_t m_per_head = param.height;
            uint32_t n_per_head = d;
            if (param.head_shape[0] != 0 && param.head_shape[1] != 0) {
                n_per_head = param.head_shape[1];
                if (param.height == param.head_shape[0] * param.n_heads && param.width == param.head_shape[1]) {
                    m_per_head = param.height;
                } else {
                    m_per_head = param.head_shape[0];
                }
            } else if (param.height % param.n_heads == 0 && param.height > d) {
                m_per_head = param.height / param.n_heads;
            }
            decrypted = output_ct.par_block_col_major_unpack(m_per_head, n_per_head, d, param.n_heads);
            auto dec_1d = decrypted.to_array_1d();
            result.output = std::vector<double>(dec_1d.data(), dec_1d.data() + dec_1d.size());
        } else if (param.dim == 0) {
            Feature0DEncrypted output_ct(context_ptr_, 0);
            output_ct.deserialize(bytes);
            output_ct.skip = param.skip;
            auto decrypted = output_ct.unpack();
            auto dec_1d = decrypted.to_array_1d();
            result.output = std::vector<double>(dec_1d.data(), dec_1d.data() + dec_1d.size());
        } else if (param.dim == 1) {
            Feature1DEncrypted output_ct(context_ptr_, 0);
            output_ct.deserialize(bytes);
            Array<double, 2> decrypted;
            if (pack_style_ == "multiplexed") {
                output_ct.invalid_fill = param.invalid_fill[0];
                decrypted = output_ct.unpack_multiplexed();
            } else {
                decrypted = output_ct.unpack();
            }
            auto dec_1d = decrypted.to_array_1d();
            result.output = std::vector<double>(dec_1d.data(), dec_1d.data() + dec_1d.size());
        } else {
            Feature2DEncrypted output_ct(context_ptr_, 0, Duo{1, 1});
            output_ct.deserialize(bytes);
            Array<double, 3> decrypted;
            if (pack_style_ == "multiplexed") {
                Duo block_shape = {task_config_["block_shape"][0], task_config_["block_shape"][1]};
                if (param.height * param.width > (int)(block_shape[0] * block_shape[1])) {
                    Duo stride = {(uint32_t)(param.height / block_shape[0]), (uint32_t)(param.width / block_shape[1])};
                    decrypted = output_ct.unpack_interleaved(block_shape, stride);
                } else {
                    output_ct.invalid_fill = param.invalid_fill;
                    decrypted = output_ct.unpack_multiplexed();
                }
            } else {
                decrypted = output_ct.unpack_multiple_channel();
            }
            auto dec_1d = decrypted.to_array_1d();
            result.output = std::vector<double>(dec_1d.data(), dec_1d.data() + dec_1d.size());
        }
        result.num_outputs = result.output.size();
        results[name] = std::move(result);
        std::cout << "[Client] Done." << std::endl;
    }

    return results;
}
