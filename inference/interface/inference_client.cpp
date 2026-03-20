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

#include "interface/inference_client.h"

#include <iostream>

InferenceClient::InferenceClient(const std::string& client_dir) : client_dir_(client_dir) {}

InferenceClient::~InferenceClient() = default;

void InferenceClient::read_configuration() {
    task_config_ = read_json((client_dir_ / "task_config.json").string());
    pack_style_ = task_config_["pack_style"];

    auto& output_param = task_config_["task_output_param"].begin().value();
    output_skip_ = output_param["skip"];

    // Read per-output parameters
    for (auto& [name, param] : task_config_["task_output_param"].items()) {
        OutputParam op;
        op.dim = param["dim"];
        op.channel = param["channel"];
        if (op.dim == 0) {
            op.skip = param["skip"];
        } else if (op.dim == 2) {
            op.height = param["shape"][0];
            op.width = param["shape"][1];
        }
        output_params_[name] = op;
    }

    // Read per-input parameters
    for (auto& [name, param] : task_config_["task_input_param"].items()) {
        InputParam ip;
        ip.dim = param["dim"];
        ip.level = param["level"];
        ip.channel = param["channel"];
        if (ip.dim == 2) {
            ip.height = param["shape"][0];
            ip.width = param["shape"][1];
        } else if (ip.dim == 0) {
            ip.skip = param.value("skip", 1);
        }
        ip.pack_num = param.value("pack_num", 0);
        input_params_[name] = ip;
    }

    // Use first input's ckks params for context setup
    auto& first_param = task_config_["task_input_param"].begin().value();
    auto ckks_config = read_json((client_dir_ / "ckks_parameter.json").string());
    std::string ckks_param_id = first_param["ckks_parameter_id"];
    poly_modulus_degree_ = ckks_config[ckks_param_id]["poly_modulus_degree"].get<int>();
    n_slots_ = poly_modulus_degree_ / 2;
    needs_btp_ = (poly_modulus_degree_ > 16384);
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
        ckks_param_ = std::make_unique<CkksParameter>(CkksParameter::create_parameter(poly_modulus_degree_));
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

        std::cout << "[Client] Encrypting input '" << name << "' (dim=" << param.dim << ")..." << std::endl;

        if (param.dim == 0) {
            auto input_array = csv_to_array<1>(csv_path);
            Feature0DEncrypted input_ct(context_ptr_, param.level);
            uint32_t input_skip = n_slots_ / param.pack_num;
            input_ct.pack(input_array, false, scale, input_skip);
            result[name] = input_ct.serialize();
        } else {
            auto input_array =
                csv_to_array<3>(csv_path, {(uint64_t)param.channel, (uint64_t)param.height, (uint64_t)param.width});
            Feature2DEncrypted input_ct(context_ptr_, param.level, Duo{1, 1});

            if (pack_style_ == "ordinary") {
                input_ct.pack(input_array, false, scale);
            } else if (param.height * param.width > n_slots_) {
                Duo block_shape = {task_config_["block_shape"][0], task_config_["block_shape"][1]};
                Duo channel_packing_factor = {(uint32_t)(param.height / block_shape[0]),
                                              (uint32_t)(param.width / block_shape[1])};
                input_ct.split_with_stride_pack(input_array, block_shape, channel_packing_factor, false, scale);
            } else {
                input_ct.par_mult_pack(input_array, false, scale);
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
        if (param.dim == 0) {
            Feature0DEncrypted output_ct(context_ptr_, 0);
            output_ct.deserialize(bytes);
            output_ct.skip = param.skip;
            auto decrypted = output_ct.unpack();
            auto dec_1d = decrypted.to_array_1d();
            result.output = std::vector<double>(dec_1d.data(), dec_1d.data() + dec_1d.size());
        } else {
            Feature2DEncrypted output_ct(context_ptr_, 0, Duo{1, 1});
            output_ct.deserialize(bytes);
            auto decrypted = output_ct.unpack();
            auto dec_1d = decrypted.to_array_1d();
            result.output = std::vector<double>(dec_1d.data(), dec_1d.data() + dec_1d.size());
        }
        result.num_outputs = result.output.size();
        results[name] = std::move(result);
        std::cout << "[Client] Done." << std::endl;
    }

    return results;
}
