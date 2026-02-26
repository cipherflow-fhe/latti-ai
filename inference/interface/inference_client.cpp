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
    auto& input_param = task_config_["task_input_param"].begin().value();
    auto& output_param = task_config_["task_output_param"].begin().value();

    level_ = input_param["level"];
    output_skip_ = output_param["skip"];
    channel_ = input_param["channel"];
    height_ = input_param["shape"][0];
    width_ = input_param["shape"][1];
    pack_style_ = task_config_["pack_style"];

    auto ckks_config = read_json((client_dir_ / "ckks_parameter.json").string());
    std::string ckks_param_id = input_param["ckks_parameter_id"];
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
        result = pub_ctx.serialize_advanced();
    }
    std::cout << "[Client] Done." << std::endl;
    return result;
}

Bytes InferenceClient::encrypt(const std::string& input_csv) const {
    // Read the input image from CSV file with shape [channel, height, width].
    auto input_array = csv_to_array<3>(input_csv, {(uint64_t)channel_, (uint64_t)height_, (uint64_t)width_});

    std::cout << "[Client] Encrypting input..." << std::endl;
    Feature2DEncrypted input_ct(context_ptr_, level_, Duo{1, 1});
    double scale = get_default_scale();

    if (pack_style_ == "ordinary") {
        input_ct.pack(input_array, false, scale);
    } else if (height_ * width_ > n_slots_) {
        Duo block_shape = {task_config_["block_shape"][0], task_config_["block_shape"][1]};
        Duo channel_packing_factor = {(uint32_t)(height_ / block_shape[0]), (uint32_t)(width_ / block_shape[1])};
        input_ct.split_with_stride_pack(input_array, block_shape, channel_packing_factor, false, scale);
    } else {
        input_ct.par_mult_pack(input_array, false, scale);
    }

    std::cout << "[Client] Done." << std::endl;
    return input_ct.serialize();
}

DecryptedOutput InferenceClient::decrypt(const Bytes& encrypted_output) const {
    std::cout << "[Client] Decrypting output..." << std::endl;
    Feature0DEncrypted output_ct(context_ptr_, 0);
    output_ct.deserialize(encrypted_output);
    output_ct.skip = output_skip_;

    auto decrypted = output_ct.unpack(DecryptType::SPARSE);
    auto dec_1d = decrypted.to_array_1d();

    DecryptedOutput result;
    result.output = std::vector<double>(dec_1d.data(), dec_1d.data() + dec_1d.size());
    result.num_outputs = result.output.size();
    std::cout << "[Client] Done." << std::endl;
    return result;
}
