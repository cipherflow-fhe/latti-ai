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

#include "interface/inference_interface.h"

#include <iostream>
#include <map>

EncryptedInference::EncryptedInference(const std::string& task_dir, bool use_gpu)
    : task_dir_(task_dir), use_gpu_(use_gpu) {}

EncryptedInference::~EncryptedInference() = default;

void EncryptedInference::read_configuration() {
    auto client_dir = task_dir_ / "client";
    task_config_ = read_json((client_dir / "task_config.json").string());
    auto& input_param = task_config_["task_input_param"].begin().value();
    auto& output_param = task_config_["task_output_param"].begin().value();

    level_ = input_param["level"];
    output_skip_ = output_param["skip"];
    channel_ = input_param["channel"];
    height_ = input_param["shape"][0];
    width_ = input_param["shape"][1];
    pack_style_ = task_config_["pack_style"];

    auto ckks_config = read_json((client_dir / "ckks_parameter.json").string());
    std::string ckks_param_id = input_param["ckks_parameter_id"];
    poly_modulus_degree_ = ckks_config[ckks_param_id]["poly_modulus_degree"].get<int>();
    n_slots_ = poly_modulus_degree_ / 2;
    needs_btp_ = (poly_modulus_degree_ > 16384);
}

void EncryptedInference::create_crypto_context() {
    std::cout << "[Context] Generating CKKS context and keys..." << std::endl;
    std::cout << "[Context] Bootstrapping: " << (needs_btp_ ? "Yes" : "No") << std::endl;
    std::cout << "[Context] Poly degree: N=" << poly_modulus_degree_ << std::endl;

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

    std::cout << "[Context] Done." << std::endl;
}

double EncryptedInference::get_default_scale() const {
    return context_ptr_->get_parameter().get_default_scale();
}

void EncryptedInference::encrypt(const std::string& input_csv_path) {
    read_configuration();
    create_crypto_context();

    // Read the input image from CSV file with shape [channel, height, width].
    input_array_ = csv_to_array<3>(input_csv_path, {(uint64_t)channel_, (uint64_t)height_, (uint64_t)width_});

    std::cout << "[Encrypt] Encrypting input..." << std::endl;
    input_ct_ = make_unique<Feature2DEncrypted>(context_ptr_, level_, Duo{1, 1});
    double scale = get_default_scale();

    if (pack_style_ == "ordinary") {
        input_ct_->pack(input_array_, false, scale);
    } else if (height_ * width_ > n_slots_) {
        Duo block_shape = {task_config_["block_shape"][0], task_config_["block_shape"][1]};
        Duo channel_packing_factor = {(uint32_t)(height_ / block_shape[0]), (uint32_t)(width_ / block_shape[1])};
        input_ct_->split_with_stride_pack(input_array_, block_shape, channel_packing_factor, false, scale);
    } else {
        input_ct_->par_mult_pack(input_array_, false, scale);
    }

    std::cout << "[Encrypt] Done." << std::endl;
}

void EncryptedInference::evaluate() {
    // Load model
    auto server_dir = task_dir_ / "server";
    init_ = std::make_unique<InitInferenceProcess>(server_dir.string() + "/", false);
    init_->init_parameters(needs_btp_);
    init_->is_lazy = false;
    init_->load_model_prepare();

    // Configure inference engine
    fp_ = std::make_unique<InferenceProcess>(init_.get(), true);
    fp_->available_keys.push_back("input");

    // Transfer crypto context
    std::map<std::string, std::unique_ptr<CkksContext>> context_map;
    if (needs_btp_) {
        context_map["param0"] = std::make_unique<CkksBtpContext>(std::move(btp_context_->shallow_copy_context()));
    } else {
        context_map["param0"] = std::make_unique<CkksContext>(std::move(ckks_context_->shallow_copy_context()));
    }
    fp_->ckks_contexts = std::move(context_map);
    fp_->set_feature("input", std::make_unique<Feature2DEncrypted>(std::move(*input_ct_)));

    // Run encrypted inference
    fp_->compute_device = use_gpu_ ? ComputeDevice::GPU : ComputeDevice::CPU;
    std::cout << "[Inference] Running encrypted inference..." << std::endl;
    std::cout << "[Inference] Device: " << (use_gpu_ ? "GPU" : "CPU") << std::endl;
    Timer timer;
    timer.start();
    fp_->run_task();
    timer.stop();
    timer.print("Encrypted inference time");
    std::cout << "[Inference] Done." << std::endl;
}

InferenceResult EncryptedInference::decrypt() {
    InferenceResult result;

    // Decrypt output
    auto encrypted_output = fp_->get_ciphertext_output_feature0D("output");
    encrypted_output.skip = output_skip_;
    auto decrypted = encrypted_output.unpack(DecryptType::SPARSE);
    auto dec_1d = decrypted.to_array_1d();
    result.encrypted_output = std::vector<double>(dec_1d.data(), dec_1d.data() + dec_1d.size());

    // Plaintext verification
    fp_->p_feature2d_x["input"] = std::move(input_array_);
    fp_->run_task_plaintext();
    result.plaintext_output = fp_->p_feature0d_x["output"];
    result.num_outputs = result.plaintext_output.size();

    return result;
}
