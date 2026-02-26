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

#include "interface/inference_server.h"

#include <iostream>
#include <map>

InferenceServer::InferenceServer(const std::string& server_dir, bool use_gpu)
    : server_dir_(server_dir), use_gpu_(use_gpu) {}

InferenceServer::~InferenceServer() = default;

void InferenceServer::import_eval_context(const Bytes& eval_context) {
    // Determine whether bootstrapping is needed from the server task config.
    auto task_config = read_json((server_dir_ / "task_config.json").string());
    auto ckks_config = read_json((server_dir_ / "ckks_parameter.json").string());
    auto& input_param = task_config["task_input_param"].begin().value();
    std::string ckks_param_id = input_param["ckks_parameter_id"];
    int poly_modulus_degree = ckks_config[ckks_param_id]["poly_modulus_degree"].get<int>();
    needs_btp_ = (poly_modulus_degree > 16384);
    channel_ = input_param["channel"];
    height_ = input_param["shape"][0];
    width_ = input_param["shape"][1];

    std::cout << "[Server] Importing evaluation context..." << std::endl;
    std::cout << "[Server] Bootstrapping: " << (needs_btp_ ? "Yes" : "No") << std::endl;

    if (needs_btp_) {
        eval_btp_context_ = std::make_unique<CkksBtpContext>(CkksBtpContext::deserialize(eval_context));
        context_ptr_ = eval_btp_context_.get();
    } else {
        eval_context_ = std::make_unique<CkksContext>(CkksContext::deserialize_advanced(eval_context));
        context_ptr_ = eval_context_.get();
    }

    std::cout << "[Server] Done." << std::endl;
}

void InferenceServer::load_model() {
    std::cout << "[Server] Loading model..." << std::endl;

    init_ = std::make_unique<InitInferenceProcess>(server_dir_.string() + "/", false);
    init_->init_parameters(needs_btp_);
    init_->is_lazy = false;
    init_->load_model_prepare();

    fp_ = std::make_unique<InferenceProcess>(init_.get(), true);
    fp_->available_keys.push_back("input");

    // Transfer eval context directly to inference engine (no shallow_copy)
    std::map<std::string, std::unique_ptr<CkksContext>> context_map;
    if (needs_btp_) {
        context_map["param0"] = std::move(eval_btp_context_);
    } else {
        context_map["param0"] = std::move(eval_context_);
    }
    fp_->ckks_contexts = std::move(context_map);
    context_ptr_ = fp_->ckks_contexts["param0"].get();

    std::cout << "[Server] Done." << std::endl;
}

Bytes InferenceServer::evaluate(const Bytes& encrypted_input) {
    // Deserialize input ciphertext
    Feature2DEncrypted input_ct(context_ptr_, 0);
    input_ct.deserialize(encrypted_input);
    fp_->set_feature("input", std::make_unique<Feature2DEncrypted>(std::move(input_ct)));

    // Run encrypted inference
    fp_->compute_device = use_gpu_ ? ComputeDevice::GPU : ComputeDevice::CPU;
    std::cout << "[Server] Running encrypted inference..." << std::endl;
    std::cout << "[Server] Device: " << (use_gpu_ ? "GPU" : "CPU") << std::endl;
    Timer timer;
    timer.start();
    fp_->run_task();
    timer.stop();
    timer.print("Encrypted inference time");
    std::cout << "[Server] Done." << std::endl;

    // Serialize output ciphertext
    auto encrypted_output = fp_->get_ciphertext_output_feature0D("output");
    return encrypted_output.serialize();
}

std::vector<double> InferenceServer::evaluate_plaintext(const std::string& input_csv) {
    auto input_array = csv_to_array<3>(input_csv, {(uint64_t)channel_, (uint64_t)height_, (uint64_t)width_});
    fp_->p_feature2d_x["input"] = std::move(input_array);
    fp_->run_task_plaintext();
    return fp_->p_feature0d_x["output"];
}
