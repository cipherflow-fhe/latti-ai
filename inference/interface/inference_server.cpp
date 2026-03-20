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

    // Store all input keys and per-input parameters
    for (auto& [name, param] : task_config["task_input_param"].items()) {
        input_keys_.push_back(name);
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

    // Store all output keys and per-output parameters
    for (auto& [name, param] : task_config["task_output_param"].items()) {
        output_keys_.push_back(name);
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
    for (auto& key : input_keys_) {
        fp_->available_keys.push_back(key);
    }

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

std::map<std::string, Bytes> InferenceServer::evaluate(const std::map<std::string, Bytes>& encrypted_inputs) {
    // Deserialize and set all input ciphertexts
    for (auto& [name, bytes] : encrypted_inputs) {
        auto it = input_params_.find(name);
        if (it == input_params_.end()) {
            throw std::runtime_error("[Server] Unknown input name: " + name);
        }
        const auto& param = it->second;

        if (param.dim == 0) {
            auto input_ct = std::make_unique<Feature0DEncrypted>(context_ptr_, 0);
            input_ct->deserialize(bytes);
            fp_->set_feature(name, std::move(input_ct));
        } else {
            auto input_ct = std::make_unique<Feature2DEncrypted>(context_ptr_, 0);
            input_ct->deserialize(bytes);
            fp_->set_feature(name, std::move(input_ct));
        }
    }

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

    // Serialize output ciphertexts
    std::map<std::string, Bytes> encrypted_outputs;
    for (auto& [name, param] : output_params_) {
        if (param.dim == 0) {
            auto output_ct = fp_->get_ciphertext_output_feature0D(name);
            encrypted_outputs[name] = output_ct.serialize();
        } else {
            auto output_ct = fp_->get_ciphertext_output_feature2D(name);
            encrypted_outputs[name] = output_ct.serialize();
        }
    }
    return encrypted_outputs;
}

std::map<std::string, std::vector<double>>
InferenceServer::evaluate_plaintext(const std::map<std::string, std::string>& input_csvs) {
    for (auto& [name, csv_path] : input_csvs) {
        auto it = input_params_.find(name);
        if (it == input_params_.end()) {
            throw std::runtime_error("[Server] Unknown input name: " + name);
        }
        const auto& param = it->second;

        if (param.dim == 0) {
            auto input_array = csv_to_array<1>(csv_path);
            fp_->p_feature0d_x[name] = input_array.to_array_1d();
        } else {
            auto input_array =
                csv_to_array<3>(csv_path, {(uint64_t)param.channel, (uint64_t)param.height, (uint64_t)param.width});
            fp_->p_feature2d_x[name] = std::move(input_array.copy());
        }
    }
    fp_->run_task_plaintext();

    std::map<std::string, std::vector<double>> results;
    for (auto& [name, param] : output_params_) {
        if (param.dim == 0) {
            results[name] = fp_->p_feature0d_x[name];
        } else {
            auto& arr = fp_->p_feature2d_x[name];
            auto arr_1d = arr.to_array_1d();
            results[name] = std::vector<double>(arr_1d.data(), arr_1d.data() + arr_1d.size());
        }
    }
    return results;
}
