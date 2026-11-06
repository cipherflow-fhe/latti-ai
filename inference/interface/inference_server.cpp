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
#include <fstream>
#include <limits>
#include <map>

#include "interface/inference_server.h"
#include "data_structs/feature_mat.h"
#ifdef INFERENCE_SDK_ENABLE_GPU
#include "lattisense/mega_ag_runners/wrapper.h"
#endif

using namespace lattisense;

namespace {

bool is_par_diagonal_pack(const std::string& mat_pack_style) {
    return mat_pack_style == "par_diagonal_pack";
}

int read_skip(const json& param, int default_value = 1) {
    if (!param.contains("skip")) {
        return default_value;
    }
    const auto& skip = param.at("skip");
    if (skip.is_array()) {
        return skip.empty() ? default_value : skip.at(0).get<int>();
    }
    return skip.get<int>();
}

}  // namespace

InferenceServer::InferenceServer(const std::string& server_dir, bool use_gpu, int gpu_device)
    : server_dir_(server_dir), use_gpu_(use_gpu), gpu_device_(gpu_device) {}

InferenceServer::~InferenceServer() {
    try {
        release_device();
    } catch (...) {
        // Destruction is best-effort and must not throw across SDK boundaries.
    }
}

void InferenceServer::import_eval_context_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("[Server] Cannot open evaluation context: " + path);
    }
    const auto size = input.tellg();
    if (size <= 0 || static_cast<uintmax_t>(size) > Bytes{}.max_size() ||
        static_cast<uintmax_t>(size) > static_cast<uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("[Server] Invalid evaluation context file size: " + path);
    }
    Bytes data(static_cast<size_t>(size));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size))) {
        throw std::runtime_error("[Server] Short read of evaluation context: " + path);
    }
    import_eval_context(data);
}

void InferenceServer::import_eval_context(const Bytes& eval_context) {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    if (fp_ || evaluating_) {
        throw std::runtime_error("RESOURCE_BUSY: release the loaded model before importing a context");
    }
    // Determine whether bootstrapping is needed from the server task config.
    auto task_config = read_json((server_dir_ / "task_config.json").string());
    auto ckks_config = read_json((server_dir_ / "ckks_parameter.json").string());
    auto& input_param = task_config["task_input_param"].begin().value();
    std::string ckks_param_id = input_param["ckks_parameter_id"];
    int poly_modulus_degree = ckks_config[ckks_param_id]["poly_modulus_degree"].get<int>();
    const bool needs_btp = task_config.value("use_btp", false);
    const std::string mat_pack_style = task_config.value<std::string>("mat_pack_style", "");

    std::vector<std::string> input_keys, output_keys;
    std::map<std::string, InputParam> input_params;
    std::map<std::string, OutputParam> output_params;

    // Store all input keys and per-input parameters
    uint32_t global_n_heads = task_config.value("n_heads", 0u);
    uint32_t global_matmul_block_size = task_config.value("matmul_block_size", 0u);
    for (auto& [name, param] : task_config["task_input_param"].items()) {
        input_keys.push_back(name);
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
            if (param.contains("head_shape")) {
                ip.head_shape = {param["head_shape"][0].get<uint32_t>(), param["head_shape"][1].get<uint32_t>()};
            }
        } else if (ip.dim == 2) {
            ip.height = param["shape"][0];
            ip.width = param["shape"][1];
        } else if (ip.dim == 1) {
            ip.length = param["shape"][0];
            ip.skip = read_skip(param);
        } else if (ip.dim == 0) {
            ip.skip = read_skip(param);
        }
        if (!ip.is_mat) {
            ip.pack_num = param.value("pack_num", 0);
        }
        input_params[name] = ip;
    }

    // Store all output keys and per-output parameters
    for (auto& [name, param] : task_config["task_output_param"].items()) {
        output_keys.push_back(name);
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
            if (param.contains("head_shape")) {
                op.head_shape = {param["head_shape"][0].get<uint32_t>(), param["head_shape"][1].get<uint32_t>()};
            }
        } else if (op.dim == 0) {
            op.skip = read_skip(param);
        } else if (op.dim == 1) {
            op.length = param["shape"][0];
        } else if (op.dim == 2) {
            op.height = param["shape"][0];
            op.width = param["shape"][1];
        }
        output_params[name] = op;
    }

    std::cout << "[Server] Importing evaluation context..." << std::endl;
    std::cout << "[Server] Bootstrapping: " << (needs_btp ? "Yes" : "No") << std::endl;

    auto context = std::make_shared<CkksContext>(CkksContext::deserialize(eval_context));
    input_keys_ = std::move(input_keys);
    output_keys_ = std::move(output_keys);
    input_params_ = std::move(input_params);
    output_params_ = std::move(output_params);
    needs_btp_ = needs_btp;
    mat_pack_style_ = mat_pack_style;
    eval_context_ = std::move(context);
    context_ptr_ = eval_context_.get();

    std::cout << "[Server] Done." << std::endl;
}

void InferenceServer::load_model() {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    if (fp_) {
        throw std::runtime_error("[Server] Model is already loaded");
    }
    if (!eval_context_) {
        throw std::runtime_error("[Server] No evaluation context imported");
    }
    if (pool_release_pending_) {
        throw std::runtime_error("RESOURCE_ERROR: retry release_device before loading the model");
    }
    std::cout << "[Server] Loading model..." << std::endl;

    auto init = std::make_unique<InitInferenceProcess>(server_dir_.string() + "/", false);
    init->init_parameters(needs_btp_);
    init->is_lazy = true;
    init->load_model_prepare();

    auto process = std::make_unique<InferenceProcess>(init.get());
    for (auto& key : input_keys_) {
        process->available_keys.push_back(key);
    }

    // The server retains the same host context across device reloads.
    process->ckks_contexts["param0"] = eval_context_;

    if (use_gpu_) {
        process->compute_device = ComputeDevice::GPU;
        process->gpu_device = gpu_device_;
    }
    process->prepare_task();
    init_ = std::move(init);
    fp_ = std::move(process);

    std::cout << "[Server] Done." << std::endl;
}

void InferenceServer::release_device() {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    if (evaluating_) {
        throw std::runtime_error("RESOURCE_BUSY: inference is running");
    }
    if (!fp_ && !pool_release_pending_) {
        return;
    }
#ifdef INFERENCE_SDK_ENABLE_GPU
    const bool release_pool = use_gpu_ || pool_release_pending_;
    const bool gpu_task = fp_ && static_cast<bool>(fp_->fhe_task_gpu_);
    if (release_pool && begin_fhe_gpu_resource_release(gpu_task) != FHE_TASK_OK) {
        throw std::runtime_error("RESOURCE_BUSY: another GPU owner is active");
    }
#endif
    fp_.reset();
    init_.reset();
#ifdef INFERENCE_SDK_ENABLE_GPU
    if (release_pool) {
        pool_release_pending_ = true;
        if (finish_fhe_gpu_resource_release(gpu_device_) != FHE_TASK_OK) {
            throw std::runtime_error("RESOURCE_ERROR: GPU resource release failed; host context retained");
        }
        pool_release_pending_ = false;
    }
#endif
}

std::map<std::string, Bytes> InferenceServer::evaluate(const std::map<std::string, Bytes>& encrypted_inputs,
                                                       lattisense::ProgressCallback progress_cb) {
    std::unique_lock<std::mutex> state_lock(state_mutex_);
    if (!fp_) {
        throw std::runtime_error("[Server] Model is not loaded");
    }
    if (evaluating_) {
        throw std::runtime_error("RESOURCE_BUSY: inference is already running");
    }
    evaluating_ = true;
    struct EvaluationGuard {
        InferenceServer& server;
        ~EvaluationGuard() {
            std::lock_guard<std::mutex> lock(server.state_mutex_);
            server.evaluating_ = false;
        }
    } evaluation_guard{*this};
    state_lock.unlock();
    // Deserialize and set all input ciphertexts
    for (auto& [name, bytes] : encrypted_inputs) {
        auto it = input_params_.find(name);
        if (it == input_params_.end()) {
            throw std::runtime_error("[Server] Unknown input name: " + name);
        }
        const auto& param = it->second;

        if (param.is_mat) {
            auto input_ct = std::make_unique<FeatureMatEncrypted>(context_ptr_, 0);
            input_ct->deserialize(bytes);
            input_ct->shape = {static_cast<uint32_t>(param.height), static_cast<uint32_t>(param.width)};
            input_ct->head_shape = param.head_shape;
            if (input_ct->matmul_block_size == 0) {
                input_ct->matmul_block_size = param.matmul_block_size;
            }
            fp_->set_feature(name, std::move(input_ct));
        } else if (param.dim == 0) {
            auto input_ct = std::make_unique<Feature0DEncrypted>(context_ptr_, 0);
            input_ct->deserialize(bytes);
            fp_->set_feature(name, std::move(input_ct));
        } else if (param.dim == 1) {
            auto input_ct = std::make_unique<Feature1DEncrypted>(context_ptr_, 0);
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
    fp_->run_task_lazy(false, progress_cb);
    timer.stop();
    timer.print("Encrypted inference time");
    std::cout << "[Server] Done." << std::endl;

    // Serialize output ciphertexts
    std::map<std::string, Bytes> encrypted_outputs;
    for (auto& [name, param] : output_params_) {
        if (param.is_mat) {
            const auto& output_ct = dynamic_cast<const FeatureMatEncrypted&>(fp_->_get_feature(name));
            encrypted_outputs[name] = output_ct.serialize();
        } else if (param.dim == 0) {
            auto output_ct = fp_->get_ciphertext_output_feature<Feature0DEncrypted>(name);
            encrypted_outputs[name] = output_ct.serialize();
        } else if (param.dim == 1) {
            auto output_ct = fp_->get_ciphertext_output_feature<Feature1DEncrypted>(name);
            encrypted_outputs[name] = output_ct.serialize();
        } else {
            auto output_ct = fp_->get_ciphertext_output_feature<Feature2DEncrypted>(name);
            encrypted_outputs[name] = output_ct.serialize();
        }
    }
    return encrypted_outputs;
}

std::map<std::string, std::vector<double>>
InferenceServer::evaluate_plaintext(const std::map<std::string, std::string>& input_csvs) {
    std::unique_lock<std::mutex> state_lock(state_mutex_);
    if (!fp_) {
        throw std::runtime_error("[Server] Model is not loaded");
    }
    if (evaluating_) {
        throw std::runtime_error("RESOURCE_BUSY: inference is already running");
    }
    evaluating_ = true;
    struct EvaluationGuard {
        InferenceServer& server;
        ~EvaluationGuard() {
            std::lock_guard<std::mutex> lock(server.state_mutex_);
            server.evaluating_ = false;
        }
    } evaluation_guard{*this};
    state_lock.unlock();
    fp_->p_feature0d_x.clear();
    fp_->p_feature1d_x.clear();
    fp_->p_feature2d_x.clear();
    fp_->available_keys.clear();

    for (auto& [name, csv_path] : input_csvs) {
        auto it = input_params_.find(name);
        if (it == input_params_.end()) {
            throw std::runtime_error("[Server] Unknown input name: " + name);
        }
        const auto& param = it->second;

        if (param.is_mat) {
            if (is_par_diagonal_pack(mat_pack_style_)) {
                auto csv_array = csv_to_array<2>(csv_path, {(uint64_t)param.width, (uint64_t)param.height});
                auto input_array = transpose_2d_array(csv_array);
                fp_->p_feature_mat_x[name] = std::move(input_array.copy());
            } else {
                auto input_array = csv_to_array<2>(csv_path, {(uint64_t)param.height, (uint64_t)param.width});
                fp_->p_feature_mat_x[name] = std::move(input_array.copy());
            }
        } else if (param.dim == 0) {
            auto input_array = csv_to_array<1>(csv_path);
            fp_->p_feature0d_x[name] = input_array.to_array_1d();
        } else if (param.dim == 1) {
            auto input_array = csv_to_array<2>(csv_path, {(uint64_t)param.channel, (uint64_t)param.length});
            fp_->p_feature1d_x[name] = std::move(input_array.copy());
        } else {
            auto input_array =
                csv_to_array<3>(csv_path, {(uint64_t)param.channel, (uint64_t)param.height, (uint64_t)param.width});
            fp_->p_feature2d_x[name] = std::move(input_array.copy());
        }
        fp_->available_keys.push_back(name);
    }
    fp_->run_task_plaintext();

    std::map<std::string, std::vector<double>> results;
    for (auto& [name, param] : output_params_) {
        if (param.is_mat) {
            if (!fp_->p_feature_mat_x.count(name)) {
                throw std::runtime_error("[Server] Plaintext output not produced: " + name);
            }
            auto& arr = fp_->p_feature_mat_x[name];
            auto arr_1d =
                is_par_diagonal_pack(mat_pack_style_) ? transpose_2d_array(arr).to_array_1d() : arr.to_array_1d();
            results[name] = std::vector<double>(arr_1d.data(), arr_1d.data() + arr_1d.size());
        } else if (param.dim == 0) {
            if (!fp_->p_feature0d_x.count(name)) {
                throw std::runtime_error("[Server] Plaintext output not produced: " + name);
            }
            results[name] = fp_->p_feature0d_x[name];
        } else if (param.dim == 1) {
            if (!fp_->p_feature1d_x.count(name)) {
                throw std::runtime_error("[Server] Plaintext output not produced: " + name);
            }
            auto& arr = fp_->p_feature1d_x[name];
            auto arr_1d = arr.to_array_1d();
            results[name] = std::vector<double>(arr_1d.data(), arr_1d.data() + arr_1d.size());
        } else {
            if (!fp_->p_feature2d_x.count(name)) {
                throw std::runtime_error("[Server] Plaintext output not produced: " + name);
            }
            auto& arr = fp_->p_feature2d_x[name];
            auto arr_1d = arr.to_array_1d();
            results[name] = std::vector<double>(arr_1d.data(), arr_1d.data() + arr_1d.size());
        }
    }
    return results;
}
