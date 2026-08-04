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

#include "interface/inference_runner.h"

#include <iostream>
#include <numeric>
#include <stdexcept>

#include "interface/runner_bundle_io.h"
#include "lattisense/cxx_sdk_v2/cxx_fhe_task.h"

namespace {

int read_skip_value(const json& param, int default_value = 1) {
    if (!param.contains("skip")) {
        return default_value;
    }
    const auto& skip = param.at("skip");
    if (skip.is_array()) {
        return skip.empty() ? default_value : skip.at(0).get<int>();
    }
    return skip.get<int>();
}

uint64_t shape_size(const json& sig) {
    const auto shape = sig.at("size").get<std::vector<uint64_t>>();
    return std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<uint64_t>());
}

std::vector<json> online_args_with_phase(const json& signature, const std::string& phase) {
    std::vector<json> result;
    for (const auto& item : signature.at("online")) {
        if (item.at("phase").get<std::string>() == phase) {
            result.push_back(item);
        }
    }
    return result;
}

}  // namespace

InferenceRunner::InferenceRunner(const std::string& runner_dir, bool use_gpu, int gpu_device)
    : runner_dir_(runner_dir), use_gpu_(use_gpu), gpu_device_(gpu_device) {}

InferenceRunner::~InferenceRunner() = default;

void InferenceRunner::read_configuration() {
    task_config_ = read_json((runner_dir_ / "task_config.json").string());
    task_signature_ = read_json((runner_dir_ / "task_signature.json").string());

    if (task_config_.value("deployment_mode", std::string("")) != "server_provisioned_runner") {
        throw std::runtime_error("[Runner] task_config.json is not marked as server_provisioned_runner");
    }
    if (task_config_.value("input_mode", std::string("")) != "plaintext") {
        throw std::runtime_error("[Runner] only plaintext input mode is supported");
    }
    if (task_config_.value("parameter_mode", std::string("")) != "encrypted_offline") {
        throw std::runtime_error("[Runner] only encrypted_offline parameter mode is supported");
    }
    if (task_config_.value("use_btp", false)) {
        throw std::runtime_error("[Runner] bootstrapping contexts are Phase 3/4 work and are not supported yet");
    }
    if (task_signature_.value("algorithm", std::string("")) != "CKKS") {
        throw std::runtime_error("[Runner] only CKKS task signatures are supported");
    }

    for (auto& [name, param] : task_config_.at("task_input_param").items()) {
        InputParam ip;
        ip.dim = param.at("dim").get<int>();
        ip.level = param.at("level").get<int>();
        ip.channel = param.value("channel", 1);
        ip.skip = read_skip_value(param);
        ip.pack_num = param.value("pack_num", 0);
        input_params_[name] = ip;
    }
    for (auto& [name, param] : task_config_.at("task_output_param").items()) {
        OutputParam op;
        op.dim = param.at("dim").get<int>();
        op.channel = param.value("channel", 1);
        op.skip = read_skip_value(param);
        output_params_[name] = op;
    }
}

void InferenceRunner::load() {
    read_configuration();

    const Bytes eval_context = read_binary_file(runner_dir_ / "eval_context.bin");
    eval_context_ = std::make_unique<fhe::CkksContext>(fhe::CkksContext::deserialize_advanced(eval_context));
    n_slots_ = eval_context_->get_parameter().get_n() / 2;
    parameter_store_ = std::make_unique<EncryptedParameterStore>(runner_dir_ / "encrypted_model_parameters");
}

std::vector<fhe::CkksPlaintextRingt> InferenceRunner::encode_plaintext_0d_input(const std::string& name,
                                                                                const std::string& csv_path,
                                                                                const json& sig) const {
    auto param_it = input_params_.find(name);
    if (param_it == input_params_.end()) {
        throw std::runtime_error("[Runner] Unknown input name: " + name);
    }
    const InputParam& param = param_it->second;
    if (param.dim != 0) {
        throw std::runtime_error("[Runner] Phase 2 plaintext runner only supports 0D inputs: " + name);
    }
    if (param.pack_num <= 0) {
        throw std::runtime_error("[Runner] 0D input is missing a positive pack_num: " + name);
    }
    if (sig.at("type").get<std::string>() != "pt_ringt") {
        throw std::runtime_error("[Runner] Phase 2 plaintext runner expects pt_ringt input signature for: " + name);
    }

    auto input_array = csv_to_array<1>(csv_path);
    const uint32_t n_in_features = input_array.get_size();
    const uint32_t n_channel_per_ct = static_cast<uint32_t>(param.pack_num);
    const uint32_t skip = static_cast<uint32_t>(n_slots_ / n_channel_per_ct);
    if (skip == 0 || n_slots_ % n_channel_per_ct != 0) {
        throw std::runtime_error("[Runner] invalid 0D input packing for: " + name);
    }

    std::vector<fhe::CkksPlaintextRingt> encoded;
    const uint64_t expected_count = shape_size(sig);
    const uint64_t actual_count = div_ceil(n_in_features, n_channel_per_ct);
    if (actual_count != expected_count) {
        throw std::runtime_error("[Runner] encoded input count mismatch for " + name + ": expected " +
                                 std::to_string(expected_count) + ", got " + std::to_string(actual_count));
    }

    encoded.reserve(actual_count);
    const double scale = eval_context_->get_parameter().get_default_scale();
    for (uint32_t pack_ct_idx = 0; pack_ct_idx < actual_count; ++pack_ct_idx) {
        std::vector<double> feature_flat(static_cast<size_t>(n_slots_), 0.0);
        for (uint32_t i = 0; i < n_channel_per_ct; ++i) {
            const uint32_t src_idx = pack_ct_idx * n_channel_per_ct + i;
            if (src_idx < n_in_features) {
                feature_flat[i * skip] = input_array.get(src_idx);
            }
        }
        encoded.push_back(eval_context_->encode_ringt(feature_flat, scale));
    }
    return encoded;
}

std::map<std::string, Bytes>
InferenceRunner::evaluate_plaintext_input(const std::map<std::string, std::string>& input_csvs,
                                          ls::ProgressCallback progress_cb) {
    if (!eval_context_ || !parameter_store_) {
        load();
    }

    std::cout << "[Runner] Running server_provisioned_runner task..." << std::endl;

    std::vector<lattisense::CxxVectorArgument> cxx_args;
    std::vector<std::vector<fhe::CkksPlaintextRingt>> online_inputs;
    std::vector<std::unique_ptr<Feature0DEncrypted>> output_features;
    std::vector<std::string> output_names;

    auto input_sigs = online_args_with_phase(task_signature_, "in");
    auto output_sigs = online_args_with_phase(task_signature_, "out");
    online_inputs.reserve(input_sigs.size());
    output_features.reserve(output_sigs.size());
    output_names.reserve(output_sigs.size());

    for (const auto& sig : input_sigs) {
        const std::string id = sig.at("id").get<std::string>();
        auto csv_it = input_csvs.find(id);
        if (csv_it == input_csvs.end()) {
            throw std::runtime_error("[Runner] Missing plaintext CSV input: " + id);
        }
        online_inputs.push_back(encode_plaintext_0d_input(id, csv_it->second, sig));
        cxx_args.push_back(lattisense::CxxVectorArgument{id, &online_inputs.back()});
    }

    for (const auto& sig : task_signature_.at("offline")) {
        const std::string id = sig.at("id").get<std::string>();
        if (sig.at("type").get<std::string>() != "ct") {
            throw std::runtime_error("[Runner] Phase 2 only supports offline ciphertext parameters: " + id);
        }
        auto& values = parameter_store_->load_argument(id);
        cxx_args.push_back(lattisense::CxxVectorArgument{id, &values});
    }

    for (const auto& sig : output_sigs) {
        const std::string id = sig.at("id").get<std::string>();
        auto param_it = output_params_.find(id);
        if (param_it == output_params_.end()) {
            throw std::runtime_error("[Runner] Unknown output name: " + id);
        }
        const OutputParam& param = param_it->second;
        if (param.dim != 0) {
            throw std::runtime_error("[Runner] Phase 2 plaintext runner only supports 0D outputs: " + id);
        }

        const int level = sig.at("level").get<int>();
        const uint64_t count = shape_size(sig);
        auto output = std::make_unique<Feature0DEncrypted>(eval_context_.get(), level);
        output->skip = param.skip;
        output->n_channel_per_ct = param.channel == 0 ? 1 : static_cast<uint32_t>(param.channel);
        if (task_config_.at("task_output_param").at(id).contains("pack_num")) {
            output->n_channel_per_ct = task_config_.at("task_output_param").at(id).at("pack_num").get<uint32_t>();
        }
        output->n_channel = static_cast<uint32_t>(param.channel);
        const double scale = eval_context_->get_parameter().get_default_scale();
        for (uint64_t i = 0; i < count; ++i) {
            output->data.push_back(eval_context_->new_ciphertext(level, scale));
        }
        cxx_args.push_back(lattisense::CxxVectorArgument{id, &output->data});
        output_names.push_back(id);
        output_features.push_back(std::move(output));
    }

    if (use_gpu_) {
#ifdef INFERENCE_SDK_ENABLE_GPU
        lattisense::FheTaskGpu task(runner_dir_.string());
        task.run(eval_context_.get(), cxx_args, progress_cb, gpu_device_);
#else
        throw std::runtime_error("[Runner] GPU support is disabled. Reconfigure with -DINFERENCE_SDK_ENABLE_GPU=ON.");
#endif
    } else {
        lattisense::FheTaskCpu task(runner_dir_.string());
        task.run(eval_context_.get(), cxx_args, progress_cb);
    }

    std::map<std::string, Bytes> encrypted_outputs;
    for (size_t i = 0; i < output_features.size(); ++i) {
        encrypted_outputs[output_names[i]] = output_features[i]->serialize();
    }
    std::cout << "[Runner] Done." << std::endl;
    return encrypted_outputs;
}
