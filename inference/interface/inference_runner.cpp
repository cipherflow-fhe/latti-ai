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

#include <algorithm>
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
    pack_style_ = task_config_.value("pack_style", std::string("ordinary"));

    if (task_config_.value("deployment_mode", std::string("")) != "server_provisioned_runner") {
        throw std::runtime_error("[Runner] task_config.json is not marked as server_provisioned_runner");
    }
    if (task_config_.value("input_mode", std::string("")) != "plaintext") {
        throw std::runtime_error("[Runner] only plaintext input mode is supported");
    }
    if (task_config_.value("parameter_mode", std::string("")) != "encrypted_offline") {
        throw std::runtime_error("[Runner] only encrypted_offline parameter mode is supported");
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
        if (ip.dim == 1) {
            ip.length = param.at("shape")[0].get<int>();
            if (param.contains("invalid_fill")) {
                ip.invalid_fill[0] = param.at("invalid_fill")[0].get<uint32_t>();
            }
        } else if (ip.dim == 2) {
            ip.height = param.at("shape")[0].get<int>();
            ip.width = param.at("shape")[1].get<int>();
            if (param.contains("skip") && param.at("skip").is_array()) {
                ip.skip2d = {param.at("skip")[0].get<uint32_t>(), param.at("skip")[1].get<uint32_t>()};
            }
            if (param.contains("invalid_fill")) {
                ip.invalid_fill = {param.at("invalid_fill")[0].get<uint32_t>(),
                                   param.at("invalid_fill")[1].get<uint32_t>()};
            }
        }
        input_params_[name] = ip;
    }
    for (auto& [name, param] : task_config_.at("task_output_param").items()) {
        OutputParam op;
        op.dim = param.at("dim").get<int>();
        op.channel = param.value("channel", 1);
        op.pack_num = param.value("pack_num", 0);
        op.skip = read_skip_value(param);
        if (op.dim == 1) {
            op.length = param.at("shape")[0].get<int>();
            if (param.contains("invalid_fill")) {
                op.invalid_fill[0] = param.at("invalid_fill")[0].get<uint32_t>();
            }
        } else if (op.dim == 2) {
            op.height = param.at("shape")[0].get<int>();
            op.width = param.at("shape")[1].get<int>();
            if (param.contains("skip") && param.at("skip").is_array()) {
                op.skip2d = {param.at("skip")[0].get<uint32_t>(), param.at("skip")[1].get<uint32_t>()};
            }
            if (param.contains("invalid_fill")) {
                op.invalid_fill = {param.at("invalid_fill")[0].get<uint32_t>(),
                                   param.at("invalid_fill")[1].get<uint32_t>()};
            }
        }
        output_params_[name] = op;
    }
}

void InferenceRunner::load() {
    read_configuration();

    const Bytes eval_context = read_binary_file(runner_dir_ / "eval_context.bin");
    if (task_config_.value("use_btp", false)) {
        eval_context_ = std::make_unique<fhe::CkksBtpContext>(fhe::CkksBtpContext::deserialize(eval_context));
    } else {
        eval_context_ = std::make_unique<fhe::CkksContext>(fhe::CkksContext::deserialize_advanced(eval_context));
    }
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
        throw std::runtime_error("[Runner] input is not 0D: " + name);
    }
    if (param.pack_num <= 0) {
        throw std::runtime_error("[Runner] 0D input is missing a positive pack_num: " + name);
    }
    if (sig.at("type").get<std::string>() != "pt_ringt") {
        throw std::runtime_error("[Runner] plaintext runner expects pt_ringt input signature for: " + name);
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

std::vector<fhe::CkksPlaintextRingt> InferenceRunner::encode_plaintext_1d_input(const std::string& name,
                                                                                const std::string& csv_path,
                                                                                const json& sig) const {
    auto param_it = input_params_.find(name);
    if (param_it == input_params_.end()) {
        throw std::runtime_error("[Runner] Unknown input name: " + name);
    }
    const InputParam& param = param_it->second;
    if (param.dim != 1) {
        throw std::runtime_error("[Runner] input is not 1D: " + name);
    }
    if (sig.at("type").get<std::string>() != "pt_ringt") {
        throw std::runtime_error("[Runner] plaintext runner expects pt_ringt input signature for: " + name);
    }

    auto input_array = csv_to_array<2>(csv_path, {(uint64_t)param.channel, (uint64_t)param.length});
    const double scale = eval_context_->get_parameter().get_default_scale();
    std::vector<fhe::CkksPlaintextRingt> encoded;

    if (pack_style_ == "ordinary") {
        const uint32_t skip = param.skip > 0 ? static_cast<uint32_t>(param.skip) : 1;
        const uint32_t shape_with_skip = static_cast<uint32_t>(param.length) * skip;
        const uint32_t n_channel_per_ct = static_cast<uint32_t>(n_slots_) / shape_with_skip;
        if (n_channel_per_ct == 0) {
            throw std::runtime_error("[Runner] invalid ordinary 1D input packing for: " + name);
        }
        const uint64_t actual_count = div_ceil(static_cast<uint32_t>(param.channel), n_channel_per_ct);
        const uint64_t expected_count = shape_size(sig);
        if (actual_count != expected_count) {
            throw std::runtime_error("[Runner] encoded 1D input count mismatch for " + name + ": expected " +
                                     std::to_string(expected_count) + ", got " + std::to_string(actual_count));
        }
        encoded.reserve(actual_count);
        for (uint32_t ct_idx = 0; ct_idx < actual_count; ++ct_idx) {
            std::vector<double> slots(static_cast<size_t>(n_slots_), 0.0);
            for (uint32_t k = 0; k < n_channel_per_ct; ++k) {
                const uint32_t channel_idx = ct_idx * n_channel_per_ct + k;
                const uint32_t source_channel = channel_idx < static_cast<uint32_t>(param.channel) ?
                                                    channel_idx :
                                                    channel_idx % static_cast<uint32_t>(param.channel);
                for (uint32_t i = 0; i < static_cast<uint32_t>(param.length); ++i) {
                    slots[k * shape_with_skip + i * skip] = input_array.get(source_channel, i);
                }
            }
            encoded.push_back(eval_context_->encode_ringt(slots, scale));
        }
        return encoded;
    }

    const uint32_t skip = param.skip > 0 ? static_cast<uint32_t>(param.skip) : 1;
    const uint32_t invalid_fill = param.invalid_fill[0] > 0 ? param.invalid_fill[0] : 1;
    const uint32_t block_size = static_cast<uint32_t>(param.length) * skip;
    const uint32_t n_channel_per_ct =
        static_cast<uint32_t>(n_slots_) / (static_cast<uint32_t>(param.length) * invalid_fill);
    const uint32_t n_channel_per_block = skip / invalid_fill;
    if (block_size == 0 || n_channel_per_ct == 0 || n_channel_per_block == 0) {
        throw std::runtime_error("[Runner] invalid multiplexed 1D input packing for: " + name);
    }
    const uint64_t actual_count = div_ceil(static_cast<uint32_t>(param.channel), n_channel_per_ct);
    const uint64_t expected_count = shape_size(sig);
    if (actual_count != expected_count) {
        throw std::runtime_error("[Runner] encoded 1D input count mismatch for " + name + ": expected " +
                                 std::to_string(expected_count) + ", got " + std::to_string(actual_count));
    }
    encoded.reserve(actual_count);
    for (uint32_t ct_idx = 0; ct_idx < actual_count; ++ct_idx) {
        std::vector<double> slots(static_cast<size_t>(n_slots_), 0.0);
        for (uint32_t j = 0; j < n_channel_per_ct; ++j) {
            const uint32_t channel = ct_idx * n_channel_per_ct + j;
            if (channel >= static_cast<uint32_t>(param.channel)) {
                continue;
            }
            const uint32_t block_idx = j / n_channel_per_block;
            const uint32_t sub_pos = j % n_channel_per_block;
            for (uint32_t data_idx = 0; data_idx < static_cast<uint32_t>(param.length); ++data_idx) {
                const uint32_t slot_idx = block_idx * block_size + data_idx * skip + sub_pos;
                slots[slot_idx] = input_array.get(channel, data_idx);
            }
        }
        encoded.push_back(eval_context_->encode_ringt(slots, scale));
    }
    return encoded;
}

std::vector<fhe::CkksPlaintextRingt> InferenceRunner::encode_plaintext_2d_input(const std::string& name,
                                                                                const std::string& csv_path,
                                                                                const json& sig) const {
    auto param_it = input_params_.find(name);
    if (param_it == input_params_.end()) {
        throw std::runtime_error("[Runner] Unknown input name: " + name);
    }
    const InputParam& param = param_it->second;
    if (param.dim != 2) {
        throw std::runtime_error("[Runner] input is not 2D: " + name);
    }
    if (sig.at("type").get<std::string>() != "pt_ringt") {
        throw std::runtime_error("[Runner] plaintext runner expects pt_ringt input signature for: " + name);
    }

    auto input_array =
        csv_to_array<3>(csv_path, {(uint64_t)param.channel, (uint64_t)param.height, (uint64_t)param.width});
    const double scale = eval_context_->get_parameter().get_default_scale();
    std::vector<fhe::CkksPlaintextRingt> encoded;

    if (pack_style_ == "ordinary") {
        const uint32_t pixels = static_cast<uint32_t>(param.height * param.width);
        const uint32_t n_channel_per_ct = static_cast<uint32_t>(n_slots_) / pixels;
        if (n_channel_per_ct == 0) {
            throw std::runtime_error("[Runner] invalid ordinary 2D input packing for: " + name);
        }
        const uint64_t actual_count = div_ceil(static_cast<uint32_t>(param.channel), n_channel_per_ct);
        const uint64_t expected_count = shape_size(sig);
        if (actual_count != expected_count) {
            throw std::runtime_error("[Runner] encoded 2D input count mismatch for " + name + ": expected " +
                                     std::to_string(expected_count) + ", got " + std::to_string(actual_count));
        }
        encoded.reserve(actual_count);
        for (uint32_t ct_idx = 0; ct_idx < actual_count; ++ct_idx) {
            std::vector<double> slots(static_cast<size_t>(n_slots_), 0.0);
            for (uint32_t k = 0; k < n_channel_per_ct; ++k) {
                const uint32_t channel_idx = ct_idx * n_channel_per_ct + k;
                const uint32_t source_channel = channel_idx < static_cast<uint32_t>(param.channel) ?
                                                    channel_idx :
                                                    channel_idx % static_cast<uint32_t>(param.channel);
                for (uint32_t i = 0; i < static_cast<uint32_t>(param.height); ++i) {
                    for (uint32_t j = 0; j < static_cast<uint32_t>(param.width); ++j) {
                        slots[k * pixels + i * static_cast<uint32_t>(param.width) + j] =
                            input_array.get(source_channel, i, j);
                    }
                }
            }
            encoded.push_back(eval_context_->encode_ringt(slots, scale));
        }
        return encoded;
    }

    if (param.height * param.width > n_slots_) {
        if (!task_config_.contains("block_shape")) {
            throw std::runtime_error("[Runner] multiplexed big 2D input requires block_shape in task_config: " + name);
        }
        const Duo block_shape = {task_config_.at("block_shape")[0].get<uint32_t>(),
                                 task_config_.at("block_shape")[1].get<uint32_t>()};
        const Duo stride = {static_cast<uint32_t>(param.height / block_shape[0]),
                            static_cast<uint32_t>(param.width / block_shape[1])};
        const uint64_t actual_count = static_cast<uint64_t>(param.channel) * prod(stride);
        const uint64_t expected_count = shape_size(sig);
        if (actual_count != expected_count) {
            throw std::runtime_error("[Runner] encoded interleaved 2D input count mismatch for " + name +
                                     ": expected " + std::to_string(expected_count) + ", got " +
                                     std::to_string(actual_count));
        }
        encoded.reserve(actual_count);
        for (uint32_t ct_idx = 0; ct_idx < actual_count; ++ct_idx) {
            std::vector<double> slots(static_cast<size_t>(n_slots_), 0.0);
            const uint32_t channel_idx = ct_idx / prod(stride);
            const uint32_t grid_idx = ct_idx % prod(stride);
            const Duo grid_idx_2d = div_mod(grid_idx, stride[1]);
            for (uint32_t x0 = 0; x0 < static_cast<uint32_t>(param.height); ++x0) {
                const uint32_t block_row_idx = x0 / stride[0];
                for (uint32_t x1 = 0; x1 < static_cast<uint32_t>(param.width); ++x1) {
                    const Duo x = {x0, x1};
                    const uint32_t block_col_idx = x1 / stride[1];
                    if (x % stride == grid_idx_2d) {
                        slots[block_row_idx * block_shape[1] + block_col_idx] = input_array.get(channel_idx, x0, x1);
                    }
                }
            }
            encoded.push_back(eval_context_->encode_ringt(slots, scale));
        }
        return encoded;
    }

    const Duo skip = param.skip2d;
    const Duo invalid_fill = param.invalid_fill;
    const uint32_t n_channel_per_block = prod(skip) / prod(invalid_fill);
    const uint32_t n_channel_per_block_col = skip[1] / invalid_fill[1];
    const uint32_t n_channel_per_ct =
        static_cast<uint32_t>(n_slots_) / (static_cast<uint32_t>(param.height * param.width) * prod(invalid_fill));
    const uint32_t n_block_per_ct = n_channel_per_ct / n_channel_per_block;
    if (n_channel_per_block == 0 || n_channel_per_block_col == 0 || n_channel_per_ct == 0 || n_block_per_ct == 0) {
        throw std::runtime_error("[Runner] invalid multiplexed 2D input packing for: " + name);
    }
    const uint64_t actual_count = div_ceil(static_cast<uint32_t>(param.channel), n_channel_per_ct);
    const uint64_t expected_count = shape_size(sig);
    if (actual_count != expected_count) {
        throw std::runtime_error("[Runner] encoded multiplexed 2D input count mismatch for " + name + ": expected " +
                                 std::to_string(expected_count) + ", got " + std::to_string(actual_count));
    }
    encoded.reserve(actual_count);
    for (uint32_t ct_idx = 0; ct_idx < actual_count; ++ct_idx) {
        std::vector<double> slots(static_cast<size_t>(n_slots_), 0.0);
        for (uint32_t block_idx = 0; block_idx < n_block_per_ct; ++block_idx) {
            for (uint32_t x0 = 0; x0 < static_cast<uint32_t>(param.height); ++x0) {
                for (uint32_t x1 = 0; x1 < static_cast<uint32_t>(param.width); ++x1) {
                    for (uint32_t channel_idx_in_block = 0; channel_idx_in_block < n_channel_per_block;
                         ++channel_idx_in_block) {
                        const uint32_t channel_idx =
                            ct_idx * n_channel_per_ct + block_idx * n_channel_per_block + channel_idx_in_block;
                        if (channel_idx >= static_cast<uint32_t>(param.channel)) {
                            continue;
                        }
                        const Duo channel_offset = div_mod(channel_idx_in_block, n_channel_per_block_col);
                        const Duo x_in_block = Duo{x0, x1} * skip + channel_offset;
                        const uint32_t slot =
                            block_idx *
                                prod(Duo{static_cast<uint32_t>(param.height), static_cast<uint32_t>(param.width)} *
                                     skip) +
                            x_in_block[0] * (static_cast<uint32_t>(param.width) * skip[1]) + x_in_block[1];
                        slots[slot] = input_array.get(channel_idx, x0, x1);
                    }
                }
            }
        }
        encoded.push_back(eval_context_->encode_ringt(slots, scale));
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
    std::vector<std::unique_ptr<FeatureEncrypted>> output_features;
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
        auto param_it = input_params_.find(id);
        if (param_it == input_params_.end()) {
            throw std::runtime_error("[Runner] Unknown input name: " + id);
        }
        const InputParam& param = param_it->second;
        if (param.dim == 0) {
            online_inputs.push_back(encode_plaintext_0d_input(id, csv_it->second, sig));
        } else if (param.dim == 1) {
            online_inputs.push_back(encode_plaintext_1d_input(id, csv_it->second, sig));
        } else if (param.dim == 2) {
            online_inputs.push_back(encode_plaintext_2d_input(id, csv_it->second, sig));
        } else {
            throw std::runtime_error("[Runner] unsupported plaintext input dimension for: " + id);
        }
        cxx_args.push_back(lattisense::CxxVectorArgument{id, &online_inputs.back()});
    }

    for (const auto& sig : task_signature_.at("offline")) {
        const std::string id = sig.at("id").get<std::string>();
        if (sig.at("type").get<std::string>() != "ct") {
            throw std::runtime_error("[Runner] only offline ciphertext parameters are supported: " + id);
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

        const int level = sig.at("level").get<int>();
        const uint64_t count = shape_size(sig);
        const double scale = eval_context_->get_parameter().get_default_scale();

        if (param.dim == 0) {
            auto output = std::make_unique<Feature0DEncrypted>(eval_context_.get(), level);
            output->skip = param.skip;
            output->n_channel_per_ct = param.pack_num > 0 ? static_cast<uint32_t>(param.pack_num) :
                                                            std::max<uint32_t>(1, static_cast<uint32_t>(param.channel));
            output->n_channel = static_cast<uint32_t>(param.channel);
            for (uint64_t i = 0; i < count; ++i) {
                output->data.push_back(eval_context_->new_ciphertext(level, scale));
            }
            cxx_args.push_back(lattisense::CxxVectorArgument{id, &output->data});
            output_names.push_back(id);
            output_features.push_back(std::move(output));
        } else if (param.dim == 1) {
            auto output = std::make_unique<Feature1DEncrypted>(
                eval_context_.get(), level, static_cast<uint32_t>(param.skip), param.invalid_fill[0]);
            output->shape = static_cast<uint32_t>(param.length);
            output->n_channel_per_ct = param.pack_num > 0 ? static_cast<uint32_t>(param.pack_num) : 1;
            output->n_channel = static_cast<uint32_t>(param.channel);
            for (uint64_t i = 0; i < count; ++i) {
                output->data.push_back(eval_context_->new_ciphertext(level, scale));
            }
            cxx_args.push_back(lattisense::CxxVectorArgument{id, &output->data});
            output_names.push_back(id);
            output_features.push_back(std::move(output));
        } else if (param.dim == 2) {
            const bool is_multiplexed = pack_style_ == "multiplexed";
            PackType packing_type = is_multiplexed ? PackType::MultiplexedPacking : PackType::MultipleChannelPacking;
            if (is_multiplexed && task_config_.contains("block_shape")) {
                const Duo block_shape = {task_config_.at("block_shape")[0].get<uint32_t>(),
                                         task_config_.at("block_shape")[1].get<uint32_t>()};
                if (static_cast<uint64_t>(param.height) * static_cast<uint64_t>(param.width) >
                    static_cast<uint64_t>(prod(block_shape))) {
                    packing_type = PackType::InterleavedPacking;
                }
            }
            auto output = std::make_unique<Feature2DEncrypted>(eval_context_.get(), level, param.skip2d,
                                                               param.invalid_fill, packing_type);
            output->shape = {static_cast<uint32_t>(param.height), static_cast<uint32_t>(param.width)};
            output->n_channel_per_ct = param.pack_num > 0 ? static_cast<uint32_t>(param.pack_num) : 1;
            output->n_channel = static_cast<uint32_t>(param.channel);
            for (uint64_t i = 0; i < count; ++i) {
                output->data.push_back(eval_context_->new_ciphertext(level, scale));
            }
            cxx_args.push_back(lattisense::CxxVectorArgument{id, &output->data});
            output_names.push_back(id);
            output_features.push_back(std::move(output));
        } else {
            throw std::runtime_error("[Runner] unsupported ciphertext output dimension for: " + id);
        }
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
