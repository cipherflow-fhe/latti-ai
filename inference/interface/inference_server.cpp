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

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <utility>

#include "fhe_layers/compute_distance_layer.h"
#include "interface/inference_server.h"

using namespace lattisense;

namespace {

std::vector<double> read_embedding_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open embedding file: " + path);
    }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::replace(text.begin(), text.end(), ',', ' ');

    std::vector<double> values;
    std::istringstream stream(text);
    double value;
    while (stream >> value) {
        values.push_back(value);
    }
    if (values.empty()) {
        throw std::runtime_error("empty embedding file: " + path);
    }
    return values;
}

void normalize_vector(std::vector<double>& values) {
    double norm2 = 0.0;
    for (double value : values) {
        norm2 += value * value;
    }
    const double norm = std::sqrt(norm2);
    if (norm <= 0.0) {
        throw std::runtime_error("gallery norm must be positive");
    }
    for (double& value : values) {
        value /= norm;
    }
}

std::vector<double> read_gallery_embedding(const std::string& path, bool normalize_gallery) {
    auto gallery = read_embedding_csv(path);
    if (normalize_gallery) {
        normalize_vector(gallery);
    }
    return gallery;
}

Feature0DEncrypted repack_feature0d_to_single_ct(CkksContext& ctx, const Feature0DEncrypted& input, uint32_t dim) {
    if (input.data.empty()) {
        throw std::runtime_error("query feature must contain at least one ciphertext");
    }
    if (input.n_channel < dim) {
        throw std::runtime_error("query feature channel count must be at least dim");
    }
    if (input.n_channel_per_ct == 0) {
        throw std::runtime_error("query feature n_channel_per_ct must be positive");
    }
    if (input.data.size() == 1 && input.skip == 1 && input.n_channel_per_ct >= dim) {
        return input.copy();
    }

    const uint32_t n_slots = ctx.get_parameter().get_n() / 2;
    const double scale = ctx.get_parameter().get_default_scale();
    CkksCiphertext acc;
    bool has_acc = false;

    for (uint32_t channel = 0; channel < dim; ++channel) {
        const uint32_t ct_idx = channel / input.n_channel_per_ct;
        const uint32_t offset = channel % input.n_channel_per_ct;
        if (ct_idx >= input.data.size()) {
            throw std::runtime_error("query feature ciphertext count is inconsistent with packing");
        }
        const uint32_t source_slot = offset * input.skip;
        const uint32_t target_slot = channel;
        if (source_slot >= n_slots || target_slot >= n_slots) {
            throw std::runtime_error("query feature slot index exceeds CKKS slot count");
        }

        std::vector<double> mask(n_slots, 0.0);
        mask[source_slot] = 1.0;
        auto mask_pt = ctx.encode(mask, input.data[ct_idx].get_level(), scale);
        auto masked = ctx.mult_plain(input.data[ct_idx], mask_pt);
        masked = ctx.rescale(masked, scale);
        auto moved = ctx.rotate(masked, static_cast<int>(source_slot) - static_cast<int>(target_slot));

        if (!has_acc) {
            acc = std::move(moved);
            has_acc = true;
        } else {
            acc = ctx.add(acc, moved);
        }
    }

    Feature0DEncrypted output(&ctx, acc.get_level());
    output.dim = 0;
    output.n_channel = dim;
    output.n_channel_per_ct = n_slots;
    output.skip = 1;
    output.data.push_back(std::move(acc));
    output.level = output.data[0].get_level();
    return output;
}

}  // namespace

InferenceServer::InferenceServer(const std::string& server_dir, bool use_gpu, int gpu_device)
    : server_dir_(server_dir), use_gpu_(use_gpu), gpu_device_(gpu_device) {}

InferenceServer::~InferenceServer() = default;

void InferenceServer::import_eval_context(const Bytes& eval_context) {
    // Determine whether bootstrapping is needed from the server task config.
    auto task_config = read_json((server_dir_ / "task_config.json").string());
    auto ckks_config = read_json((server_dir_ / "ckks_parameter.json").string());
    auto& input_param = task_config["task_input_param"].begin().value();
    std::string ckks_param_id = input_param["ckks_parameter_id"];
    int poly_modulus_degree = ckks_config[ckks_param_id]["poly_modulus_degree"].get<int>();
    needs_btp_ = task_config.value("use_btp", false);

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
        } else if (ip.dim == 1) {
            ip.length = param["shape"][0];
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
        } else if (op.dim == 1) {
            op.length = param["shape"][0];
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
    init_->is_lazy = true;
    init_->load_model_prepare();

    fp_ = std::make_unique<InferenceProcess>(init_.get());
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

    if (use_gpu_) {
        fp_->compute_device = ComputeDevice::GPU;
        fp_->gpu_device = gpu_device_;
    }
    fp_->prepare_task();

    std::cout << "[Server] Done." << std::endl;
}

std::map<std::string, Bytes> InferenceServer::evaluate(const std::map<std::string, Bytes>& encrypted_inputs,
                                                       lattisense::ProgressCallback progress_cb) {
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
        if (param.dim == 0) {
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

Bytes InferenceServer::compute_distance(const std::string& feature_name,
                                        const std::string& gallery_path,
                                        bool normalize_gallery,
                                        double norm2_min,
                                        double norm2_max,
                                        int nr_iterations) {
    if (!needs_btp_) {
        throw std::runtime_error("compute_distance requires a bootstrapping context");
    }
    auto out_it = output_params_.find(feature_name);
    if (out_it == output_params_.end()) {
        throw std::runtime_error("[Server] Unknown output feature name: " + feature_name);
    }
    if (out_it->second.dim != 0) {
        throw std::runtime_error("compute_distance expects a 0D output feature");
    }

    auto gallery = read_gallery_embedding(gallery_path, normalize_gallery);
    const uint32_t dim = out_it->second.channel;
    if (gallery.size() != dim) {
        throw std::runtime_error("gallery size must match output feature channel count");
    }

    auto query = fp_->get_ciphertext_output_feature<Feature0DEncrypted>(feature_name);
    auto& btp_context = static_cast<CkksBtpContext&>(*context_ptr_);

    Timer timer;
    timer.start();
    auto packed_query = repack_feature0d_to_single_ct(btp_context, query, dim);
    ComputeDistanceLayer distance_layer(btp_context.get_parameter(), dim, norm2_min, norm2_max, nr_iterations);
    distance_layer.prepare_weight(gallery, packed_query.level);
    auto distance = distance_layer.run(btp_context, packed_query);
    timer.stop();
    timer.print("Encrypted distance time");
    return distance.serialize();
}

double InferenceServer::compute_distance_plaintext(const std::string& feature_name,
                                                  const std::string& gallery_path,
                                                  bool normalize_gallery,
                                                  double norm2_min,
                                                  double norm2_max,
                                                  int nr_iterations) {
    auto out_it = output_params_.find(feature_name);
    if (out_it == output_params_.end()) {
        throw std::runtime_error("[Server] Unknown output feature name: " + feature_name);
    }
    if (out_it->second.dim != 0) {
        throw std::runtime_error("compute_distance expects a 0D output feature");
    }

    auto gallery = read_gallery_embedding(gallery_path, normalize_gallery);
    const uint32_t dim = out_it->second.channel;
    if (gallery.size() != dim) {
        throw std::runtime_error("gallery size must match output feature channel count");
    }

    auto query_it = fp_->p_feature0d_x.find(feature_name);
    if (query_it == fp_->p_feature0d_x.end()) {
        throw std::runtime_error("plaintext feature is not available: " + feature_name);
    }

    ComputeDistanceLayer distance_layer(context_ptr_->get_parameter(), dim, norm2_min, norm2_max, nr_iterations);
    return distance_layer.run_plaintext(query_it->second, gallery);
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
        } else if (param.dim == 1) {
            auto input_array = csv_to_array<2>(csv_path, {(uint64_t)param.channel, (uint64_t)param.length});
            fp_->p_feature1d_x[name] = std::move(input_array.copy());
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
        } else if (param.dim == 1) {
            auto& arr = fp_->p_feature1d_x[name];
            auto arr_1d = arr.to_array_1d();
            results[name] = std::vector<double>(arr_1d.data(), arr_1d.data() + arr_1d.size());
        } else {
            auto& arr = fp_->p_feature2d_x[name];
            auto arr_1d = arr.to_array_1d();
            results[name] = std::vector<double>(arr_1d.data(), arr_1d.data() + arr_1d.size());
        }
    }
    return results;
}
