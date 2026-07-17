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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <type_traits>

#include "interface/inference_server.h"
#ifdef INFERENCE_SDK_ENABLE_MPC
#include "mpc_wrapper/mpc_data_transmission.h"
#endif

using namespace lattisense;

InferenceServer::InferenceServer(const std::string& server_dir, bool use_gpu, int gpu_device)
    : server_dir_(server_dir), use_gpu_(use_gpu), gpu_device_(gpu_device) {}

InferenceServer::~InferenceServer() = default;

double InferenceServer::get_last_mpc_time_ms() const {
    if (!init_) {
        return 0.0;
    }
    return init_->total_fpga_time;
}

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

void InferenceServer::import_eval_context_ckks(const Bytes& eval_context) {
    auto task_config = read_json((server_dir_ / "task_config.json").string());

    needs_btp_ = false;

    input_keys_.clear();
    input_params_.clear();
    output_keys_.clear();
    output_params_.clear();

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

    std::cout << "[Server] Importing CKKS evaluation context..." << std::endl;
    eval_context_ = std::make_unique<CkksContext>(CkksContext::deserialize_advanced(eval_context));
    context_ptr_ = eval_context_.get();
    std::cout << "[Server] Done." << std::endl;
}

void InferenceServer::load_model() {
    std::cout << "[Server] Loading model..." << std::endl;

    init_ = std::make_unique<InitInferenceProcess>(server_dir_.string() + "/", false);
    init_->init_parameters(needs_btp_);
    init_->is_lazy = true;
    init_->load_model_prepare();
    std::cout << "[Server] Hybrid pipeline: " << (init_->has_hybrid_pipeline() ? "enabled" : "disabled")
              << std::endl;

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

void InferenceServer::load_model_for_mpc_sdk() {
#ifndef INFERENCE_SDK_ENABLE_MPC
    throw std::runtime_error("MPC support is disabled. Reconfigure with -DINFERENCE_SDK_ENABLE_MPC=ON to enable it.");
#else
    std::cout << "[Server] Loading model for SDK MPC executor..." << std::endl;

    init_ = std::make_unique<InitInferenceProcess>(server_dir_.string() + "/", false);
    init_->init_parameters(needs_btp_);
    init_->is_lazy = true;
    init_->load_model_prepare();

    fp_ = std::make_unique<InferenceProcess>(init_.get());
    for (auto& key : input_keys_) {
        fp_->available_keys.push_back(key);
    }

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

    std::cout << "[Server] Done." << std::endl;
#endif
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
    if (init_->has_hybrid_pipeline()) {
        fp_->run_task_hybrid_pipeline(false, progress_cb);
    } else {
        fp_->run_task_lazy(false, progress_cb);
    }
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

std::map<std::string, Bytes> InferenceServer::evaluate_mpc_sdk(const std::map<std::string, Bytes>& encrypted_inputs) {
#ifndef INFERENCE_SDK_ENABLE_MPC
    throw std::runtime_error("MPC support is disabled. Reconfigure with -DINFERENCE_SDK_ENABLE_MPC=ON to enable it.");
#else
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

    fp_->compute_device = use_gpu_ ? ComputeDevice::GPU : ComputeDevice::CPU;
    std::cout << "[Server] Running SDK encrypted inference with MPC refresh..." << std::endl;
    std::cout << "[Server] Device: " << (use_gpu_ ? "GPU" : "CPU") << std::endl;
    Timer timer;
    timer.start();
    if (init_->has_hybrid_pipeline()) {
        std::cout << "[Server] Dispatch: hybrid MPC pipeline" << std::endl;
        fp_->run_task_hybrid_pipeline(true);
    } else {
        std::cout << "[Server] Dispatch: SDK MPC layer executor" << std::endl;
        fp_->run_task_sdk(true);
    }
    timer.stop();
    timer.print("SDK MPC encrypted inference time");

    send_mpc_end();

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
#endif
}

void InferenceServer::dump_intermediate_plaintexts(const std::string& output_path) const {
    if (!fp_ || !init_) {
        throw std::runtime_error("[Server] Model is not loaded");
    }

    std::ofstream ofs(output_path);
    if (!ofs.is_open()) {
        throw std::runtime_error("[Server] Cannot open layer dump file: " + output_path);
    }

    ofs << std::setprecision(12);
    const auto& json_features = init_->json_features;
    const auto& pack_style = init_->pack_style;
    const Duo block_shape = init_->block_shape;
    constexpr int kDumpLimit = 10;
    constexpr int kChannelDumpLimit = 64;

    auto write_limited = [&](const auto& flat) {
        using FlatType = std::decay_t<decltype(flat)>;
        int total = 0;
        if constexpr (std::is_same_v<FlatType, std::vector<double>>) {
            total = static_cast<int>(flat.size());
        } else {
            total = flat.get_size();
        }

        int count = std::min(kDumpLimit, total);
        ofs << "total_values=" << total << "\n";
        ofs << "first_values=";
        for (int i = 0; i < count; i++) {
            if (i > 0) {
                ofs << ',';
            }
            if constexpr (std::is_same_v<FlatType, std::vector<double>>) {
                ofs << flat[i];
            } else {
                ofs << flat.get_data()[i];
            }
        }
        ofs << "\n";
    };

    auto write_channel_first_values = [&](const Array<double, 3>& arr) {
        if (arr.get_size() == 0) {
            ofs << "channel_first_values=\n";
            return;
        }
        const auto shape = arr.get_shape();
        int count = std::min(kChannelDumpLimit, static_cast<int>(shape[0]));
        ofs << "channel_first_values=";
        for (int c = 0; c < count; c++) {
            if (c > 0) {
                ofs << ',';
            }
            ofs << arr.get(c, 0, 0);
        }
        ofs << "\n";
    };

    for (const auto& name : fp_->encrypted_feature_order_) {
        auto feature_it = fp_->intermediate_result_.find(name);
        if (feature_it == fp_->intermediate_result_.end()) {
            continue;
        }
        const auto& feature_ptr = feature_it->second;
        if (!feature_ptr) {
            continue;
        }
        auto json_it = json_features.find(name);
        if (json_it == json_features.end()) {
            continue;
        }
        FeatureNode feature_node(*json_it);

        ofs << "FEATURE " << name << "\n";
        ofs << "dim=" << feature_node.dim << ", channel=" << feature_node.channel << ", shape=["
            << feature_node.shape[0] << "," << feature_node.shape[1] << "], skip=[" << feature_node.skip[0] << ","
            << feature_node.skip[1] << "], level=" << feature_node.level << "\n";
        if (feature_node.dim == 0) {
            auto feature = dynamic_cast<const Feature0DEncrypted&>(*feature_ptr).copy();
            feature.skip = feature_node.skip[0];
            auto arr = feature.unpack().to_array_1d();
            write_limited(arr);
        } else if (feature_node.dim == 1) {
            auto feature = dynamic_cast<const Feature1DEncrypted&>(*feature_ptr).copy();
            Array<double, 2> arr;
            if (pack_style == "multiplexed") {
                feature.invalid_fill = feature_node.invalid_fill[0];
                arr = feature.unpack_multiplexed();
            } else {
                arr = feature.unpack();
            }
            auto flat = arr.to_array_1d();
            write_limited(flat);
        } else if (feature_node.dim == 2) {
            auto feature = dynamic_cast<const Feature2DEncrypted&>(*feature_ptr).copy();
            Array<double, 3> arr;
            if (pack_style == "multiplexed") {
                if (feature_node.shape[0] * feature_node.shape[1] > block_shape[0] * block_shape[1]) {
                    Duo stride = {(uint32_t)(feature_node.shape[0] / block_shape[0]),
                                  (uint32_t)(feature_node.shape[1] / block_shape[1])};
                    arr = feature.unpack_interleaved(block_shape, stride);
                } else {
                    feature.invalid_fill = feature_node.invalid_fill;
                    arr = feature.unpack_multiplexed();
                }
            } else {
                arr = feature.unpack_multiple_channel();
            }
            auto flat = arr.to_array_1d();
            write_limited(flat);
            write_channel_first_values(arr);
        }
        ofs << "\n";
    }

    std::cout << "[Server] Wrote layer plaintext dump to " << output_path << std::endl;
}

void InferenceServer::dump_plaintext_intermediates(const std::map<std::string, std::string>& input_csvs,
                                                   const std::string& output_path) {
    if (!fp_ || !init_) {
        throw std::runtime_error("[Server] Model is not loaded");
    }

    fp_->p_feature0d_x.clear();
    fp_->p_feature1d_x.clear();
    fp_->p_feature2d_x.clear();
    fp_->available_keys.clear();
    fp_->plaintext_feature_order_.clear();

    for (auto& key : input_keys_) {
        fp_->available_keys.push_back(key);
    }

    for (const auto& name : input_keys_) {
        auto csv_it = input_csvs.find(name);
        if (csv_it == input_csvs.end()) {
            throw std::runtime_error("[Server] Missing input CSV for: " + name);
        }
        auto it = input_params_.find(name);
        if (it == input_params_.end()) {
            throw std::runtime_error("[Server] Unknown input name: " + name);
        }
        const auto& param = it->second;
        const auto& csv_path = csv_it->second;

        fp_->plaintext_feature_order_.push_back(name);
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

    if (init_->has_hybrid_pipeline()) {
        fp_->run_task_plaintext_hybrid_pipeline();
    } else {
        fp_->run_task_plaintext();
    }

    std::ofstream ofs(output_path);
    if (!ofs.is_open()) {
        throw std::runtime_error("[Server] Cannot open plaintext dump file: " + output_path);
    }

    ofs << std::setprecision(12);
    json json_features = init_->json_features;
    if (init_->has_hybrid_pipeline()) {
        for (const auto& graph : init_->subgraphs()) {
            for (const auto& item : graph.json_features.items()) {
                json_features[item.key()] = item.value();
            }
        }
    }
    constexpr int kDumpLimit = 10;
    constexpr int kChannelDumpLimit = 64;

    auto write_limited = [&](const auto& flat) {
        using FlatType = std::decay_t<decltype(flat)>;
        int total = 0;
        if constexpr (std::is_same_v<FlatType, std::vector<double>>) {
            total = static_cast<int>(flat.size());
        } else {
            total = flat.get_size();
        }

        int count = std::min(kDumpLimit, total);
        ofs << "total_values=" << total << "\n";
        ofs << "first_values=";
        for (int i = 0; i < count; i++) {
            if (i > 0) {
                ofs << ',';
            }
            if constexpr (std::is_same_v<FlatType, std::vector<double>>) {
                ofs << flat[i];
            } else {
                ofs << flat.get_data()[i];
            }
        }
        ofs << "\n";
    };

    auto write_channel_first_values = [&](const Array<double, 3>& arr) {
        if (arr.get_size() == 0) {
            ofs << "channel_first_values=\n";
            return;
        }
        const auto shape = arr.get_shape();
        int count = std::min(kChannelDumpLimit, static_cast<int>(shape[0]));
        ofs << "channel_first_values=";
        for (int c = 0; c < count; c++) {
            if (c > 0) {
                ofs << ',';
            }
            ofs << arr.get(c, 0, 0);
        }
        ofs << "\n";
    };

    for (const auto& name : fp_->plaintext_feature_order_) {
        auto json_it = json_features.find(name);
        if (json_it == json_features.end()) {
            continue;
        }
        FeatureNode feature_node(*json_it);

        ofs << "FEATURE " << name << "\n";
        ofs << "dim=" << feature_node.dim << ", channel=" << feature_node.channel << ", shape=["
            << feature_node.shape[0] << "," << feature_node.shape[1] << "], skip=[" << feature_node.skip[0] << ","
            << feature_node.skip[1] << "], level=" << feature_node.level << "\n";

        if (feature_node.dim == 0) {
            auto it = fp_->p_feature0d_x.find(name);
            if (it != fp_->p_feature0d_x.end()) {
                write_limited(it->second);
            }
        } else if (feature_node.dim == 1) {
            auto it = fp_->p_feature1d_x.find(name);
            if (it != fp_->p_feature1d_x.end()) {
                auto flat = it->second.to_array_1d();
                write_limited(flat);
            }
        } else if (feature_node.dim == 2) {
            auto it = fp_->p_feature2d_x.find(name);
            if (it != fp_->p_feature2d_x.end()) {
                auto flat = it->second.to_array_1d();
                write_limited(flat);
                write_channel_first_values(it->second);
            }
        }
        ofs << "\n";
    }

    std::cout << "[Server] Wrote plaintext reference dump to " << output_path << std::endl;
}

void InferenceServer::dump_task_output_debug(const std::map<std::string, std::string>& input_csvs,
                                             const std::string& output_path) {
    if (!fp_ || !init_) {
        throw std::runtime_error("[Server] Model is not loaded");
    }

    fp_->p_feature0d_x.clear();
    fp_->p_feature1d_x.clear();
    fp_->p_feature2d_x.clear();
    fp_->available_keys.clear();
    fp_->plaintext_feature_order_.clear();

    for (auto& key : input_keys_) {
        fp_->available_keys.push_back(key);
    }

    for (const auto& name : input_keys_) {
        auto csv_it = input_csvs.find(name);
        if (csv_it == input_csvs.end()) {
            throw std::runtime_error("[Server] Missing input CSV for: " + name);
        }
        auto it = input_params_.find(name);
        if (it == input_params_.end()) {
            throw std::runtime_error("[Server] Unknown input name: " + name);
        }
        const auto& param = it->second;
        const auto& csv_path = csv_it->second;

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

    if (init_->has_hybrid_pipeline()) {
        fp_->run_task_plaintext_hybrid_pipeline();
    } else {
        fp_->run_task_plaintext();
    }

    json json_features = init_->json_features;
    if (init_->has_hybrid_pipeline()) {
        for (const auto& graph : init_->subgraphs()) {
            for (const auto& item : graph.json_features.items()) {
                json_features[item.key()] = item.value();
            }
        }
    }

    struct TaskOutputRef {
        int graph_idx;
        std::string graph_name;
        std::string feature_name;
    };
    std::vector<TaskOutputRef> task_outputs;
    if (init_->has_hybrid_pipeline()) {
        const auto& graphs = init_->subgraphs();
        for (int i = 0; i < (int)graphs.size(); i++) {
            for (const auto& feature : graphs[i].json_data["output_feature"]) {
                task_outputs.push_back({i, graphs[i].name, feature.get<std::string>()});
            }
        }
    } else {
        for (const auto& feature : init_->json_data["output_feature"]) {
            task_outputs.push_back({0, "0", feature.get<std::string>()});
        }
    }

    const auto pack_style = init_->pack_style;
    const Duo block_shape = init_->block_shape;
    auto encrypted_unpacked_values = [&](const std::string& name, const FeatureNode& feature_node) {
        auto feature_it = fp_->intermediate_result_.find(name);
        if (feature_it == fp_->intermediate_result_.end() || !feature_it->second) {
            return std::vector<double>{};
        }

        if (feature_node.dim == 0) {
            auto feature = dynamic_cast<const Feature0DEncrypted&>(*feature_it->second).copy();
            feature.n_channel = feature_node.channel;
            feature.n_channel_per_ct = feature_node.pack_channel_per_ciphertext;
            feature.skip = feature_node.skip[0];
            auto arr = feature.unpack();
            return arr.to_array_1d();
        }
        if (feature_node.dim == 1) {
            auto feature = dynamic_cast<const Feature1DEncrypted&>(*feature_it->second).copy();
            feature.n_channel = feature_node.channel;
            feature.n_channel_per_ct = feature_node.pack_channel_per_ciphertext;
            feature.shape = feature_node.shape[0];
            feature.skip = feature_node.skip[0];
            feature.invalid_fill = feature_node.invalid_fill[0];
            Array<double, 2> arr;
            if (pack_style == "multiplexed") {
                arr = feature.unpack_multiplexed();
            } else {
                arr = feature.unpack();
            }
            return arr.to_array_1d();
        }

        auto feature = dynamic_cast<const Feature2DEncrypted&>(*feature_it->second).copy();
        feature.n_channel = feature_node.channel;
        feature.n_channel_per_ct = feature_node.pack_channel_per_ciphertext;
        feature.shape = feature_node.shape;
        feature.skip = feature_node.skip;
        feature.invalid_fill = feature_node.invalid_fill;
        Array<double, 3> arr;
        if (pack_style == "multiplexed") {
            if (feature_node.shape[0] * feature_node.shape[1] > block_shape[0] * block_shape[1]) {
                Duo stride = {(uint32_t)(feature_node.shape[0] / block_shape[0]),
                              (uint32_t)(feature_node.shape[1] / block_shape[1])};
                arr = feature.unpack_interleaved(block_shape, stride);
            } else {
                arr = feature.unpack_multiplexed();
            }
        } else {
            arr = feature.unpack_multiple_channel();
        }
        return arr.to_array_1d();
    };

    auto plaintext_values = [&](const std::string& name, const FeatureNode& feature_node) {
        if (feature_node.dim == 0) {
            auto it = fp_->p_feature0d_x.find(name);
            return it == fp_->p_feature0d_x.end() ? std::vector<double>{} : it->second;
        }
        if (feature_node.dim == 1) {
            auto it = fp_->p_feature1d_x.find(name);
            return it == fp_->p_feature1d_x.end() ? std::vector<double>{} : it->second.to_array_1d();
        }
        auto it = fp_->p_feature2d_x.find(name);
        return it == fp_->p_feature2d_x.end() ? std::vector<double>{} : it->second.to_array_1d();
    };

    std::ofstream ofs(output_path);
    if (!ofs.is_open()) {
        throw std::runtime_error("[Server] Cannot open task output debug file: " + output_path);
    }

    constexpr int kDumpLimit = 10;
    ofs << std::setprecision(12);
    auto write_values = [&](const std::string& label, const std::vector<double>& values) {
        ofs << label << " total_values=" << values.size() << " first_values=";
        int count = std::min(kDumpLimit, static_cast<int>(values.size()));
        for (int i = 0; i < count; i++) {
            if (i > 0) {
                ofs << ',';
            }
            ofs << values[i];
        }
        ofs << "\n";
    };

    for (const auto& output : task_outputs) {
        auto json_it = json_features.find(output.feature_name);
        if (json_it == json_features.end()) {
            continue;
        }
        FeatureNode feature_node(*json_it);
        ofs << "TASK " << output.graph_idx << " name=" << output.graph_name << " output=" << output.feature_name
            << "\n";
        ofs << "dim=" << feature_node.dim << ", channel=" << feature_node.channel << ", shape=["
            << feature_node.shape[0] << "," << feature_node.shape[1] << "], skip=[" << feature_node.skip[0] << ","
            << feature_node.skip[1] << "], level=" << feature_node.level << "\n";
        write_values("encrypted_unpacked", encrypted_unpacked_values(output.feature_name, feature_node));
        write_values("plaintext_unpacked", plaintext_values(output.feature_name, feature_node));
        ofs << "\n";
    }

    std::cout << "[Server] Wrote task output debug dump to " << output_path << std::endl;
}

std::map<std::string, std::vector<double>>
InferenceServer::evaluate_plaintext(const std::map<std::string, std::string>& input_csvs) {
    fp_->p_feature0d_x.clear();
    fp_->p_feature1d_x.clear();
    fp_->p_feature2d_x.clear();
    fp_->available_keys.clear();
    fp_->plaintext_feature_order_.clear();
    for (auto& key : input_keys_) {
        fp_->available_keys.push_back(key);
    }

    for (const auto& name : input_keys_) {
        auto csv_it = input_csvs.find(name);
        if (csv_it == input_csvs.end()) {
            throw std::runtime_error("[Server] Missing input CSV for: " + name);
        }
        auto it = input_params_.find(name);
        if (it == input_params_.end()) {
            throw std::runtime_error("[Server] Unknown input name: " + name);
        }
        const auto& param = it->second;
        const auto& csv_path = csv_it->second;

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
    if (init_->has_hybrid_pipeline()) {
        fp_->run_task_plaintext_hybrid_pipeline();
    } else {
        fp_->run_task_plaintext();
    }

    std::map<std::string, std::vector<double>> results;
    for (auto& [name, param] : output_params_) {
        if (param.dim == 0) {
            auto it = fp_->p_feature0d_x.find(name);
            if (it == fp_->p_feature0d_x.end()) {
                throw std::runtime_error("[Server] Missing plaintext output: " + name);
            }
            results[name] = it->second;
        } else if (param.dim == 1) {
            auto it = fp_->p_feature1d_x.find(name);
            if (it == fp_->p_feature1d_x.end()) {
                throw std::runtime_error("[Server] Missing plaintext output: " + name);
            }
            auto& arr = it->second;
            auto arr_1d = arr.to_array_1d();
            results[name] = std::vector<double>(arr_1d.data(), arr_1d.data() + arr_1d.size());
        } else {
            auto it = fp_->p_feature2d_x.find(name);
            if (it == fp_->p_feature2d_x.end()) {
                throw std::runtime_error("[Server] Missing plaintext output: " + name);
            }
            auto& arr = it->second;
            auto arr_1d = arr.to_array_1d();
            results[name] = std::vector<double>(arr_1d.data(), arr_1d.data() + arr_1d.size());
        }
    }
    return results;
}
