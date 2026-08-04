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

#include "interface/inference_provisioner.h"

#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "fhe_layers/dense_packed_layer.h"
#include "inference_task/inference_process.h"
#include "interface/runner_bundle_io.h"
#include "util/serial.h"

namespace {

uint64_t shape_size(const json& sig) {
    const auto shape = sig.at("size").get<std::vector<uint64_t>>();
    return std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<uint64_t>());
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

void write_json_file(const std::filesystem::path& path, const json& value) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open JSON file for writing: " + path.string());
    }
    out << value.dump(4);
}

void copy_if_exists(const std::filesystem::path& src, const std::filesystem::path& dst) {
    if (std::filesystem::exists(src)) {
        std::filesystem::create_directories(dst.parent_path());
        std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);
    }
}

}  // namespace

InferenceProvisioner::InferenceProvisioner(const std::string& task_dir) : task_dir_(task_dir) {
    resolve_directories();
}

InferenceProvisioner::~InferenceProvisioner() = default;

void InferenceProvisioner::resolve_directories() {
    if (std::filesystem::exists(task_dir_ / "server" / "task_config.json")) {
        server_dir_ = task_dir_ / "server";
        provisioner_dir_ = task_dir_ / "provisioner";
    } else if (std::filesystem::exists(task_dir_ / "task_config.json") &&
               std::filesystem::exists(task_dir_ / "nn_layers_ct_0.json")) {
        server_dir_ = task_dir_;
        provisioner_dir_ = task_dir_.parent_path() / "provisioner";
    } else {
        throw std::runtime_error("[Provisioner] --task-dir must be a task root containing server/ or a server dir");
    }
    std::filesystem::create_directories(provisioner_dir_);
}

fhe::CkksParameter InferenceProvisioner::make_parameter() const {
    const auto& first_input = task_config_.at("task_input_param").begin().value();
    const std::string ckks_param_id = first_input.at("ckks_parameter_id").get<std::string>();
    const auto& entry = ckks_config_.at(ckks_param_id);
    const int poly_modulus_degree = entry.at("poly_modulus_degree").get<int>();

    fhe::CkksParameter param(0);
    if (entry.contains("q") && entry.contains("p")) {
        const auto q = entry.at("q").get<std::vector<uint64_t>>();
        const auto p = entry.at("p").get<std::vector<uint64_t>>();
        param = fhe::CkksParameter::create_custom_parameter(poly_modulus_degree, q, p);
    } else {
        param = fhe::CkksParameter::create_parameter(poly_modulus_degree);
    }
    if (entry.contains("log_slots")) {
        param.set_log_slots(entry.at("log_slots").get<int>());
    }
    return param;
}

void InferenceProvisioner::setup_or_load_keys() {
    task_config_ = read_json((server_dir_ / "task_config.json").string());
    ckks_config_ = read_json((server_dir_ / "ckks_parameter.json").string());

    if (task_config_.value("use_btp", false)) {
        throw std::runtime_error("[Provisioner] bootstrapping contexts are Phase 3/4 work and are not supported yet");
    }

    auto secret_context_path = provisioner_dir_ / "secret_context.bin";
    if (!std::filesystem::exists(secret_context_path) && std::filesystem::exists(provisioner_dir_ / "secret_key.bin")) {
        secret_context_path = provisioner_dir_ / "secret_key.bin";
    }
    if (std::filesystem::exists(secret_context_path)) {
        std::cout << "[Provisioner] Loading existing server-owned secret context..." << std::endl;
        context_ = std::make_unique<fhe::CkksContext>(
            fhe::CkksContext::deserialize_advanced(read_binary_file(secret_context_path)));
        if (secret_context_path.filename() != "secret_context.bin") {
            write_binary_file(provisioner_dir_ / "secret_context.bin", read_binary_file(secret_context_path));
        }
        return;
    }

    std::cout << "[Provisioner] Generating server-owned CKKS keys..." << std::endl;
    auto param = make_parameter();
    context_ = std::make_unique<fhe::CkksContext>(fhe::CkksContext::create_random_context(param));
    context_->gen_rotation_keys();

    const Bytes full_context = context_->serialize_advanced();
    write_binary_file(secret_context_path, full_context);
    write_binary_file(provisioner_dir_ / "secret_key.bin", full_context);
}

void InferenceProvisioner::copy_private_and_runner_configs(const std::filesystem::path& runner_dir) const {
    json provisioner_config = task_config_;
    provisioner_config["deployment_mode"] = "server_provisioned_runner";
    provisioner_config["input_mode"] = "plaintext";
    provisioner_config["parameter_mode"] = "encrypted_offline";
    provisioner_config["decryptor"] = "provisioner";
    provisioner_config["deployment_role"] = "provisioner";
    write_json_file(provisioner_dir_ / "task_config.json", provisioner_config);

    copy_if_exists(server_dir_ / "ckks_parameter.json", provisioner_dir_ / "ckks_parameter.json");
    copy_if_exists(server_dir_ / "model_parameters.h5", provisioner_dir_ / "model_parameters.h5");

    json runner_config = std::filesystem::exists(runner_dir / "task_config.json") ?
                             read_json((runner_dir / "task_config.json").string()) :
                             task_config_;
    runner_config["deployment_mode"] = "server_provisioned_runner";
    runner_config["input_mode"] = "plaintext";
    runner_config["parameter_mode"] = "encrypted_offline";
    runner_config["decryptor"] = "provisioner";
    runner_config["deployment_role"] = "runner";
    write_json_file(runner_dir / "task_config.json", runner_config);

    copy_if_exists(server_dir_ / "ckks_parameter.json", runner_dir / "ckks_parameter.json");
    copy_if_exists(server_dir_ / "nn_layers_ct_0.json", runner_dir / "nn_layers_ct_0.json");

    if (!std::filesystem::exists(runner_dir / "mega_ag.json") ||
        !std::filesystem::exists(runner_dir / "task_signature.json")) {
        throw std::runtime_error("[Provisioner] runner directory must already contain mega_ag.json and "
                                 "task_signature.json; run gen_mega_ag.py in server_provisioned_runner mode first");
    }
}

std::vector<fhe::CkksCiphertext> InferenceProvisioner::encrypt_dense_argument(const std::string& arg_id,
                                                                              const json& sig,
                                                                              fhe::CkksContext& context) const {
    const bool is_weight = starts_with(arg_id, "densew_");
    const bool is_bias = starts_with(arg_id, "denseb_");
    if (!is_weight && !is_bias) {
        throw std::runtime_error("[Provisioner] Phase 2 only supports encrypted DensePackedLayer args, got: " + arg_id);
    }

    const std::string prefix = is_weight ? "densew_" : "denseb_";
    const std::string layer_id = arg_id.substr(prefix.size());
    const int encrypt_level = sig.at("level").get<int>();
    const auto shape = sig.at("size").get<std::vector<uint64_t>>();

    InitInferenceProcess init(server_dir_.string(), false);
    init.init_parameters(false);
    init.is_lazy = true;
    init.load_model_prepare();

    auto& layer = init.get_layer<DensePackedLayer>(layer_id);
    if (!layer.normal_dense || layer.is_1d_multiplexed) {
        throw std::runtime_error("[Provisioner] Phase 2 only supports ordinary 0D DensePackedLayer args: " + arg_id);
    }

    const uint64_t count = shape_size(sig);
    std::vector<fhe::CkksCiphertext> encrypted;
    encrypted.reserve(count);
    for (uint64_t flat_idx = 0; flat_idx < count; ++flat_idx) {
        fhe::CkksPlaintextRingt ringt;
        if (is_weight) {
            if (shape.size() != 2) {
                throw std::runtime_error("[Provisioner] dense weight signature must be rank 2: " + arg_id);
            }
            const uint32_t out_idx = static_cast<uint32_t>(flat_idx / shape[1]);
            const uint32_t weight_idx = static_cast<uint32_t>(flat_idx % shape[1]);
            ringt = layer.generate_weight_0d_pt_for_indices(context, out_idx, weight_idx);
        } else {
            if (shape.size() != 1) {
                throw std::runtime_error("[Provisioner] dense bias signature must be rank 1: " + arg_id);
            }
            ringt = layer.generate_bias_0d_pt_for_index(context, static_cast<uint32_t>(flat_idx));
        }
        auto pt = context.ringt_to_pt(ringt, encrypt_level);
        encrypted.push_back(context.encrypt_asymmetric(pt));
    }
    return encrypted;
}

void InferenceProvisioner::write_encrypted_argument(const std::filesystem::path& parameter_root,
                                                    const std::string& arg_id,
                                                    const std::vector<fhe::CkksCiphertext>& values,
                                                    json& manifest,
                                                    const json& sig) const {
    const std::string filename = arg_id + ".bin";
    std::stringstream ss;
    uint64_t count = values.size();
    ss_write(ss, count);
    for (const auto& ct : values) {
        Bytes ct_bytes = ct.serialize(context_->get_parameter());
        ss_write_vector(ss, ct_bytes);
    }
    write_binary_file(parameter_root / filename, ss_to_bytes(ss));

    manifest["arguments"][arg_id] = {
        {"id", arg_id},           {"type", "ct"},     {"level", sig.at("level").get<int>()},
        {"size", sig.at("size")}, {"file", filename},
    };
}

void InferenceProvisioner::export_runner_bundle(const std::string& runner_dir_arg) {
    setup_or_load_keys();

    const std::filesystem::path runner_dir(runner_dir_arg);
    std::filesystem::create_directories(runner_dir);
    copy_private_and_runner_configs(runner_dir);

    task_signature_ = read_json((runner_dir / "task_signature.json").string());
    if (task_signature_.value("algorithm", std::string("")) != "CKKS") {
        throw std::runtime_error("[Provisioner] only CKKS task signatures are supported");
    }

    auto eval_context = context_->make_public_context(false, true, true);
    Bytes eval_bytes = eval_context.serialize_advanced();
    write_binary_file(provisioner_dir_ / "eval_context.bin", eval_bytes);
    write_binary_file(runner_dir / "eval_context.bin", eval_bytes);

    const auto parameter_root = runner_dir / "encrypted_model_parameters";
    std::filesystem::create_directories(parameter_root);

    json manifest = {
        {"format", "latti-ai.encrypted-parameter-store.v1"},
        {"arguments", json::object()},
    };

    for (const auto& sig : task_signature_.at("offline")) {
        const std::string arg_id = sig.at("id").get<std::string>();
        if (sig.at("type").get<std::string>() != "ct") {
            throw std::runtime_error("[Provisioner] Phase 2 only supports ciphertext offline args: " + arg_id);
        }
        std::cout << "[Provisioner] Encrypting offline parameter " << arg_id << "..." << std::endl;
        auto encrypted = encrypt_dense_argument(arg_id, sig, *context_);
        write_encrypted_argument(parameter_root, arg_id, encrypted, manifest, sig);
    }

    write_json_file(parameter_root / "manifest.json", manifest);
    write_json_file(provisioner_dir_ / "encrypted_parameter_manifest.json", manifest);

    if (std::filesystem::exists(runner_dir / "secret_key.bin") ||
        std::filesystem::exists(runner_dir / "secret_context.bin") ||
        std::filesystem::exists(runner_dir / "model_parameters.h5")) {
        throw std::runtime_error("[Provisioner] runner bundle contains secret/model files; remove them before use: " +
                                 runner_dir.string());
    }

    std::cout << "[Provisioner] Runner bundle exported to " << runner_dir << std::endl;
}
