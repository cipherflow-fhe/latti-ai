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

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "fhe_ops_lib/fhe_lib_v2.h"
#include "util.h"

namespace fhe = fhe_ops_lib;

class InitInferenceProcess;

class InferenceProvisioner {
public:
    explicit InferenceProvisioner(const std::string& task_dir);
    ~InferenceProvisioner();

    InferenceProvisioner(const InferenceProvisioner&) = delete;
    InferenceProvisioner& operator=(const InferenceProvisioner&) = delete;
    InferenceProvisioner(InferenceProvisioner&&) = default;
    InferenceProvisioner& operator=(InferenceProvisioner&&) = default;

    void setup_or_load_keys();
    void export_runner_bundle(const std::string& runner_dir);

private:
    std::filesystem::path task_dir_;
    std::filesystem::path server_dir_;
    std::filesystem::path provisioner_dir_;
    json task_config_;
    json ckks_config_;
    json task_signature_;
    std::unique_ptr<fhe::CkksContext> context_;

    void resolve_directories();
    fhe::CkksParameter make_parameter() const;
    void copy_private_and_runner_configs(const std::filesystem::path& runner_dir) const;
    struct ParameterArgumentInfo {
        std::string source_id;
        std::string layer_id;
        std::string param_kind;
        int coeff_idx = -1;
    };

    ParameterArgumentInfo parse_parameter_argument(const std::string& arg_id, const InitInferenceProcess& init) const;
    std::vector<fhe::CkksCiphertext> encrypt_parameter_argument(const std::string& arg_id,
                                                                const json& sig,
                                                                fhe::CkksContext& context,
                                                                InitInferenceProcess& init) const;
    void write_encrypted_argument(const std::filesystem::path& parameter_root,
                                  const std::string& arg_id,
                                  const std::vector<fhe::CkksCiphertext>& values,
                                  json& manifest,
                                  const json& sig,
                                  const ParameterArgumentInfo& info) const;
};
