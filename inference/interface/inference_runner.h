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
#include <map>
#include <memory>
#include <string>

#include "data_structs/feature.h"
#include "interface/encrypted_parameter_store.h"
#include "interface/inference_client.h"
#include "lattisense/cxx_sdk_v2/cxx_fhe_task.h"
#include "util.h"

namespace ls = lattisense;
namespace fhe = fhe_ops_lib;

class InferenceRunner {
public:
    explicit InferenceRunner(const std::string& runner_dir, bool use_gpu = false, int gpu_device = 0);
    ~InferenceRunner();

    InferenceRunner(const InferenceRunner&) = delete;
    InferenceRunner& operator=(const InferenceRunner&) = delete;
    InferenceRunner(InferenceRunner&&) = default;
    InferenceRunner& operator=(InferenceRunner&&) = default;

    void load();
    std::map<std::string, Bytes> evaluate_plaintext_input(const std::map<std::string, std::string>& input_csvs,
                                                          ls::ProgressCallback progress_cb = nullptr);

private:
    std::filesystem::path runner_dir_;
    bool use_gpu_;
    int gpu_device_;

    json task_config_;
    json task_signature_;
    std::map<std::string, InputParam> input_params_;
    std::map<std::string, OutputParam> output_params_;
    int n_slots_ = 0;
    std::unique_ptr<fhe::CkksContext> eval_context_;
    std::unique_ptr<EncryptedParameterStore> parameter_store_;

    void read_configuration();
    std::vector<fhe::CkksPlaintextRingt>
    encode_plaintext_0d_input(const std::string& name, const std::string& csv_path, const json& sig) const;
};
