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

#include <cxx_sdk_v2/cxx_fhe_task.h>
#include "data_structs/feature.h"
#include "inference_task/inference_process.h"
#include "util.h"

using namespace cxx_sdk_v2;

/// Result of an encrypted inference run.
struct InferenceResult {
    std::vector<double> encrypted_output;
    std::vector<double> plaintext_output;
    int num_outputs;
};

/// High-level encrypted inference interface.
///
/// Encapsulates the full pipeline: context creation, encryption,
/// model loading, inference, and decryption.  Auto-detects whether
/// bootstrapping is needed from the task configuration files.
///
/// Usage:
///   EncryptedInference engine("./task");
///   engine.encrypt("./task/client/img.csv");
///   engine.evaluate();
///   auto result = engine.decrypt();
class EncryptedInference {
public:
    /// @param task_dir  Path to the task directory (contains client/ and server/).
    /// @param use_gpu   Whether to use GPU acceleration.
    explicit EncryptedInference(const std::string& task_dir = "./task", bool use_gpu = false);
    ~EncryptedInference();

    EncryptedInference(const EncryptedInference&) = delete;
    EncryptedInference& operator=(const EncryptedInference&) = delete;
    EncryptedInference(EncryptedInference&&) = default;
    EncryptedInference& operator=(EncryptedInference&&) = default;

    /// Read input, create crypto context, and encrypt.
    void encrypt(const std::string& input_csv_path);

    /// Load model and run encrypted inference.
    void evaluate();

    /// Decrypt output and run plaintext verification.
    InferenceResult decrypt();

private:
    std::filesystem::path task_dir_;
    bool use_gpu_;

    // Parsed from task_config.json and ckks_parameter.json
    int level_ = 0;
    int output_skip_ = 0;
    int channel_ = 0;
    int height_ = 0;
    int width_ = 0;
    int n_slots_ = 0;
    int poly_modulus_degree_ = 0;
    bool needs_btp_ = false;
    std::string pack_style_;
    nlohmann::ordered_json task_config_;

    // Crypto objects (btp and non-btp, only one is populated)
    std::unique_ptr<CkksParameter> ckks_param_;
    std::unique_ptr<CkksBtpParameter> btp_param_;
    CkksContext* context_ptr_ = nullptr;
    std::unique_ptr<CkksContext> ckks_context_;
    std::unique_ptr<CkksBtpContext> btp_context_;

    // Inference objects
    std::unique_ptr<InitInferenceProcess> init_;
    std::unique_ptr<InferenceProcess> fp_;
    std::unique_ptr<Feature2DEncrypted> input_ct_;
    Array<double, 3> input_array_;

    void read_configuration();
    void create_crypto_context();
    double get_default_scale() const;
};
