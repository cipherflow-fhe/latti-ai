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
#include "util.h"

using namespace cxx_sdk_v2;

/// Result of decrypting an encrypted inference output.
struct DecryptedOutput {
    std::vector<double> output;
    int num_outputs;
};

/// Client-side encrypted inference interface.
///
/// Holds the secret key and is responsible for:
/// - Generating the full CKKS key pair
/// - Exporting a public-only evaluation context for the server
/// - Encrypting input data
/// - Decrypting inference results
///
class InferenceClient {
public:
    /// @param client_dir  Path to the client directory (contains task_config.json, ckks_parameter.json).
    explicit InferenceClient(const std::string& client_dir);
    ~InferenceClient();

    InferenceClient(const InferenceClient&) = delete;
    InferenceClient& operator=(const InferenceClient&) = delete;
    InferenceClient(InferenceClient&&) = default;
    InferenceClient& operator=(InferenceClient&&) = default;

    /// Read configuration and generate crypto context with keys.
    void setup();

    /// Export a public-only evaluation context (serialized bytes).
    /// The server uses this to perform encrypted computation without the secret key.
    Bytes export_eval_context() const;

    /// Encrypt input from a CSV file and return serialized ciphertext.
    Bytes encrypt(const std::string& input_csv) const;

    /// Decrypt serialized encrypted output from the server.
    DecryptedOutput decrypt(const Bytes& encrypted_output) const;

private:
    std::filesystem::path client_dir_;

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

    std::unique_ptr<CkksParameter> ckks_param_;
    std::unique_ptr<CkksBtpParameter> btp_param_;
    CkksContext* context_ptr_ = nullptr;
    std::unique_ptr<CkksContext> ckks_context_;
    std::unique_ptr<CkksBtpContext> btp_context_;

    void read_configuration();
    void create_crypto_context();
    double get_default_scale() const;
};
