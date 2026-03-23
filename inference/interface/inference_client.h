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

/// Per-output parameters read from task_config.json.
struct OutputParam {
    int dim = 0;
    int channel = 0;
    int skip = 1;    // dim=0 only (scalar skip)
    int height = 0;  // dim=2 only
    int width = 0;   // dim=2 only
};

/// Per-input parameters read from task_config.json.
struct InputParam {
    int dim = 0;
    int level = 0;
    int channel = 0;
    int height = 0;    // dim=2 only
    int width = 0;     // dim=2 only
    int skip = 1;      // dim=0 only
    int pack_num = 0;  // n_channel_per_ct
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

    /// Encrypt inputs from CSV files and return serialized ciphertexts.
    /// @param input_csvs  Map of input name -> CSV file path.
    /// @return Map of input name -> serialized ciphertext bytes.
    std::map<std::string, Bytes> encrypt(const std::map<std::string, std::string>& input_csvs) const;

    /// Decrypt serialized encrypted outputs from the server.
    std::map<std::string, DecryptedOutput> decrypt(const std::map<std::string, Bytes>& encrypted_outputs) const;

private:
    std::filesystem::path client_dir_;

    int output_skip_ = 0;
    std::map<std::string, OutputParam> output_params_;
    int n_slots_ = 0;
    int poly_modulus_degree_ = 0;
    bool needs_btp_ = false;
    std::string pack_style_;
    nlohmann::ordered_json task_config_;
    std::map<std::string, InputParam> input_params_;

    std::unique_ptr<CkksParameter> ckks_param_;
    std::unique_ptr<CkksBtpParameter> btp_param_;
    CkksContext* context_ptr_ = nullptr;
    std::unique_ptr<CkksContext> ckks_context_;
    std::unique_ptr<CkksBtpContext> btp_context_;

    void read_configuration();
    void create_crypto_context();
    double get_default_scale() const;
};
