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

#include <cxx_sdk_v2/cxx_fhe_task.h>
#include "data_structs/feature.h"
#include "inference_task/inference_process.h"
#include "util.h"

using namespace cxx_sdk_v2;

/// Server-side encrypted inference interface.
///
/// Holds the model and a public-only evaluation context.
/// Responsible for:
/// - Importing the evaluation context from the client
/// - Loading the model weights and computation graph
/// - Running encrypted inference on ciphertext inputs
///
class InferenceServer {
public:
    /// @param server_dir  Path to the server directory (contains model weights, ergs, task_config.json).
    /// @param use_gpu     Whether to use GPU acceleration.
    explicit InferenceServer(const std::string& server_dir, bool use_gpu = false);
    ~InferenceServer();

    InferenceServer(const InferenceServer&) = delete;
    InferenceServer& operator=(const InferenceServer&) = delete;
    InferenceServer(InferenceServer&&) = default;
    InferenceServer& operator=(InferenceServer&&) = default;

    /// Import evaluation context (public keys only) from serialized bytes.
    void import_eval_context(const Bytes& eval_context);

    /// Load model weights and computation graph.
    void load_model();

    /// Run encrypted inference on serialized ciphertext input.
    /// Returns serialized ciphertext output.
    Bytes evaluate(const Bytes& encrypted_input);

    /// Run plaintext inference on a CSV input file (for verification).
    std::vector<double> evaluate_plaintext(const std::string& input_csv);

private:
    std::filesystem::path server_dir_;
    bool use_gpu_;
    bool needs_btp_ = false;
    int channel_ = 0;
    int height_ = 0;
    int width_ = 0;

    CkksContext* context_ptr_ = nullptr;
    std::unique_ptr<CkksContext> eval_context_;
    std::unique_ptr<CkksBtpContext> eval_btp_context_;

    std::unique_ptr<InitInferenceProcess> init_;
    std::unique_ptr<InferenceProcess> fp_;
};
