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

#include <memory>

#include "data_structs/feature.h"
#include "inference_task/inference_process.h"
#include "interface/inference_client.h"
#include "util.h"

namespace ls = lattisense;

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
    /// @param gpu_device  GPU device index (default 0).
    explicit InferenceServer(const std::string& server_dir, bool use_gpu = false, int gpu_device = 0);
    ~InferenceServer();

    InferenceServer(const InferenceServer&) = delete;
    InferenceServer& operator=(const InferenceServer&) = delete;
    InferenceServer(InferenceServer&&) = default;
    InferenceServer& operator=(InferenceServer&&) = default;

    /// Import evaluation context (public keys only) from serialized bytes.
    void import_eval_context(const Bytes& eval_context);

    /// Import evaluation context as a regular CKKS context even if task_config.json has use_btp=true.
    void import_eval_context_ckks(const Bytes& eval_context);

    /// Load model weights and computation graph.
    void load_model();

    /// Load model for the hand-written SDK/MPC executor without task_signature.json.
    void load_model_for_mpc_sdk();

    /// Run encrypted inference on serialized ciphertext inputs.
    /// @param encrypted_inputs  Map of input name -> serialized ciphertext bytes.
    /// @param progress_cb  Optional callback for progress reporting (completed, total).
    /// Returns map of output name -> serialized ciphertext bytes.
    std::map<std::string, Bytes> evaluate(const std::map<std::string, Bytes>& encrypted_inputs,
                                          ls::ProgressCallback progress_cb = nullptr);

    /// Run encrypted inference through the SDK executor with MPC refresh enabled.
    /// Requires an active MPC client connected through the global SCI channel.
    std::map<std::string, Bytes> evaluate_mpc_sdk(const std::map<std::string, Bytes>& encrypted_inputs);

    /// Last SDK MPC refresh time in milliseconds.
    double get_last_mpc_time_ms() const;

    /// Dump all encrypted intermediate features as plaintext for local debugging.
    /// Requires a full CKKS context with the secret key.
    void dump_intermediate_plaintexts(const std::string& output_path) const;

    /// Run plaintext inference and dump all plaintext intermediate features.
    void dump_plaintext_intermediates(const std::map<std::string, std::string>& input_csvs,
                                      const std::string& output_path);

    /// Dump each task output's first ciphertext slots and matching plaintext slots.
    void dump_task_output_debug(const std::map<std::string, std::string>& input_csvs,
                                const std::string& output_path);

    /// Run plaintext inference on CSV input files (for verification).
    std::map<std::string, std::vector<double>> evaluate_plaintext(const std::map<std::string, std::string>& input_csvs);

private:
    std::filesystem::path server_dir_;
    bool use_gpu_;
    int gpu_device_;
    bool needs_btp_ = false;

    std::vector<std::string> input_keys_;
    std::map<std::string, InputParam> input_params_;
    std::vector<std::string> output_keys_;
    std::map<std::string, OutputParam> output_params_;
    ls::CkksContext* context_ptr_ = nullptr;
    std::unique_ptr<ls::CkksContext> eval_context_;
    std::unique_ptr<ls::CkksBtpContext> eval_btp_context_;

    std::unique_ptr<InitInferenceProcess> init_;
    std::unique_ptr<InferenceProcess> fp_;
};
