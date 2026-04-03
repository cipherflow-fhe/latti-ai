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

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "catch.hpp"
#include "interface/inference_client.h"
#include "interface/inference_server.h"
#include "util.h"

namespace fs = std::filesystem;

static const fs::path e2e_base_path = "../hetero_e2e";

static void run_e2e_test(const fs::path& test_dir, bool use_gpu) {
    using clock = std::chrono::high_resolution_clock;
    std::string test_name = test_dir.filename().string();
    std::string device_tag = use_gpu ? "GPU" : "CPU";
    fs::path client_dir = test_dir / "task" / "client";
    fs::path server_dir = test_dir / "task" / "server";

    // Read task_config to get input names and build CSV map
    auto task_config = read_json((client_dir / "task_config.json").string());
    std::map<std::string, std::string> input_csvs;
    for (auto& [name, param] : task_config["task_input_param"].items()) {
        fs::path csv_path = client_dir / (name + ".csv");
        REQUIRE(fs::exists(csv_path));
        input_csvs[name] = csv_path.string();
    }

    // Timing records: label -> duration in ms
    struct TimingRecord {
        const char* label;
        double ms;
    };
    std::vector<TimingRecord> timings;
    auto measure = [&](const char* label, auto&& fn) {
        auto t0 = clock::now();
        auto result = fn();
        auto t1 = clock::now();
        timings.push_back({label, std::chrono::duration<double, std::milli>(t1 - t0).count()});
        return result;
    };
    auto measure_void = [&](const char* label, auto&& fn) {
        auto t0 = clock::now();
        fn();
        auto t1 = clock::now();
        timings.push_back({label, std::chrono::duration<double, std::milli>(t1 - t0).count()});
    };

    // Client: generate keys and encrypt input
    InferenceClient client(client_dir.string());
    measure_void("Key generation", [&] { client.setup(); });
    auto eval_ctx = measure("Export eval context", [&] { return client.export_eval_context(); });
    auto encrypted_inputs = measure("Encrypt input", [&] { return client.encrypt(input_csvs); });

    // Server: import context, load model, run encrypted inference
    InferenceServer server(server_dir.string(), use_gpu);
    measure_void("Import eval context", [&] { server.import_eval_context(eval_ctx); });
    measure_void("Load model", [&] { server.load_model(); });
    auto encrypted_outputs = measure("Evaluate (encrypted)", [&] { return server.evaluate(encrypted_inputs); });

    // Client: decrypt results
    auto results = measure("Decrypt output", [&] { return client.decrypt(encrypted_outputs); });

    // Server: run plaintext reference inference
    auto plaintext_outputs = measure("Evaluate (plaintext)", [&] { return server.evaluate_plaintext(input_csvs); });

    // Print timing summary
    double total_ms = 0.0;
    for (auto& t : timings)
        total_ms += t.ms;
    std::cout << "\n[" << test_name << "] (" << device_tag << ") Timing:" << std::endl;
    for (auto& t : timings) {
        std::cout << "  " << std::left << std::setw(24) << t.label << std::right << std::setw(10) << std::fixed
                  << std::setprecision(2) << t.ms << " ms" << std::endl;
    }
    std::cout << "  " << std::string(34, '-') << std::endl;
    std::cout << "  " << std::left << std::setw(24) << "Total" << std::right << std::setw(10) << std::fixed
              << std::setprecision(2) << total_ms << " ms" << std::endl;

    // Compare encrypted vs plaintext outputs
    constexpr double tolerance = 0.1;
    for (auto& [name, result] : results) {
        auto pt_it = plaintext_outputs.find(name);
        REQUIRE(pt_it != plaintext_outputs.end());
        auto& pt_output = pt_it->second;

        int count = std::min(result.num_outputs, static_cast<int>(pt_output.size()));
        REQUIRE(count > 0);

        double max_abs_err = 0.0;
        int max_err_idx = 0;
        for (int i = 0; i < count; i++) {
            double abs_err = fabs(result.output[i] - pt_output[i]);
            if (abs_err > max_abs_err) {
                max_abs_err = abs_err;
                max_err_idx = i;
            }
        }

        // Print first 10 plaintext/ciphertext values for manual comparison
        int preview_count = std::min(count, 10);
        std::cout << std::defaultfloat << std::setprecision(4);
        std::cout << "[" << test_name << "] Output: " << name << " (" << device_tag << ", " << count << " values)"
                  << std::endl;
        std::cout << "  Plaintext : ";
        for (int i = 0; i < preview_count; i++) {
            std::cout << pt_output[i] << (i < preview_count - 1 ? ", " : "");
        }
        std::cout << std::endl;
        std::cout << "  Ciphertext: ";
        for (int i = 0; i < preview_count; i++) {
            std::cout << result.output[i] << (i < preview_count - 1 ? ", " : "");
        }
        std::cout << std::endl;
        std::cout << "  Max error : " << std::scientific << max_abs_err << " at index " << max_err_idx << std::endl;

        INFO("Test: " << test_name << " (" << device_tag << "), Output: " << name << ", Max error: " << max_abs_err
                      << " at index " << max_err_idx);
        REQUIRE(max_abs_err < tolerance);
    }
}
