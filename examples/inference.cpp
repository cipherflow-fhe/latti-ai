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

// Unified encrypted inference example.
// Works for any task (MNIST, CIFAR-10, ImageNet, etc.) by specifying --task-dir.
// Demonstrates the InferenceClient / InferenceServer separation.

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "interface/inference_client.h"
#include "interface/inference_server.h"

using namespace std;

int main(int argc, char* argv[]) {
    string task_dir;
    vector<string> input_args;
    bool use_gpu = false;
    bool verify = false;
    constexpr double tolerance = 0.1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gpu") == 0) {
            use_gpu = true;
        } else if (strcmp(argv[i], "--task-dir") == 0 && i + 1 < argc) {
            task_dir = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_args.push_back(argv[++i]);
        } else if (strcmp(argv[i], "--verify") == 0) {
            verify = true;
        }
    }

    if (task_dir.empty() || input_args.empty()) {
        cerr << "Usage: " << argv[0] << " --task-dir <path> --input [name=]<path> [--input ...] [--gpu] [--verify]"
             << endl;
        return 1;
    }

    // Read input names from task_config
    auto task_config = read_json(task_dir + "/client/task_config.json");
    vector<string> input_names;
    for (auto& [name, _] : task_config["task_input_param"].items()) {
        input_names.push_back(name);
    }

    // Build input CSV map: supports "name=path" or positional mapping
    map<string, string> input_csvs;
    for (size_t i = 0; i < input_args.size(); i++) {
        auto eq_pos = input_args[i].find('=');
        if (eq_pos != string::npos) {
            input_csvs[input_args[i].substr(0, eq_pos)] = input_args[i].substr(eq_pos + 1);
        } else if (i < input_names.size()) {
            input_csvs[input_names[i]] = input_args[i];
        } else {
            cerr << "Too many --input arguments. Expected " << input_names.size() << " inputs." << endl;
            return 1;
        }
    }

    cout << "========== Encrypted Inference ==========" << endl;
    cout << "Task directory: " << task_dir << endl;
    for (auto& [name, path] : input_csvs) {
        cout << "Input [" << name << "]: " << path << endl;
    }
    cout << "Device:         " << (use_gpu ? "GPU" : "CPU") << endl;
    if (verify) {
        cout << "Verify mode:    ON (tolerance = " << tolerance << ")" << endl;
    }
    cout << endl;

    // --- Client side: generate keys and encrypt input ---
    cout << "[Step 1/5] Setting up client (key generation)..." << endl;
    InferenceClient client(task_dir + "/client");
    client.setup();
    auto eval_ctx = client.export_eval_context();

    cout << "[Step 2/5] Encrypting input..." << endl;
    auto encrypted_inputs = client.encrypt(input_csvs);

    // In actual scenarios, the client sends eval_ctx and encrypted_inputs to the server over the network.

    // --- Server side: load model and run encrypted inference ---
    cout << "[Step 3/5] Server loading model and importing context..." << endl;
    InferenceServer server(task_dir + "/server", use_gpu);
    server.import_eval_context(eval_ctx);
    server.load_model();

    cout << "[Step 4/5] Running encrypted inference..." << endl;
    auto encrypted_outputs = server.evaluate(encrypted_inputs);

    // In actual scenarios, the server sends encrypted_outputs back to the client over the network.

    // --- Client side: decrypt result ---
    cout << "[Step 5/5] Decrypting result..." << endl;
    auto results = client.decrypt(encrypted_outputs);
    cout << endl;

    // --- Display results ---
    cout << "========== Results ==========" << endl;
    for (auto& [name, result] : results) {
        print_double_message(result.output.data(), ("Encrypted output [" + name + "]").c_str(), 1);
    }

    auto plaintext_outputs = server.evaluate_plaintext(input_csvs);
    for (auto& [name, plaintext_output] : plaintext_outputs) {
        print_double_message(plaintext_output.data(), ("Plaintext output [" + name + "]").c_str(), 1);
    }

    if (verify) {
        for (auto& [name, result] : results) {
            auto pt_it = plaintext_outputs.find(name);
            if (pt_it == plaintext_outputs.end())
                continue;
            auto& plaintext_output = pt_it->second;

            cout << endl;
            cout << "========== Verification [" << name << "] ==========" << endl;
            int count = min(result.num_outputs, (int)plaintext_output.size());
            double max_abs_err = 0.0;
            double sum_abs_err = 0.0;
            int max_err_idx = 0;
            for (int i = 0; i < count; i++) {
                double abs_err = fabs(result.output[i] - plaintext_output[i]);
                sum_abs_err += abs_err;
                if (abs_err > max_abs_err) {
                    max_abs_err = abs_err;
                    max_err_idx = i;
                }
            }
            double avg_abs_err = count > 0 ? sum_abs_err / count : 0.0;

            cout << "Elements compared: " << count << endl;
            cout << fixed << setprecision(8);

            cout << endl;
            cout << setw(8) << "Index" << setw(18) << "Encrypted" << setw(18) << "Plaintext" << setw(18) << "Abs Error"
                 << endl;
            cout << string(62, '-') << endl;
            for (int i = 0; i < count; i++) {
                double abs_err = fabs(result.output[i] - plaintext_output[i]);
                cout << setw(8) << i << setw(18) << result.output[i] << setw(18) << plaintext_output[i] << setw(18)
                     << abs_err;
                if (abs_err > tolerance) {
                    cout << "  <-- EXCEEDS TOLERANCE";
                }
                cout << endl;
            }

            cout << string(62, '-') << endl;
            cout << "Max absolute error: " << max_abs_err << " (at index " << max_err_idx << ")" << endl;
            cout << "Avg absolute error: " << avg_abs_err << endl;
            cout << "Tolerance:          " << tolerance << endl;
            cout << endl;

            if (max_abs_err > tolerance) {
                cout << "Result: FAIL" << endl;
                return 1;
            }
            cout << "Result: PASS" << endl;
        }
    }

    return 0;
}
