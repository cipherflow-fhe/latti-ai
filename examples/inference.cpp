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

#include "interface/inference_client.h"
#include "interface/inference_server.h"

using namespace std;

int main(int argc, char* argv[]) {
    string task_dir;
    string input_csv;
    bool use_gpu = false;
    bool verify = false;
    constexpr double tolerance = 0.1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gpu") == 0) {
            use_gpu = true;
        } else if (strcmp(argv[i], "--task-dir") == 0 && i + 1 < argc) {
            task_dir = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_csv = argv[++i];
        } else if (strcmp(argv[i], "--verify") == 0) {
            verify = true;
        }
    }

    if (task_dir.empty() || input_csv.empty()) {
        cerr << "Usage: " << argv[0] << " --task-dir <path> --input <path> [--gpu] [--verify]" << endl;
        return 1;
    }

    cout << "========== Encrypted Inference ==========" << endl;
    cout << "Task directory: " << task_dir << endl;
    cout << "Input file:     " << input_csv << endl;
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
    auto encrypted_input = client.encrypt(input_csv);

    // In actual scenarios, the client sends eval_ctx and encrypted_input to the server over the network.

    // --- Server side: load model and run encrypted inference ---
    cout << "[Step 3/5] Server loading model and importing context..." << endl;
    InferenceServer server(task_dir + "/server", use_gpu);
    server.import_eval_context(eval_ctx);
    server.load_model();

    cout << "[Step 4/5] Running encrypted inference..." << endl;
    auto encrypted_output = server.evaluate(encrypted_input);

    // In actual scenarios, the server sends encrypted_output back to the client over the network.

    // --- Client side: decrypt result ---
    cout << "[Step 5/5] Decrypting result..." << endl;
    auto result = client.decrypt(encrypted_output);
    cout << endl;

    // --- Display results ---
    cout << "========== Results ==========" << endl;
    print_double_message(result.output.data(), "Encrypted output", 10);

    auto plaintext_output = server.evaluate_plaintext(input_csv);
    print_double_message(plaintext_output.data(), "Plaintext output", 10);

    if (verify) {
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

        cout << endl;
        cout << "========== Verification ==========" << endl;
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

    return 0;
}
