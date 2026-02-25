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

#include <cstring>
#include <iostream>
#include <string>

#include "interface/inference_interface.h"

using namespace std;

int main(int argc, char* argv[]) {
    string task_dir;
    string input_csv;
    bool use_gpu = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gpu") == 0) {
            use_gpu = true;
        } else if (strcmp(argv[i], "--task-dir") == 0 && i + 1 < argc) {
            task_dir = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_csv = argv[++i];
        }
    }

    if (task_dir.empty() || input_csv.empty()) {
        cerr << "Usage: " << argv[0] << " --task-dir <path> --input <path> [--gpu]" << endl;
        return 1;
    }

    cout << "========== Encrypted Inference ==========" << endl;
    cout << "Task directory: " << task_dir << endl;
    cout << "Input file:     " << input_csv << endl;

    EncryptedInference engine(task_dir, use_gpu);

    engine.encrypt(input_csv);
    engine.evaluate();
    auto result = engine.decrypt();

    print_double_message(result.encrypted_output.data(), "Encrypted output", result.num_outputs);
    print_double_message(result.plaintext_output.data(), "Plaintext output", result.num_outputs);

    return 0;
}
