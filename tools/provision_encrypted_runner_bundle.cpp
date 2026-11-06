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

#include <cstring>
#include <iostream>
#include <string>

#include "interface/inference_provisioner.h"

int main(int argc, char* argv[]) {
    std::string task_dir;
    std::string out_dir;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--task-dir") == 0 && i + 1 < argc) {
            task_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_dir = argv[++i];
        }
    }

    if (task_dir.empty() || out_dir.empty()) {
        std::cerr << "Usage: " << argv[0] << " --task-dir <task-root> --out <task-root/runner>" << std::endl;
        return 1;
    }

    try {
        InferenceProvisioner provisioner(task_dir);
        provisioner.export_runner_bundle(out_dir);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 2;
    }
    return 0;
}
