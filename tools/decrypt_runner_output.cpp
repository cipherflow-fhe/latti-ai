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

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "interface/inference_client.h"
#include "interface/runner_bundle_io.h"

int main(int argc, char* argv[]) {
    std::string task_dir;
    std::string cipher_path;
    std::string output_csv;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--task-dir") == 0 && i + 1 < argc) {
            task_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--cipher") == 0 && i + 1 < argc) {
            cipher_path = argv[++i];
        } else if (std::strcmp(argv[i], "--output-csv") == 0 && i + 1 < argc) {
            output_csv = argv[++i];
        }
    }

    if (task_dir.empty() || cipher_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " --task-dir <task-root/provisioner> --cipher <output.ct>"
                  << " [--output-csv <full-logits.csv>]" << std::endl;
        return 1;
    }

    try {
        std::filesystem::path provisioner_dir(task_dir);
        InferenceClient decryptor(task_dir);
        if (std::filesystem::exists(provisioner_dir / "full_context.bin")) {
            decryptor.load_full_context(read_binary_file(provisioner_dir / "full_context.bin"));
        } else {
            decryptor.load_context(read_binary_file(provisioner_dir / "eval_context.bin"),
                                   read_binary_file(provisioner_dir / "secret_key.bin"));
        }
        auto encrypted_outputs = read_named_bytes_bundle(cipher_path);
        auto results = decryptor.decrypt(encrypted_outputs);

        if (!output_csv.empty()) {
            std::ofstream csv(output_csv);
            if (!csv) throw std::runtime_error("Cannot open output CSV: " + output_csv);
            csv << "output,index,value\n" << std::setprecision(std::numeric_limits<double>::max_digits10);
            for (const auto& [name, result] : results) {
                std::string escaped_name;
                for (char c : name) escaped_name += c == '"' ? "\"\"" : std::string(1, c);
                for (size_t i = 0; i < result.output.size(); ++i) {
                    csv << '"' << escaped_name << "\"," << i << ',' << result.output[i] << '\n';
                }
            }
            csv.close();
            if (!csv) throw std::runtime_error("Cannot write output CSV: " + output_csv);
        }

        for (const auto& [name, result] : results) {
            std::cout << "Decrypted output [" << name << "] = [";
            int print_count = std::min(result.num_outputs, 10);
            for (int i = 0; i < print_count; ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << std::fixed << std::setprecision(8) << result.output[i];
            }
            if (result.num_outputs > print_count) {
                std::cout << ", ...";
            }
            std::cout << "] (num_outputs=" << result.num_outputs << ")" << std::endl;

            if (!result.output.empty()) {
                auto max_it = std::max_element(result.output.begin(), result.output.end());
                std::cout << "Top-1 [" << name << "]: index=" << std::distance(result.output.begin(), max_it)
                          << ", value=" << std::fixed << std::setprecision(8) << *max_it << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 2;
    }
    return 0;
}
