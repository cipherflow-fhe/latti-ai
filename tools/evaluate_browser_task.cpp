// Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Execution-only adapter: every task, key and ciphertext comes from the browser.

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include "interface/inference_runner.h"
#include "interface/inference_server.h"
#include "interface/runner_bundle_io.h"

int main(int argc, char* argv[]) {
    try {
        std::string task, mode, evaluation_context, output;
        bool gpu = false;
        int device = 0;
        std::map<std::string, std::string> inputs;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--gpu") {
                gpu = true;
                continue;
            }
            if (i + 1 == argc)
                throw std::runtime_error("Missing option value: " + flag);
            const std::string value = argv[++i];
            if (flag == "--task-dir")
                task = value;
            else if (flag == "--mode")
                mode = value;
            else if (flag == "--eval-context")
                evaluation_context = value;
            else if (flag == "--output")
                output = value;
            else if (flag == "--device")
                device = std::stoi(value);
            else if (flag == "--input") {
                const auto separator = value.find('=');
                if (separator == 0 || separator == std::string::npos || separator + 1 == value.size() ||
                    !inputs.emplace(value.substr(0, separator), value.substr(separator + 1)).second)
                    throw std::runtime_error("Invalid or duplicate named input");
            } else
                throw std::runtime_error("Unknown option: " + flag);
        }
        if (task.empty() || output.empty() || inputs.empty())
            throw std::runtime_error("Required: --task-dir --mode --output --input ID=FILE [--gpu --device N]");
        std::map<std::string, Bytes> result;
        if (mode == "client_encrypted_input") {
            if (evaluation_context.empty())
                throw std::runtime_error("Input mode requires --eval-context");
            InferenceServer server(task, gpu, device);
            server.import_eval_context_file(evaluation_context);
            server.load_model();
            std::map<std::string, Bytes> ciphertexts;
            for (const auto& [name, file] : inputs)
                ciphertexts.emplace(name, read_binary_file(file));
            result = server.evaluate(ciphertexts);
        } else if (mode == "server_provisioned_runner") {
            InferenceRunner runner(task, gpu, device);
            runner.load();
            result = runner.evaluate_plaintext_input(inputs);
        } else
            throw std::runtime_error("Invalid deployment mode");
        write_named_bytes_bundle(output, result);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
