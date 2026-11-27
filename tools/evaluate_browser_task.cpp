// Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Execution-only adapter: every task, key and ciphertext comes from the browser.

#include <iostream>
#include <chrono>
#include <fstream>
#include <set>
#include <map>
#include <stdexcept>
#include <string>

#include "interface/inference_runner.h"
#include "interface/inference_server.h"
#include "interface/runner_bundle_io.h"

int main(int argc, char* argv[]) {
    try {
        std::string task, mode, evaluation_context, output, batch_manifest, output_directory;
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
            else if (flag == "--batch-manifest")
                batch_manifest = value;
            else if (flag == "--output-dir")
                output_directory = value;
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
        if (!batch_manifest.empty()) {
            if (task.empty() || output_directory.empty() || mode != "server_provisioned_runner" || !inputs.empty() ||
                !output.empty())
                throw std::runtime_error(
                    "Batch requires --task-dir --mode server_provisioned_runner --batch-manifest --output-dir");
            const auto batch = read_json(batch_manifest);
            const auto& samples = batch.at("samples");
            if (!samples.is_array() || samples.empty())
                throw std::runtime_error("Empty batch");
            const auto root = std::filesystem::path(batch_manifest).parent_path();
            const auto config = read_json((std::filesystem::path(task) / "task_config.json").string());
            std::set<std::string> expected_inputs;
            for (const auto& item : config.at("task_input_param").items())
                expected_inputs.insert(item.key());
            std::set<std::string> seen;
            // Validate all paths and sample IDs before evaluating any sample.
            for (const auto& sample : samples) {
                const auto id = sample.at("case_id").get<std::string>();
                if (id.empty() || id == "." || id == ".." || id.find_first_of("/\\:") != std::string::npos ||
                    id.find('\0') != std::string::npos || !seen.insert(id).second)
                    throw std::runtime_error("Invalid or duplicate batch case ID");
                if (!sample.at("inputs").is_object())
                    throw std::runtime_error("Batch inputs must be an object");
                std::set<std::string> provided_inputs;
                for (const auto& item : sample.at("inputs").items())
                    provided_inputs.insert(item.key());
                if (provided_inputs != expected_inputs)
                    throw std::runtime_error("Batch input feature set differs");
                for (const auto& item : sample.at("inputs").items()) {
                    const auto value = item.value().get<std::string>();
                    if (value.find_first_of("\\:") != std::string::npos || value.find('\0') != std::string::npos)
                        throw std::runtime_error("Invalid batch input path");
                    auto file = std::filesystem::path(item.value().get<std::string>());
                    if (file.empty() || file.is_absolute())
                        throw std::runtime_error("Batch input path must be relative");
                    for (const auto& part : file)
                        if (part == ".." || part == ".")
                            throw std::runtime_error("Unsafe batch input path");
                    const auto resolved = std::filesystem::weakly_canonical(root / file);
                    const auto relative = resolved.lexically_relative(std::filesystem::weakly_canonical(root));
                    if (relative.empty() || *relative.begin() == "..")
                        throw std::runtime_error("Batch input path escapes its directory");
                    if (!std::filesystem::is_regular_file(root / file))
                        throw std::runtime_error("Missing batch ciphertext");
                }
            }
            if (std::filesystem::exists(output_directory))
                throw std::runtime_error("Batch output directory must be new");
            std::filesystem::create_directories(output_directory);
            InferenceRunner runner(task, gpu, device);
            runner.load();
            json records = json::array();
            for (const auto& sample : samples) {
                const auto id = sample.at("case_id").get<std::string>();
                std::map<std::string, std::string> files;
                for (const auto& item : sample.at("inputs").items())
                    files.emplace(item.key(), (root / item.value().get<std::string>()).string());
                const auto start = std::chrono::steady_clock::now();
                auto result = runner.evaluate_inputs(files);
                if (result.size() != 1)
                    throw std::runtime_error("Batch result archive requires one output feature");
                const auto filename = id + ".result.bin";
                write_binary_file(std::filesystem::path(output_directory) / filename, result.begin()->second);
                records.push_back(
                    {{"case_id", id},
                     {"file", filename},
                     {"output_id", result.begin()->first},
                     {"duration_ms",
                      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count()}});
            }
            std::ofstream manifest(std::filesystem::path(output_directory) / "batch_manifest.json");
            manifest
                << json({{"schema", "latti-ai.runner-batch.v1"}, {"state", "committed"}, {"samples", records}}).dump(2)
                << "\n";
            if (!manifest)
                throw std::runtime_error("Cannot write batch manifest");
            return 0;
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
            result = runner.evaluate_inputs(inputs);
        } else
            throw std::runtime_error("Invalid deployment mode");
        write_named_bytes_bundle(output, result);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
