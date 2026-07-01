#pragma once

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "fhe_ops_lib/utils.h"
#include "interface/inference_client.h"
#include "mpc_wrapper/mpc_wrapper.h"
#include "mpc_wrapper/mpc_data_transmission.h"

#ifndef LATTIAI_SOURCE_DIR
#define LATTIAI_SOURCE_DIR "."
#endif

inline std::string default_task_dir() {
    return std::string(LATTIAI_SOURCE_DIR) + "/data/cifar10/task";
}

inline std::string default_input_csv() {
    return std::string(LATTIAI_SOURCE_DIR) + "/examples/test_cifar10/task/client/img.csv";
}

inline std::map<std::string, std::string> build_input_csvs(const std::string& task_dir,
                                                           const std::vector<std::string>& input_args) {
    auto task_config = read_json(task_dir + "/client/task_config.json");
    std::vector<std::string> input_names;
    for (auto& [name, _] : task_config["task_input_param"].items()) {
        input_names.push_back(name);
    }

    std::map<std::string, std::string> input_csvs;
    for (size_t i = 0; i < input_args.size(); i++) {
        auto eq_pos = input_args[i].find('=');
        if (eq_pos != std::string::npos) {
            input_csvs[input_args[i].substr(0, eq_pos)] = input_args[i].substr(eq_pos + 1);
        } else if (i < input_names.size()) {
            input_csvs[input_names[i]] = input_args[i];
        } else {
            throw std::runtime_error("Too many --input arguments");
        }
    }
    return input_csvs;
}

inline void print_outputs(const std::map<std::string, DecryptedOutput>& results) {
    std::cout << "\n========== Results ==========" << std::endl;
    for (auto& [name, result] : results) {
        fhe_ops_lib::print_double_message(result.output.data(), ("Encrypted output [" + name + "]").c_str(), 1);
    }
}

inline bool verify_outputs(const std::map<std::string, DecryptedOutput>& results,
                           const PlaintextMap& plaintext_outputs,
                           double tolerance) {
    bool pass = true;
    for (auto& [name, result] : results) {
        auto pt_it = plaintext_outputs.find(name);
        if (pt_it == plaintext_outputs.end()) {
            continue;
        }

        const auto& plaintext_output = pt_it->second;
        int count = std::min(result.num_outputs, (int)plaintext_output.size());
        double max_abs_err = 0.0;
        double sum_abs_err = 0.0;
        int max_err_idx = 0;
        for (int i = 0; i < count; i++) {
            double abs_err = std::fabs(result.output[i] - plaintext_output[i]);
            sum_abs_err += abs_err;
            if (abs_err > max_abs_err) {
                max_abs_err = abs_err;
                max_err_idx = i;
            }
        }
        double avg_abs_err = count > 0 ? sum_abs_err / count : 0.0;

        std::cout << "\n========== Verification [" << name << "] ==========" << std::endl;
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "Elements compared: " << count << std::endl;
        std::cout << "Max absolute error: " << max_abs_err << " (at index " << max_err_idx << ")" << std::endl;
        std::cout << "Avg absolute error: " << avg_abs_err << std::endl;
        std::cout << "Tolerance:          " << tolerance << std::endl;

        if (max_abs_err > tolerance) {
            pass = false;
        }
    }
    return pass;
}
