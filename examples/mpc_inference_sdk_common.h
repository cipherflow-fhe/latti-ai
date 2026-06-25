#pragma once

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "fhe_mpc.h"
#include "SCI/src/globals.h"
#include "fhe_ops_lib/utils.h"
#include "interface/inference_client.h"

#ifndef LATTIAI_SOURCE_DIR
#define LATTIAI_SOURCE_DIR "."
#endif

using EncryptedBytesMap = std::map<std::string, Bytes>;
using PlaintextMap = std::map<std::string, std::vector<double>>;

inline std::string default_task_dir() {
    return std::string(LATTIAI_SOURCE_DIR) + "/data/cifar10/task";
}

inline std::string default_input_csv() {
    return std::string(LATTIAI_SOURCE_DIR) + "/examples/test_cifar10/task/client/img.csv";
}

inline void init_mpc_party(int party_id, int port_in) {
    party = party_id;
    port = port_in;
    address = "127.0.0.1";
    num_threads = 1;
    bitlength = RING_MOD_BIT;
    StartComputation();
}

inline void send_string(DataTransmission& data_trans, const std::string& value) {
    size_t size = value.size();
    data_trans.send_data(&size, sizeof(size));
    if (size > 0) {
        data_trans.send_data(value.data(), size);
    }
}

inline std::string receive_string(DataTransmission& data_trans) {
    size_t size = 0;
    data_trans.recv_data(&size, sizeof(size));
    std::string value(size, '\0');
    if (size > 0) {
        data_trans.recv_data(value.data(), size);
    }
    return value;
}

inline void send_encrypted_map(DataTransmission& data_trans, const EncryptedBytesMap& values) {
    size_t size = values.size();
    data_trans.send_data(&size, sizeof(size));
    for (const auto& [name, bytes] : values) {
        std::cout << "[Transport] Sending ciphertext [" << name << "], bytes=" << bytes.size() << std::endl;
        send_string(data_trans, name);
        data_trans.send_bytes(bytes);
    }
    data_trans.flush();
}

inline EncryptedBytesMap receive_encrypted_map(DataTransmission& data_trans) {
    size_t size = 0;
    data_trans.recv_data(&size, sizeof(size));
    EncryptedBytesMap values;
    for (size_t i = 0; i < size; i++) {
        std::string name = receive_string(data_trans);
        values[name] = data_trans.receive_bytes();
        std::cout << "[Transport] Received ciphertext [" << name << "], bytes=" << values[name].size() << std::endl;
    }
    return values;
}

inline void send_plaintext_map(DataTransmission& data_trans, const PlaintextMap& values) {
    size_t size = values.size();
    data_trans.send_data(&size, sizeof(size));
    for (const auto& [name, data] : values) {
        send_string(data_trans, name);
        size_t data_size = data.size();
        data_trans.send_data(&data_size, sizeof(data_size));
        if (data_size > 0) {
            data_trans.send_data(data.data(), data_size * sizeof(double));
        }
    }
    data_trans.flush();
}

inline PlaintextMap receive_plaintext_map(DataTransmission& data_trans) {
    size_t size = 0;
    data_trans.recv_data(&size, sizeof(size));
    PlaintextMap values;
    for (size_t i = 0; i < size; i++) {
        std::string name = receive_string(data_trans);
        size_t data_size = 0;
        data_trans.recv_data(&data_size, sizeof(data_size));
        std::vector<double> data(data_size);
        if (data_size > 0) {
            data_trans.recv_data(data.data(), data_size * sizeof(double));
        }
        values[name] = std::move(data);
    }
    return values;
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
