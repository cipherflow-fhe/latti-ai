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

#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "SCI/src/globals.h"
#include "fhe_ops_lib/utils.h"
#include "inference_task/inference_process_client.h"
#include "interface/inference_client.h"
#include "interface/inference_server.h"

#ifndef LATTIAI_SOURCE_DIR
#define LATTIAI_SOURCE_DIR "."
#endif

using namespace std;
using namespace lattisense;
using namespace fhe_ops_lib;

namespace {

using EncryptedBytesMap = map<string, Bytes>;
using PlaintextMap = map<string, vector<double>>;

void print_outputs(const map<string, DecryptedOutput>& results);
bool verify_outputs(const map<string, DecryptedOutput>& results,
                    const map<string, vector<double>>& plaintext_outputs,
                    double tolerance);

void print_usage(const char* argv0) {
    cerr << "Usage: " << argv0
         << " [--task-dir <path>] [--input [name=]<path>] [--port <port>] [--gpu] [--verify]\n"
         << "Defaults:\n"
         << "  --task-dir " << LATTIAI_SOURCE_DIR << "/data/cifar10/task\n"
         << "  --input    " << LATTIAI_SOURCE_DIR << "/examples/test_cifar10/task/client/img.csv\n"
         << "  --port     12309" << endl;
}

void init_mpc_party(int party_id, int port_in) {
    party = party_id;
    port = port_in;
    address = "127.0.0.1";
    num_threads = 1;
    bitlength = RING_MOD_BIT;
    StartComputation();
}

void send_string(sci::NetIO* io_in, const string& value) {
    size_t size = value.size();
    io_in->send_data(&size, sizeof(size));
    if (size > 0) {
        io_in->send_data(value.data(), size);
    }
}

string receive_string(sci::NetIO* io_in) {
    size_t size = 0;
    io_in->recv_data(&size, sizeof(size));
    string value(size, '\0');
    if (size > 0) {
        io_in->recv_data(value.data(), size);
    }
    return value;
}

void send_encrypted_map(DataTransmission& data_trans, const EncryptedBytesMap& values) {
    size_t size = values.size();
    data_trans.io_in->send_data(&size, sizeof(size));
    for (const auto& [name, bytes] : values) {
        cout << "[Transport] Sending ciphertext [" << name << "], bytes=" << bytes.size() << endl;
        send_string(data_trans.io_in, name);
        data_trans.send_bytes(bytes);
    }
    data_trans.io_in->flush();
}

EncryptedBytesMap receive_encrypted_map(DataTransmission& data_trans) {
    size_t size = 0;
    data_trans.io_in->recv_data(&size, sizeof(size));
    EncryptedBytesMap values;
    for (size_t i = 0; i < size; i++) {
        string name = receive_string(data_trans.io_in);
        values[name] = data_trans.receive_bytes();
        cout << "[Transport] Received ciphertext [" << name << "], bytes=" << values[name].size() << endl;
    }
    return values;
}

void send_plaintext_map(sci::NetIO* io_in, const PlaintextMap& values) {
    size_t size = values.size();
    io_in->send_data(&size, sizeof(size));
    for (const auto& [name, data] : values) {
        send_string(io_in, name);
        size_t data_size = data.size();
        io_in->send_data(&data_size, sizeof(data_size));
        if (data_size > 0) {
            io_in->send_data(data.data(), data_size * sizeof(double));
        }
    }
}

PlaintextMap receive_plaintext_map(sci::NetIO* io_in) {
    size_t size = 0;
    io_in->recv_data(&size, sizeof(size));
    PlaintextMap values;
    for (size_t i = 0; i < size; i++) {
        string name = receive_string(io_in);
        size_t data_size = 0;
        io_in->recv_data(&data_size, sizeof(data_size));
        vector<double> data(data_size);
        if (data_size > 0) {
            io_in->recv_data(data.data(), data_size * sizeof(double));
        }
        values[name] = move(data);
    }
    return values;
}

map<string, string> build_input_csvs(const string& task_dir, const vector<string>& input_args) {
    auto task_config = read_json(task_dir + "/client/task_config.json");
    vector<string> input_names;
    for (auto& [name, _] : task_config["task_input_param"].items()) {
        input_names.push_back(name);
    }

    map<string, string> input_csvs;
    for (size_t i = 0; i < input_args.size(); i++) {
        auto eq_pos = input_args[i].find('=');
        if (eq_pos != string::npos) {
            input_csvs[input_args[i].substr(0, eq_pos)] = input_args[i].substr(eq_pos + 1);
        } else if (i < input_names.size()) {
            input_csvs[input_names[i]] = input_args[i];
        } else {
            throw runtime_error("Too many --input arguments");
        }
    }
    return input_csvs;
}

map<string, unique_ptr<CkksContext>> make_client_context_map(const string& task_dir, const Bytes& full_context) {
    map<string, unique_ptr<CkksContext>> ckks_contexts;
    ckks_contexts["param0"] = make_unique<CkksContext>(CkksContext::deserialize_advanced(full_context));
    return ckks_contexts;
}

struct CkksClientRuntime {
    explicit CkksClientRuntime(const string& client_dir_in) : client_dir(client_dir_in) {
        task_config = read_json(client_dir + "/task_config.json");
        pack_style = task_config["pack_style"];

        for (auto& [name, param] : task_config["task_input_param"].items()) {
            InputParam ip;
            ip.dim = param["dim"];
            ip.level = param["level"];
            ip.channel = param["channel"];
            if (ip.dim == 2) {
                ip.height = param["shape"][0];
                ip.width = param["shape"][1];
            } else if (ip.dim == 1) {
                ip.length = param["shape"][0];
            } else if (ip.dim == 0) {
                ip.skip = param.value("skip", 1);
            }
            ip.pack_num = param.value("pack_num", 0);
            input_params[name] = ip;
        }

        for (auto& [name, param] : task_config["task_output_param"].items()) {
            OutputParam op;
            op.dim = param["dim"];
            op.channel = param["channel"];
            if (op.dim == 0) {
                op.skip = param["skip"];
            } else if (op.dim == 1) {
                op.length = param["shape"][0];
                if (param.contains("invalid_fill")) {
                    op.invalid_fill[0] = param["invalid_fill"][0];
                    if (param["invalid_fill"].size() > 1) {
                        op.invalid_fill[1] = param["invalid_fill"][1];
                    }
                }
            } else if (op.dim == 2) {
                op.height = param["shape"][0];
                op.width = param["shape"][1];
                if (param.contains("invalid_fill")) {
                    op.invalid_fill = {param["invalid_fill"][0], param["invalid_fill"][1]};
                }
            }
            output_params[name] = op;
        }

        auto ckks_config = read_json(client_dir + "/ckks_parameter.json");
        auto& first_param = task_config["task_input_param"].begin().value();
        string ckks_param_id = first_param["ckks_parameter_id"];
        auto& ckks_entry = ckks_config[ckks_param_id];
        poly_modulus_degree = ckks_entry["poly_modulus_degree"].get<int>();
        n_slots = poly_modulus_degree / 2;
    }

    void setup() {
        cout << "[Client] Generating regular CKKS context and keys..." << endl;
        cout << "[Client] Poly degree: N=" << poly_modulus_degree << endl;
        cout << "[Client] Creating CKKS parameters..." << endl;
        param = make_unique<CkksParameter>(CkksParameter::create_parameter(poly_modulus_degree));
        cout << "[Client] Creating CKKS parameters ok..." << endl;
        int max_level = param->get_max_level();
        cout << "[Client] Creating CKKS context at level " << max_level << "..." << endl;
        context = make_unique<CkksContext>(CkksContext::create_random_context(*param, max_level));
        cout << "[Client] Generating rotation keys at level " << max_level << "..." << endl;
        context->gen_rotation_keys(max_level);
        cout << "[Client] Done." << endl;
    }

    Bytes export_eval_context() const {
        auto pub_ctx = context->make_public_context(false, true, true);
        return pub_ctx.serialize_advanced();
    }

    Bytes export_full_context() const { return context->serialize_advanced(); }

    EncryptedBytesMap encrypt(const map<string, string>& input_csvs) const {
        EncryptedBytesMap result;
        double scale = context->get_parameter().get_default_scale();

        for (auto& [name, csv_path] : input_csvs) {
            auto it = input_params.find(name);
            if (it == input_params.end()) {
                throw runtime_error("[Client] Unknown input name: " + name);
            }
            const auto& param_in = it->second;

            cout << "[Client] Encrypting input '" << name << "' (dim=" << param_in.dim << ")..." << endl;
            if (param_in.dim == 0) {
                auto input_array = csv_to_array<1>(csv_path);
                Feature0DEncrypted input_ct(context.get(), param_in.level);
                uint32_t input_skip = n_slots / param_in.pack_num;
                input_ct.pack(input_array, false, scale, input_skip);
                result[name] = input_ct.serialize();
            } else if (param_in.dim == 1) {
                auto input_array = csv_to_array<2>(csv_path, {(uint64_t)param_in.channel, (uint64_t)param_in.length});
                uint32_t skip =
                    param_in.pack_num > 0 ? (uint32_t)(n_slots / (param_in.length * param_in.pack_num)) : 1;
                Feature1DEncrypted input_ct(context.get(), param_in.level, skip);
                if (pack_style == "ordinary") {
                    input_ct.pack(input_array, false, scale);
                } else {
                    input_ct.pack_multiplexed(input_array, false, scale);
                }
                result[name] = input_ct.serialize();
            } else {
                auto input_array = csv_to_array<3>(
                    csv_path, {(uint64_t)param_in.channel, (uint64_t)param_in.height, (uint64_t)param_in.width});
                Feature2DEncrypted input_ct(context.get(), param_in.level, Duo{1, 1});
                if (pack_style == "ordinary") {
                    input_ct.pack_multiple_channel(input_array, false, scale);
                } else if (param_in.height * param_in.width > n_slots) {
                    Duo block_shape = {task_config["block_shape"][0], task_config["block_shape"][1]};
                    Duo channel_packing_factor = {(uint32_t)(param_in.height / block_shape[0]),
                                                  (uint32_t)(param_in.width / block_shape[1])};
                    input_ct.pack_interleaved(input_array, block_shape, channel_packing_factor, false, scale);
                } else {
                    input_ct.pack_multiplexed(input_array, false, scale);
                }
                result[name] = input_ct.serialize();
            }
            cout << "[Client] Done." << endl;
        }

        return result;
    }

    map<string, DecryptedOutput> decrypt(const EncryptedBytesMap& encrypted_outputs) const {
        map<string, DecryptedOutput> results;
        for (auto& [name, bytes] : encrypted_outputs) {
            auto it = output_params.find(name);
            if (it == output_params.end()) {
                throw runtime_error("[Client] Unknown output name: " + name);
            }
            const auto& param_out = it->second;

            cout << "[Client] Decrypting output '" << name << "' (dim=" << param_out.dim << ")..." << endl;
            DecryptedOutput result;
            if (param_out.dim == 0) {
                Feature0DEncrypted output_ct(context.get(), 0);
                output_ct.deserialize(bytes);
                output_ct.skip = param_out.skip;
                auto decrypted = output_ct.unpack();
                auto dec_1d = decrypted.to_array_1d();
                result.output = vector<double>(dec_1d.data(), dec_1d.data() + dec_1d.size());
            } else if (param_out.dim == 1) {
                Feature1DEncrypted output_ct(context.get(), 0);
                output_ct.deserialize(bytes);
                Array<double, 2> decrypted;
                if (pack_style == "multiplexed") {
                    output_ct.invalid_fill = param_out.invalid_fill[0];
                    decrypted = output_ct.unpack_multiplexed();
                } else {
                    decrypted = output_ct.unpack();
                }
                auto dec_1d = decrypted.to_array_1d();
                result.output = vector<double>(dec_1d.data(), dec_1d.data() + dec_1d.size());
            } else {
                Feature2DEncrypted output_ct(context.get(), 0, Duo{1, 1});
                output_ct.deserialize(bytes);
                Array<double, 3> decrypted;
                if (pack_style == "multiplexed") {
                    Duo block_shape = {task_config["block_shape"][0], task_config["block_shape"][1]};
                    if (param_out.height * param_out.width > (int)(block_shape[0] * block_shape[1])) {
                        Duo stride = {(uint32_t)(param_out.height / block_shape[0]),
                                      (uint32_t)(param_out.width / block_shape[1])};
                        decrypted = output_ct.unpack_interleaved(block_shape, stride);
                    } else {
                        output_ct.invalid_fill = param_out.invalid_fill;
                        decrypted = output_ct.unpack_multiplexed();
                    }
                } else {
                    decrypted = output_ct.unpack_multiple_channel();
                }
                auto dec_1d = decrypted.to_array_1d();
                result.output = vector<double>(dec_1d.data(), dec_1d.data() + dec_1d.size());
            }
            result.num_outputs = result.output.size();
            results[name] = move(result);
            cout << "[Client] Done." << endl;
        }
        return results;
    }

    string client_dir;
    string pack_style;
    json task_config;
    map<string, InputParam> input_params;
    map<string, OutputParam> output_params;
    int poly_modulus_degree = 0;
    int n_slots = 0;
    unique_ptr<CkksParameter> param;
    unique_ptr<CkksContext> context;
};

void run_mpc_client_process(const string& task_dir,
                            const map<string, string>& input_csvs,
                            int mpc_port,
                            bool verify) {
    try {
        this_thread::sleep_for(chrono::milliseconds(200));
        cout << "[Client] Starting MPC channel..." << endl;
        init_mpc_party(CLIENT, mpc_port);
        cout << "[Client] MPC channel ready." << endl;

        CkksClientRuntime client(task_dir + "/client");
        client.setup();
        auto eval_ctx = client.export_eval_context();
        cout << "[Client] Exported eval context, bytes=" << eval_ctx.size() << endl;
        auto full_ctx = client.export_full_context();
        cout << "[Client] Exported full context, bytes=" << full_ctx.size() << endl;
        auto encrypted_inputs = client.encrypt(input_csvs);

        DataTransmission data_trans(io);
        cout << "[Client] Sending eval context..." << endl;
        data_trans.send_bytes(eval_ctx);
        data_trans.io_in->flush();
        cout << "[Client] Sending encrypted inputs..." << endl;
        send_encrypted_map(data_trans, encrypted_inputs);
        cout << "[Client] Initial payload sent; entering MPC process loop." << endl;

        auto ckks_contexts = make_client_context_map(task_dir, full_ctx);
        process(&ckks_contexts);
        cout << "[Client] MPC process loop finished; waiting encrypted outputs." << endl;

        auto encrypted_outputs = receive_encrypted_map(data_trans);
        auto results = client.decrypt(encrypted_outputs);
        print_outputs(results);

        if (verify) {
            auto plaintext_outputs = receive_plaintext_map(data_trans.io_in);
            for (auto& [name, plaintext_output] : plaintext_outputs) {
                fhe_ops_lib::print_double_message(plaintext_output.data(), ("Plaintext output [" + name + "]").c_str(),
                                                  1);
            }
            if (!verify_outputs(results, plaintext_outputs, 0.1)) {
                cout << "Result: FAIL" << endl;
                _exit(1);
            }
            cout << "Result: PASS" << endl;
        }
    } catch (const exception& e) {
        cerr << "[MPC Client] " << e.what() << endl;
        _exit(1);
    }
    _exit(0);
}

void print_outputs(const map<string, DecryptedOutput>& results) {
    cout << "\n========== Results ==========" << endl;
    for (auto& [name, result] : results) {
        fhe_ops_lib::print_double_message(result.output.data(), ("Encrypted output [" + name + "]").c_str(), 1);
    }
}

bool verify_outputs(const map<string, DecryptedOutput>& results,
                    const map<string, vector<double>>& plaintext_outputs,
                    double tolerance) {
    bool pass = true;
    for (auto& [name, result] : results) {
        auto pt_it = plaintext_outputs.find(name);
        if (pt_it == plaintext_outputs.end()) {
            continue;
        }

        const auto& plaintext_output = pt_it->second;
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

        cout << "\n========== Verification [" << name << "] ==========" << endl;
        cout << fixed << setprecision(8);
        cout << "Elements compared: " << count << endl;
        cout << "Max absolute error: " << max_abs_err << " (at index " << max_err_idx << ")" << endl;
        cout << "Avg absolute error: " << avg_abs_err << endl;
        cout << "Tolerance:          " << tolerance << endl;

        if (max_abs_err > tolerance) {
            pass = false;
        }
    }
    return pass;
}

}  // namespace

int main(int argc, char* argv[]) {
    string task_dir = string(LATTIAI_SOURCE_DIR) + "/data/cifar10/task";
    vector<string> input_args = {string(LATTIAI_SOURCE_DIR) + "/examples/test_cifar10/task/client/img.csv"};
    int mpc_port = 12309;
    bool use_gpu = false;
    bool verify = false;
    constexpr double tolerance = 0.1;
    pid_t client_pid = -1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gpu") == 0) {
            use_gpu = true;
        } else if (strcmp(argv[i], "--verify") == 0) {
            verify = true;
        } else if (strcmp(argv[i], "--task-dir") == 0 && i + 1 < argc) {
            task_dir = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            if (input_args.size() == 1 &&
                input_args[0] == string(LATTIAI_SOURCE_DIR) + "/examples/test_cifar10/task/client/img.csv") {
                input_args.clear();
            }
            input_args.push_back(argv[++i]);
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            mpc_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        auto input_csvs = build_input_csvs(task_dir, input_args);

        cout << "========== SDK MPC Encrypted Inference ==========" << endl;
        cout << "Task directory: " << task_dir << endl;
        for (auto& [name, path] : input_csvs) {
            cout << "Input [" << name << "]: " << path << endl;
        }
        cout << "MPC address:    127.0.0.1:" << mpc_port << endl;
        cout << "Device:         " << (use_gpu ? "GPU" : "CPU") << endl;
        cout << endl;

        client_pid = fork();
        if (client_pid < 0) {
            throw runtime_error("fork() failed");
        }
        if (client_pid == 0) {
            run_mpc_client_process(task_dir, input_csvs, mpc_port, verify);
        }

        cout << "[Step 1/4] Starting MPC server channel..." << endl;
        init_mpc_party(SERVER, mpc_port);
        cout << "[Server] MPC channel ready." << endl;

        DataTransmission data_trans(io);
        cout << "[Step 2/4] Receiving client context and encrypted inputs..." << endl;
        auto eval_ctx = data_trans.receive_bytes();
        cout << "[Server] Received eval context, bytes=" << eval_ctx.size() << endl;
        auto encrypted_inputs = receive_encrypted_map(data_trans);
        cout << "[Server] Received all encrypted inputs." << endl;

        cout << "[Step 3/4] Server loading model and importing context..." << endl;
        InferenceServer server(task_dir + "/server", use_gpu, 0);
        server.import_eval_context_ckks(eval_ctx);
        server.load_model();

        cout << "[Step 4/4] Running SDK MPC inference..." << endl;
        auto encrypted_outputs = server.evaluate_mpc_sdk(encrypted_inputs);
        cout << "[Server] Sending encrypted outputs..." << endl;
        send_encrypted_map(data_trans, encrypted_outputs);
        cout << "[Server] Encrypted outputs sent." << endl;

        if (verify) {
            auto plaintext_outputs = server.evaluate_plaintext(input_csvs);
            send_plaintext_map(data_trans.io_in, plaintext_outputs);
        }

        int status = 0;
        if (waitpid(client_pid, &status, 0) < 0) {
            throw runtime_error("waitpid() failed");
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            throw runtime_error("MPC client process failed");
        }
        client_pid = -1;
    } catch (const exception& e) {
        if (client_pid > 0) {
            kill(client_pid, SIGTERM);
            waitpid(client_pid, nullptr, 0);
        }
        cerr << "[Error] " << e.what() << endl;
        return 1;
    }

    return 0;
}
