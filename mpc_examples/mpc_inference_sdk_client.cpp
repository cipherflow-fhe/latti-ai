/*
 * Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "mpc_adapter/inference_process_client.h"
#include "mpc_inference_sdk_common.h"

using namespace std;
using namespace lattisense;
using namespace fhe_ops_lib;

namespace {

void print_usage(const char* argv0) {
    cerr << "Usage: " << argv0 << " [--task-dir <path>] [--input [name=]<path>] [--port <port>] [--verify]\n"
         << "Defaults:\n"
         << "  --task-dir " << default_task_dir() << "\n"
         << "  --input    " << default_input_csv() << "\n"
         << "  --port     12309" << endl;
}

map<string, unique_ptr<CkksContext>> make_client_context_map(const Bytes& full_context) {
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
        cout << "[Client] Creating CKKS parameters ok." << endl;
        double default_scale = param->get_default_scale();
        cout << "[Client] Default scale: " << default_scale << " (bits=" << log2(default_scale) << ")" << endl;
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

}  // namespace

int main(int argc, char* argv[]) {
    string task_dir = default_task_dir();
    vector<string> input_args = {default_input_csv()};
    int mpc_port = 12309;
    bool verify = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verify") == 0) {
            verify = true;
        } else if (strcmp(argv[i], "--task-dir") == 0 && i + 1 < argc) {
            task_dir = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            if (input_args.size() == 1 && input_args[0] == default_input_csv()) {
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

        cout << "========== SDK MPC Client ==========" << endl;
        cout << "Task directory: " << task_dir << endl;
        for (auto& [name, path] : input_csvs) {
            cout << "Input [" << name << "]: " << path << endl;
        }
        cout << "MPC address:    127.0.0.1:" << mpc_port << endl;
        cout << endl;

        CkksClientRuntime client(task_dir + "/client");
        client.setup();
        auto encrypted_inputs = client.encrypt(input_csvs);

        cout << "[Client] Starting MPC channel..." << endl;
        init_mpc_party(MPC_CLIENT, mpc_port);
        cout << "[Client] MPC channel ready." << endl;

        MpcDataTransmission mpc_trans = MpcDataTransmission::current();
        bool server_needs_full_context = mpc_trans.receive_dump_flag();
        auto server_ctx = server_needs_full_context ? client.export_full_context() : client.export_eval_context();
        cout << "[Client] Sending " << (server_needs_full_context ? "full" : "public")
             << " context, bytes=" << server_ctx.size() << endl;
        mpc_trans.send_context_bytes(server_ctx);
        cout << "[Client] Sending encrypted inputs..." << endl;
        mpc_trans.send_encrypted_map(encrypted_inputs);
        cout << "[Client] Initial payload sent; entering MPC process loop." << endl;

        auto full_ctx = client.export_full_context();
        auto ckks_contexts = make_client_context_map(full_ctx);
        InferenceMpcClient(ckks_contexts).run();
        cout << "[Client] MPC process loop finished; waiting encrypted outputs." << endl;

        auto encrypted_outputs = mpc_trans.receive_encrypted_map();
        auto results = client.decrypt(encrypted_outputs);
        print_outputs(results);

        if (verify) {
            auto plaintext_outputs = mpc_trans.receive_plaintext_map();
            for (auto& [name, plaintext_output] : plaintext_outputs) {
                fhe_ops_lib::print_double_message(plaintext_output.data(), ("Plaintext output [" + name + "]").c_str(),
                                                  1);
            }
            if (!verify_outputs(results, plaintext_outputs, 0.1)) {
                cout << "Result: FAIL" << endl;
                return 1;
            }
            cout << "Result: PASS" << endl;
        }
    } catch (const exception& e) {
        cerr << "[Client][Error] " << e.what() << endl;
        return 1;
    }

    return 0;
}
