/*
 * Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "interface/inference_server.h"
#include "mpc_inference_sdk_common.h"

using namespace std;

namespace {

bool g_dump_intermediate_plaintexts = false;

bool is_enabled_env(const char* value) {
    return value != nullptr && strcmp(value, "0") != 0 && strcmp(value, "false") != 0 && strcmp(value, "FALSE") != 0;
}

void print_usage(const char* argv0) {
    cerr << "Usage: " << argv0
         << " [--task-dir <path>] [--input [name=]<path>] [--port <port>] [--gpu] [--verify]"
            " [--dump-intermediates] [--no-dump-intermediates]"
            " [--dump-layers <txt>] [--dump-plaintext <txt>]\n"
         << "Defaults:\n"
         << "  --task-dir " << default_task_dir() << "\n"
         << "  --input    " << default_input_csv() << "\n"
         << "  --port     12309\n"
         << "  dump intermediates disabled unless --dump-intermediates or LATTI_MPC_DUMP_INTERMEDIATES is set\n"
         << "  --dump-layers mpc_layer_dump.txt\n"
         << "  --dump-plaintext mpc_plaintext_dump.txt" << endl;
}

}  // namespace

int main(int argc, char* argv[]) {
    string task_dir = default_task_dir();
    vector<string> input_args = {default_input_csv()};
    int mpc_port = 12309;
    bool use_gpu = false;
    bool verify = false;
    string dump_layers_path = "mpc_layer_dump.txt";
    string dump_plaintext_path = "mpc_plaintext_dump.txt";
    g_dump_intermediate_plaintexts = is_enabled_env(getenv("LATTI_MPC_DUMP_INTERMEDIATES"));

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gpu") == 0) {
            use_gpu = true;
        } else if (strcmp(argv[i], "--verify") == 0) {
            verify = true;
        } else if (strcmp(argv[i], "--dump-intermediates") == 0) {
            g_dump_intermediate_plaintexts = true;
        } else if (strcmp(argv[i], "--no-dump-intermediates") == 0) {
            g_dump_intermediate_plaintexts = false;
        } else if (strcmp(argv[i], "--task-dir") == 0 && i + 1 < argc) {
            task_dir = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            if (input_args.size() == 1 && input_args[0] == default_input_csv()) {
                input_args.clear();
            }
            input_args.push_back(argv[++i]);
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            mpc_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dump-layers") == 0 && i + 1 < argc) {
            dump_layers_path = argv[++i];
            g_dump_intermediate_plaintexts = true;
        } else if (strcmp(argv[i], "--dump-plaintext") == 0 && i + 1 < argc) {
            dump_plaintext_path = argv[++i];
            g_dump_intermediate_plaintexts = true;
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

        cout << "========== SDK MPC Server ==========" << endl;
        cout << "Task directory: " << task_dir << endl;
        cout << "MPC address:    127.0.0.1:" << mpc_port << endl;
        cout << "Device:         " << (use_gpu ? "GPU" : "CPU") << endl;
        cout << "Layer dumps:    " << (g_dump_intermediate_plaintexts ? "enabled" : "disabled") << endl;
        cout << endl;

        cout << "[Server] Starting MPC channel..." << endl;
        init_mpc_party(MPC_SERVER, mpc_port);
        cout << "[Server] MPC channel ready." << endl;

        MpcDataTransmission mpc_trans = MpcDataTransmission::current();
        mpc_trans.send_dump_flag(g_dump_intermediate_plaintexts);

        cout << "[Server] Receiving client " << (g_dump_intermediate_plaintexts ? "full" : "public")
             << " context and encrypted inputs..." << endl;
        auto eval_ctx = mpc_trans.receive_context_bytes();
        cout << "[Server] Received " << (g_dump_intermediate_plaintexts ? "full" : "public")
             << " context, bytes=" << eval_ctx.size() << endl;
        auto encrypted_inputs = mpc_trans.receive_encrypted_map();
        cout << "[Server] Received all encrypted inputs." << endl;

        cout << "[Server] Loading model and importing context..." << endl;
        InferenceServer server(task_dir + "/server", use_gpu, 0);
        server.import_eval_context_ckks(eval_ctx);
        server.load_model_for_mpc_sdk();

        cout << "[Server] Running SDK MPC inference..." << endl;
        // EndComputation();
        auto encrypted_outputs = server.evaluate_mpc_sdk(encrypted_inputs);
        // EndComputation();
        cout << "[Server] Pure MPC refresh time: " << server.get_last_mpc_time_ms() << " ms" << endl;
        if (g_dump_intermediate_plaintexts) {
            server.dump_intermediate_plaintexts(dump_layers_path);
            server.dump_plaintext_intermediates(input_csvs, dump_plaintext_path);
        }

        cout << "[Server] Sending encrypted outputs..." << endl;
        mpc_trans.send_encrypted_map(encrypted_outputs);
        cout << "[Server] Encrypted outputs sent." << endl;

        if (verify) {
            auto plaintext_outputs = server.evaluate_plaintext(input_csvs);
            mpc_trans.send_plaintext_map(plaintext_outputs);
        }
    } catch (const exception& e) {
        cerr << "[Server][Error] " << e.what() << endl;
        return 1;
    }

    return 0;
}
