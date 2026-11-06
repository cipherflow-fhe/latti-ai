// Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#ifdef __linux__
#include <atomic>
#include <cerrno>
#include <cstring>
#include <dlfcn.h>
#include <csignal>
#include <sys/resource.h>
#endif
#include "fhe_api.h"
#include "interface/inference_client.h"
#include "interface/inference_server.h"
#include "util/serial.h"

using namespace fhe_ops_lib;
namespace fs = std::filesystem;

#ifdef __linux__
namespace {
std::atomic<const char*> fail_publication{nullptr};
}
// Test-local filesystem fault injection; native production publication is unchanged.
extern "C" int link(const char* source, const char* destination) {
    const char* fault = fail_publication.load();
    if (fault && fs::path(destination).filename() == fault &&
        fail_publication.compare_exchange_strong(fault, nullptr)) {
        errno = EIO;
        return -1;
    }
    const auto original = reinterpret_cast<int (*)(const char*, const char*)>(dlsym(RTLD_NEXT, "link"));
    return original(source, destination);
}
#endif

TEST_CASE("streamed contexts use v6 files and preserve client keys", "[streaming_context]") {
    const auto root = fs::temp_directory_path() / ("latti-stream-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    struct Cleanup { fs::path path; ~Cleanup() { fs::remove_all(path); } } cleanup{root};
    for (const auto ring : {RingType::Standard, RingType::ConjugateInvariant}) {
        const auto param = CkksParameter::create_parameter(12, ring);
        auto client = CkksContext::create_client_context(param);
        const auto eval_path = root / (ring == RingType::Standard ? "eval-std" : "eval-ci");
        const auto full_path = root / (ring == RingType::Standard ? "full-std" : "full-ci");
        client.export_eval_context_to_file(eval_path.string());
        client.export_full_context_to_file(full_path.string());
        REQUIRE_THROWS(client.export_eval_context_to_file(eval_path.string()));
        REQUIRE_THROWS(client.export_full_context_to_file(full_path.string()));
        REQUIRE_THROWS(client.export_full_context_to_file((root / "missing" / "full").string()));

        std::ifstream eval_file(eval_path, std::ios::binary);
        Bytes eval_bytes((std::istreambuf_iterator<char>(eval_file)), {});
        auto server = CkksContext::deserialize(eval_bytes);
        std::ifstream full_file(full_path, std::ios::binary);
        Bytes full_bytes((std::istreambuf_iterator<char>(full_file)), {});
        const json task = {{"pack_style", "ordinary"}, {"use_btp", false},
            {"task_input_param", {{"input", {{"dim", 0}, {"level", 1}, {"pack_num", 1},
                                              {"ckks_parameter_id", "param0"}}}}},
            {"task_output_param", {{"output", {{"dim", 0}}}}}};
        const json params = {{"param0", {{"poly_modulus_degree", param.n()},
            {"ring_type", static_cast<int>(ring)}, {"log_default_scale", param.log_default_scale()},
            {"q", param.q()}, {"p", param.p()}}}};
        auto multi_task = task;
        multi_task["task_input_param"]["other"] = multi_task["task_input_param"]["input"];
        multi_task["task_input_param"]["other"]["ckks_parameter_id"] = "param1";
        auto multi_params = params;
        multi_params["param1"] = params["param0"];
        std::ofstream(root / "task_config.json") << multi_task;
        std::ofstream(root / "ckks_parameter.json") << multi_params;
        REQUIRE_NOTHROW(InferenceClient(root.string()));
        multi_params["param1"]["log_default_scale"] = param.log_default_scale() - 1;
        std::ofstream(root / "ckks_parameter.json") << multi_params;
        REQUIRE_THROWS(InferenceClient(root.string()));
        multi_task["task_input_param"].erase("other");
        multi_task["task_output_param"]["output"]["ckks_parameter_id"] = "param1";
        std::ofstream(root / "task_config.json") << multi_task;
        REQUIRE_THROWS(InferenceClient(root.string()));
        std::ofstream(root / "task_config.json") << task;
        std::ofstream(root / "ckks_parameter.json") << params;
        InferenceClient app_client(root.string());
        app_client.load_full_context(full_bytes);
        REQUIRE(app_client.export_full_context() == full_bytes);
        Bytes invalid = full_bytes;
        std::fill(invalid.begin(), invalid.begin() + 8, 0xff);
        REQUIRE_THROWS(app_client.load_full_context(invalid));
        REQUIRE_THROWS(app_client.load_full_context(Bytes{}));
        invalid = full_bytes;
        invalid.push_back(0);
        REQUIRE_THROWS(app_client.load_full_context(invalid));
        REQUIRE(app_client.export_full_context() == full_bytes);
        auto other_params = params;
        other_params["param0"]["log_default_scale"] = param.log_default_scale() - 1;
        std::ofstream(root / "ckks_parameter.json") << other_params;
        InferenceClient mismatched(root.string());
        REQUIRE_THROWS(mismatched.load_full_context(full_bytes));
        std::ofstream(root / "ckks_parameter.json") << params;
        InferenceClient setup_client(root.string());
        REQUIRE_NOTHROW(setup_client.setup());
        REQUIRE(fs::exists(root / "key_manifest.json"));
        REQUIRE_THROWS(setup_client.export_eval_context());
        const auto setup_full = setup_client.export_full_context();
        setup_client.release();
        REQUIRE_NOTHROW(setup_client.setup());
        REQUIRE(setup_client.export_full_context() == setup_full);
        REQUIRE_NOTHROW(InferenceServer(root.string()).import_eval_context_file((root / "eval_context.bin").string()));
        fs::remove(root / "key_manifest.json");
        REQUIRE_THROWS(setup_client.setup());
        REQUIRE(fs::exists(root / "full_context.bin"));
        fs::remove(root / "eval_context.bin");
        fs::remove(root / "full_context.bin");
#ifdef __linux__
        for (const char* fault : {"full_context.bin", "key_manifest.json"}) {
            INFO(fault);
            fail_publication = fault;
            REQUIRE_THROWS(setup_client.setup());
            REQUIRE(fail_publication.load() == nullptr);
            REQUIRE_FALSE(fs::exists(root / "eval_context.bin"));
            REQUIRE_FALSE(fs::exists(root / "full_context.bin"));
            REQUIRE_FALSE(fs::exists(root / "key_manifest.json"));
            REQUIRE_FALSE(fs::exists(root / ".key-setup.lock"));
            for (const auto& entry : fs::directory_iterator(root))
                REQUIRE(entry.path().filename().string().find(".key-setup-") != 0);
        }
        // Kernel file-size limit forces a real output write/flush failure.
        struct FileLimit {
            rlimit original;
            decltype(std::signal(SIGXFSZ, SIG_IGN)) handler;
            FileLimit() {
                getrlimit(RLIMIT_FSIZE, &original);
                handler = std::signal(SIGXFSZ, SIG_IGN);
                auto limited = original;
                limited.rlim_cur = 4096;
                if (setrlimit(RLIMIT_FSIZE, &limited) != 0) throw std::runtime_error("setrlimit failed");
            }
            ~FileLimit() { setrlimit(RLIMIT_FSIZE, &original); std::signal(SIGXFSZ, handler); }
        };
        {
            FileLimit limit;
            REQUIRE_THROWS(setup_client.setup());
        }
        REQUIRE_FALSE(fs::exists(root / "eval_context.bin"));
        REQUIRE_FALSE(fs::exists(root / "full_context.bin"));
        REQUIRE_FALSE(fs::exists(root / "key_manifest.json"));
        REQUIRE_NOTHROW(setup_client.setup());
        fs::remove(root / "eval_context.bin");
        fs::remove(root / "full_context.bin");
        fs::remove(root / "key_manifest.json");
#endif
        InferenceServer app_server(root.string());
        REQUIRE_NOTHROW(app_server.import_eval_context_file(eval_path.string()));
        REQUIRE_THROWS(app_server.import_eval_context_file((root / "missing").string()));
        std::stringstream full;
        bytes_to_ss(full_bytes, full);
        Bytes context_bytes, secret_bytes;
        ss_read_vector(full, &context_bytes);
        ss_read_vector(full, &secret_bytes);
        REQUIRE(full.peek() == std::char_traits<char>::eof());
        auto restored = CkksContext::deserialize(context_bytes);
        restored.set_secret_key(SecretKey::deserialize(secret_bytes));
        REQUIRE(restored.public_key().serialize() == client.public_key().serialize());

        std::vector<double> input(param.max_slots(), 0.25);
        input[1] = 0.5;
        auto ciphertext = client.encrypt(fhe_api::encode(client, input, 1, param.default_scale()));
        auto rotated = server.rotate(ciphertext, 1);
        std::vector<double> decoded;
        restored.decode(restored.decrypt(rotated), decoded);
        REQUIRE(decoded[0] == Approx(0.5).margin(1e-5));
        REQUIRE(decoded[1] == Approx(0.25).margin(1e-5));
    }
}
