#include <cxx_sdk_v2/cxx_fhe_task.h>
#include "fhe_api.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sys/resource.h>
#include <filesystem>
using namespace lattisense;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
json read(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);
    json j; f >> j; return j;
}
void write_bytes(const std::string& path, const Bytes& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Usage: ckks_step_gpu TASK_DIR");
        const std::string dir = argv[1];
        auto m = read(dir + "/experiment.json");
        auto data = read(dir + "/data.json");
        auto begin = Clock::now();
        auto param = CkksParameter::create_parameter(m.at("log_n").get<int>());
        auto ctx = CkksContext::create_random_context(param, EncryptorType::PUBLIC_KEY,
                                                    m.value("bootstrapping_count", 0) > 0);
        std::vector<int32_t> rotations;
        for (int i = 0; i < param.log_max_slots(); ++i) rotations.push_back(1 << i);
        ctx.gen_rotation_keys(rotations);
        auto keyed = Clock::now();
        // Persist representative serialized artifacts for storage accounting.
        // Secret material is measured separately and is never sent to the GPU task.
        write_bytes(dir + "/storage_context.bin", ctx.serialize());
        write_bytes(dir + "/storage_public_key.bin", ctx.public_key().serialize());
        write_bytes(dir + "/storage_secret_key.bin", ctx.secret_key().serialize());
        if (m.value("bootstrapping_count", 0) > 0)
            write_bytes(dir + "/storage_bootstrap_keys.bin", ctx.bootstrapping_evaluation_keys().serialize());
        std::map<std::string, CkksCiphertext> ciphertexts;
        std::map<std::string, CkksPlaintext> plaintexts;
        for (auto it = m["inputs"].begin(); it != m["inputs"].end(); ++it) {
            std::string name = it.key();
            auto values = data.at(name).get<std::vector<double>>();
            int level = it.value()["level"];
            double scale = it.value()["scale"];
            if (name.find("coefficient") != std::string::npos) {
                values.resize(param.max_slots(), values.at(0));
                plaintexts.emplace(name, fhe_api::encode_ringt(ctx, values, scale));
            } else {
                // Weights are scalar replicated over all slots, features are zero-padded.
                double fill = name[0] == 'w' ? values.at(0) : 0.0;
                values.resize(param.max_slots(), fill);
                auto pt = fhe_api::encode(ctx, values, level, scale);
                ciphertexts.emplace(name, ctx.encrypt(pt));
                write_bytes(dir + "/storage_ciphertext_" + name + ".bin", ciphertexts.at(name).serialize());
            }
        }
        for (auto out : m["outputs"]) {
            auto ct = CkksCiphertext(param, out["level"].get<int>());
            ciphertexts.emplace(out["name"].get<std::string>(), std::move(ct));
        }
        std::vector<CxxVectorArgument> args;
        auto signature = read(dir + "/task_signature.json");
        for (auto arg : signature.at("online")) {
            std::string name = arg.at("id");
            if (ciphertexts.count(name)) args.push_back({name, &ciphertexts.at(name)});
            else args.push_back({name, &plaintexts.at(name)});
        }
        auto encrypted = Clock::now();
        FheTaskGpu task(dir);
        task.run(&ctx, args, nullptr, 0);
        auto evaluated = Clock::now();
        json results;
        for (auto out : m["outputs"]) {
            std::string name = out["name"];
            auto pt = ctx.decrypt(ciphertexts.at(name));
            std::vector<double> values; ctx.decode(pt, values);
            if (name.rfind("updated_w", 0) == 0) results["weights"].push_back(values.at(0));
            else results["checkpoint_weights"][name] = values.at(0);
        }
        auto end = Clock::now();
        auto seconds = [](auto a, auto b) { return std::chrono::duration<double>(b-a).count(); };
        results["keygen_seconds"] = seconds(begin, keyed);
        results["encode_encrypt_seconds"] = seconds(keyed, encrypted);
        results["gpu_task_wall_seconds"] = seconds(encrypted, evaluated);
        results["decrypt_seconds"] = seconds(evaluated, end);
        results["total_seconds"] = seconds(begin, end);
        struct rusage usage{}; getrusage(RUSAGE_SELF, &usage);
        results["host_peak_rss_bytes"] = usage.ru_maxrss * 1024L;
        std::ofstream(dir + "/result.json") << results.dump(2);
        std::cout << results.dump(2) << std::endl;
    } catch (const std::exception& e) { std::cerr << e.what() << std::endl; return 1; }
}
