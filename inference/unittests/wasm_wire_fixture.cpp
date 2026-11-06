// Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "interface/inference_client.h"

namespace fs = std::filesystem;
static Bytes read_bytes(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot read fixture: " + path.string());
    return Bytes((std::istreambuf_iterator<char>(file)), {});
}

// Called by the Go acceptance test with current v6 task/keys/ciphertext files.
// Native reads the Go payload unchanged, checks plaintext, and encrypts a fresh
// payload for the reverse direction under the same restored client key pair.
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: wasm_wire_fixture fixture-directory");
        const fs::path root(argv[1]);
        InferenceClient client(root.string());
        client.load_full_context(read_bytes(root / "full.bin"));
        const auto result = client.decrypt({{"output", read_bytes(root / "go.ct")}}).at("output").output;
        std::ifstream expected_file(root / "output-values.json");
        json expected;
        expected_file >> expected;
        const auto values = expected.get<std::vector<double>>();
        if (result.size() != values.size()) throw std::runtime_error("native output count differs");
        for (size_t i = 0; i < values.size(); ++i) {
            if (!std::isfinite(result[i]) || std::abs(result[i] - values[i]) > 0.001)
                throw std::runtime_error("native value differs at " + std::to_string(i));
        }
        const auto ciphertext = client.encrypt({{"input", (root / "values.csv").string()}}).at("input");
        std::ofstream output(root / "native.ct", std::ios::binary);
        output.write(reinterpret_cast<const char*>(ciphertext.data()), ciphertext.size());
        if (!output) throw std::runtime_error("native fixture write failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
