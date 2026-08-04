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

#include "interface/runner_bundle_io.h"

#include <fstream>
#include <stdexcept>

#include "util/serial.h"

namespace {

const std::string kNamedBytesBundleMagic = "latti-ai.runner-output.v1";

}  // namespace

Bytes read_binary_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open binary file: " + path.string());
    }
    in.seekg(0, std::ios::end);
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size < 0) {
        throw std::runtime_error("Failed to determine binary file size: " + path.string());
    }
    Bytes bytes(static_cast<size_t>(size));
    if (size > 0) {
        in.read(reinterpret_cast<char*>(bytes.data()), size);
    }
    if (!in.good() && !in.eof()) {
        throw std::runtime_error("Failed to read binary file: " + path.string());
    }
    return bytes;
}

void write_binary_file(const std::filesystem::path& path, const Bytes& bytes) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open binary file for writing: " + path.string());
    }
    if (!bytes.empty()) {
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    if (!out.good()) {
        throw std::runtime_error("Failed to write binary file: " + path.string());
    }
}

void write_named_bytes_bundle(const std::filesystem::path& path, const std::map<std::string, Bytes>& values) {
    std::stringstream ss;
    ss_write_string(ss, kNamedBytesBundleMagic);
    uint64_t count = values.size();
    ss_write(ss, count);
    for (const auto& [name, bytes] : values) {
        ss_write_string(ss, name);
        ss_write_vector(ss, bytes);
    }
    write_binary_file(path, ss_to_bytes(ss));
}

std::map<std::string, Bytes> read_named_bytes_bundle(const std::filesystem::path& path) {
    Bytes bytes = read_binary_file(path);
    std::stringstream ss;
    bytes_to_ss(bytes, ss);

    std::string magic;
    ss_read_string(ss, &magic);
    if (magic != kNamedBytesBundleMagic) {
        throw std::runtime_error("Unsupported runner output bundle format: " + path.string());
    }

    uint64_t count = 0;
    ss_read(ss, &count);
    std::map<std::string, Bytes> values;
    for (uint64_t i = 0; i < count; ++i) {
        std::string name;
        Bytes value;
        ss_read_string(ss, &name);
        ss_read_vector(ss, &value);
        values.emplace(std::move(name), std::move(value));
    }
    return values;
}
