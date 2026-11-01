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
#include <cmath>
#include <stdexcept>

#include "util/serial.h"

namespace {

const std::string kNamedBytesBundleMagic = "latti-ai.runner-output.v1";

}  // namespace

json parameter_layout_metadata(const json& graph, const std::string& layer_id, const json& signature) {
    const auto& layer = graph.at("layer").at(layer_id);
    const auto id = signature.at("id").get<std::string>();
    if (id.rfind("dense", 0) == 0 && layer.contains("dense_seed_tiles"))
        return layer.at("dense_seed_tiles");
    if (id.rfind("poly_reluw_", 0) == 0 && layer.contains("poly_seed_layout"))
        return layer.at("poly_seed_layout");
    if (id.rfind("convw_", 0) != 0 || !layer.contains("seed_weight_subgrid_stride")) return nullptr;
    return {{"kind", layer.value("seed_weight_layout", std::string("kernel"))},
            {"stride", layer.at("seed_weight_subgrid_stride")},
            {"group_size", layer.at("seed_weight_subgrid_stride").at(0).get<uint64_t>() *
                           layer.at("seed_weight_subgrid_stride").at(1).get<uint64_t>()},
            {"kernel_shape", layer.at("kernel_shape")}, {"storage_shape", signature.at("size")},
            {"input_geometry", graph.at("feature").at(layer.at("feature_input").at(0).get<std::string>())},
            {"output_geometry", graph.at("feature").at(layer.at("feature_output").at(0).get<std::string>())}};
}

void validate_parameter_manifest(const json& manifest, const json& signature, const json& graph) {
    const auto& parameters = signature.contains("encrypted_parameters") ? signature.at("encrypted_parameters") : signature.at("offline");
    size_t expected_count = 0;
    for (const auto& sig : parameters) {
        if (sig.at("type") != "ct") continue;
        ++expected_count;
        const auto id = sig.at("id").get<std::string>();
        const auto& arg = manifest.at("arguments").at(id);
        for (const auto* key : {"size", "level", "type"}) {
            if (arg.at(key) != sig.at(key)) throw std::runtime_error("Parameter manifest/signature mismatch: " + id + " " + key);
        }
        // Layouts are JSON objects: insertion order is not part of their contract.
        if (arg.contains("layout") && nlohmann::json(arg.at("layout")) !=
            nlohmann::json(parameter_layout_metadata(graph, arg.at("layer_id"), sig)))
            throw std::runtime_error("Parameter manifest/graph layout mismatch: " + id);
        if (sig.contains("layout") && (!arg.contains("layout") ||
            nlohmann::json(sig.at("layout")) != nlohmann::json(arg.at("layout"))))
            throw std::runtime_error("Parameter manifest/signature layout mismatch: " + id);
        if (sig.contains("scale") && !arg.contains("scale"))
            throw std::runtime_error("Missing parameter scale: " + id);
        if (arg.contains("scale")) {
            const double scale = arg.at("scale").get<double>();
            if (!std::isfinite(scale) || scale <= 0) throw std::runtime_error("Invalid parameter scale: " + id);
            if (sig.contains("scale") && std::abs(scale / sig.at("scale").get<double>() - 1.0) > 1e-10)
                throw std::runtime_error("Parameter manifest/signature scale mismatch: " + id);
        }
    }
    if (manifest.at("arguments").size() != expected_count)
        throw std::runtime_error("Parameter manifest has unexpected arguments");
}

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
