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
#include <set>

#include "util/serial.h"

namespace {

const std::string kNamedBytesBundleMagic = "latti-ai.runner-output.v1";

}  // namespace

json parameter_layout_metadata(const json& graph, const std::string& layer_id, const json& signature) {
    const auto& layer = graph.at("layer").at(layer_id);
    const auto id = signature.at("id").get<std::string>();
    if (signature.contains("generator") &&
        signature.at("generator").value("type", std::string()) == "row_chunk_seed") {
        const auto& layout = layer.at("pcm_seed_layout");
        const auto& feature = graph.at("feature").at(layer.at("feature_input")[0].get<std::string>());
        const auto d = layout.at("d").get<uint64_t>();
        const auto heads = layout.at("H").get<uint64_t>();
        const auto slots = layout.at("C").get<uint64_t>() * layout.at("rho").get<uint64_t>();
        if (!d || (d & (d - 1)) || !heads || slots < d*d ||
            layout.at("weight_shape") != layer.at("weight_shape") ||
            layout.at("head_shape") != feature.at("head_shape") ||
            feature.at("head_shape")[1] != d)
            throw std::runtime_error("Invalid PCM parameter seed geometry: " + layer_id);
        uint64_t padded = 1;
        while (padded < heads) padded *= 2;
        const auto lanes = std::min(padded, slots/(d*d));
        if (lanes == 1) padded = heads;
        const auto chunk = std::min(slots, padded*d*d);
        const auto groups = padded/lanes;
        const auto repeats = slots/chunk;
        const auto rows = layer.at("weight_shape")[0].get<uint64_t>();
        const auto cols = layer.at("weight_shape")[1].get<uint64_t>();
        const auto kr = (rows + heads*d - 1)/(heads*d);
        const auto kc = (cols + heads*d - 1)/(heads*d);
        const auto blocks = std::max(kr, kc);
        const auto requests = blocks*groups*heads*d;
        const auto per_seed = d*repeats;
        if (std::min(kr, kc) != 1 || (lanes & (lanes-1)) || slots % chunk ||
            layout.at("Hp") != padded || layout.at("u") != lanes || layout.at("C") != chunk ||
            layout.at("G") != groups || layout.at("rho") != repeats || layout.at("K") != blocks ||
            layout.at("request_count") != requests || layout.at("requests_per_seed") != per_seed ||
            layout.at("seed_count") != (requests + per_seed-1)/per_seed ||
            layout.at("request_order_version") != 1 ||
            signature.at("generator").at("pcm_seed_layout") != layout ||
            signature.at("scale") != layout.at("scale"))
            throw std::runtime_error("PCM parameter seed layout differs from geometry: " + layer_id);
        if (layout.at("kind") != "pcm_row_chunk_v1" ||
            layout.at("seed_level").get<int>() != layout.at("use_site_level").get<int>() + 1 ||
            signature.at("level") != layout.at("seed_level") ||
            signature.at("generator").at("seed_id").get<uint64_t>() >= layout.at("seed_count").get<uint64_t>())
            throw std::runtime_error("Invalid PCM parameter seed layout: " + layer_id);
        return layout;
    }
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
    std::map<std::string, std::set<uint64_t>> seed_ids;
    for (const auto& sig : parameters) {
        if (sig.at("type") != "ct") continue;
        ++expected_count;
        const auto id = sig.at("id").get<std::string>();
        const auto& arg = manifest.at("arguments").at(id);
        if (sig.contains("generator") && sig.at("generator").value("type", std::string()) == "row_chunk_seed") {
            const auto layer_id = sig.at("layer_id").get<std::string>();
            if (!seed_ids[layer_id].insert(sig.at("generator").at("seed_id").get<uint64_t>()).second)
                throw std::runtime_error("Duplicate PCM seed: " + id);
        }
        for (const auto* key : {"size", "level", "type"}) {
            if (arg.at(key) != sig.at(key)) throw std::runtime_error("Parameter manifest/signature mismatch: " + id + " " + key);
        }
        if (sig.contains("use_site_level") && arg.at("use_site_level") != sig.at("use_site_level"))
            throw std::runtime_error("Parameter manifest/signature use-site mismatch: " + id);
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
    for (const auto& [layer_id, ids] : seed_ids)
        if (ids.size() != graph.at("layer").at(layer_id).at("pcm_seed_layout").at("seed_count").get<uint64_t>())
            throw std::runtime_error("Incomplete PCM seed set: " + layer_id);
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
