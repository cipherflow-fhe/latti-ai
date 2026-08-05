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

#include "interface/encrypted_parameter_store.h"

#include <fstream>
#include <numeric>
#include <stdexcept>

#include "interface/runner_bundle_io.h"
#include "util/serial.h"

namespace {

uint64_t shape_size(const std::vector<uint64_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<uint64_t>());
}

}  // namespace

EncryptedParameterStore::EncryptedParameterStore(std::filesystem::path root) : root_(std::move(root)) {
    const auto manifest_path = root_ / "manifest.json";
    manifest_ = read_json(manifest_path.string());

    if (!manifest_.contains("arguments") || !manifest_["arguments"].is_object()) {
        throw std::runtime_error("Encrypted parameter manifest is missing object field 'arguments': " +
                                 manifest_path.string());
    }

    for (auto& [arg_id, value] : manifest_["arguments"].items()) {
        Entry entry;
        entry.id = value.value("id", arg_id);
        entry.file = value.at("file").get<std::string>();
        entry.level = value.at("level").get<int>();
        entry.shape = value.at("size").get<std::vector<uint64_t>>();
        if (value.contains("elements")) {
            const auto expected = shape_size(entry.shape);
            if (value["elements"].size() != expected) {
                throw std::runtime_error("Encrypted parameter manifest element count mismatch for " + arg_id);
            }
            entry.elements.reserve(value["elements"].size());
            for (const auto& item : value["elements"]) {
                ElementSpan span;
                span.offset = item.at("offset").get<uint64_t>();
                span.length = item.at("length").get<uint64_t>();
                entry.elements.push_back(span);
            }
        }
        if (entry.id != arg_id) {
            throw std::runtime_error("Encrypted parameter manifest id mismatch for argument: " + arg_id);
        }
        entries_[arg_id] = std::move(entry);
    }
}

bool EncryptedParameterStore::has_argument(const std::string& arg_id) const {
    return entries_.find(arg_id) != entries_.end();
}

std::vector<fhe_ops_lib::CkksCiphertext>& EncryptedParameterStore::load_argument(const std::string& arg_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto cache_it = cache_.find(arg_id);
    if (cache_it != cache_.end()) {
        return cache_it->second;
    }

    auto entry_it = entries_.find(arg_id);
    if (entry_it == entries_.end()) {
        throw std::runtime_error("Encrypted parameter argument not found in manifest: " + arg_id);
    }
    const Entry& entry = entry_it->second;

    Bytes bytes = read_binary_file(root_ / entry.file);
    std::stringstream ss;
    bytes_to_ss(bytes, ss);

    uint64_t count = 0;
    ss_read(ss, &count);
    const uint64_t expected = shape_size(entry.shape);
    if (count != expected) {
        throw std::runtime_error("Encrypted parameter count mismatch for " + arg_id + ": expected " +
                                 std::to_string(expected) + ", got " + std::to_string(count));
    }

    std::vector<fhe_ops_lib::CkksCiphertext> values;
    values.reserve(count);
    for (uint64_t i = 0; i < count; ++i) {
        Bytes ct_bytes;
        ss_read_vector(ss, &ct_bytes);
        auto ct = fhe_ops_lib::CkksCiphertext::deserialize(ct_bytes);
        if (ct.get_level() != entry.level) {
            throw std::runtime_error("Encrypted parameter level mismatch for " + arg_id + ": expected " +
                                     std::to_string(entry.level) + ", got " + std::to_string(ct.get_level()));
        }
        values.push_back(std::move(ct));
    }

    auto inserted = cache_.emplace(arg_id, std::move(values));
    return inserted.first->second;
}

std::shared_ptr<fhe_ops_lib::CkksCiphertext> EncryptedParameterStore::load_element(const std::string& arg_id,
                                                                                   uint64_t flat_index) {
    bool use_whole_argument_fallback = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto entry_it = entries_.find(arg_id);
        if (entry_it == entries_.end()) {
            throw std::runtime_error("Encrypted parameter argument not found in manifest: " + arg_id);
        }
        const Entry& entry = entry_it->second;
        const uint64_t expected = shape_size(entry.shape);
        if (flat_index >= expected) {
            throw std::runtime_error("Encrypted parameter flat index out of range for " + arg_id + ": " +
                                     std::to_string(flat_index));
        }

        if (entry.elements.empty()) {
            use_whole_argument_fallback = true;
        } else {
            auto& cached_values = element_cache_[arg_id];
            if (cached_values.empty()) {
                cached_values.resize(expected);
            }
            if (cached_values[flat_index]) {
                return cached_values[flat_index];
            }

            const ElementSpan& span = entry.elements.at(flat_index);
            std::ifstream in(root_ / entry.file, std::ios::binary);
            if (!in.is_open()) {
                throw std::runtime_error("Failed to open encrypted parameter file: " + (root_ / entry.file).string());
            }

            in.seekg(0, std::ios::end);
            const auto file_size = static_cast<uint64_t>(in.tellg());
            if (span.offset > file_size || span.length > file_size - span.offset) {
                throw std::runtime_error("Encrypted parameter byte range out of file for " + arg_id + "[" +
                                         std::to_string(flat_index) + "]");
            }

            Bytes ct_bytes(static_cast<size_t>(span.length));
            in.seekg(static_cast<std::streamoff>(span.offset), std::ios::beg);
            in.read(reinterpret_cast<char*>(ct_bytes.data()), static_cast<std::streamsize>(ct_bytes.size()));
            if (!in) {
                throw std::runtime_error("Failed to read encrypted parameter element " + arg_id + "[" +
                                         std::to_string(flat_index) + "]");
            }

            auto ct = std::make_shared<fhe_ops_lib::CkksCiphertext>(fhe_ops_lib::CkksCiphertext::deserialize(ct_bytes));
            if (ct->get_level() != entry.level) {
                throw std::runtime_error("Encrypted parameter level mismatch for " + arg_id + ": expected " +
                                         std::to_string(entry.level) + ", got " + std::to_string(ct->get_level()));
            }
            cached_values[flat_index] = ct;
            return ct;
        }
    }

    if (use_whole_argument_fallback) {
        auto& values = load_argument(arg_id);
        if (flat_index >= values.size()) {
            throw std::runtime_error("Encrypted parameter flat index out of range for " + arg_id + ": " +
                                     std::to_string(flat_index));
        }
        return std::shared_ptr<fhe_ops_lib::CkksCiphertext>(&values.at(flat_index),
                                                            [](fhe_ops_lib::CkksCiphertext*) {});
    }

    throw std::runtime_error("Failed to load encrypted parameter element " + arg_id + "[" + std::to_string(flat_index) +
                             "]");
}
