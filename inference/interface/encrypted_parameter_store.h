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

#pragma once

#include <filesystem>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "fhe_ops_lib/fhe_lib_v2.h"
#include "util.h"

class EncryptedParameterStore {
public:
    // Account retained ciphertexts by their serialized payload size. Graph-held
    // shared_ptrs remain valid after eviction and are released by the executor.
    explicit EncryptedParameterStore(std::filesystem::path root, uint64_t cache_capacity_bytes = 64ULL << 20);

    bool has_argument(const std::string& arg_id) const;
    std::vector<fhe_ops_lib::CkksCiphertext>& load_argument(const std::string& arg_id);
    std::shared_ptr<fhe_ops_lib::CkksCiphertext> load_element(const std::string& arg_id, uint64_t flat_index);
    json cache_statistics() const;
    const json& manifest() const {
        return manifest_;
    }

private:
    struct ElementSpan {
        uint64_t offset = 0;
        uint64_t length = 0;
    };

    struct Entry {
        std::string id;
        std::filesystem::path file;
        int level = -1;
        double scale = 0.0;  // Absent in legacy manifests.
        std::vector<uint64_t> shape;
        std::vector<ElementSpan> elements;
    };

    using ElementKey = std::pair<std::string, uint64_t>;
    struct CachedElement {
        std::shared_ptr<fhe_ops_lib::CkksCiphertext> value;
        uint64_t bytes;
        std::list<ElementKey>::iterator position;
    };

    std::filesystem::path root_;
    json manifest_;
    std::map<std::string, Entry> entries_;
    std::map<std::string, std::vector<fhe_ops_lib::CkksCiphertext>> cache_;
    std::map<ElementKey, CachedElement> element_cache_;
    std::list<ElementKey> lru_;
    uint64_t cache_capacity_bytes_;
    uint64_t cache_bytes_ = 0, peak_cache_bytes_ = 0;
    uint64_t canonical_loads_ = 0, cache_hits_ = 0, cache_evictions_ = 0;
    mutable std::mutex mutex_;
};
