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
#include <map>
#include <string>
#include <vector>

#include "fhe_ops_lib/fhe_lib_v2.h"
#include "util.h"

class EncryptedParameterStore {
public:
    explicit EncryptedParameterStore(std::filesystem::path root);

    bool has_argument(const std::string& arg_id) const;
    std::vector<fhe_ops_lib::CkksCiphertext>& load_argument(const std::string& arg_id);
    const json& manifest() const {
        return manifest_;
    }

private:
    struct Entry {
        std::string id;
        std::filesystem::path file;
        int level = -1;
        std::vector<uint64_t> shape;
    };

    std::filesystem::path root_;
    json manifest_;
    std::map<std::string, Entry> entries_;
    std::map<std::string, std::vector<fhe_ops_lib::CkksCiphertext>> cache_;
};
