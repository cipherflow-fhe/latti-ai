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

#include "util.h"

Bytes read_binary_file(const std::filesystem::path& path);
void write_binary_file(const std::filesystem::path& path, const Bytes& bytes);

void write_named_bytes_bundle(const std::filesystem::path& path, const std::map<std::string, Bytes>& values);
std::map<std::string, Bytes> read_named_bytes_bundle(const std::filesystem::path& path);
