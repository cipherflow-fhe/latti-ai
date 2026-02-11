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

#include <cstdint>
#include <vector>
#include <sstream>
#include <string>

#include "types.h"

template <typename T> void ss_write(std::stringstream& ss, const T& x) {
    ss.write(reinterpret_cast<const char*>(&x), sizeof(x));
}

template <typename T> void ss_write_vector(std::stringstream& ss, const std::vector<T>& x) {
    size_t length = x.size();
    ss.write(reinterpret_cast<const char*>(&length), sizeof(length));
    ss.write(reinterpret_cast<const char*>(x.data()), length * sizeof(T));
}

inline void ss_write_string(std::stringstream& ss, const std::string& x) {
    size_t length = x.size();
    ss.write(reinterpret_cast<const char*>(&length), sizeof(length));
    ss.write(reinterpret_cast<const char*>(x.data()), length * sizeof(char));
}

template <typename T> void ss_read(std::stringstream& ss, T* x) {
    ss.read(reinterpret_cast<char*>(x), sizeof(*x));
}

template <typename T> void ss_read_vector(std::stringstream& ss, std::vector<T>* x) {
    size_t length;
    ss.read(reinterpret_cast<char*>(&length), sizeof(length));
    x->resize(length);
    ss.read(reinterpret_cast<char*>(x->data()), length * sizeof(T));
}

inline void ss_read_string(std::stringstream& ss, std::string* x) {
    size_t length;
    ss.read(reinterpret_cast<char*>(&length), sizeof(length));
    char* buffer = new char[length];
    ss.read(reinterpret_cast<char*>(buffer), length * sizeof(char));
    x->assign(buffer, length);
    delete[] buffer;
}

inline void bytes_to_ss(const Bytes& bytes, std::stringstream& ss) {
    ss.write((const char*)bytes.data(), bytes.size());
}

inline Bytes ss_to_bytes(std::stringstream& ss) {
    Bytes bytes(ss.tellp());
    ss.seekg(0);
    ss.read((char*)bytes.data(), ss.tellp());
    return bytes;
}
