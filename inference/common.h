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
#include <iostream>
#include <array>
#include <cstdarg>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "util/array_io.h"
#include "util/helper.h"
#include "util/ndarray.h"
#include "util/serial.h"
#include "util/timer.h"
#include "util/types.h"

// config
#include "lattisense/lib/nlohmann/json.hpp"
#include <fstream>
#include <sys/stat.h>

#define th_nums 16
const bool is_print = false;

const int SIGMA = 32;
const int ENC_TO_SHARE_SCALE_BIT = 31;
const double ENC_TO_SHARE_SCALE = pow(2, ENC_TO_SHARE_SCALE_BIT);

const int DEFAULT_SCALE_BIT = 34;
const double DEFAULT_SCALE = pow(2, DEFAULT_SCALE_BIT);

const int RING_MOD_BIT = 44;

const uint64_t RING_MOD = 1UL << RING_MOD_BIT;

const int T_SCALE_BIT = 6;
const uint64_t T_SCALE = 1UL << T_SCALE_BIT;
const Duo BLOCK_SHAPE = {64, 64};

inline std::string str(const Duo& x) {
    return "(" + std::to_string(x[0]) + ',' + std::to_string(x[1]) + ')';
}

template <typename T>
bool get_config_from_file(const std::string& id, const std::filesystem::path& filename, T* result) {
    std::ifstream fs;
    fs.open(filename);
    nlohmann::json json = nlohmann::json::parse(fs);
    fs.close();

    if (json.contains(id)) {
        *result = json.at(id).get<T>();
        return true;
    } else {
        return false;
    }
}

template <typename T> T get_config(const std::string& id) {
    std::filesystem::path source_path(SOURCE_PATH);
    std::filesystem::path config_path = source_path / "config.json";
    std::filesystem::path config_user_path = source_path / "config_user.json";

    T result;
    if (std::filesystem::exists(config_user_path)) {
        if (get_config_from_file<T>(id, config_user_path, &result)) {
            return result;
        }
    }
    if (std::filesystem::exists(config_path)) {
        if (get_config_from_file<T>(id, config_path, &result)) {
            return result;
        }
    }
    throw "Config id not found.";
    return 0;
}

inline void _print_dev_internal(const char* filename, int line, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

#define print_dev(...) _print_dev_internal(__FILE__, __LINE__, __VA_ARGS__)

inline int get_param_id(const std::string& param) {
    std::string sub_in = param.substr(5);
    int param_id = stoi(sub_in);
    return param_id;
}

inline uint64_t gen_random_uint(int n_bit) {
    uint64_t result = 0;
    for (int i = 0; i < n_bit; i++) {
        result = result * 2 + (random() % 2 == 0 ? 0 : 1);
    }
    return result;
}

inline void write_file_common(std::string file_name, const Array1D& data) {
    std::ofstream outputFilePlain(file_name);
    for (int i = 0; i < data.size(); i++)
        outputFilePlain << data[i] << std::endl;
}

template <int dim> inline void save_data(std::string file_name, Array<double, dim>& data) {
    std::ofstream outputFilePlain(file_name);
    for (int i = 0; i < data.get_size(); i++)
        outputFilePlain << data.get_data()[i] << std::endl;
}
