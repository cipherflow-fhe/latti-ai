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

#include <math.h>
#include <random>
#include "ndarray.h"
#include <fstream>
#include <iomanip>
#include <string>
#include <cstring>

template <int dim>
Array<double, dim>
csv_to_array(const std::string& filename, const std::array<uint64_t, dim>& shape, double factor = 1.0) {
    std::ifstream filestream(filename);
    if (!filestream.is_open()) {
        std::fprintf(stderr, "[ERROR] failed to open file <%s> : %s (errno=%d)\n", filename.c_str(), strerror(errno),
                     errno);
        std::exit(EXIT_FAILURE);
    }
    Array<double, dim> result(shape);
    std::string line;
    std::string cell;
    uint64_t counter = 0;
    while (getline(filestream, line)) {
        std::stringstream line_stream(line);
        while (getline(line_stream, cell, ',')) {
            double x;
            sscanf(cell.c_str(), "%lf", &x);
            result.set(counter, x * factor);
            counter++;
        }
    }
    return result;
}

template <int dim>
void array_to_csv(const Array<double, dim>& arr, const std::string& filename, double factor = 1.0, int precision = 6) {
    static_assert(dim == 3, "Array dimension must be 3 for this function");
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        printf("%s, file created failed。", filename.c_str());
        return;
    }
    auto shape = arr.get_shape();
    const uint64_t C = shape[0];
    const uint64_t H = shape[1];
    const uint64_t W = shape[2];
    outfile << std::fixed << std::setprecision(precision);
    for (uint64_t c = 0; c < C; c++) {
        outfile << arr.get(c, 0, 0) * factor;
        for (uint64_t hw = 1; hw < H * W; hw++) {
            const uint64_t h = hw / W;
            const uint64_t w = hw % W;

            outfile << "," << arr.get(c, h, w) * factor;
        }
        outfile << "\n";
    }
    outfile.close();
}

template <int dim> Array<double, 1> csv_to_array(const std::string& filename, double factor = 1.0) {
    assert(dim == 1);
    std::ifstream filestream(filename);
    if (!filestream.is_open()) {
        printf("%s, failed to open file.", filename.c_str());
        exit(0);
    }
    std::vector<double> data;
    std::string line;
    std::string cell;
    while (getline(filestream, line)) {
        std::stringstream line_stream(line);
        while (getline(line_stream, cell, ',')) {
            double x;
            sscanf(cell.c_str(), "%lf", &x);
            data.push_back(x * factor);
        }
    }
    return Array<double, 1>::move_from_array_1d(std::move(data));
}

template <int dim> Array<double, dim> gen_random_array(const std::array<uint64_t, dim>& shape, double scale) {
    Array<double, dim> result(shape);
    uint64_t s = result.get_size();
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < s; i++) {
        result.set(i, scale * dis(gen));
    }
    return result;
}

template <int dim> Array<double, dim> gen_random_array_positive(const std::array<uint64_t, dim>& shape, double scale) {
    Array<double, dim> result(shape);
    uint64_t s = result.get_size();
    for (int i = 0; i < s; i++) {
        result.set(i, scale * ((rand() % 10000 / 10000.0) * 0.9 + 0.1));
    }
    return result;
}

inline double max_value(Array<double, 3>& values) {
    auto temp = values.to_array_1d();
    double max_v = temp[0];
    for (int i = 1; i < temp.size(); i++) {
        if (max_v < temp[i]) {
            max_v = temp[i];
        }
    }
    return max_v;
}

inline double min_value(Array<double, 3>& values) {
    auto temp = values.to_array_1d();
    double min_v = temp[0];
    for (int i = 1; i < temp.size(); i++) {
        if (min_v > temp[i]) {
            min_v = temp[i];
        }
    }
    return min_v;
}
