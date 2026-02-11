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
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include "lattisense/lib/nlohmann/json.hpp"
#include "common.h"

using json = nlohmann::ordered_json;

struct SplitInfo {
    Array4D result;
    std::vector<std::vector<int>> patch_x_start_end;
    std::vector<std::vector<int>> patch_y_start_end;
};

extern double total_num;
extern double mpc_num;
extern double mpc_fpga_num;

json read_json(std::string filename);

uint64_t mod_sub(uint64_t x, uint64_t y, uint64_t mod, bool is_print = false);

void vec_to_share(Array1D& vec, Array1DUint& share1, Array1DUint& share2, int scale_ord, uint64_t ring_mod);

inline Array1D L2_normal(const Array1D& x) {
    double sum = 0;
    for (int i = 0; i < x.size(); i++) {
        sum += pow((x[i]), 2);
    }
    sum = pow(sum, 0.5);
    Array1D res;
    for (int i = 0; i < x.size(); i++) {
        res.push_back(x[i] / sum);
    }
    return res;
};

inline bool f_equal(double a, double b) {
    const double eps = 1e-8;
    if (fabs(b) < eps) {
        return fabs(a - b) < eps;
    } else {
        return fabs((a - b) / b) < eps;
    }
}

uint64_t double_to_uint64(double input, double scale, uint64_t ring_mod);

template <int dim>
Array<uint64_t, dim> array_double_to_uint64(const Array<double, dim>& x, int scale_ord, uint64_t ring_mod) {
    double scale = pow(2, scale_ord);
    Array<uint64_t, dim> result(x.get_shape());
    uint64_t s = x.get_size();
    for (int i = 0; i < s; i++) {
        double value = x.get(i);
        uint64_t y = double_to_uint64(value, scale, ring_mod);
        result.set(i, y);
    }
    return result;
}

Array<double, 4> transpose_weight(const Array<double, 4>& weight);

void print_array(const Array<double, 3>& arr, std::ostream& out = std::cout);

void print_array_to_file(const Array<double, 3>& arr, const std::string& filename, bool append = true);

Array<double, 3> upsample_with_zero(const Array<double, 3>& x, const Duo& stride);
