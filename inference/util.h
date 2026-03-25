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
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "util/array_io.h"
#include "util/numeric.h"
#include "util/ndarray.h"
#include "util/serial.h"
#include "util/timer.h"
#include "util/types.h"

#include "lattisense/lib/nlohmann/json.hpp"

using json = nlohmann::ordered_json;

#define th_nums 16

const int SIGMA = 32;
const int ENC_TO_SHARE_SCALE_BIT = 31;
const double ENC_TO_SHARE_SCALE = pow(2, ENC_TO_SHARE_SCALE_BIT);

const int DEFAULT_SCALE_BIT = 34;
const double DEFAULT_SCALE = pow(2, DEFAULT_SCALE_BIT);

const int RING_MOD_BIT = 44;
const uint64_t RING_MOD = 1UL << RING_MOD_BIT;

const Duo BLOCK_SHAPE = {64, 64};

inline uint64_t gen_random_uint(int n_bit) {
    uint64_t result = 0;
    for (int i = 0; i < n_bit; i++) {
        result = result * 2 + (random() % 2 == 0 ? 0 : 1);
    }
    return result;
}

json read_json(std::string filename);

uint64_t mod_sub(uint64_t x, uint64_t y, uint64_t mod);

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

Array<double, 3> upsample_with_zero(const Array<double, 3>& x, const Duo& stride);
