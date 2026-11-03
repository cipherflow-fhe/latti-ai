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
#include "../data_structs/feature_mat.h"
#include "layer_util.h"
#include <cassert>
#include <cmath>
#include <stdexcept>

// Slot coordinates shared by PCM pointwise layers. Keep in sync with feature_mat.cpp.
struct ParBlockColMajorLayout {
    uint32_t rows, cols, head_cols, heads, d, slots, u, groups, chunk, br, bc, n_mb, cts_per_mb;
    ParBlockColMajorLayout(Duo shape, Duo head_shape, uint32_t h, uint32_t block, uint32_t n_slot)
        : rows(shape[0]), cols(shape[1]), head_cols(head_shape[1]), heads(h), d(block), slots(n_slot) {
        assert(rows && cols && head_cols && heads && head_shape[0] == rows);
        assert(d && !(d & (d - 1)) && slots >= d * d);
        uint32_t hp = next_pow2(heads);
        if (slots >= hp * d * d) {
            u = hp;
            chunk = hp * d * d;
            groups = 1;
        } else {
            u = slots / (d * d);
            chunk = slots;
            if (u == 1)
                hp = heads;
            groups = hp / u;
        }
        assert(u && groups && slots % chunk == 0 && (u == 1 || hp % u == 0));
        br = div_ceil(rows, d);
        bc = div_ceil(head_cols, d);
        n_mb = div_ceil(cols, heads * head_cols);
        cts_per_mb = br * bc * groups;
    }
    uint32_t total_cts() const {
        return n_mb * cts_per_mb;
    }
    int column(uint32_t idx, uint32_t slot) const {
        assert(idx < total_cts() && slot < slots);
        uint32_t mb = idx / cts_per_mb, local = idx % cts_per_mb;
        uint32_t g = local % groups, block = local / groups;
        uint32_t bi = block % br, bj = block / br;
        uint32_t pos = slot % chunk, h = g * u + pos % u;
        uint32_t row = bi * d + (pos / u) % d, col = bj * d + (pos / u) / d;
        uint32_t global = mb * heads * head_cols + h * head_cols + col;
        return h < heads && row < rows && col < head_cols && global < cols ? int(global) : -1;
    }
    std::vector<double> constant(uint32_t idx, double value) const {
        std::vector<double> result(slots, 0.0);
        for (uint32_t s = 0; s < slots; s++)
            if (column(idx, s) >= 0)
                result[s] = value;
        return result;
    }
};

inline double pcm_checked_scale(double scale) {
    if (!std::isfinite(scale) || scale <= 0)
        throw std::runtime_error("Invalid PCM plaintext scale");
    return scale;
}
