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

#include "layer_util.h"

#include <map>
#include <utility>

using namespace lattisense;

std::vector<CkksCiphertext>
populate_rotations_1_side(CkksContext& ctx, const CkksCiphertext& x, int n_rotation, int unit) {
    std::vector<CkksCiphertext> result;
    result.reserve(1 + n_rotation);
    result.push_back(x.copy());

    std::vector<int32_t> rotation_steps;
    rotation_steps.reserve(n_rotation);
    for (int i = 1; i <= n_rotation; i++) {
        rotation_steps.push_back(i * unit);
    }

    std::map<int32_t, CkksCiphertext> rotated_map = ctx.rotate(x, rotation_steps);
    for (auto& [step, ct] : rotated_map) {
        result.push_back(std::move(ct));
    }

    return result;
}

std::vector<CkksCiphertext>
populate_rotations_2_sides(CkksContext& ctx, const CkksCiphertext& x, int n_rotation, int unit) {
    std::vector<CkksCiphertext> result;
    result.reserve(n_rotation);
    const int filter_center = n_rotation / 2;

    std::vector<int32_t> rotation_steps;
    for (int i = -filter_center; i < n_rotation - filter_center; i++) {
        if (i != 0) {
            rotation_steps.push_back(i * unit);
        }
    }

    std::map<int32_t, CkksCiphertext> rotated_map = ctx.rotate(x, rotation_steps);

    if (-filter_center < 0) {
        for (int i = -filter_center; i < 0; i++) {
            result.push_back(std::move(rotated_map.at(i * unit)));
        }
    }
    result.push_back(x.copy());
    if (n_rotation - filter_center > 1) {
        for (int i = 1; i < n_rotation - filter_center; i++) {
            result.push_back(std::move(rotated_map.at(i * unit)));
        }
    }

    return result;
}

uint32_t next_pow2(uint32_t x) {
    uint32_t p = 1;
    while (p < x) {
        p *= 2;
    }
    return p;
}
