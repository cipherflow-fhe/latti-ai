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

#include <cxx_sdk_v2/cxx_fhe_task.h>

#include <vector>

namespace ls = lattisense;

std::vector<ls::CkksCiphertext>
populate_rotations_1_side(ls::CkksContext& ctx, const ls::CkksCiphertext& x, int n_rotation, int unit);

std::vector<ls::CkksCiphertext>
populate_rotations_2_sides(ls::CkksContext& ctx, const ls::CkksCiphertext& x, int n_rotation, int unit);

uint32_t next_pow2(uint32_t x);
