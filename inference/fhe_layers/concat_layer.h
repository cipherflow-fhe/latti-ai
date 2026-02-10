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
#include <stdio.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include "common.h"
#include "data_structs/feature2d.h"

class ConcatLayer {
public:
    ConcatLayer();
    Feature2DEncrypted run(CkksContext& ctx, const Feature2DEncrypted& x1, const Feature2DEncrypted& x2);
    Feature2DEncrypted run_multiple_inputs(CkksContext& ctx, const std::vector<Feature2DEncrypted>& inputs);
    Array<double, 3> concatenate_channels(const Array<double, 3>& x1, const Array<double, 3>& x2);
    Array<double, 3> concatenate_channels_multiple_inputs(const std::vector<Array<double, 3>>& inputs);
};
