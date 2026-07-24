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

#include "layer.h"
#include "../data_structs/feature.h"
#include "data_structs/constants.h"

#include <array>
#include <cstdint>
#include <vector>

class Conv2DLayer : public Layer {
public:
    Conv2DLayer(const ls::CkksParameter& param,
                const Duo& input_shape,
                Array<double, 4>&& weight,
                Array<double, 1>&& bias,
                const Duo& stride);

    Array<double, 3> run_plaintext(const Array<double, 3>& x, double multiplier = 1.0);

    Array<double, 4> weight_;

    Array<double, 1> bias_;

protected:
    uint32_t n_out_channel_;

    uint32_t n_in_channel_;

    uint32_t n_groups_ = 1;

    Duo input_shape_;

    Duo kernel_shape_;

    Duo stride_;

    double compute_output_element(uint32_t out_ch,
                                  const Duo& output_pos,
                                  const Array<double, 3>& padded_input,
                                  double weight_scale) const;
};
