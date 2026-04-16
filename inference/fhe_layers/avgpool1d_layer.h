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
#include <stdint.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include "layer.h"
#include "util.h"
#include "data_structs/feature1d.h"

class Avgpool1DLayer : public Layer {
public:
    using Layer::prepare_weight;
    using Layer::prepare_weight_lazy;

    Avgpool1DLayer(uint32_t shape_in, uint32_t stride_in);
    Feature1DEncrypted run_adaptive_avgpool(ls::CkksContext& ctx, const Feature1DEncrypted& x);
    Array<double, 2> run_plaintext(const Array<double, 2>& x);
    Array<double, 2> run_plaintext_multiplexed(const Array<double, 2>& x);
    std::vector<double> select_tensor(int num) const;
    void prepare_weight(const ls::CkksParameter& param_in,
                        int n_channel_per_ct,
                        int n_channel,
                        int level,
                        uint32_t skip_in,
                        uint32_t shape_in);
    void prepare_weight_lazy(const ls::CkksParameter& param_in,
                             int n_channel_per_ct,
                             int n_channel,
                             int level,
                             uint32_t skip_in,
                             uint32_t shape_in);
    ls::CkksPlaintextRingt generate_select_tensor_pt_for_index(ls::CkksContext& ctx, int i) const;
    Feature1DEncrypted run_multiplexed_avgpool(ls::CkksContext& ctx, const Feature1DEncrypted& x);
    std::vector<ls::CkksPlaintextRingt> select_tensor_pt;
    uint32_t shape;
    uint32_t stride;
    uint32_t skip;
    uint32_t n_block_per_ct;
};
