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
#include "layer.h"
#include "util.h"
#include "data_structs/feature2d.h"

class UpsampleNearestLayer : public Layer {
public:
    UpsampleNearestLayer(const ls::CkksParameter& param_in,
                         const Duo& shape_in,
                         const Duo& skip_in,
                         const Duo& upsample_factor_in,
                         const uint32_t& n_channel_per_ct_in,
                         const uint32_t& level_in);
    std::vector<double> select_tensor(int num) const;
    void prepare_weight();
    void prepare_weight_lazy();
    Feature2DEncrypted run(ls::CkksContext& ctx, const Feature2DEncrypted& x);
    Array<double, 3> run_plaintext(const Array<double, 3>& x);

    // Helper function to generate select_tensor plaintext on-demand
    ls::CkksPlaintextRingt generate_select_tensor_pt_for_index(ls::CkksContext& ctx, int idx) const;

    std::vector<ls::CkksPlaintextRingt> select_tensor_pt;
    Duo upsample_factor;
    Duo shape;
    Duo skip;
    uint32_t n_channel_per_ct;
    uint32_t n_block_per_ct;

private:
    // Cached values for on-demand generation
    uint32_t cached_block_size;
    uint32_t cached_skip_div_upsample_0;
    uint32_t cached_skip_div_upsample_1;
};
