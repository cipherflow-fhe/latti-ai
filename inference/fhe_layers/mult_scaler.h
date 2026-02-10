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
#include <cstdint>
#include "common.h"
#include "data_structs/feature2d.h"
#include "util.h"

class MultScalarLayer {
public:
    MultScalarLayer(const CkksParameter& param_in,
                    const Duo& input_shape_in,
                    const Array<double, 1>& weight_in,
                    const Duo& skip_in,
                    uint32_t n_channel_per_ct_in,
                    uint32_t level_in,
                    const Duo& upsample_factor_in = {1, 1},
                    const Duo& block_expansion_in = {1, 1});
    ~MultScalarLayer();
    virtual void prepare_weight();
    std::vector<CkksCiphertext> run_core(CkksContext& ctx, const std::vector<CkksCiphertext>& x);
    Feature2DEncrypted run(CkksContext& ctx, const Feature2DEncrypted& x);
    virtual Array<double, 3> run_plaintext(const Array<double, 3>& x);

    CkksParameter param;
    Duo input_shape;
    Duo skip;
    Duo pre_skip;
    Duo upsample_factor;
    Duo block_expansion;
    Duo block_shape;
    Array<double, 1> weight;
    uint32_t n_channel_per_ct;
    uint32_t level;
    uint32_t n_block_per_ct;
    vector<CkksPlaintextRingt> weight_pt;
};
