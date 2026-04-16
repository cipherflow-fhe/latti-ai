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

class Avgpool2DLayer : public Layer {
public:
    using Layer::prepare_weight;
    using Layer::prepare_weight_lazy;

    Avgpool2DLayer(const Duo& shape_in, const Duo& stride_in);
    Feature2DEncrypted run(ls::CkksContext& ctx, const Feature2DEncrypted& x);
    Feature2DEncrypted run_adaptive_avgpool(ls::CkksContext& ctx, const Feature2DEncrypted& x);
    Array<double, 3> run_plaintext(const Array<double, 3>& x);
    Array<double, 3> run_plaintext_multiplexed(const Array<double, 3>& x);
    std::vector<double> select_tensor(int num) const;
    void prepare_weight(const ls::CkksParameter& param_in,
                        int n_channel_per_ct,
                        int n_channel,
                        int level,
                        const Duo& skip_in,
                        const Duo& shape_in);
    void prepare_weight_lazy(const ls::CkksParameter& param_in,
                             int n_channel_per_ct,
                             int n_channel,
                             int level,
                             const Duo& skip_in,
                             const Duo& shape_in);
    ls::CkksPlaintextRingt generate_select_tensor_pt_for_index(ls::CkksContext& ctx, int i) const;
    Feature2DEncrypted run_multiplexed_avgpool(ls::CkksContext& ctx, const Feature2DEncrypted& x);
    Feature2DEncrypted run_split_avgpool(ls::CkksContext& ctx, const Feature2DEncrypted& x, const Duo block_expansion);
    void prepare_weight_repack(const ls::CkksParameter& param_in,
                               uint32_t n_channel,
                               int level,
                               const Duo& second_stage_stride,
                               const Duo& block_shape_in);
    std::vector<ls::CkksPlaintextRingt> select_tensor_pt;
    ls::CkksPlaintextRingt repack_mask_pt;
    Duo shape;
    Duo stride;
    Duo block_shape;
    Duo skip;
    uint32_t n_block_per_ct;
    bool need_repack = false;
    uint32_t n_channel_ = 0;
};
