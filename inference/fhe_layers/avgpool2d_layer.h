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
    Avgpool2DLayer(const Duo& shape_in, const Duo& stride_in);
    Feature2DEncrypted run(ls::CkksContext& ctx, const Feature2DEncrypted& x);
    Feature2DEncrypted run_adaptive_avgpool(ls::CkksContext& ctx, const Feature2DEncrypted& x);
    Array<double, 3> plaintext_call(const Array<double, 3>& x);
    Array<double, 3> plaintext_call_multiplexed(const Array<double, 3>& x);
    std::vector<double> select_tensor(int num);
    void prepare_weight(const ls::CkksParameter& param_in,
                        int n_channel_per_ct,
                        int n_channel,
                        int level,
                        const Duo& skip_in,
                        const Duo& shape_in);
    Feature2DEncrypted run_multiplexed_avgpool(ls::CkksContext& ctx, const Feature2DEncrypted& x);
    Feature2DEncrypted run_split_avgpool(ls::CkksContext& ctx, const Feature2DEncrypted& x, const Duo block_expansion);
    std::vector<ls::CkksPlaintextRingt> select_tensor_pt;
    Duo shape;
    Duo stride;
    Duo skip;
    uint32_t n_block_per_ct;
};
