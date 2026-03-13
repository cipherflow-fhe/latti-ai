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

#include "../data_structs/feature.h"
#include "data_structs/constants.h"

#include <array>
#include <cstdint>
#include <vector>

class Conv1DPackedLayer {
public:
    Conv1DPackedLayer(const CkksParameter& param_in,
                      const uint32_t input_shape_in,
                      const Array<double, 3>& weight_in,
                      const Array<double, 1>& bias_in,
                      const uint32_t stride_in,
                      const uint32_t skip_in,
                      uint32_t pack_in,
                      uint32_t level_in,
                      double residual_scale = 1.0);
    ~Conv1DPackedLayer();

    virtual void prepare_weight();
    void prepare_weight_lazy();

    virtual Feature1DEncrypted run(CkksContext& ctx, Feature1DEncrypted& x);
    Array<double, 2> plaintext_call(const Array<double, 2>& x);

    CkksPlaintextRingt generate_weight_pt_for_indices(CkksContext& ctx, int ct_idx, int j, int k) const;
    CkksPlaintextRingt generate_bias_pt_for_index(CkksContext& ctx, int bpt_idx) const;

    uint32_t input_shape;
    uint32_t skip;
    uint32_t stride;

    Array<double, 3> weight;
    Array<double, 1> bias;
    CkksParameter param;
    std::vector<std::vector<std::vector<CkksPlaintextRingt>>> weight_pt;
    std::vector<CkksPlaintextRingt> bias_pt;
    bool normal_conv = true;

private:
    std::vector<CkksCiphertext> run_core(CkksContext& ctx, std::vector<CkksCiphertext>& x);
    void mult_add(CkksContext* ctx,
                  vector<vector<CkksCiphertext>>& rotated_x,
                  uint32_t start,
                  uint32_t end,
                  vector<CkksCiphertext>& result);
    uint32_t kernel_shape;
    uint32_t n_channel_in;
    uint32_t n_channel_out;
    uint32_t n_channel_per_ct;
    uint32_t n_packed_in_channel;
    uint32_t n_packed_out_channel;
    uint32_t level;
    double weight_scale;
    std::vector<std::vector<double>> kernel_masks_;
};
