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

#include "conv2d_layer.h"
#include "data_structs/feature.h"
#include <vector>

class Conv2DPackedDepthwiseLayer : public Conv2DLayer {
public:
    Conv2DPackedDepthwiseLayer(const CkksParameter& param,
                               const Duo& input_shape,
                               const Array<double, 4>& weight,
                               const Array<double, 1>& bias,
                               const Duo& stride,
                               const Duo& skip,
                               uint32_t n_channel_per_ct,
                               uint32_t level,
                               double residual_scale = 1.0);

    ~Conv2DPackedDepthwiseLayer() override = default;

    // Disable copy, enable move
    Conv2DPackedDepthwiseLayer(const Conv2DPackedDepthwiseLayer&) = delete;
    Conv2DPackedDepthwiseLayer& operator=(const Conv2DPackedDepthwiseLayer&) = delete;
    Conv2DPackedDepthwiseLayer(Conv2DPackedDepthwiseLayer&&) noexcept = default;
    Conv2DPackedDepthwiseLayer& operator=(Conv2DPackedDepthwiseLayer&&) noexcept = default;

    void prepare_weight();
    void prepare_weight_lazy();

    virtual Feature2DEncrypted run(CkksContext& ctx, const Feature2DEncrypted& x);

    virtual Array<double, 3> run_plaintext(const Array<double, 3>& x, double multiplier = 1.0);

    std::vector<std::vector<CkksPlaintextRingt>> weight_pt_;

    std::vector<CkksPlaintextRingt> bias_pt_;

    // Helper functions to generate weights/bias on-demand (for lazy mode)
    CkksPlaintextRingt generate_weight_pt_for_indices(CkksContext& ctx, int ct_idx, int j) const;
    CkksPlaintextRingt generate_bias_pt_for_index(CkksContext& ctx, int bpt_idx) const;

private:
    std::vector<CkksCiphertext> run_core(CkksContext& ctx, const std::vector<CkksCiphertext>& x) const;

    void mult_add(CkksContext* ctx,
                  const std::vector<std::vector<CkksCiphertext>>& rotated_x,
                  uint32_t start,
                  uint32_t end,
                  std::vector<CkksCiphertext>& result) const;

    double compute_depthwise_element(uint32_t out_ch,
                                     uint32_t out_i,
                                     uint32_t out_j,
                                     const Array<double, 3>& padded_input,
                                     double weight_scale) const;

    uint32_t n_channel_per_ct_;

    uint32_t n_packed_in_ct_;

    uint32_t n_packed_out_ct_;

    uint32_t level_;

    double modified_scale_;

    // Cached values for on-demand generation (lazy mode)
    int N = 0;
    uint32_t cached_input_block_size = 0;
    // cppcheck-suppress duplInheritedMember
    std::vector<std::vector<double>> kernel_masks_;
    // cppcheck-suppress duplInheritedMember
    std::vector<int> input_rotate_units_;
};
