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

class InverseMultiplexedConv2DLayer : public Layer {
public:
    InverseMultiplexedConv2DLayer(const ls::CkksParameter& param_in,
                                  const Duo& input_shape_in,
                                  const Array<double, 4>& weight_in,
                                  const Array<double, 1>& bias_in,
                                  const Array<int, 1>& padding_in,
                                  const Duo& stride_in,
                                  const Duo& stride_next_in,
                                  const Duo& skip_in,
                                  const Duo& block_shape_in,
                                  uint32_t level_in,
                                  double residual_scale = 1.0);
    virtual void prepare_weight();
    virtual void prepare_weight_lazy();

    virtual Feature2DEncrypted run(ls::CkksContext& ctx, const Feature2DEncrypted& x);

    virtual Array<double, 3> run_plaintext(const Array<double, 3>& x, double multiplier = 1.0);

    std::vector<std::vector<std::vector<ls::CkksPlaintextRingt>>> weight_pt;
    std::vector<ls::CkksPlaintextRingt> bias_pt;
    ls::CkksPlaintextRingt repack_mask_pt;
    bool normal_conv = true;
    // Helper functions to generate weights/bias on-demand
    ls::CkksPlaintextRingt generate_weight_pt_for_indices(ls::CkksContext& ctx,
                                                          int out_channel_idx,
                                                          int in_channel_idx,
                                                          int kernel_count) const;
    ls::CkksPlaintextRingt generate_bias_pt_for_index(ls::CkksContext& ctx, int out_channel_idx) const;
    ls::CkksPlaintextRingt generate_repack_mask_pt(ls::CkksContext& ctx) const;
    std::vector<uint32_t> get_used_input_indices() const;

private:
    std::vector<ls::CkksCiphertext> run_core(ls::CkksContext& ctx, const std::vector<ls::CkksCiphertext>& x);

    int N;
    uint32_t n_out_channel;
    uint32_t n_in_channel;
    Duo input_shape;
    Duo kernel_shape;
    Duo block_shape;
    Duo stride;
    Duo stride_next;
    Duo skip;
    Duo padding_shape;
    Duo orig_stride;
    bool need_repack = false;
    Array<double, 4> weight;
    Array<double, 1> bias;
    std::vector<std::vector<double>> kernel_masks;
    std::vector<int32_t> input_rotate_steps;
    std::vector<int> input_rotate_units;
    std::vector<int> input_rotate_ranges;
    double weight_scale;

    // Cached values for on-demand generation
    int cached_input_block_size = 0;
    int cached_kernel_total_count = 0;
    int cached_total_block_size = 0;
};
