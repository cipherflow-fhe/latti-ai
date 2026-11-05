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
#include "conv2d_layer.h"
#include "../data_structs/feature.h"

class InverseMultiplexedConv2DLayerDepthwise : public Conv2DLayer {
public:
    InverseMultiplexedConv2DLayerDepthwise(const ls::CkksParameter& param_in,
                                           const Duo& input_shape_in,
                                           Array<double, 4>&& weight_in,
                                           Array<double, 1>&& bias_in,
                                           const Array<int, 1>& padding_in,
                                           const Duo& stride_in,
                                           const Duo& block_shape_in,
                                           uint32_t level_in,
                                           double residual_scale = 1.0,
                                          const Duo& seed_weight_subgrid_stride_in = {1, 1},
                                  bool canonical_seed_kernels = false);
    void prepare_weight() override;
    void prepare_weight_lazy() override;

    virtual Feature2DEncrypted run(ls::CkksContext& ctx, const Feature2DEncrypted& x);

    std::vector<std::vector<ls::CkksPlaintext>> weight_pt;
    std::vector<ls::CkksPlaintext> bias_pt;
    ls::CkksPlaintext repack_mask_pt;
    bool normal_conv = true;
    bool need_repack = false;

    ls::CkksPlaintext
    generate_weight_pt_for_indices(ls::CkksContext& ctx, int out_channel_idx, int kernel_count) const;
    ls::CkksPlaintext generate_bias_pt_for_index(ls::CkksContext& ctx, int out_channel_idx) const;
    ls::CkksPlaintext generate_repack_mask_pt(ls::CkksContext& ctx) const;
    std::vector<uint32_t> get_used_input_indices() const;

    ls::CkksPlaintextRingt generate_weight_pt_for_seed_indices(ls::CkksContext& ctx, int ct_idx, int seed_kernel) const;
    ls::CkksPlaintextRingt generate_mask_pt_for_seed_indices(ls::CkksContext& ctx, int subgrid,
                                                            int kernel_index, bool boundary) const;
    bool use_seed_weight_pack() const { return use_seed_weight_pack_; }
    uint32_t seed_weight_kernel_count() const {
        const uint32_t count = prod(kernel_shape_) * (canonical_seed_kernels_ ? 1 : prod(output_step));
        return use_seed_weight_pack_ ? div_ceil(count, prod(seed_weight_subgrid_stride_)) : count;
    }
    uint32_t seed_weight_level_cost() const {
        return use_seed_weight_pack_ ? (kernel_shape_ != Duo{1, 1} || pad_ != DuoInt{0, 0} ? 2 : 1) : 0;
    }

private:
    std::vector<ls::CkksCiphertext> run_core(ls::CkksContext& ctx, const std::vector<ls::CkksCiphertext>& x);

    int N;
    Duo block_shape;
    Duo output_step;
    Duo input_step;
    DuoInt pad_;
    Duo ct_stride_;
    std::vector<int32_t> input_rotate_steps;
    std::vector<int> input_rotate_units;
    std::vector<int> input_rotate_ranges;
    Duo seed_weight_subgrid_stride_;
    bool use_seed_weight_pack_ = false;
    bool canonical_seed_kernels_ = false;
    std::vector<double> generate_weight_values_for_indices(int ct_idx, int kernel_index, bool boundary) const;
    std::vector<double> seed_mask_values(int subgrid, int kernel_index, bool boundary) const;
    double weight_scale;

    // Cached values for on-demand generation
    int cached_input_block_size = 0;
    int cached_kernel_total_count = 0;
    int cached_total_block_size = 0;
};
