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

class MultiplexedConv2DPackedLayerDepthwise : public Conv2DLayer {
public:
    MultiplexedConv2DPackedLayerDepthwise(const ls::CkksParameter& param_in,
                                          const Duo& input_shape_in,
                                          Array<double, 4>&& weight_in,
                                          Array<double, 1>&& bias_in,
                                          const Duo& stride_in,
                                          const Duo& skip_in,
                                          uint32_t n_channel_per_ct_in,
                                          uint32_t level_in,
                                          double residual_scale = 1.0,
                                          const Duo& seed_weight_subgrid_stride_in = {1, 1},
                                          bool joint_seed_requests = false);

    void prepare_weight() override;
    void prepare_weight_lazy() override;

    virtual Feature2DEncrypted run(ls::CkksContext& ctx, const Feature2DEncrypted& x);
    virtual std::vector<double> select_tensor(int num) const;

    std::vector<std::vector<ls::CkksPlaintextRingt>> weight_pt;
    std::vector<ls::CkksPlaintextRingt> bias_pt;
    std::vector<ls::CkksPlaintextRingt> mask_pt;
    bool normal_conv = true;

    // Helper functions to generate weights/bias/mask on-demand
    ls::CkksPlaintextRingt
    generate_weight_pt_for_indices(ls::CkksContext& ctx, int n_packed_out_channel_idx, int kernel_idx) const;
    ls::CkksPlaintextRingt generate_bias_pt_for_index(ls::CkksContext& ctx, int bpt_idx) const;
    ls::CkksPlaintextRingt generate_mask_pt_for_indices(ls::CkksContext& ctx, int ct_idx, int i) const;
    static std::vector<double> mask_tensor_for_config(const Duo& input_shape, const Duo& skip,
                                                     const Duo& stride, uint32_t channels_per_ct,
                                                     uint32_t slots, int ct_idx, int local_idx);

    // Getter for n_channel_per_ct (needed for CustomData executors)
    uint32_t get_n_channel_per_ct() const {
        return n_channel_per_ct;
    }

    ls::CkksPlaintextRingt generate_weight_pt_for_seed_indices(ls::CkksContext& ctx, int ct_idx, int seed_kernel) const;
    ls::CkksPlaintextRingt generate_mask_pt_for_seed_indices(ls::CkksContext& ctx, int subgrid,
                                                            int kernel_index, bool boundary) const;
    bool use_seed_weight_pack() const { return use_seed_weight_pack_; }
    bool joint_seed_weights() const { return use_seed_weight_pack_ && joint_seed_requests_; }
    uint32_t seed_weight_count() const {
        return joint_seed_weights()
            ? div_ceil(n_packed_in_channel * prod(kernel_shape_), prod(seed_weight_subgrid_stride_))
            : n_packed_in_channel * seed_weight_kernel_count();
    }
    uint32_t seed_weight_kernel_count() const {
        const uint32_t count = prod(kernel_shape_);
        return use_seed_weight_pack_ ? div_ceil(count, prod(seed_weight_subgrid_stride_)) : count;
    }
    uint32_t seed_weight_level_cost() const {
        return use_seed_weight_pack_ ? (kernel_shape_ != Duo{1, 1} ? 2 : 1) : 0;
    }

private:
    static std::vector<double> select_tensor_for_config(const Duo& input_shape, const Duo& skip,
                                                       const Duo& stride, uint32_t channels_per_ct, int num);
    std::vector<ls::CkksCiphertext> run_core(ls::CkksContext& ctx, const std::vector<ls::CkksCiphertext>& x);
    uint32_t n_channel_per_ct;
    uint32_t n_packed_in_channel;
    uint32_t n_packed_out_channel;
    uint32_t n_block_per_ct;
    bool need_repack_;
    Duo seed_weight_subgrid_stride_;
    bool use_seed_weight_pack_ = false;
    bool joint_seed_requests_ = false;
    std::vector<double> generate_weight_values_for_indices(int ct_idx, int kernel_index, bool boundary) const;
    std::vector<double> seed_mask_values(int subgrid, int kernel_index, bool boundary) const;
    double weight_scale;
    Duo skip_;
    std::vector<std::vector<double>> kernel_masks_;
    std::vector<int> input_rotate_units_;

    // Cached values for on-demand generation
    Duo cached_input_shape_ct = {0, 0};
    int cached_input_block_size = 0;
    int cached_kernel_size = 0;
    int cached_skip_prod = 0;
    // For bias generation
    Duo cached_bias_skip;
    int cached_bias_n_channel_per_ct = 0;
    int cached_total_block_size = 0;
};
