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

class MultiplexedConv2DPackedLayer : public Conv2DLayer {
public:
    MultiplexedConv2DPackedLayer(const ls::CkksParameter& param_in,
                                 const Duo& input_shape_in,
                                 Array<double, 4>&& weight_in,
                                 Array<double, 1>&& bias_in,
                                 const Duo& stride_in,
                                 const Duo& skip_in,
                                 uint32_t n_channel_per_ct_in,
                                 uint32_t level_in,
                                 double residual_scale = 1.0,
                                 const Duo& external_upsample_factor_in = {1, 1},
                                 const Duo& seed_weight_subgrid_stride_in = {1, 1},
                                 bool joint_seed_requests = false);

    void prepare_weight() override {
        prepare_weight_for_post_skip_rotation();
    }
    void prepare_weight_lazy() override {
        prepare_weight_for_post_skip_rotation_lazy();
    }
    // virtual void prepare_weight_for_reduct_rot();
    virtual void prepare_weight_for_post_skip_rotation();
    virtual void prepare_weight_for_post_skip_rotation_lazy();

    virtual Feature2DEncrypted run(ls::CkksContext& ctx, const Feature2DEncrypted& x);
    virtual Feature2DEncrypted run_for_post_skip_rotation(ls::CkksContext& ctx, const Feature2DEncrypted& x);
    virtual Feature2DEncrypted run_for_reduct_rot(ls::CkksContext& ctx, const Feature2DEncrypted& x);
    virtual std::vector<double> select_tensor(int num) const;
    static std::vector<double> select_tensor_for_config(const Duo& input_shape,
                                                        const Duo& skip,
                                                        const Duo& stride,
                                                        uint32_t n_channel_per_ct,
                                                        const Duo& external_upsample_factor,
                                                        int num);
    static std::vector<double> seed_mask_tensor_for_config(const Duo& input_shape, const Duo& skip,
                                                           const Duo& stride, uint32_t n_channel_per_ct,
                                                           const Duo& external_upsample_factor, uint32_t a,
                                                           uint32_t b, int subgrid, int kernel_index,
                                                           const Duo& kernel_shape,
                                                           bool kernel_boundary);

    std::vector<std::vector<std::vector<ls::CkksPlaintextRingt>>> weight_pt;
    std::vector<ls::CkksPlaintextRingt> bias_pt;
    // post_skip_rotation: block-i masks (one per block position in output CT),
    // shared across all ct_idx. Size = min(n_block_per_ct, n_out_channel_).
    std::vector<ls::CkksPlaintextRingt> mask_pt;

    std::vector<std::vector<double>> mask_channel;

    bool normal_conv = true;

    // Helper functions to generate weights/bias/mask on-demand (post_skip_rotation path)
    ls::CkksPlaintextRingt generate_weight_pt_for_indices(ls::CkksContext& ctx, int ct_idx, int j, int k) const;
    ls::CkksPlaintextRingt generate_weight_pt_for_seed_indices(ls::CkksContext& ctx, int ct_idx, int j, int k) const;
    ls::CkksPlaintextRingt generate_bias_pt_for_index(ls::CkksContext& ctx, int bpt_idx) const;
    ls::CkksPlaintextRingt generate_mask_pt_for_indices(ls::CkksContext& ctx, int i) const;

    // Helper functions to generate weights/bias/mask on-demand (reduct_rot path)
    ls::CkksPlaintextRingt
    generate_weight_pt_for_indices_reduct_rot(ls::CkksContext& ctx, int ct_idx, int j, int k) const;
    ls::CkksPlaintextRingt generate_bias_pt_for_index_reduct_rot(ls::CkksContext& ctx, int bpt_idx) const;
    ls::CkksPlaintextRingt generate_mask_pt_for_indices_reduct_rot(ls::CkksContext& ctx, int ct_idx, int i) const;

    bool use_reduct_rot = false;

    // Selected from the public layer shape. This changes only the local
    // input-shift/weight decomposition; Feature2DEncrypted stays canonical.
    bool use_periodic_pack() const { return use_periodic_pack_; }
    uint32_t periodic_block_count() const { return periodic_block_count_; }
    bool use_seed_weight_pack() const { return use_seed_weight_pack_; }
    bool joint_seed_weights() const { return use_seed_weight_pack_ && joint_seed_requests_; }
    uint32_t seed_weight_level_cost() const {
        return use_seed_weight_pack_ ? (kernel_shape_ != Duo{1, 1} ? 2 : 1) : 0;
    }
    uint32_t seed_weight_count() const {
        return div_ceil(div_ceil(n_out_channel_, n_block_per_ct) * legacy_weight_input_count() * prod(kernel_shape_),
                        prod(seed_weight_subgrid_stride_));
    }

private:
    std::vector<ls::CkksCiphertext> run_core(ls::CkksContext& ctx, const std::vector<ls::CkksCiphertext>& x);
    std::vector<ls::CkksCiphertext> run_core_for_post_skip_rotation(ls::CkksContext& ctx,
                                                                    const std::vector<ls::CkksCiphertext>& x);
    std::vector<ls::CkksCiphertext> run_core_for_reduct_rot(ls::CkksContext& ctx,
                                                            const std::vector<ls::CkksCiphertext>& x);
    std::vector<ls::CkksCiphertext> expand_periodic_inputs(
        ls::CkksContext& ctx, const std::vector<ls::CkksCiphertext>& x) const;
    uint32_t input_block_count() const;
    uint32_t legacy_weight_input_count() const;
    uint32_t weight_input_count() const;

    int N;
    uint32_t n_channel_per_ct;
    uint32_t n_packed_in_channel;
    uint32_t n_packed_out_channel;
    uint32_t n_block_per_ct;
    uint32_t periodic_channel_count_ = 0;
    uint32_t periodic_block_count_ = 0;
    bool use_periodic_pack_ = false;
    Duo seed_weight_subgrid_stride_ = {1, 1};
    bool use_seed_weight_pack_ = false;
    bool joint_seed_requests_ = false;
    int bias_level_down = 0;
    bool need_repack_;
    double weight_scale;
    Duo skip_;
    Duo external_upsample_factor;
    Duo zero_inserted_skip;
    std::vector<std::vector<double>> kernel_masks_;
    std::vector<int> input_rotate_units_;

    // Cached values for on-demand generation
    Duo cached_input_shape_ct = {0, 0};
    int cached_input_block_size = 0;
    int cached_kernel_size = 0;
    int cached_total_skip = 0;
    // For bias generation
    Duo cached_bias_skip;
    int cached_skip_prod = 0;
    int cached_bias_n_channel_per_ct = 0;
    int cached_total_block_size = 0;
    std::vector<double> generate_weight_values_for_indices(int ct_idx, int j, int k,
                                                            bool apply_kernel_boundary) const;
    // For reduct_rot
    uint32_t cached_skip_out_prod = 0;
};

ls::CkksCiphertext sum_slot(ls::CkksContext& ctx, ls::CkksCiphertext& x, uint32_t m, uint32_t p);
