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
#include "../data_structs/feature.h"

class ParMultiplexedConv2DPackedLayer : public Conv2DLayer {
public:
    ParMultiplexedConv2DPackedLayer(const CkksParameter& param_in,
                                    const Duo& input_shape_in,
                                    const Array<double, 4>& weight_in,
                                    const Array<double, 1>& bias_in,
                                    const Duo& stride_in,
                                    const Duo& skip_in,
                                    uint32_t n_channel_per_ct_in,
                                    uint32_t level_in,
                                    double residual_scale = 1.0,
                                    const Duo& upsample_factor_in = {1, 1});

    ~ParMultiplexedConv2DPackedLayer();
    virtual void prepare_weight_for_reduct_rot();
    virtual void prepare_weight_for_post_skip_rotation();
    virtual void prepare_weight_for_post_skip_rotation_lazy();

    virtual Feature2DEncrypted run(CkksContext& ctx, const Feature2DEncrypted& x);
    virtual Feature2DEncrypted run_for_post_skip_rotation(CkksContext& ctx, const Feature2DEncrypted& x);
    virtual Feature2DEncrypted run_for_reduct_rot(CkksContext& ctx, const Feature2DEncrypted& x);
    virtual vector<double> select_tensor(int num) const;

    std::vector<std::vector<std::vector<CkksPlaintextRingt>>> weight_pt;
    std::vector<CkksPlaintextRingt> bias_pt;
    std::vector<std::vector<CkksPlaintextRingt>> mask_pt;

    vector<vector<double>> mask_channel;

    bool normal_conv = true;

    // Helper functions to generate weights/bias/mask on-demand
    CkksPlaintextRingt generate_weight_pt_for_indices(CkksContext& ctx, int ct_idx, int j, int k) const;
    CkksPlaintextRingt generate_bias_pt_for_index(CkksContext& ctx, int bpt_idx) const;
    CkksPlaintextRingt generate_mask_pt_for_indices(CkksContext& ctx, int ct_idx, int i) const;

private:
    std::vector<CkksCiphertext> run_core(CkksContext& ctx, const std::vector<CkksCiphertext>& x);
    std::vector<CkksCiphertext> run_core_for_post_skip_rotation(CkksContext& ctx, const std::vector<CkksCiphertext>& x);
    std::vector<CkksCiphertext> run_core_for_reduct_rot(CkksContext& ctx, const std::vector<CkksCiphertext>& x);

    int N;
    uint32_t n_channel_per_ct;
    uint32_t n_packed_in_channel;
    uint32_t n_packed_out_channel;
    uint32_t n_block_per_ct;
    uint32_t level;
    int bias_level_down = 0;
    double weight_scale;
    Duo upsample_factor;
    Duo zero_inserted_skip;

    // Cached values for on-demand generation
    uint32_t cached_input_shape_ct[2] = {0, 0};
    int cached_input_block_size = 0;
    int cached_kernel_size = 0;
    int cached_total_skip = 0;
    // For bias generation
    Duo cached_bias_skip;
    int cached_skip_prod = 0;
    int cached_bias_n_channel_per_ct = 0;
    int cached_total_block_size = 0;
};

CkksCiphertext sum_slot(CkksContext& ctx, CkksCiphertext& x, uint32_t m, uint32_t p);
