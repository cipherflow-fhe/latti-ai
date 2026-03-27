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
#include "util.h"
#include "../data_structs/feature.h"

class DensePackedLayer : public Layer {
public:
    DensePackedLayer(const ls::CkksParameter& param_in,
                     const Array<double, 2>& weight_in,
                     const Array<double, 1>& bias_in,
                     uint32_t pack_in,
                     uint32_t level_in,
                     int mark_in,
                     double residual_scale = 1.0);
    virtual void prepare_weight_0d_skip(uint32_t skip_0d);
    virtual void prepare_weight_0d_skip_lazy(uint32_t skip_0d);
    virtual void prepare_weight_for_2d_multiplexed(const Duo& input_shape_in,
                                                   const Duo& skip_in,
                                                   const Duo& invalid_fill_in = {1, 1});
    virtual void prepare_weight_for_2d_multiplexed_lazy(const Duo& input_shape_in,
                                                        const Duo& skip_in,
                                                        const Duo& invalid_fill_in = {1, 1});

    virtual Feature0DEncrypted run_0d_skip(ls::CkksContext& ctx, const Feature0DEncrypted& x);
    virtual Feature0DEncrypted run_2d_multiplexed(ls::CkksContext& ctx, const Feature0DEncrypted& x);
    Array<double, 1> plaintext_call(const Array<double, 1>& x, double multiplier = 1.0);

    std::vector<std::vector<ls::CkksPlaintextRingt>> weight_pt;
    std::vector<ls::CkksPlaintextRingt> bias_pt;

    // For lazy generation
    std::vector<std::vector<std::vector<double>>> weight_rearranged;
    std::vector<std::vector<double>> bias_rearranged;

    bool normal_dense = true;

    // Helper functions for prepare_weight_0d_lazy
    ls::CkksPlaintextRingt
    generate_weight_0d_pt_for_indices(ls::CkksContext& ctx, uint32_t packed_out_idx, uint32_t weight_idx) const;
    ls::CkksPlaintextRingt generate_bias_0d_pt_for_index(ls::CkksContext& ctx, uint32_t packed_out_idx) const;

    // Helper functions for prepare_weight_for_mult_pack_lazy
    ls::CkksPlaintextRingt generate_weight_pt_mult_pack_for_indices(ls::CkksContext& ctx,
                                                                    int packed_out_feature_idx,
                                                                    int n_block_input_idx) const;
    ls::CkksPlaintextRingt generate_bias_pt_mult_pack_for_index(ls::CkksContext& ctx, int packed_out_feature_idx) const;

protected:
    uint32_t n_out_feature;
    uint32_t n_in_feature;
    Array<double, 2> weight;
    Array<double, 1> bias;
    uint32_t n_channel_per_ct;
    uint32_t n_packed_in_feature;
    uint32_t n_packed_out_feature;
    int mark;
    double modified_scale;

    // Cached values for prepare_weight_for_mult_pack_lazy
    Duo special_input_shape = {0, 0};
    Duo special_skip = {0, 0};
    uint32_t input_shape_ct_mult[2] = {0, 0};
    int n_block_per_ct = 0;
    int n_block_input = 0;
    int N_half = 0;
    Duo special_invalid_fill = {1, 1};

    // 0D specific
    uint32_t skip = 0;
    uint32_t bsgs_bs = 0;
    uint32_t bsgs_gs = 0;

    // core function
    virtual std::vector<ls::CkksCiphertext> run_core_0d(ls::CkksContext& ctx, const std::vector<ls::CkksCiphertext>& x);
    virtual std::vector<ls::CkksCiphertext> run_core_mult_pack(ls::CkksContext& ctx,
                                                               const std::vector<ls::CkksCiphertext>& x);
};
