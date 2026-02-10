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
#include "common.h"
#include "../data_structs/feature.h"

using namespace cxx_sdk_v2;

class DensePackedLayer {
public:
    DensePackedLayer(const CkksParameter& param_in,
                     const Duo& input_shape_in,
                     const Duo& skip_in,
                     const Array<double, 2>& weight_in,
                     const Array<double, 1>& bias_in,
                     uint32_t pack_in,
                     uint32_t level_in,
                     int mark_in,
                     double residual_scale = 1.0);
    ~DensePackedLayer();
    virtual void prepare_weight1();
    virtual void prepare_weight1_lazy();
    virtual void prepare_weight_for_mult_pack();
    virtual void prepare_weight_for_mult_pack_lazy();
    virtual std::vector<CkksCiphertext> call(CkksContext& ctx, const std::vector<CkksCiphertext>& x);
    virtual Feature0DEncrypted call(CkksContext& ctx, const Feature2DEncrypted& x);
    virtual Feature0DEncrypted call(CkksContext& ctx, const Feature0DEncrypted& x);
    virtual std::vector<CkksCiphertext> run_core_mult_pack(CkksContext& ctx, const std::vector<CkksCiphertext>& x);
    virtual Feature0DEncrypted run_mult_park(CkksContext& ctx, const Feature2DEncrypted& x);
    virtual Feature0DEncrypted run_mult_park(CkksContext& ctx, const Feature0DEncrypted& x);
    Array<double, 1> plaintext_call(const Array<double, 1>& x, double multiplier = 1.0);
    std::vector<std::vector<CkksPlaintextRingt>> weight_pt;
    std::vector<CkksPlaintextRingt> bias_pt;

    // For lazy generation
    std::vector<std::vector<std::vector<double>>> weight_rearranged;
    std::vector<std::vector<double>> bias_rearranged;

    std::string project_path;

    CkksParameter param;
    bool normal_dense = true;
    double ckks_scale_in = DEFAULT_SCALE;
    double ckks_scale_out = DEFAULT_SCALE;

    // Helper functions for prepare_weight1_lazy
    CkksPlaintextRingt
    generate_weight1_pt_for_indices(CkksContext& ctx, int packed_out_feature_idx, int in_feature_idx) const;
    CkksPlaintextRingt generate_bias1_pt_for_index(CkksContext& ctx, int packed_out_feature_idx) const;

    // Helper functions for prepare_weight_for_mult_pack_lazy
    CkksPlaintextRingt
    generate_weight_pt_mult_pack_for_indices(CkksContext& ctx, int packed_out_feature_idx, int n_block_input_idx) const;
    CkksPlaintextRingt generate_bias_pt_mult_pack_for_index(CkksContext& ctx, int packed_out_feature_idx) const;

protected:
    uint32_t input_shape[2];
    uint32_t skip[2];
    uint32_t n_out_feature;
    uint32_t n_in_feature;
    Array<double, 2> weight;
    Array<double, 1> bias;
    uint32_t pack;
    uint32_t n_packed_in_feature;
    uint32_t n_packed_out_feature;
    uint32_t level;
    int mark;
    double modified_scale;
    // Cached values for prepare_weight1_lazy
    uint32_t cached_input_shape_ct_1[2] = {0, 0};
    int cached_per_channel_num = 0;

    // Cached values for prepare_weight_for_mult_pack_lazy
    uint32_t cached_input_shape_ct_mult[2] = {0, 0};
    int cached_n_num_pre_ct = 0;
    int cached_n_block_input = 0;
    int cached_N_half = 0;
};
