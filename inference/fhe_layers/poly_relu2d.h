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

class PolyRelu {
public:
    PolyRelu(const CkksParameter& param_in,
             const Duo& input_shape_in,
             const int order_in,
             const Array<double, 2>& weight_in,
             const Duo& skip_in,
             uint32_t n_channel_per_ct_in,
             uint32_t level_in,
             const Duo& upsample_factor_in = {1, 1},
             const Duo& block_expansion_in = {1, 1},
             bool is_ordinary_pack_in = false);

    ~PolyRelu();

    virtual void prepare_weight();
    virtual void prepare_weight_for_non_absorb_case();
    virtual void prepare_weight_lazy();
    virtual void prepare_weight_for_non_absorb_case_lazy();

    // Helper functions to generate weights on-demand (for lazy mode)
    CkksPlaintextRingt generate_weight_pt_for_indices(CkksContext& ctx, int idx, int n_packed_out_channel_idx) const;
    CkksPlaintextRingt
    generate_weight_pt_for_non_absorb_indices(CkksContext& ctx, int idx, int n_packed_out_channel_idx) const;

    virtual Feature2DEncrypted run(CkksContext& ctx, const Feature2DEncrypted& x);
    virtual Feature2DEncrypted run_for_non_absorb_case(CkksContext& ctx, const Feature2DEncrypted& x);
    std::vector<CkksCiphertext> run_core(CkksContext& ctx, const std::vector<CkksCiphertext>& x);
    std::vector<CkksCiphertext> run_core_for_non_absorb_case(CkksContext& ctx, const std::vector<CkksCiphertext>& x);
    virtual Array<double, 3> run_plaintext(const Array<double, 3>& x);
    virtual Array<double, 3> run_plaintext_for_non_absorb_case(const Array<double, 3>& x);

    CkksParameter param;
    Duo input_shape;
    Duo skip;
    Array<double, 2> weight;
    uint32_t n_channel_per_ct;
    uint32_t level;
    int order;
    int n_block_per_ct;
    Duo pre_skip;
    Duo block_expansion;
    Duo block_shape;
    Duo upsample_factor;
    vector<vector<CkksPlaintextRingt>> weight_pt;
    bool is_ordinary_pack;

private:
    // Cached values for on-demand generation
    int N;
    int cached_skip_prod;
    int cached_channel;
    int cached_n_packed_out_channel;
    int cached_total_block_size;
    map<int, double> cached_coeff_scale;  // For order==4 case
    map<int, int> cached_level_order;     // For order==4 case
};
