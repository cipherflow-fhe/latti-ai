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
#include "poly_relu_base.h"

class PolyRelu : public PolyReluBase {
public:
    PolyRelu(const CkksParameter& param_in,
             const Duo& input_shape_in,
             const int order_in,
             const Array<double, 2>& weight_in,
             const Duo& skip_in,
             uint32_t n_channel_per_ct_in,
             uint32_t level_in,
             const Duo& zero_skip_in = {1, 1},
             const Duo& block_expansion_in = {1, 1},
             bool is_ordinary_pack_in = false);

    ~PolyRelu() override;

    void prepare_weight();
    void prepare_weight_lazy();
    void prepare_weight_bsgs();
    void prepare_weight_bsgs_lazy();
    void prepare_weight_hornor();
    void prepare_weight_hornor_lazy();

    CkksPlaintextRingt generate_weight_pt_for_indices(CkksContext& ctx, int idx, int n_packed_out_channel_idx) const;
    CkksPlaintextRingt
    generate_weight_pt_for_non_absorb_indices(CkksContext& ctx, int idx, int n_packed_out_channel_idx) const;
    CkksPlaintextRingt
    generate_weight_pt_for_bsgs(CkksContext& ctx, int idx, int n_packed_out_channel_idx) const override;

    Feature2DEncrypted run(CkksContext& ctx, const Feature2DEncrypted& x);
    std::vector<CkksCiphertext> run_core(CkksContext& ctx, const std::vector<CkksCiphertext>& x);
    Feature2DEncrypted run_bsgs(CkksContext& ctx, const Feature2DEncrypted& x);
    Feature2DEncrypted run_horner(CkksContext& ctx, const Feature2DEncrypted& x);
    std::vector<CkksCiphertext> run_core_horner(CkksContext& ctx, const std::vector<CkksCiphertext>& x);
    Array<double, 3> run_plaintext_absorb_case(const Array<double, 3>& x);
    Array<double, 3> run_plaintext_for_non_absorb_case(const Array<double, 3>& x);

    Duo input_shape;
    Duo skip;
    int n_block_per_ct;
    Duo pre_skip;
    Duo block_expansion;
    Duo block_shape;
    Duo zero_skip;
    bool is_ordinary_pack;

private:
    int cached_skip_prod;
    int cached_n_packed_out_channel;
    int cached_total_block_size;
    map<int, double> cached_coeff_scale;
    map<int, int> cached_level_order;
};
