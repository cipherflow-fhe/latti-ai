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
#include <map>
#include <set>

struct PowerInfo {
    int depth;     // depth
    int level;     // level
    double scale;  // scale
    int decomp_a;  // x^n = x^a * x^b
    int decomp_b;
    bool computed;  // is compute
};

class PolyRelu {
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

    ~PolyRelu();

    virtual void prepare_weight();
    virtual void prepare_weight_lazy();
    virtual void prepare_weight_bsgs();
    virtual void prepare_weight_bsgs_lazy();
    virtual void prepare_weight_hornor();
    virtual void prepare_weight_hornor_lazy();
    virtual void prepare_weight_for_feature0d();
    virtual void prepare_weight_for_feature0d_lazy();

    // Helper functions to generate weights on-demand (for lazy mode)
    CkksPlaintextRingt generate_weight_pt_for_indices(CkksContext& ctx, int idx, int n_packed_out_channel_idx) const;
    CkksPlaintextRingt
    generate_weight_pt_for_non_absorb_indices(CkksContext& ctx, int idx, int n_packed_out_channel_idx) const;
    CkksPlaintextRingt
    generate_weight_pt_for_bsgs_indices(CkksContext& ctx, int idx, int n_packed_out_channel_idx) const;
    CkksPlaintextRingt generate_weight_pt_for_feature0d_indices(CkksContext& ctx, int idx, int ct_idx) const;

    virtual Feature2DEncrypted run(CkksContext& ctx, const Feature2DEncrypted& x);
    std::vector<CkksCiphertext> run_core(CkksContext& ctx, const std::vector<CkksCiphertext>& x);
    virtual Feature2DEncrypted run_bsgs(CkksContext& ctx, const Feature2DEncrypted& x);
    virtual Feature0DEncrypted run_bsgs(CkksContext& ctx, const Feature0DEncrypted& x);
    std::vector<CkksCiphertext> run_core_bsgs(CkksContext& ctx, const std::vector<CkksCiphertext>& x);
    virtual Feature2DEncrypted run_horner(CkksContext& ctx, const Feature2DEncrypted& x);
    std::vector<CkksCiphertext> run_core_horner(CkksContext& ctx, const std::vector<CkksCiphertext>& x);
    virtual Array<double, 3> run_plaintext_absorb_case(const Array<double, 3>& x);
    virtual Array<double, 3> run_plaintext_for_non_absorb_case(const Array<double, 3>& x);
    virtual Array<double, 1> run_plaintext_for_non_absorb_case_0d(const Array<double, 1>& x);

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
    Duo zero_skip;
    vector<vector<CkksPlaintextRingt>> weight_pt;
    bool is_ordinary_pack;
    bool is_feature_0d = false;

    // BSGS parameters (public for inspection)
    int baby_steps = 0;
    int bsgs_giant_steps = 0;

    static int compute_bsgs_level_cost(int order);

private:
    // Cached values for on-demand generation
    int N;
    int cached_skip_prod;
    int cached_channel;
    int cached_n_packed_out_channel;
    int cached_total_block_size;
    map<int, double> cached_coeff_scale;  // For order==4 case
    map<int, int> cached_level_order;     // For order==4 case

    // BSGS private methods
    void init_bsgs();
    void analyze_all_powers_bsgs();
    void determine_required_powers_bsgs();
    void compute_coefficient_scales_bsgs(std::map<int, double>& coeff_scale, std::map<int, int>& level_order);

    void compute_all_powers();
    void compute_power(int n);
    PowerInfo get_power_info(int n) const;
    void analyze_depth_distribution() const;

    // BSGS private members
    std::vector<double> modulus;
    std::map<int, PowerInfo> powers;
    std::set<int> required_powers;
    std::vector<double> baby_poly_output_scale;
    std::vector<int> baby_poly_output_level;
    int bsgs_output_level = 0;
    std::map<int, double> cached_bsgs_coeff_scale;
    std::map<int, int> cached_bsgs_level_order;
    bool bsgs_initialized = false;
};
