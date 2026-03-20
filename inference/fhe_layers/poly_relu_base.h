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
    int depth;
    int level;
    double scale;
    int decomp_a;
    int decomp_b;
    bool computed;
};

class PolyReluBase {
public:
    PolyReluBase(const CkksParameter& param_in,
                 const Array<double, 2>& weight_in,
                 uint32_t n_channel_per_ct_in,
                 uint32_t level_in,
                 int order_in);

    virtual ~PolyReluBase();

    CkksParameter param;
    Array<double, 2> weight;
    uint32_t n_channel_per_ct;
    uint32_t level;
    int order;
    vector<vector<CkksPlaintextRingt>> weight_pt;

    int baby_steps = 0;
    int bsgs_giant_steps = 0;

    static int compute_bsgs_level_cost(int order);

    virtual CkksPlaintextRingt generate_weight_pt_for_bsgs(CkksContext& ctx, int idx, int ct_idx) const = 0;

protected:
    int N;
    int cached_channel;

    // BSGS infrastructure
    void init_bsgs();
    void compute_all_powers();
    void compute_power(int n);
    PowerInfo get_power_info(int n) const;
    void analyze_depth_distribution() const;
    void determine_required_powers_bsgs();
    void compute_coefficient_scales_bsgs(std::map<int, double>& coeff_scale, std::map<int, int>& level_order);

    std::vector<CkksCiphertext> run_core_bsgs(CkksContext& ctx, const std::vector<CkksCiphertext>& x);

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

class PolyRelu0D : public PolyReluBase {
public:
    PolyRelu0D(const CkksParameter& param_in,
               const Array<double, 2>& weight_in,
               uint32_t n_channel_per_ct_in,
               uint32_t level_in,
               int order_in,
               int skip_in);

    ~PolyRelu0D() override;

    void prepare_weight();
    void prepare_weight_lazy();

    CkksPlaintextRingt generate_weight_pt_for_bsgs(CkksContext& ctx, int idx, int ct_idx) const override;

    Feature0DEncrypted run(CkksContext& ctx, const Feature0DEncrypted& x);
    Array<double, 1> run_plaintext(const Array<double, 1>& x);

    int ciphertext_skip;
};
