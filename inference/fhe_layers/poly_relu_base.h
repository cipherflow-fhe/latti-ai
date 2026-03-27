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

class PolyReluBase : public Layer {
public:
    PolyReluBase(const ls::CkksParameter& param_in,
                 const Array<double, 2>& weight_in,
                 uint32_t n_channel_per_ct_in,
                 uint32_t level_in,
                 int order_in);

    Array<double, 2> weight;
    uint32_t n_channel_per_ct;
    int order;
    std::vector<std::vector<ls::CkksPlaintextRingt>> weight_pt;

    int baby_steps = 0;
    int bsgs_giant_steps = 0;

    static int compute_bsgs_level_cost(int order);

    virtual ls::CkksPlaintextRingt generate_weight_pt_for_bsgs(ls::CkksContext& ctx, int idx, int ct_idx) const = 0;

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

    std::vector<ls::CkksCiphertext> run_core_bsgs(ls::CkksContext& ctx, const std::vector<ls::CkksCiphertext>& x);

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
    // n_channel_per_ct is derived from skip_in: N/2 / skip_in
    // This matches both encoding cases:
    //   Case 1 (Feature0DEncrypted::pack): n_channel_per_ct = N/2 / skip
    //   Case 2 (ReshapeLayer):             n_channel_per_ct = N/2 / (shape[0]*skip2d[0]*shape[1]*skip2d[1])
    PolyRelu0D(const ls::CkksParameter& param_in,
               const Array<double, 2>& weight_in,
               uint32_t level_in,
               int order_in,
               int skip_in);

    // Mode 1: direct 0D pack — channel ch at slot ch*ciphertext_skip
    void prepare_weight_0d_skip();
    void prepare_weight_0d_skip_lazy();

    // Mode 2: from reshape of 2D with shape>1 — mirrors DensePackedLayer::prepare_weight_for_multiplexed
    // input_shape_in: [H, W] spatial dims of the original 2D feature
    // skip_in:        [s0, s1] skip of the original 2D feature
    void prepare_weight_2d_multiplexed(const Duo& input_shape_in, const Duo& skip_in);
    void prepare_weight_2d_multiplexed_lazy(const Duo& input_shape_in, const Duo& skip_in);

    ls::CkksPlaintextRingt generate_weight_pt_for_bsgs(ls::CkksContext& ctx, int idx, int ct_idx) const override;

    Feature0DEncrypted run(ls::CkksContext& ctx, const Feature0DEncrypted& x);
    Array<double, 1> run_plaintext(const Array<double, 1>& x);

    int ciphertext_skip;
    bool is_multiplexed = false;

private:
    // Helper for Mode 1 lazy generation
    ls::CkksPlaintextRingt generate_weight_pt_skip0d(ls::CkksContext& ctx, int idx, int ct_idx) const;

    // Helper for Mode 2 lazy generation
    ls::CkksPlaintextRingt generate_weight_pt_multiplexed(ls::CkksContext& ctx, int idx, int ct_idx) const;

    // Cached values for Mode 2 (multiplexed)
    Duo special_input_shape = {0, 0};  // [H, W]
    Duo special_skip = {1, 1};         // [s0, s1]
    int block_size = 0;                // H*s0 * W*s1 = ciphertext_skip
    int n_channel_per_ct_mux = 0;
};
