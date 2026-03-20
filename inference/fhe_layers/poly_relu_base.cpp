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

#include "poly_relu_base.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <functional>

// ======================== PolyReluBase ========================

PolyReluBase::PolyReluBase(const CkksParameter& param_in,
                           const Array<double, 2>& weight_in,
                           uint32_t n_channel_per_ct_in,
                           uint32_t level_in,
                           int order_in)
    : param(param_in.copy()), weight(weight_in.copy()) {
    order = order_in;
    level = level_in;
    n_channel_per_ct = n_channel_per_ct_in;
    N = param_in.get_n();
    cached_channel = weight.get_shape()[1];
}

PolyReluBase::~PolyReluBase() {}

void PolyReluBase::compute_all_powers() {
    powers[1] = {0, (int)level, param.get_default_scale(), 0, 0, true};
    for (int n = 2; n <= order; n++) {
        compute_power(n);
    }
}

void PolyReluBase::compute_power(int n) {
    if (powers.find(n) != powers.end() && powers[n].computed) {
        return;
    }

    int best_depth = std::numeric_limits<int>::max();
    int best_a = 1, best_b = n - 1;

    for (int a = 1; a <= n / 2; a++) {
        int b = n - a;
        if (powers.find(a) == powers.end())
            compute_power(a);
        if (powers.find(b) == powers.end())
            compute_power(b);

        int depth = std::max(powers[a].depth, powers[b].depth) + 1;

        if (depth < best_depth) {
            best_depth = depth;
            best_a = a;
            best_b = b;
        } else if (depth == best_depth && std::abs(a - b) < std::abs(best_a - best_b)) {
            best_a = a;
            best_b = b;
        }
    }

    int result_level = std::min(powers[best_a].level, powers[best_b].level) - 1;
    double result_scale = powers[best_a].scale * powers[best_b].scale;

    if (result_level >= 0 && result_level + 1 < (int)modulus.size()) {
        result_scale = result_scale / modulus[result_level + 1];
    }

    powers[n] = {best_depth, result_level, result_scale, best_a, best_b, true};
}

PowerInfo PolyReluBase::get_power_info(int n) const {
    auto it = powers.find(n);
    if (it != powers.end()) {
        return it->second;
    }
    return {-1, -1, 0.0, 0, 0, false};
}

void PolyReluBase::analyze_depth_distribution() const {
    std::map<int, std::vector<int>> depth_groups;
    for (int i = 1; i <= order; i++) {
        auto it = powers.find(i);
        if (it != powers.end() && it->second.computed) {
            depth_groups[it->second.depth].push_back(i);
        }
    }

    std::cout << "\n=== analyse depth ===" << std::endl;
    for (const auto& [depth, powers_list] : depth_groups) {
        std::cout << "depth " << depth << ": ";
        for (int p : powers_list) {
            std::cout << "x^" << p << " ";
        }
        std::cout << std::endl;
    }
}

int PolyReluBase::compute_bsgs_level_cost(int order) {
    if (order <= 1)
        return 1;
    int baby_steps = (int)ceil(sqrt(order + 1));
    int giant_steps = (int)ceil((double)(order + 1) / baby_steps);

    struct PInfo {
        int depth, a, b;
    };
    std::map<int, PInfo> pinfo;
    pinfo[1] = {0, 0, 0};
    for (int n = 2; n <= order; n++) {
        int best_d = std::numeric_limits<int>::max(), best_a = 1, best_b = n - 1;
        for (int a = 1; a <= n / 2; a++) {
            int b = n - a;
            int d = std::max(pinfo[a].depth, pinfo[b].depth) + 1;
            if (d < best_d || (d == best_d && std::abs(a - b) < std::abs(best_a - best_b))) {
                best_d = d;
                best_a = a;
                best_b = b;
            }
        }
        pinfo[n] = {best_d, best_a, best_b};
    }

    std::set<int> required, to_compute;
    for (int i = 1; i <= baby_steps; i++) {
        required.insert(i);
        to_compute.insert(i);
    }
    for (int g = 1; g < giant_steps; g++) {
        int gp = g * baby_steps;
        if (gp <= order) {
            required.insert(gp);
            to_compute.insert(gp);
        }
    }
    std::function<void(int)> add_deps = [&](int n) {
        if (n <= 1)
            return;
        if (pinfo[n].a > 1) {
            to_compute.insert(pinfo[n].a);
            add_deps(pinfo[n].a);
        }
        if (pinfo[n].b > 1) {
            to_compute.insert(pinfo[n].b);
            add_deps(pinfo[n].b);
        }
    };
    for (int p : std::set<int>(required))
        add_deps(p);

    std::map<int, int> pd;
    pd[1] = 0;
    for (int n : to_compute) {
        if (n <= 1)
            continue;
        pd[n] = std::max(pd[pinfo[n].a], pd[pinfo[n].b]) + 1;
    }
    int max_d = 0;
    for (int p : required)
        max_d = std::max(max_d, pd[p]);
    return max_d + 1;
}

void PolyReluBase::init_bsgs() {
    if (bsgs_initialized)
        return;

    baby_steps = (int)ceil(sqrt(order + 1));
    bsgs_giant_steps = (int)ceil((double)(order + 1) / baby_steps);

    modulus.clear();
    for (int i = 0; i <= (int)level; i++) {
        modulus.push_back(param.get_q(i));
    }
    powers.clear();
    compute_all_powers();

    determine_required_powers_bsgs();
    compute_coefficient_scales_bsgs(cached_bsgs_coeff_scale, cached_bsgs_level_order);

    bsgs_initialized = true;
}

void PolyReluBase::determine_required_powers_bsgs() {
    required_powers.clear();

    for (int i = 1; i <= baby_steps; i++) {
        required_powers.insert(i);
    }

    for (int g = 1; g < bsgs_giant_steps; g++) {
        int giant_power = g * baby_steps;
        if (giant_power <= order) {
            required_powers.insert(giant_power);
        }
    }
}

void PolyReluBase::compute_coefficient_scales_bsgs(std::map<int, double>& coeff_scale,
                                                   std::map<int, int>& level_order) {
    double S = param.get_default_scale();

    int max_depth = 0, max_power_level = level;
    for (int p : required_powers) {
        PowerInfo info = get_power_info(p);
        if (info.depth > max_depth) {
            max_depth = info.depth;
            max_power_level = info.level;
        }
    }
    bsgs_output_level = max_power_level - 1;

    baby_poly_output_scale.resize(bsgs_giant_steps);
    baby_poly_output_level.resize(bsgs_giant_steps);

    for (int g = 0; g < bsgs_giant_steps; g++) {
        if (g == 0) {
            baby_poly_output_scale[g] = S;
            baby_poly_output_level[g] = bsgs_output_level;
        } else {
            int giant_power = g * baby_steps;
            if (giant_power > order)
                break;

            PowerInfo gp_info = get_power_info(giant_power);
            int level_mult = bsgs_output_level + 1;
            baby_poly_output_level[g] = level_mult;
            baby_poly_output_scale[g] = S * param.get_q(level_mult) / gp_info.scale;
        }

        int start_idx = g * baby_steps;
        int end_idx = std::min(start_idx + baby_steps - 1, order);

        double target_scale = baby_poly_output_scale[g];
        int target_level = baby_poly_output_level[g];

        for (int idx = start_idx; idx <= end_idx; idx++) {
            int baby_step = idx - start_idx;

            if (baby_step == 0) {
                level_order[idx] = target_level;
                coeff_scale[idx] = target_scale;
            } else {
                PowerInfo x_info = get_power_info(baby_step);
                coeff_scale[idx] = target_scale * param.get_q(target_level + 1) / x_info.scale;
                level_order[idx] = target_level + 1;
            }
        }
    }
}

std::vector<CkksCiphertext> PolyReluBase::run_core_bsgs(CkksContext& ctx, const std::vector<CkksCiphertext>& x) {
    std::vector<CkksCiphertext> result(x.size());

    if (order <= 0) {
        throw std::runtime_error("Order must be at least 1");
    }

    parallel_for(x.size(), th_nums, ctx, [&](CkksContext& ctx_copy, int x_idx) {
        // 1. Compute all required powers using optimal decomposition
        std::map<int, CkksCiphertext> x_powers;
        x_powers[1] = x[x_idx].copy();
        if (x_powers[1].is_empty()) {
            throw std::runtime_error("BSGS: input x[" + std::to_string(x_idx) + "] has invalid handle");
        }

        std::set<int> powers_to_compute;
        for (int p : required_powers) {
            powers_to_compute.insert(p);
            std::function<void(int)> add_dependencies = [&](int n) {
                if (n <= 1)
                    return;
                PowerInfo info = get_power_info(n);
                if (info.decomp_a > 1) {
                    powers_to_compute.insert(info.decomp_a);
                    add_dependencies(info.decomp_a);
                }
                if (info.decomp_b > 1) {
                    powers_to_compute.insert(info.decomp_b);
                    add_dependencies(info.decomp_b);
                }
            };
            add_dependencies(p);
        }

        for (int i = 2; i <= order; i++) {
            if (powers_to_compute.find(i) == powers_to_compute.end()) {
                continue;
            }

            PowerInfo info = get_power_info(i);
            int a = info.decomp_a;
            int b = info.decomp_b;

            auto x_a = x_powers[a].copy();
            auto x_b = x_powers[b].copy();

            int target_level = std::min(x_a.get_level(), x_b.get_level());
            while (x_a.get_level() > target_level) {
                x_a = ctx_copy.drop_level(x_a);
            }
            while (x_b.get_level() > target_level) {
                x_b = ctx_copy.drop_level(x_b);
            }

            x_powers[i] = ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(x_a, x_b)),
                                           ctx_copy.get_parameter().get_default_scale());
            if (x_powers[i].is_empty()) {
                throw std::runtime_error("BSGS: x_powers[" + std::to_string(i) + "] is empty after computation");
            }
        }

        // 2. BSGS: P(x) = P0(x) + P1(x)*x^baby + P2(x)*x^(2*baby) + ...
        std::vector<CkksCiphertext> baby_polys(bsgs_giant_steps);
        std::vector<bool> baby_poly_initialized(bsgs_giant_steps, false);
        std::vector<bool> baby_poly_has_terms(bsgs_giant_steps, false);

        for (int g = 0; g < bsgs_giant_steps; g++) {
            int target_level = baby_poly_output_level[g];
            double target_scale = baby_poly_output_scale[g];

            for (int b = 0; b < baby_steps; b++) {
                int idx = g * baby_steps + b;
                if (idx > order)
                    break;

                if (b == 0) {
                    continue;
                } else {
                    baby_poly_has_terms[g] = true;

                    auto x_copy = x_powers[b].copy();
                    while (x_copy.get_level() > target_level + 1) {
                        x_copy = ctx_copy.drop_level(x_copy);
                    }

                    CkksCiphertext term;
                    if (weight_pt.empty()) {
                        auto coeff_pt_rt = generate_weight_pt_for_bsgs(ctx_copy, idx, x_idx);
                        auto coeff_pt = ctx_copy.ringt_to_mul(coeff_pt_rt, x_copy.get_level());
                        term = ctx_copy.rescale(ctx_copy.mult_plain_mul(x_copy, coeff_pt), target_scale);
                    } else {
                        auto coeff_pt = ctx_copy.ringt_to_mul(weight_pt[idx][x_idx], x_copy.get_level());
                        term = ctx_copy.rescale(ctx_copy.mult_plain_mul(x_copy, coeff_pt), target_scale);
                    }

                    if (!baby_poly_initialized[g]) {
                        baby_polys[g] = term.copy();
                        baby_poly_initialized[g] = true;
                    } else {
                        if (baby_polys[g].is_empty() || term.is_empty()) {
                            throw std::runtime_error("BSGS baby_poly add: g=" + std::to_string(g) +
                                                     " b_empty=" + std::to_string(baby_polys[g].is_empty()) +
                                                     " t_empty=" + std::to_string(term.is_empty()));
                        }
                        baby_polys[g] = ctx_copy.add(baby_polys[g], term);
                    }
                }
            }

            int const_idx = g * baby_steps;
            if (const_idx <= order && baby_poly_has_terms[g]) {
                if (weight_pt.empty()) {
                    auto coeff_pt = generate_weight_pt_for_bsgs(ctx_copy, const_idx, x_idx);
                    baby_polys[g] = ctx_copy.add_plain_ringt(baby_polys[g], coeff_pt);
                } else {
                    baby_polys[g] = ctx_copy.add_plain_ringt(baby_polys[g], weight_pt[const_idx][x_idx]);
                }
            }
        }

        // 3. Combine: result = P0 + P1*x^baby + P2*x^(2*baby) + ...
        if (baby_polys[0].is_empty()) {
            throw std::runtime_error("BSGS: baby_polys[0] is empty before combine, x_idx=" + std::to_string(x_idx));
        }
        result[x_idx] = baby_polys[0].copy();
        if (result[x_idx].is_empty()) {
            throw std::runtime_error("BSGS: result[x_idx] empty after copy from baby_polys[0], x_idx=" +
                                     std::to_string(x_idx));
        }

        for (int g = 1; g < bsgs_giant_steps; g++) {
            int giant_power = g * baby_steps;
            if (giant_power > order)
                break;

            auto x_giant = x_powers[giant_power].copy();
            int mult_level = bsgs_output_level + 1;

            while (x_giant.get_level() > mult_level) {
                x_giant = ctx_copy.drop_level(x_giant);
            }

            CkksCiphertext term;

            if (baby_poly_has_terms[g]) {
                auto baby_poly_copy = baby_polys[g].copy();
                while (baby_poly_copy.get_level() > mult_level) {
                    baby_poly_copy = ctx_copy.drop_level(baby_poly_copy);
                }
                term = ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(baby_poly_copy, x_giant)),
                                        ctx_copy.get_parameter().get_default_scale());
            } else {
                int const_idx = g * baby_steps;
                if (const_idx <= order) {
                    if (weight_pt.empty()) {
                        auto coeff_pt_rt = generate_weight_pt_for_bsgs(ctx_copy, const_idx, x_idx);
                        auto coeff_pt = ctx_copy.ringt_to_mul(coeff_pt_rt, x_giant.get_level());
                        term = ctx_copy.rescale(ctx_copy.mult_plain_mul(x_giant, coeff_pt),
                                                ctx_copy.get_parameter().get_default_scale());
                    } else {
                        auto coeff_pt = ctx_copy.ringt_to_mul(weight_pt[const_idx][x_idx], x_giant.get_level());
                        term = ctx_copy.rescale(ctx_copy.mult_plain_mul(x_giant, coeff_pt),
                                                ctx_copy.get_parameter().get_default_scale());
                    }
                }
            }

            result[x_idx] = ctx_copy.add(result[x_idx], term);
            if (result[x_idx].is_empty()) {
                throw std::runtime_error("BSGS combine: result empty after add, g=" + std::to_string(g));
            }
        }
    });

    return result;
}

// ======================== PolyRelu0D ========================

PolyRelu0D::PolyRelu0D(const CkksParameter& param_in,
                       const Array<double, 2>& weight_in,
                       uint32_t n_channel_per_ct_in,
                       uint32_t level_in,
                       int order_in,
                       int ciphertext_skip_in)
    : PolyReluBase(param_in, weight_in, n_channel_per_ct_in, level_in, order_in), ciphertext_skip(ciphertext_skip_in) {}

PolyRelu0D::~PolyRelu0D() {}

void PolyRelu0D::prepare_weight() {
    init_bsgs();

    int channel = weight.get_shape()[1];
    int n_packed_out_channel = div_ceil(channel, n_channel_per_ct);
    weight_pt.resize(order + 1);

    CkksContext ctx = CkksContext::create_empty_context(this->param);
    ctx.resize_copies(order + 1);
    parallel_for(order + 1, th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        for (int ct_idx = 0; ct_idx < n_packed_out_channel; ct_idx++) {
            vector<double> feature_tmp_pack(ctx_copy.get_parameter().get_n() / 2, 0.0);
            for (int ch = 0; ch < (int)n_channel_per_ct; ch++) {
                int channel_idx = ct_idx * n_channel_per_ct + ch;
                if (channel_idx >= channel)
                    continue;
                feature_tmp_pack[ch * ciphertext_skip] = weight.get(idx, channel_idx);
            }
            double pack_scale = cached_bsgs_coeff_scale[idx];
            weight_pt[idx].push_back(ctx_copy.encode_ringt(feature_tmp_pack, pack_scale));
        }
    });
}

void PolyRelu0D::prepare_weight_lazy() {
    init_bsgs();
    weight_pt.clear();
}

CkksPlaintextRingt PolyRelu0D::generate_weight_pt_for_bsgs(CkksContext& ctx, int idx, int ct_idx) const {
    vector<double> feature_tmp_pack(N / 2, 0.0);
    for (int ch = 0; ch < (int)n_channel_per_ct; ch++) {
        int channel_idx = ct_idx * n_channel_per_ct + ch;
        if (channel_idx >= cached_channel)
            continue;
        feature_tmp_pack[ch * ciphertext_skip] = weight.get(idx, channel_idx);
    }
    double pack_scale = cached_bsgs_coeff_scale.at(idx);
    return ctx.encode_ringt(feature_tmp_pack, pack_scale);
}

Feature0DEncrypted PolyRelu0D::run(CkksContext& ctx, const Feature0DEncrypted& x) {
    Feature0DEncrypted result(&ctx, x.level);
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.skip = x.skip;
    result.data = run_core_bsgs(ctx, x.data);
    result.level = result.data[0].get_level();
    return result;
}

Array<double, 1> PolyRelu0D::run_plaintext(const Array<double, 1>& x) {
    int n_channel = x.get_size();
    Array<double, 1> result({(uint64_t)n_channel});
    for (int ch = 0; ch < n_channel; ch++) {
        double p = weight.get(0, ch);
        for (int k = 1; k <= order; k++) {
            p += weight.get(k, ch) * pow(x.get(ch), k);
        }
        result.set(ch, p);
    }
    return result;
}
