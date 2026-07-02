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

#include "par_upper_diagonal_poly.h"
#include "layer_util.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <utility>

using namespace std;
using namespace lattisense;

namespace {

bool is_power_of_two(uint32_t x) {
    return x > 0 && (x & (x - 1)) == 0;
}

void validate_upper_diag_geometry(uint32_t n_prepad,
                                  uint32_t total_cols,
                                  uint32_t m_prepad,
                                  uint32_t H_prepad,
                                  uint32_t n_slot) {
    assert(n_prepad > 0);
    assert(total_cols > 0);
    assert(m_prepad > 0);
    assert(H_prepad > 0);

    uint32_t H = next_pow2(H_prepad);
    uint32_t m = next_pow2(m_prepad);
    uint32_t n = next_pow2(n_prepad);
    assert(m > 0 && is_power_of_two(m));
    assert(n >= m);
    assert(n % m == 0);

    uint32_t segment_len = H * n;
    assert(segment_len > 0);
    assert(n_slot % segment_len == 0);
    uint32_t c = n_slot / segment_len;
    assert(c > 0);
    assert(m % c == 0);
}

bool is_valid_slot(uint32_t mb,
                   uint32_t ct_local,
                   uint32_t local_diag,
                   uint32_t t,
                   uint32_t h,
                   uint32_t n_prepad,
                   uint32_t total_cols,
                   uint32_t m_prepad,
                   uint32_t H_prepad,
                   uint32_t m,
                   uint32_t c,
                   uint32_t packed_extent) {
    uint32_t diag_idx = ct_local * c + local_diag;
    uint32_t row = t;
    uint32_t local_col = (diag_idx + t) % m;
    uint32_t global_col = mb * packed_extent + h * m_prepad + local_col;
    return h < H_prepad && row < n_prepad && local_col < m_prepad && global_col < total_cols;
}

}  // namespace

ParUpperDiagonalPoly::ParUpperDiagonalPoly(const CkksParameter& param,
                                           Duo shape,
                                           Duo head_shape,
                                           uint32_t n_heads,
                                           uint32_t init_level,
                                           Array<double, 2>&& coeffs,
                                           uint32_t order_in)
    : MatPolyBase(param, move(coeffs), init_level, static_cast<int>(order_in)) {
    assert(order > 0);
    assert(coeffs_.get_shape()[0] == order + 1);

    n_prepad_ = shape[0];
    total_cols_ = shape[1];
    assert(head_shape[0] == n_prepad_);
    m_prepad_ = head_shape[1];
    H_prepad_ = n_heads;
    n_slot_ = param_.get_n() / 2;
    assert(coeffs_.get_shape()[1] >= total_cols_);
    validate_upper_diag_geometry(n_prepad_, total_cols_, m_prepad_, H_prepad_, n_slot_);

    H_ = next_pow2(H_prepad_);
    m_ = next_pow2(m_prepad_);
    n_ = next_pow2(n_prepad_);
    packed_extent_ = H_prepad_ * m_prepad_;
    segment_len_ = H_ * n_;
    c_ = n_slot_ / segment_len_;
    cts_per_mb_ = m_ / c_;
    n_mb_ = div_ceil(total_cols_, packed_extent_);
}

uint32_t ParUpperDiagonalPoly::ct_index(uint32_t mb, uint32_t ct_local) const {
    return mb * cts_per_mb_ + ct_local;
}

uint32_t ParUpperDiagonalPoly::total_cts() const {
    return n_mb_ * cts_per_mb_;
}

vector<double> ParUpperDiagonalPoly::build_coeff_vec(uint32_t coeff_idx, uint32_t mb, uint32_t ct_local) const {
    assert(coeff_idx <= static_cast<uint32_t>(order));
    assert(ct_local < cts_per_mb_);

    vector<double> vec(n_slot_, 0.0);
    for (uint32_t local_diag = 0; local_diag < c_; local_diag++) {
        uint32_t diag_idx = ct_local * c_ + local_diag;
        uint32_t segment_base = local_diag * segment_len_;
        for (uint32_t t = 0; t < n_; t++) {
            for (uint32_t h = 0; h < H_; h++) {
                uint32_t local_col = (diag_idx + t) % m_;
                uint32_t global_col = mb * packed_extent_ + h * m_prepad_ + local_col;
                if (is_valid_slot(mb, ct_local, local_diag, t, h, n_prepad_, total_cols_, m_prepad_, H_prepad_, m_, c_,
                                  packed_extent_)) {
                    vec[segment_base + t * H_ + h] = coeffs_.get(coeff_idx, global_col);
                }
            }
        }
    }
    return vec;
}

CkksPlaintextRingt
ParUpperDiagonalPoly::generate_weight_pt_for_bsgs(CkksContext& ctx, int coeff_idx, int ct_idx) const {
    assert(coeff_idx >= 0 && coeff_idx <= order);
    assert(ct_idx >= 0 && static_cast<uint32_t>(ct_idx) < total_cts());

    uint32_t mb = static_cast<uint32_t>(ct_idx) / cts_per_mb_;
    uint32_t ct_local = static_cast<uint32_t>(ct_idx) % cts_per_mb_;
    double pack_scale = cached_bsgs_coeff_scale.at(coeff_idx);
    return ctx.encode_ringt(build_coeff_vec(static_cast<uint32_t>(coeff_idx), mb, ct_local), pack_scale);
}

CkksPlaintextRingt
ParUpperDiagonalPoly::generate_weight_pt_for_stockmeyer(CkksContext& ctx, int coeff_idx, int ct_idx) const {
    assert(coeff_idx >= 0 && coeff_idx <= order);
    assert(ct_idx >= 0 && static_cast<uint32_t>(ct_idx) < total_cts());

    uint32_t mb = static_cast<uint32_t>(ct_idx) / cts_per_mb_;
    uint32_t ct_local = static_cast<uint32_t>(ct_idx) % cts_per_mb_;
    double pack_scale = cached_stockmeyer_coeff_scale.at(coeff_idx);
    return ctx.encode_ringt(build_coeff_vec(static_cast<uint32_t>(coeff_idx), mb, ct_local), pack_scale);
}

void ParUpperDiagonalPoly::prepare_weight() {
    init_bsgs();

    uint32_t n_ct = total_cts();
    weight_pt_.resize(order + 1);
    CkksContext ctx = CkksContext::create_empty_context(param_);

    parallel_for(order + 1, th_nums, ctx, [&](CkksContext& ctx_copy, int coeff_idx) {
        weight_pt_[coeff_idx].resize(n_ct);
        for (uint32_t ct_idx = 0; ct_idx < n_ct; ct_idx++) {
            weight_pt_[coeff_idx][ct_idx] = generate_weight_pt_for_bsgs(ctx_copy, coeff_idx, ct_idx);
        }
    });
}

void ParUpperDiagonalPoly::prepare_weight_lazy() {
    init_bsgs();
    weight_pt_.clear();
}

void ParUpperDiagonalPoly::prepare_weight_stockmeyer() {
    init_stockmeyer();

    uint32_t n_ct = total_cts();
    stockmeyer_weight_pt_.resize(order + 1);
    CkksContext ctx = CkksContext::create_empty_context(param_);

    parallel_for(order + 1, th_nums, ctx, [&](CkksContext& ctx_copy, int coeff_idx) {
        stockmeyer_weight_pt_[coeff_idx].resize(n_ct);
        for (uint32_t ct_idx = 0; ct_idx < n_ct; ct_idx++) {
            stockmeyer_weight_pt_[coeff_idx][ct_idx] = generate_weight_pt_for_stockmeyer(ctx_copy, coeff_idx, ct_idx);
        }
    });
}

void ParUpperDiagonalPoly::prepare_weight_stockmeyer_lazy() {
    init_stockmeyer();
    stockmeyer_weight_pt_.clear();
}

vector<CkksCiphertext> ParUpperDiagonalPoly::run_core(CkksContext& ctx, const vector<CkksCiphertext>& x) {
    return run_core_bsgs(ctx, x);
}

vector<CkksCiphertext> ParUpperDiagonalPoly::run_core_bsgs(CkksContext& ctx, const vector<CkksCiphertext>& x) {
    init_bsgs();
    vector<CkksCiphertext> result(x.size());

    if (order <= 0) {
        throw runtime_error("ParUpperDiagonalPoly: order must be at least 1");
    }

    parallel_for(x.size(), th_nums, ctx, [&](CkksContext& ctx_copy, int x_idx) {
        map<int, CkksCiphertext> x_powers;
        x_powers[1] = x[x_idx].copy();
        if (x_powers[1].is_empty()) {
            throw runtime_error("ParUpperDiagonalPoly BSGS: input x[" + to_string(x_idx) + "] has invalid handle");
        }

        set<int> powers_to_compute;
        for (int p : required_powers) {
            powers_to_compute.insert(p);
            function<void(int)> add_dependencies = [&](int n) {
                if (n <= 1) {
                    return;
                }
                MatPolyPowerInfo info = get_power_info(n);
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

            MatPolyPowerInfo info = get_power_info(i);
            auto x_a = x_powers[info.decomp_a].copy();
            auto x_b = x_powers[info.decomp_b].copy();

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
                throw runtime_error("ParUpperDiagonalPoly BSGS: x_powers[" + to_string(i) +
                                    "] is empty after computation");
            }
        }

        vector<CkksCiphertext> baby_polys(bsgs_giant_steps);
        vector<bool> baby_poly_initialized(bsgs_giant_steps, false);
        vector<bool> baby_poly_has_terms(bsgs_giant_steps, false);

        for (int g = 0; g < bsgs_giant_steps; g++) {
            int target_level = baby_poly_output_level[g];
            double target_scale = baby_poly_output_scale[g];

            for (int b = 0; b < baby_steps; b++) {
                int coeff_idx = g * baby_steps + b;
                if (coeff_idx > order) {
                    break;
                }
                if (b == 0) {
                    continue;
                }

                baby_poly_has_terms[g] = true;
                auto x_copy = x_powers[b].copy();
                while (x_copy.get_level() > target_level + 1) {
                    x_copy = ctx_copy.drop_level(x_copy);
                }

                CkksCiphertext term;
                if (weight_pt_.empty()) {
                    auto coeff_pt_rt = generate_weight_pt_for_bsgs(ctx_copy, coeff_idx, x_idx);
                    auto coeff_pt = ctx_copy.ringt_to_mul(coeff_pt_rt, x_copy.get_level());
                    term = ctx_copy.rescale(ctx_copy.mult_plain_mul(x_copy, coeff_pt), target_scale);
                } else {
                    auto coeff_pt = ctx_copy.ringt_to_mul(weight_pt_[coeff_idx][x_idx], x_copy.get_level());
                    term = ctx_copy.rescale(ctx_copy.mult_plain_mul(x_copy, coeff_pt), target_scale);
                }

                if (!baby_poly_initialized[g]) {
                    baby_polys[g] = term.copy();
                    baby_poly_initialized[g] = true;
                } else {
                    if (baby_polys[g].is_empty() || term.is_empty()) {
                        throw runtime_error("ParUpperDiagonalPoly BSGS baby add: g=" + to_string(g));
                    }
                    baby_polys[g] = ctx_copy.add(baby_polys[g], term);
                }
            }

            int const_idx = g * baby_steps;
            if (const_idx <= order && baby_poly_has_terms[g]) {
                if (weight_pt_.empty()) {
                    auto coeff_pt = generate_weight_pt_for_bsgs(ctx_copy, const_idx, x_idx);
                    baby_polys[g] = ctx_copy.add_plain_ringt(baby_polys[g], coeff_pt);
                } else {
                    baby_polys[g] = ctx_copy.add_plain_ringt(baby_polys[g], weight_pt_[const_idx][x_idx]);
                }
            }
        }

        if (baby_polys[0].is_empty()) {
            throw runtime_error("ParUpperDiagonalPoly BSGS: baby_polys[0] is empty before combine, x_idx=" +
                                to_string(x_idx));
        }
        result[x_idx] = baby_polys[0].copy();
        if (result[x_idx].is_empty()) {
            throw runtime_error("ParUpperDiagonalPoly BSGS: result[" + to_string(x_idx) + "] is empty");
        }

        for (int g = 1; g < bsgs_giant_steps; g++) {
            int giant_power = g * baby_steps;
            if (giant_power > order) {
                break;
            }

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
                    if (weight_pt_.empty()) {
                        auto coeff_pt_rt = generate_weight_pt_for_bsgs(ctx_copy, const_idx, x_idx);
                        auto coeff_pt = ctx_copy.ringt_to_mul(coeff_pt_rt, x_giant.get_level());
                        term = ctx_copy.rescale(ctx_copy.mult_plain_mul(x_giant, coeff_pt),
                                                ctx_copy.get_parameter().get_default_scale());
                    } else {
                        auto coeff_pt = ctx_copy.ringt_to_mul(weight_pt_[const_idx][x_idx], x_giant.get_level());
                        term = ctx_copy.rescale(ctx_copy.mult_plain_mul(x_giant, coeff_pt),
                                                ctx_copy.get_parameter().get_default_scale());
                    }
                }
            }

            result[x_idx] = ctx_copy.add(result[x_idx], term);
            if (result[x_idx].is_empty()) {
                throw runtime_error("ParUpperDiagonalPoly BSGS combine: result empty after add, g=" + to_string(g));
            }
        }
    });

    return result;
}

vector<CkksCiphertext> ParUpperDiagonalPoly::run_core_stockmeyer(CkksContext& ctx, const vector<CkksCiphertext>& x) {
    init_stockmeyer();
    vector<CkksCiphertext> result(x.size());

    if (order != 15 && order != 31) {
        throw runtime_error("ParUpperDiagonalPoly Stockmeyer currently supports only order 15 and 31");
    }

    parallel_for(x.size(), th_nums, ctx, [&](CkksContext& ctx_copy, int x_idx) {
        map<int, CkksCiphertext> x_powers;
        x_powers[1] = x[x_idx].copy();
        if (x_powers[1].is_empty()) {
            throw runtime_error("ParUpperDiagonalPoly Stockmeyer: input x[" + to_string(x_idx) +
                                "] has invalid handle");
        }

        auto drop_to_level = [&](CkksCiphertext ct, int target_level, const string& label) -> CkksCiphertext {
            if (target_level < 0) {
                throw runtime_error("ParUpperDiagonalPoly Stockmeyer " + label + ": negative target level");
            }
            while (ct.get_level() > target_level) {
                ct = ctx_copy.drop_level(ct);
            }
            if (ct.get_level() != target_level) {
                throw runtime_error("ParUpperDiagonalPoly Stockmeyer " + label + ": cannot raise level from " +
                                    to_string(ct.get_level()) + " to " + to_string(target_level));
            }
            return ct;
        };

        auto square_power = [&](int power, int half_power) {
            auto half = x_powers.at(half_power).copy();
            x_powers[power] = ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(half, half)),
                                               ctx_copy.get_parameter().get_default_scale());
            if (x_powers[power].is_empty()) {
                throw runtime_error("ParUpperDiagonalPoly Stockmeyer: x^" + to_string(power) +
                                    " is empty after computation");
            }
        };

        square_power(2, 1);
        square_power(4, 2);
        square_power(8, 4);
        if (order == 31) {
            square_power(16, 8);
        }

        auto get_coeff_mul = [&](int coeff_idx, int level) {
            auto it = cached_stockmeyer_level_order.find(coeff_idx);
            if (it == cached_stockmeyer_level_order.end() || it->second != level) {
                throw runtime_error("ParUpperDiagonalPoly Stockmeyer: coeff " + to_string(coeff_idx) +
                                    " expected at level " +
                                    to_string(it == cached_stockmeyer_level_order.end() ? -1 : it->second) + ", got " +
                                    to_string(level));
            }

            if (stockmeyer_weight_pt_.empty()) {
                auto coeff_pt_rt = generate_weight_pt_for_stockmeyer(ctx_copy, coeff_idx, x_idx);
                return ctx_copy.ringt_to_mul(coeff_pt_rt, level);
            }
            return ctx_copy.ringt_to_mul(stockmeyer_weight_pt_[coeff_idx][x_idx], level);
        };

        auto add_const_coeff = [&](const CkksCiphertext& acc, int coeff_idx) -> CkksCiphertext {
            if (stockmeyer_weight_pt_.empty()) {
                auto coeff_pt = generate_weight_pt_for_stockmeyer(ctx_copy, coeff_idx, x_idx);
                return ctx_copy.add_plain_ringt(acc, coeff_pt);
            }
            return ctx_copy.add_plain_ringt(acc, stockmeyer_weight_pt_[coeff_idx][x_idx]);
        };

        auto multiply_plain_term = [&](const CkksCiphertext& power_ct, int coeff_idx, int mult_level,
                                       double target_scale, const string& label) -> CkksCiphertext {
            auto power_copy = drop_to_level(power_ct.copy(), mult_level, label + "_power");
            auto coeff_pt = get_coeff_mul(coeff_idx, power_copy.get_level());
            auto term = ctx_copy.rescale(ctx_copy.mult_plain_mul(power_copy, coeff_pt), target_scale);
            if (term.is_empty()) {
                throw runtime_error("ParUpperDiagonalPoly Stockmeyer: empty term in " + label);
            }
            return term;
        };

        auto eval_baby = [&](int baby_idx) -> CkksCiphertext {
            int base = baby_idx * stockmeyer_baby_steps;
            if (base + 3 > order) {
                throw runtime_error("ParUpperDiagonalPoly Stockmeyer: incomplete baby polynomial " +
                                    to_string(baby_idx));
            }

            int target_level = stockmeyer_baby_poly_output_level[baby_idx];
            double target_scale = stockmeyer_baby_poly_output_scale[baby_idx];
            double c3x_scale =
                target_scale * ctx_copy.get_parameter().get_q(target_level + 1) / stockmeyer_powers.at(2).scale;

            auto c1x = multiply_plain_term(x_powers.at(1), base + 1, target_level + 1, target_scale,
                                           "P" + to_string(baby_idx) + "_c1x");
            auto c2x2 = multiply_plain_term(x_powers.at(2), base + 2, target_level + 1, target_scale,
                                            "P" + to_string(baby_idx) + "_c2x2");
            auto acc = ctx_copy.add(c1x, c2x2);

            auto c3x = multiply_plain_term(x_powers.at(1), base + 3, target_level + 2, c3x_scale,
                                           "P" + to_string(baby_idx) + "_c3x");
            auto x2_for_c3 =
                drop_to_level(x_powers.at(2).copy(), target_level + 1, "P" + to_string(baby_idx) + "_x2_for_c3");
            auto c3x3 = ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(c3x, x2_for_c3)), target_scale);
            if (c3x3.is_empty()) {
                throw runtime_error("ParUpperDiagonalPoly Stockmeyer: empty cubic term in P" + to_string(baby_idx));
            }

            acc = ctx_copy.add(acc, c3x3);
            acc = add_const_coeff(acc, base);
            if (acc.is_empty()) {
                throw runtime_error("ParUpperDiagonalPoly Stockmeyer: empty baby polynomial P" + to_string(baby_idx));
            }
            return acc;
        };

        auto combine_with_power = [&](const CkksCiphertext& left, const CkksCiphertext& right, int power,
                                      int target_level, double target_scale, const string& label) -> CkksCiphertext {
            auto left_copy = drop_to_level(left.copy(), target_level, label + "_left");
            int mult_level = target_level + 1;
            auto right_copy = drop_to_level(right.copy(), mult_level, label + "_right");
            auto power_copy = drop_to_level(x_powers.at(power).copy(), mult_level, label + "_x" + to_string(power));

            auto term = ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(right_copy, power_copy)), target_scale);
            if (term.is_empty()) {
                throw runtime_error("ParUpperDiagonalPoly Stockmeyer: empty product in " + label);
            }

            auto combined = ctx_copy.add(left_copy, term);
            if (combined.is_empty()) {
                throw runtime_error("ParUpperDiagonalPoly Stockmeyer: empty combined node " + label);
            }
            return combined;
        };

        vector<CkksCiphertext> P(stockmeyer_n_baby_polys);
        for (int j = 0; j < stockmeyer_n_baby_polys; j++) {
            P[j] = eval_baby(j);
        }

        int Lout = stockmeyer_output_level;
        double S = ctx_copy.get_parameter().get_default_scale();
        if (order == 15) {
            auto G0 = combine_with_power(P[0], P[1], 4, Lout, S, "G0");
            auto G1 = combine_with_power(P[2], P[3], 4, Lout + 1, stockmeyer_baby_poly_output_scale[2], "G1");
            result[x_idx] = combine_with_power(G0, G1, 8, Lout, S, "out");
        } else {
            auto G0 = combine_with_power(P[0], P[1], 4, Lout, S, "G0");
            auto G1 = combine_with_power(P[2], P[3], 4, Lout + 1, stockmeyer_baby_poly_output_scale[2], "G1");
            auto G2 = combine_with_power(P[4], P[5], 4, Lout + 1, stockmeyer_baby_poly_output_scale[4], "G2");
            auto G3 = combine_with_power(P[6], P[7], 4, Lout + 2, stockmeyer_baby_poly_output_scale[6], "G3");
            auto H0 = combine_with_power(G0, G1, 8, Lout, S, "H0");
            auto H1 = combine_with_power(G2, G3, 8, Lout + 1, stockmeyer_baby_poly_output_scale[4], "H1");
            result[x_idx] = combine_with_power(H0, H1, 16, Lout, S, "out");
        }

        if (result[x_idx].is_empty()) {
            throw runtime_error("ParUpperDiagonalPoly Stockmeyer: result[" + to_string(x_idx) + "] is empty");
        }
    });

    return result;
}

FeatureMatEncrypted ParUpperDiagonalPoly::run(CkksContext& ctx, const FeatureMatEncrypted& x) {
    init_bsgs();
    assert(x.level == level_);
    assert(x.shape[0] == n_prepad_ && x.shape[1] == total_cols_);
    assert(x.head_shape[0] == n_prepad_ && x.head_shape[1] == m_prepad_);
    assert(x.matmul_block_size == m_);
    assert(x.data.size() == total_cts());

    FeatureMatEncrypted result(&ctx, bsgs_output_level);
    result.shape = x.shape;
    result.head_shape = x.head_shape;
    result.matmul_block_size = x.matmul_block_size;
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.data = run_core(ctx, x.data);
    result.level = result.data[0].get_level();
    return result;
}

FeatureMatEncrypted ParUpperDiagonalPoly::run_stockmeyer(CkksContext& ctx, const FeatureMatEncrypted& x) {
    init_stockmeyer();
    assert(x.level == level_);
    assert(x.shape[0] == n_prepad_ && x.shape[1] == total_cols_);
    assert(x.head_shape[0] == n_prepad_ && x.head_shape[1] == m_prepad_);
    assert(x.matmul_block_size == m_);
    assert(x.data.size() == total_cts());

    FeatureMatEncrypted result(&ctx, stockmeyer_output_level);
    result.shape = x.shape;
    result.head_shape = x.head_shape;
    result.matmul_block_size = x.matmul_block_size;
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.data = run_core_stockmeyer(ctx, x.data);
    result.level = result.data[0].get_level();
    return result;
}

Array<double, 2> ParUpperDiagonalPoly::run_plaintext(const Array<double, 2>& x) const {
    auto shape = x.get_shape();
    if (shape[0] != n_prepad_ || shape[1] != total_cols_) {
        throw runtime_error("ParUpperDiagonalPoly plaintext input shape mismatch");
    }

    Array<double, 2> result({n_prepad_, total_cols_});
    for (uint32_t i = 0; i < n_prepad_; i++) {
        for (uint32_t j = 0; j < total_cols_; j++) {
            double v = x.get(i, j);
            double v_power = 1.0;
            double out = coeffs_.get(0, j);
            for (int k = 1; k <= order; k++) {
                v_power *= v;
                out += coeffs_.get(k, j) * v_power;
            }
            result.set(i, j, out);
        }
    }
    return result;
}
