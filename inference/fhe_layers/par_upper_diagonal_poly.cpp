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
ParUpperDiagonalPoly::generate_weight_pt_for_stockmeyer(CkksContext& ctx, int coeff_idx, int ct_idx) const {
    assert(coeff_idx >= 0 && coeff_idx <= order);
    assert(ct_idx >= 0 && static_cast<uint32_t>(ct_idx) < total_cts());

    uint32_t mb = static_cast<uint32_t>(ct_idx) / cts_per_mb_;
    uint32_t ct_local = static_cast<uint32_t>(ct_idx) % cts_per_mb_;
    double pack_scale = cached_stockmeyer_coeff_scale.at(coeff_idx);
    return ctx.encode_ringt(build_coeff_vec(static_cast<uint32_t>(coeff_idx), mb, ct_local), pack_scale);
}

void ParUpperDiagonalPoly::prepare_weight() {
    prepare_weight_stockmeyer();
}

void ParUpperDiagonalPoly::prepare_weight_lazy() {
    prepare_weight_stockmeyer_lazy();
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
    return run_core_stockmeyer(ctx, x);
}

vector<CkksCiphertext> ParUpperDiagonalPoly::run_core_stockmeyer(CkksContext& ctx, const vector<CkksCiphertext>& x) {
    init_stockmeyer();
    vector<CkksCiphertext> result(x.size());

    if (order <= 0 || order >= 64) {
        throw runtime_error("ParUpperDiagonalPoly Stockmeyer supports only 1 <= order < 64");
    }

    struct StockmeyerNode {
        CkksCiphertext ct;
        int const_coeff_idx = -1;
        int target_level = -1;
        double target_scale = 0.0;
        bool has_ct = false;
        bool const_only = false;
    };

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

        for (const auto& kv : stockmeyer_powers) {
            int power = kv.first;
            const auto& info = kv.second;
            if (power <= 1) {
                continue;
            }
            square_power(power, info.decomp_a);
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

        auto clone_node = [](const StockmeyerNode& node) -> StockmeyerNode {
            StockmeyerNode cloned;
            cloned.const_coeff_idx = node.const_coeff_idx;
            cloned.target_level = node.target_level;
            cloned.target_scale = node.target_scale;
            cloned.has_ct = node.has_ct;
            cloned.const_only = node.const_only;
            if (node.has_ct) {
                cloned.ct = node.ct.copy();
            }
            return cloned;
        };

        auto eval_baby_node = [&](int baby_idx) -> StockmeyerNode {
            int base = baby_idx * stockmeyer_baby_steps;
            if (base > order) {
                throw runtime_error("ParUpperDiagonalPoly Stockmeyer: missing baby polynomial " + to_string(baby_idx));
            }

            StockmeyerNode node;
            int target_level = stockmeyer_baby_poly_output_level[baby_idx];
            double target_scale = stockmeyer_baby_poly_output_scale[baby_idx];
            node.target_level = target_level;
            node.target_scale = target_scale;

            CkksCiphertext acc;
            bool acc_initialized = false;
            auto add_term = [&](const CkksCiphertext& term, const string& label) {
                if (term.is_empty()) {
                    throw runtime_error("ParUpperDiagonalPoly Stockmeyer: empty baby term in " + label);
                }
                if (!acc_initialized) {
                    acc = term.copy();
                    acc_initialized = true;
                } else {
                    acc = ctx_copy.add(acc, term);
                }
                if (acc.is_empty()) {
                    throw runtime_error("ParUpperDiagonalPoly Stockmeyer: empty baby accumulator in " + label);
                }
            };

            if (base + 1 <= order) {
                auto c1x = multiply_plain_term(x_powers.at(1), base + 1, target_level + 1, target_scale,
                                               "P" + to_string(baby_idx) + "_c1x");
                add_term(c1x, "P" + to_string(baby_idx) + "_c1x");
            }

            if (base + 2 <= order) {
                auto c2x2 = multiply_plain_term(x_powers.at(2), base + 2, target_level + 1, target_scale,
                                                "P" + to_string(baby_idx) + "_c2x2");
                add_term(c2x2, "P" + to_string(baby_idx) + "_c2x2");
            }

            if (base + 3 <= order) {
                double c3x_scale =
                    target_scale * ctx_copy.get_parameter().get_q(target_level + 1) / stockmeyer_powers.at(2).scale;
                auto c3x = multiply_plain_term(x_powers.at(1), base + 3, target_level + 2, c3x_scale,
                                               "P" + to_string(baby_idx) + "_c3x");
                auto x2_for_c3 =
                    drop_to_level(x_powers.at(2).copy(), target_level + 1, "P" + to_string(baby_idx) + "_x2_for_c3");
                auto c3x3 = ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(c3x, x2_for_c3)), target_scale);
                if (c3x3.is_empty()) {
                    throw runtime_error("ParUpperDiagonalPoly Stockmeyer: empty cubic term in P" + to_string(baby_idx));
                }
                add_term(c3x3, "P" + to_string(baby_idx) + "_c3x3");
            }

            if (acc_initialized) {
                node.ct = add_const_coeff(acc, base);
                if (node.ct.is_empty()) {
                    throw runtime_error("ParUpperDiagonalPoly Stockmeyer: empty baby polynomial P" +
                                        to_string(baby_idx));
                }
                node.has_ct = true;
                return node;
            }

            node.const_coeff_idx = base;
            node.const_only = true;
            return node;
        };

        auto combine_with_power = [&](const StockmeyerNode& left, const StockmeyerNode& right, int power,
                                      const string& label) -> StockmeyerNode {
            if (!left.has_ct) {
                throw runtime_error("ParUpperDiagonalPoly Stockmeyer: left node is not ciphertext in " + label);
            }

            StockmeyerNode combined_node;
            combined_node.target_level = left.target_level;
            combined_node.target_scale = left.target_scale;

            auto left_copy = drop_to_level(left.ct.copy(), left.target_level, label + "_left");
            int mult_level = left.target_level + 1;
            if (right.target_level != mult_level) {
                throw runtime_error("ParUpperDiagonalPoly Stockmeyer: right node target level mismatch in " + label);
            }
            auto power_copy = drop_to_level(x_powers.at(power).copy(), mult_level, label + "_x" + to_string(power));

            CkksCiphertext term;
            if (right.has_ct) {
                auto right_copy = drop_to_level(right.ct.copy(), mult_level, label + "_right");
                term = ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(right_copy, power_copy)), left.target_scale);
            } else if (right.const_only) {
                auto coeff_pt = get_coeff_mul(right.const_coeff_idx, power_copy.get_level());
                term = ctx_copy.rescale(ctx_copy.mult_plain_mul(power_copy, coeff_pt), left.target_scale);
            } else {
                throw runtime_error("ParUpperDiagonalPoly Stockmeyer: empty right node in " + label);
            }

            if (term.is_empty()) {
                throw runtime_error("ParUpperDiagonalPoly Stockmeyer: empty product in " + label);
            }

            auto combined = ctx_copy.add(left_copy, term);
            if (combined.is_empty()) {
                throw runtime_error("ParUpperDiagonalPoly Stockmeyer: empty combined node " + label);
            }
            combined_node.ct = move(combined);
            combined_node.has_ct = true;
            return combined_node;
        };

        vector<StockmeyerNode> nodes;
        nodes.reserve(stockmeyer_n_baby_polys);
        for (int j = 0; j < stockmeyer_n_baby_polys; j++) {
            nodes.push_back(eval_baby_node(j));
        }

        int combine_power = stockmeyer_baby_steps;
        int combine_round = 0;
        while (nodes.size() > 1) {
            vector<StockmeyerNode> next_nodes;
            next_nodes.reserve((nodes.size() + 1) / 2);

            for (size_t i = 0; i < nodes.size(); i += 2) {
                if (i + 1 >= nodes.size()) {
                    next_nodes.push_back(clone_node(nodes[i]));
                    continue;
                }

                next_nodes.push_back(
                    combine_with_power(nodes[i], nodes[i + 1], combine_power,
                                       "combine_" + to_string(combine_round) + "_" + to_string(i / 2)));
            }

            nodes = move(next_nodes);
            combine_power *= 2;
            combine_round++;
        }

        if (nodes.empty() || !nodes[0].has_ct) {
            throw runtime_error("ParUpperDiagonalPoly Stockmeyer: result is not a ciphertext");
        }
        result[x_idx] = nodes[0].ct.copy();

        if (result[x_idx].is_empty()) {
            throw runtime_error("ParUpperDiagonalPoly Stockmeyer: result[" + to_string(x_idx) + "] is empty");
        }
    });

    return result;
}

FeatureMatEncrypted ParUpperDiagonalPoly::run(CkksContext& ctx, const FeatureMatEncrypted& x) {
    return run_stockmeyer(ctx, x);
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
