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

#include "par_upper_diagonal_layernorm.h"
#include "layer_util.h"
#include <cassert>
#include <cmath>
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

// ============================================================
// ParUpperDiagonalLNStats
// ============================================================

ParUpperDiagonalLNStats::ParUpperDiagonalLNStats(const CkksParameter& param,
                                                 Duo shape,
                                                 Duo head_shape,
                                                 uint32_t n_heads,
                                                 uint32_t init_level,
                                                 double eps,
                                                 double inv_var)
    : Layer(param), eps_(eps), inv_var_(inv_var) {
    assert(init_level >= 4);
    level_ = init_level;
    n_prepad_ = shape[0];
    total_cols_ = shape[1];
    assert(head_shape[0] == n_prepad_);
    m_prepad_ = head_shape[1];
    H_prepad_ = n_heads;
    n_slot_ = param_.get_n() / 2;
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

uint32_t ParUpperDiagonalLNStats::ct_index(uint32_t mb, uint32_t ct_local) const {
    return mb * cts_per_mb_ + ct_local;
}

uint32_t ParUpperDiagonalLNStats::total_cts() const {
    return n_mb_ * cts_per_mb_;
}

vector<double> ParUpperDiagonalLNStats::build_h0_mask(double value) const {
    // Applied after local-diagonal aggregation; local_diag lanes are working copies here.
    vector<double> mask(n_slot_, 0.0);
    for (uint32_t local_diag = 0; local_diag < c_; local_diag++) {
        uint32_t segment_base = local_diag * segment_len_;
        for (uint32_t t = 0; t < n_prepad_; t++) {
            mask[segment_base + t * H_] = value;
        }
    }
    return mask;
}

vector<double> ParUpperDiagonalLNStats::build_valid_mask(uint32_t mb, uint32_t ct_local, double value) const {
    assert(ct_local < cts_per_mb_);
    vector<double> mask(n_slot_, 0.0);
    for (uint32_t local_diag = 0; local_diag < c_; local_diag++) {
        uint32_t segment_base = local_diag * segment_len_;
        for (uint32_t t = 0; t < n_; t++) {
            for (uint32_t h = 0; h < H_; h++) {
                if (is_valid_slot(mb, ct_local, local_diag, t, h, n_prepad_, total_cols_, m_prepad_, H_prepad_, m_, c_,
                                  packed_extent_)) {
                    mask[segment_base + t * H_ + h] = value;
                }
            }
        }
    }
    return mask;
}

CkksPlaintextRingt ParUpperDiagonalLNStats::generate_pt(CkksContext& ctx,
                                                        uint32_t pt_idx,
                                                        uint32_t mb,
                                                        uint32_t ct_local,
                                                        uint32_t) const {
    double D = param_.get_default_scale();
    double q_L = param_.get_q(level_);
    double q_L1 = param_.get_q(level_ - 1);
    double q_L2 = param_.get_q(level_ - 2);
    double n_cols = static_cast<double>(total_cols_);
    double input_scale = sqrt(inv_var_) / n_cols;

    if (pt_idx == 0) {
        return ctx.encode_ringt(build_h0_mask(input_scale), q_L);
    }
    if (pt_idx == 1) {
        return ctx.encode_ringt(build_h0_mask(n_cols * input_scale * input_scale), q_L);
    }
    if (pt_idx == 2) {
        return ctx.encode_ringt(build_valid_mask(mb, ct_local, 1.0), q_L1 / D * q_L2);
    }
    if (pt_idx == 3) {
        return ctx.encode_ringt(build_valid_mask(mb, ct_local, eps_ * inv_var_), D);
    }
    throw runtime_error("ParUpperDiagonalLNStats: unknown pt_idx " + to_string(pt_idx));
}

void ParUpperDiagonalLNStats::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    h0_mask_pt_ = generate_pt(ctx, 0);

    inv_n_pt_.resize(total_cts());
    iv_pt_.resize(total_cts());
    eps_add_pt_.resize(total_cts());
    for (uint32_t mb = 0; mb < n_mb_; mb++) {
        for (uint32_t ct_local = 0; ct_local < cts_per_mb_; ct_local++) {
            uint32_t idx = ct_index(mb, ct_local);
            inv_n_pt_[idx] = generate_pt(ctx, 1, mb, ct_local);
            iv_pt_[idx] = generate_pt(ctx, 2, mb, ct_local);
            eps_add_pt_[idx] = generate_pt(ctx, 3, mb, ct_local);
        }
    }
}

CkksCiphertext ParUpperDiagonalLNStats::reduce_cols_in_ct(CkksContext& ctx,
                                                          const CkksCiphertext& ct,
                                                          double rescale_target,
                                                          const CkksPlaintextRingt& h0_mask_pt) const {
    CkksCiphertext result = ct.copy();
    for (uint32_t step = 1; step < c_; step <<= 1) {
        result = ctx.add(result, ctx.rotate(result, (int)(step * segment_len_)));
    }
    for (uint32_t step = 1; step < H_; step <<= 1) {
        result = ctx.add(result, ctx.rotate(result, (int)step));
    }

    auto mask_mul = ctx.ringt_to_mul(h0_mask_pt, result.get_level());
    auto masked = ctx.rescale(ctx.mult_plain_mul(result, mask_mul), rescale_target);

    CkksCiphertext replicated = masked.copy();
    for (uint32_t step = 1; step < H_; step <<= 1) {
        replicated = ctx.add(replicated, ctx.rotate(replicated, -(int)step));
    }
    return replicated;
}

vector<CkksCiphertext> ParUpperDiagonalLNStats::run(CkksContext& ctx, const FeatureMatEncrypted& x) {
    assert(x.level == level_);
    assert(x.shape[0] == n_prepad_ && x.shape[1] == total_cols_);
    assert(x.head_shape[0] == n_prepad_ && x.head_shape[1] == m_prepad_);
    assert(x.matmul_block_size == m_);
    assert(x.data.size() == total_cts());

    double D = param_.get_default_scale();
    uint32_t n_ct = total_cts();

    vector<CkksCiphertext> partial_sum_x(n_ct);
    vector<CkksCiphertext> x_sq(n_ct);
    parallel_for(n_ct, th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        partial_sum_x[idx] = reduce_cols_in_ct(ctx_copy, x.data[idx], D, h0_mask_pt_);  // sqrt(inv_var) * mean(x)
        auto prod = ctx_copy.mult(x.data[idx], x.data[idx]);
        x_sq[idx] =
            ctx_copy.rescale(ctx_copy.relinearize(prod), D / param_.get_q(level_) * D);  // level L-1, scale D/q_L * D
    });

    CkksCiphertext sum_x = partial_sum_x[0].copy();
    for (uint32_t idx = 1; idx < n_ct; idx++) {
        sum_x = ctx.add(sum_x, partial_sum_x[idx]);  // level L-1, scale D
    }

    auto sq_sum_x_raw = ctx.mult(sum_x, sum_x);
    CkksCiphertext sq_sum_x =
        ctx.rescale(ctx.relinearize(sq_sum_x_raw), D / param_.get_q(level_ - 1) * D);  // level L-2

    vector<CkksCiphertext> partial_sum_x_sq(n_ct);
    parallel_for(n_ct, th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        partial_sum_x_sq[idx] = reduce_cols_in_ct(ctx_copy, x_sq[idx], D / param_.get_q(level_ - 1) * D,
                                                  inv_n_pt_[idx]);  // inv_var * mean(x^2), level L-2
    });

    CkksCiphertext sum_x_sq = partial_sum_x_sq[0].copy();
    for (uint32_t idx = 1; idx < n_ct; idx++) {
        sum_x_sq = ctx.add(sum_x_sq, partial_sum_x_sq[idx]);  // level L-2, scale D/q_{L-1} *D
    }

    CkksCiphertext variance = ctx.sub(sum_x_sq, sq_sum_x);  // inv_var * (mean(x^2) - mean(x)^2)

    vector<CkksCiphertext> a_cts(n_ct);
    parallel_for(n_ct, th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        auto pt_one = ctx_copy.ringt_to_mul(iv_pt_[idx], level_ - 2);
        a_cts[idx] = ctx_copy.rescale(ctx_copy.mult_plain_mul(variance, pt_one), D);  // level L-3, scale D
        a_cts[idx] = ctx_copy.add_plain_ringt(a_cts[idx], eps_add_pt_[idx]);          // level L-3, scale D
    });

    return a_cts;  // level L-3, scale D
}

Array<double, 2> ParUpperDiagonalLNStats::run_plaintext(const Array<double, 2>& x) const {
    Array<double, 2> result({n_prepad_, total_cols_});
    for (uint32_t i = 0; i < n_prepad_; i++) {
        double sum_x = 0.0;
        double sum_x2 = 0.0;
        for (uint32_t j = 0; j < total_cols_; j++) {
            double v = x.get(i, j);
            sum_x += v;
            sum_x2 += v * v;
        }
        double n_cols = static_cast<double>(total_cols_);
        double mean = sum_x / n_cols;
        double var = sum_x2 / n_cols - mean * mean;
        double a = (var + eps_) * inv_var_;
        for (uint32_t j = 0; j < total_cols_; j++) {
            result.set(i, j, a);
        }
    }
    return result;
}

// ============================================================
// ParUpperDiagonalLNXCentered
// ============================================================

ParUpperDiagonalLNXCentered::ParUpperDiagonalLNXCentered(const CkksParameter& param,
                                                         Duo shape,
                                                         Duo head_shape,
                                                         uint32_t n_heads,
                                                         uint32_t init_level)
    : Layer(param) {
    assert(init_level >= 2);
    level_ = init_level;
    n_prepad_ = shape[0];
    total_cols_ = shape[1];
    assert(head_shape[0] == n_prepad_);
    m_prepad_ = head_shape[1];
    H_prepad_ = n_heads;
    n_slot_ = param_.get_n() / 2;
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

uint32_t ParUpperDiagonalLNXCentered::ct_index(uint32_t mb, uint32_t ct_local) const {
    return mb * cts_per_mb_ + ct_local;
}

uint32_t ParUpperDiagonalLNXCentered::total_cts() const {
    return n_mb_ * cts_per_mb_;
}

vector<double> ParUpperDiagonalLNXCentered::build_h0_mask() const {
    // Applied after local-diagonal aggregation; local_diag lanes are working copies here.
    vector<double> mask(n_slot_, 0.0);
    for (uint32_t local_diag = 0; local_diag < c_; local_diag++) {
        uint32_t segment_base = local_diag * segment_len_;
        for (uint32_t t = 0; t < n_prepad_; t++) {
            mask[segment_base + t * H_] = 1.0;
        }
    }
    return mask;
}

vector<double> ParUpperDiagonalLNXCentered::build_valid_mask(uint32_t mb, uint32_t ct_local, double value) const {
    assert(ct_local < cts_per_mb_);
    vector<double> mask(n_slot_, 0.0);
    for (uint32_t local_diag = 0; local_diag < c_; local_diag++) {
        uint32_t segment_base = local_diag * segment_len_;
        for (uint32_t t = 0; t < n_; t++) {
            for (uint32_t h = 0; h < H_; h++) {
                if (is_valid_slot(mb, ct_local, local_diag, t, h, n_prepad_, total_cols_, m_prepad_, H_prepad_, m_, c_,
                                  packed_extent_)) {
                    mask[segment_base + t * H_ + h] = value;
                }
            }
        }
    }
    return mask;
}

CkksPlaintextRingt ParUpperDiagonalLNXCentered::generate_pt(CkksContext& ctx,
                                                            uint32_t pt_idx,
                                                            uint32_t mb,
                                                            uint32_t ct_local,
                                                            uint32_t) const {
    if (pt_idx == 0) {
        return ctx.encode_ringt(build_h0_mask(), param_.get_q(level_));
    }
    if (pt_idx == 1) {
        return ctx.encode_ringt(build_valid_mask(mb, ct_local, 1.0 / static_cast<double>(total_cols_)),
                                param_.get_q(level_ - 1));
    }
    throw runtime_error("ParUpperDiagonalLNXCentered: unknown pt_idx " + to_string(pt_idx));
}

void ParUpperDiagonalLNXCentered::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    h0_mask_pt_ = generate_pt(ctx, 0);

    inv_n_pt_.resize(total_cts());
    for (uint32_t mb = 0; mb < n_mb_; mb++) {
        for (uint32_t ct_local = 0; ct_local < cts_per_mb_; ct_local++) {
            inv_n_pt_[ct_index(mb, ct_local)] = generate_pt(ctx, 1, mb, ct_local);
        }
    }
}

CkksCiphertext ParUpperDiagonalLNXCentered::reduce_cols_in_ct(CkksContext& ctx,
                                                              const CkksCiphertext& ct,
                                                              double rescale_target) const {
    CkksCiphertext result = ct.copy();
    for (uint32_t step = 1; step < c_; step <<= 1) {
        result = ctx.add(result, ctx.rotate(result, (int)(step * segment_len_)));
    }
    for (uint32_t step = 1; step < H_; step <<= 1) {
        result = ctx.add(result, ctx.rotate(result, (int)step));
    }

    auto mask_mul = ctx.ringt_to_mul(h0_mask_pt_, result.get_level());
    auto masked = ctx.rescale(ctx.mult_plain_mul(result, mask_mul), rescale_target);

    CkksCiphertext replicated = masked.copy();
    for (uint32_t step = 1; step < H_; step <<= 1) {
        replicated = ctx.add(replicated, ctx.rotate(replicated, -(int)step));
    }
    return replicated;
}

vector<CkksCiphertext> ParUpperDiagonalLNXCentered::run(CkksContext& ctx, const FeatureMatEncrypted& x) {
    assert(x.level == level_);
    assert(x.shape[0] == n_prepad_ && x.shape[1] == total_cols_);
    assert(x.head_shape[0] == n_prepad_ && x.head_shape[1] == m_prepad_);
    assert(x.matmul_block_size == m_);
    assert(x.data.size() == total_cts());

    double D = param_.get_default_scale();
    uint32_t n_ct = total_cts();

    vector<CkksCiphertext> partial_sum_x(n_ct);
    parallel_for(n_ct, th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        partial_sum_x[idx] = reduce_cols_in_ct(ctx_copy, x.data[idx], D);
    });  // level L-1, scale D

    CkksCiphertext sum_x = partial_sum_x[0].copy();
    for (uint32_t idx = 1; idx < n_ct; idx++) {
        sum_x = ctx.add(sum_x, partial_sum_x[idx]);  // level L-1, scale D
    }

    vector<CkksCiphertext> mean_cts(n_ct);
    parallel_for(n_ct, th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        auto pt_inv_n = ctx_copy.ringt_to_mul(inv_n_pt_[idx], level_ - 1);
        mean_cts[idx] = ctx_copy.rescale(ctx_copy.mult_plain_mul(sum_x, pt_inv_n), D);  // level L-2, scale D
    });

    vector<CkksCiphertext> x_centered(n_ct);
    parallel_for(n_ct, th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        auto x_drop = ctx_copy.drop_level(x.data[idx], 2);
        x_centered[idx] = ctx_copy.sub(x_drop, mean_cts[idx]);
    });

    return x_centered;  // level L-2, scale D
}

Array<double, 2> ParUpperDiagonalLNXCentered::run_plaintext(const Array<double, 2>& x) const {
    Array<double, 2> result({n_prepad_, total_cols_});
    for (uint32_t i = 0; i < n_prepad_; i++) {
        double sum_x = 0.0;
        for (uint32_t j = 0; j < total_cols_; j++) {
            sum_x += x.get(i, j);
        }
        double n_cols = static_cast<double>(total_cols_);
        double mean = sum_x / n_cols;
        for (uint32_t j = 0; j < total_cols_; j++) {
            result.set(i, j, x.get(i, j) - mean);
        }
    }
    return result;
}

// ============================================================
// ParUpperDiagonalLNMinimaxInit
// ============================================================

ParUpperDiagonalLNMinimaxInit::ParUpperDiagonalLNMinimaxInit(const CkksParameter& param,
                                                             Duo shape,
                                                             Duo head_shape,
                                                             uint32_t n_heads,
                                                             uint32_t input_level,
                                                             double c0,
                                                             double c1,
                                                             double c2)
    : Layer(param), c0_(c0), c1_(c1), c2_(c2) {
    assert(input_level >= 2);
    level_ = input_level;
    n_prepad_ = shape[0];
    total_cols_ = shape[1];
    assert(head_shape[0] == n_prepad_);
    m_prepad_ = head_shape[1];
    H_prepad_ = n_heads;
    n_slot_ = param_.get_n() / 2;
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

uint32_t ParUpperDiagonalLNMinimaxInit::ct_index(uint32_t mb, uint32_t ct_local) const {
    return mb * cts_per_mb_ + ct_local;
}

uint32_t ParUpperDiagonalLNMinimaxInit::total_cts() const {
    return n_mb_ * cts_per_mb_;
}

vector<double> ParUpperDiagonalLNMinimaxInit::build_valid_mask(uint32_t mb, uint32_t ct_local, double value) const {
    assert(ct_local < cts_per_mb_);
    vector<double> mask(n_slot_, 0.0);
    for (uint32_t local_diag = 0; local_diag < c_; local_diag++) {
        uint32_t segment_base = local_diag * segment_len_;
        for (uint32_t t = 0; t < n_; t++) {
            for (uint32_t h = 0; h < H_; h++) {
                if (is_valid_slot(mb, ct_local, local_diag, t, h, n_prepad_, total_cols_, m_prepad_, H_prepad_, m_, c_,
                                  packed_extent_)) {
                    mask[segment_base + t * H_ + h] = value;
                }
            }
        }
    }
    return mask;
}

CkksPlaintextRingt ParUpperDiagonalLNMinimaxInit::generate_pt(CkksContext& ctx,
                                                              uint32_t pt_idx,
                                                              uint32_t mb,
                                                              uint32_t ct_local,
                                                              uint32_t) const {
    double D = param_.get_default_scale();
    double q_L = param_.get_q(level_);
    if (pt_idx == 0) {
        return ctx.encode_ringt(build_valid_mask(mb, ct_local, c0_), D);
    }
    if (pt_idx == 1) {
        return ctx.encode_ringt(build_valid_mask(mb, ct_local, c1_), q_L);
    }
    if (pt_idx == 2) {
        return ctx.encode_ringt(build_valid_mask(mb, ct_local, c2_), q_L / D * param_.get_q(level_ - 1));
    }
    throw runtime_error("ParUpperDiagonalLNMinimaxInit: unknown pt_idx " + to_string(pt_idx));
}

void ParUpperDiagonalLNMinimaxInit::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    c0_add_pt_.resize(total_cts());
    c1_pt_.resize(total_cts());
    c2_norm_pt_.resize(total_cts());
    for (uint32_t mb = 0; mb < n_mb_; mb++) {
        for (uint32_t ct_local = 0; ct_local < cts_per_mb_; ct_local++) {
            uint32_t idx = ct_index(mb, ct_local);
            c0_add_pt_[idx] = generate_pt(ctx, 0, mb, ct_local);
            c1_pt_[idx] = generate_pt(ctx, 1, mb, ct_local);
            c2_norm_pt_[idx] = generate_pt(ctx, 2, mb, ct_local);
        }
    }
}

vector<CkksCiphertext> ParUpperDiagonalLNMinimaxInit::run(CkksContext& ctx, const vector<CkksCiphertext>& a_cts) {
    assert(a_cts.size() == total_cts());
    double D = param_.get_default_scale();
    vector<CkksCiphertext> y_cts(a_cts.size());

    parallel_for(a_cts.size(), th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        auto a_sq_raw = ctx_copy.mult(a_cts[idx], a_cts[idx]);
        auto a_sq = ctx_copy.rescale(ctx_copy.relinearize(a_sq_raw), D / param_.get_q(level_) * D);

        auto c2_mul = ctx_copy.ringt_to_mul(c2_norm_pt_[idx], level_ - 1);
        auto c2a2 = ctx_copy.rescale(ctx_copy.mult_plain_mul(a_sq, c2_mul), D);  // level L-2, scale D

        auto c1_mul = ctx_copy.ringt_to_mul(c1_pt_[idx], level_);
        auto c1a = ctx_copy.rescale(ctx_copy.mult_plain_mul(a_cts[idx], c1_mul), D);  // level L-1, scale D

        auto y0 = ctx_copy.add(ctx_copy.drop_level(c1a), c2a2);
        y0 = ctx_copy.add_plain_ringt(y0, c0_add_pt_[idx]);
        y_cts[idx] = move(y0);
    });

    return y_cts;  // level L-2, scale D
}

Array<double, 2> ParUpperDiagonalLNMinimaxInit::run_plaintext(const Array<double, 2>& a) const {
    auto shape = a.get_shape();
    Array<double, 2> result(shape);
    for (uint64_t i = 0; i < shape[0]; i++) {
        for (uint64_t j = 0; j < shape[1]; j++) {
            double v = a.get(i, j);
            result.set(i, j, c0_ + c1_ * v + c2_ * v * v);
        }
    }
    return result;
}

// ============================================================
// ParUpperDiagonalLNGoldschmidt
// ============================================================

ParUpperDiagonalLNGoldschmidt::ParUpperDiagonalLNGoldschmidt(const CkksParameter& param,
                                                             Duo shape,
                                                             Duo head_shape,
                                                             uint32_t n_heads,
                                                             uint32_t input_level)
    : Layer(param) {
    assert(input_level >= 3);
    level_ = input_level;
    n_prepad_ = shape[0];
    total_cols_ = shape[1];
    assert(head_shape[0] == n_prepad_);
    m_prepad_ = head_shape[1];
    H_prepad_ = n_heads;
    n_slot_ = param_.get_n() / 2;
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

uint32_t ParUpperDiagonalLNGoldschmidt::ct_index(uint32_t mb, uint32_t ct_local) const {
    return mb * cts_per_mb_ + ct_local;
}

uint32_t ParUpperDiagonalLNGoldschmidt::total_cts() const {
    return n_mb_ * cts_per_mb_;
}

vector<double> ParUpperDiagonalLNGoldschmidt::build_valid_mask(uint32_t mb, uint32_t ct_local, double value) const {
    assert(ct_local < cts_per_mb_);
    vector<double> mask(n_slot_, 0.0);
    for (uint32_t local_diag = 0; local_diag < c_; local_diag++) {
        uint32_t segment_base = local_diag * segment_len_;
        for (uint32_t t = 0; t < n_; t++) {
            for (uint32_t h = 0; h < H_; h++) {
                if (is_valid_slot(mb, ct_local, local_diag, t, h, n_prepad_, total_cols_, m_prepad_, H_prepad_, m_, c_,
                                  packed_extent_)) {
                    mask[segment_base + t * H_ + h] = value;
                }
            }
        }
    }
    return mask;
}

CkksPlaintextRingt ParUpperDiagonalLNGoldschmidt::generate_pt(CkksContext& ctx,
                                                              uint32_t pt_idx,
                                                              uint32_t mb,
                                                              uint32_t ct_local,
                                                              uint32_t) const {
    double D = param_.get_default_scale();
    double q_L = param_.get_q(level_);
    double q_L1 = param_.get_q(level_ - 1);
    double q_L2 = param_.get_q(level_ - 2);

    if (pt_idx == 0) {
        return ctx.encode_ringt(build_valid_mask(mb, ct_local, 3.0), D / q_L * D / q_L1 * D);
    }
    if (pt_idx == 1) {
        return ctx.encode_ringt(build_valid_mask(mb, ct_local, 0.5), q_L / D * q_L / D * q_L1 / D * q_L2);
    }
    throw runtime_error("ParUpperDiagonalLNGoldschmidt: unknown pt_idx " + to_string(pt_idx));
}

void ParUpperDiagonalLNGoldschmidt::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    three_pt_.resize(total_cts());
    half_norm_pt_.resize(total_cts());
    for (uint32_t mb = 0; mb < n_mb_; mb++) {
        for (uint32_t ct_local = 0; ct_local < cts_per_mb_; ct_local++) {
            uint32_t idx = ct_index(mb, ct_local);
            three_pt_[idx] = generate_pt(ctx, 0, mb, ct_local);
            half_norm_pt_[idx] = generate_pt(ctx, 1, mb, ct_local);
        }
    }
}

vector<CkksCiphertext> ParUpperDiagonalLNGoldschmidt::run(CkksContext& ctx,
                                                          const vector<CkksCiphertext>& y_cts,
                                                          const vector<CkksCiphertext>& a_cts) {
    assert(y_cts.size() == total_cts());
    assert(a_cts.size() == total_cts());
    double D = param_.get_default_scale();
    vector<CkksCiphertext> y_new(y_cts.size());

    parallel_for(y_cts.size(), th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        auto a_drop = a_cts[idx].copy();
        if (a_drop.get_level() > (int)level_) {
            a_drop = ctx_copy.drop_level(a_drop, a_drop.get_level() - (int)level_);
        }

        auto ya =
            ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(y_cts[idx], a_drop)), D / param_.get_q(level_) * D);
        auto yy =
            ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(y_cts[idx], y_cts[idx])), D / param_.get_q(level_) * D);

        double S_prod = D / param_.get_q(level_) * D / param_.get_q(level_) * D / param_.get_q(level_ - 1) * D;
        auto ya_yy = ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(ya, yy)), S_prod);  // level L-2, scale S_prod

        auto three_mul = ctx_copy.ringt_to_mul(three_pt_[idx], level_);
        auto three_y =
            ctx_copy.rescale(ctx_copy.mult_plain_mul(y_cts[idx], three_mul), S_prod);  // level L-1, scale S_prod
        auto diff = ctx_copy.sub(ctx_copy.drop_level(three_y), ya_yy);                 // level L-2, scale S_prod

        auto half_mul = ctx_copy.ringt_to_mul(half_norm_pt_[idx], level_ - 2);
        y_new[idx] = ctx_copy.rescale(ctx_copy.mult_plain_mul(diff, half_mul), D);  // level L-3, scale D
    });

    return y_new;  // level L-3, scale D
}

Array<double, 2> ParUpperDiagonalLNGoldschmidt::run_plaintext(const Array<double, 2>& y,
                                                              const Array<double, 2>& a) const {
    auto shape = y.get_shape();
    Array<double, 2> result(shape);
    for (uint64_t i = 0; i < shape[0]; i++) {
        for (uint64_t j = 0; j < shape[1]; j++) {
            double yv = y.get(i, j);
            double av = a.get(i, j);
            result.set(i, j, 0.5 * yv * (3.0 - av * yv * yv));
        }
    }
    return result;
}

// ============================================================
// ParUpperDiagonalLNAffine
// ============================================================

ParUpperDiagonalLNAffine::ParUpperDiagonalLNAffine(const CkksParameter& param,
                                                   Duo shape,
                                                   Duo head_shape,
                                                   uint32_t n_heads,
                                                   uint32_t y_level,
                                                   double inv_std,
                                                   Array<double, 1>&& gamma,
                                                   Array<double, 1>&& beta)
    : Layer(param), y_level_(y_level), inv_std_(inv_std), gamma_vals_(move(gamma)), beta_vals_(move(beta)) {
    assert(y_level >= 2);
    level_ = y_level;
    n_prepad_ = shape[0];
    total_cols_ = shape[1];
    assert(head_shape[0] == n_prepad_);
    m_prepad_ = head_shape[1];
    H_prepad_ = n_heads;
    n_slot_ = param_.get_n() / 2;
    assert(gamma_vals_.get_size() >= total_cols_);
    assert(beta_vals_.get_size() >= total_cols_);
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

uint32_t ParUpperDiagonalLNAffine::ct_index(uint32_t mb, uint32_t ct_local) const {
    return mb * cts_per_mb_ + ct_local;
}

uint32_t ParUpperDiagonalLNAffine::total_cts() const {
    return n_mb_ * cts_per_mb_;
}

vector<double> ParUpperDiagonalLNAffine::build_valid_weight(uint32_t mb,
                                                            uint32_t ct_local,
                                                            const Array<double, 1>& values,
                                                            double factor) const {
    assert(ct_local < cts_per_mb_);
    vector<double> weight(n_slot_, 0.0);
    for (uint32_t local_diag = 0; local_diag < c_; local_diag++) {
        uint32_t diag_idx = ct_local * c_ + local_diag;
        uint32_t segment_base = local_diag * segment_len_;
        for (uint32_t t = 0; t < n_; t++) {
            for (uint32_t h = 0; h < H_; h++) {
                uint32_t local_col = (diag_idx + t) % m_;
                uint32_t global_col = mb * packed_extent_ + h * m_prepad_ + local_col;
                if (is_valid_slot(mb, ct_local, local_diag, t, h, n_prepad_, total_cols_, m_prepad_, H_prepad_, m_, c_,
                                  packed_extent_)) {
                    weight[segment_base + t * H_ + h] = factor * values.get(global_col);
                }
            }
        }
    }
    return weight;
}

CkksPlaintextRingt ParUpperDiagonalLNAffine::generate_pt(CkksContext& ctx,
                                                         uint32_t pt_idx,
                                                         uint32_t mb,
                                                         uint32_t ct_local,
                                                         uint32_t) const {
    double D = param_.get_default_scale();
    if (pt_idx == 0) {
        return ctx.encode_ringt(build_valid_weight(mb, ct_local, gamma_vals_, inv_std_),
                                param_.get_q(y_level_) / D * param_.get_q(y_level_ - 1));
    }
    if (pt_idx == 1) {
        return ctx.encode_ringt(build_valid_weight(mb, ct_local, beta_vals_, 1.0), D);
    }
    throw runtime_error("ParUpperDiagonalLNAffine: unknown pt_idx " + to_string(pt_idx));
}

void ParUpperDiagonalLNAffine::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    gamma_pt_.resize(total_cts());
    beta_add_pt_.resize(total_cts());
    for (uint32_t mb = 0; mb < n_mb_; mb++) {
        for (uint32_t ct_local = 0; ct_local < cts_per_mb_; ct_local++) {
            uint32_t idx = ct_index(mb, ct_local);
            gamma_pt_[idx] = generate_pt(ctx, 0, mb, ct_local);
            beta_add_pt_[idx] = generate_pt(ctx, 1, mb, ct_local);
        }
    }
}

FeatureMatEncrypted ParUpperDiagonalLNAffine::run(CkksContext& ctx,
                                                  const vector<CkksCiphertext>& x_centered,
                                                  const vector<CkksCiphertext>& y_cts) {
    assert(x_centered.size() == total_cts());
    assert(y_cts.size() == total_cts());

    double D = param_.get_default_scale();
    FeatureMatEncrypted result(&ctx, y_level_ - 2);
    result.shape = {n_prepad_, total_cols_};
    result.head_shape = {n_prepad_, m_prepad_};
    result.matmul_block_size = m_;
    result.data.resize(total_cts());

    parallel_for(total_cts(), th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        auto gamma_mul = ctx_copy.ringt_to_mul(gamma_pt_[idx], y_level_);
        auto yw = ctx_copy.rescale(ctx_copy.mult_plain_mul(y_cts[idx], gamma_mul),
                                   param_.get_q(y_level_ - 1));  // level L-1, scale q_{L-1}

        auto xc = x_centered[idx].copy();
        if (xc.get_level() > (int)(y_level_ - 1)) {
            xc = ctx_copy.drop_level(xc, xc.get_level() - (int)(y_level_ - 1));
        }

        auto out = ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(xc, yw)), D);  // level L-2, scale D
        result.data[idx] = ctx_copy.add_plain_ringt(out, beta_add_pt_[idx]);          // level L-2, scale D
    });

    result.level = y_level_ - 2;
    return result;  // level L-2, scale D
}

Array<double, 2> ParUpperDiagonalLNAffine::run_plaintext(const Array<double, 2>& x_centered,
                                                         const Array<double, 2>& y) const {
    Array<double, 2> result({n_prepad_, total_cols_});
    uint64_t y_cols = y.get_shape()[1];
    for (uint32_t i = 0; i < n_prepad_; i++) {
        for (uint32_t j = 0; j < total_cols_; j++) {
            double yv = y.get(i, y_cols == 1 ? 0 : j);
            double out = x_centered.get(i, j) * yv * inv_std_ * gamma_vals_.get(j) + beta_vals_.get(j);
            result.set(i, j, out);
        }
    }
    return result;
}
