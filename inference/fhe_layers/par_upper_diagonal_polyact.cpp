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

#include "par_upper_diagonal_polyact.h"
#include "layer_util.h"
#include <cassert>
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
// ParUpperDiagonalPolyActRNGamma
// ============================================================

ParUpperDiagonalPolyActRNGamma::ParUpperDiagonalPolyActRNGamma(const CkksParameter& param,
                                                               Duo shape,
                                                               Duo head_shape,
                                                               uint32_t n_heads,
                                                               uint32_t init_level,
                                                               Array<double, 1>&& gamma)
    : Layer(param), gamma_vals_(move(gamma)) {
    level_ = init_level;
    n_prepad_ = shape[0];
    total_cols_ = shape[1];
    assert(head_shape[0] == n_prepad_);
    m_prepad_ = head_shape[1];
    H_prepad_ = n_heads;
    n_slot_ = param_.get_n() / 2;
    assert(gamma_vals_.get_size() >= total_cols_);
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

uint32_t ParUpperDiagonalPolyActRNGamma::ct_index(uint32_t mb, uint32_t ct_local) const {
    return mb * cts_per_mb_ + ct_local;
}

uint32_t ParUpperDiagonalPolyActRNGamma::total_cts() const {
    return n_mb_ * cts_per_mb_;
}

vector<double> ParUpperDiagonalPolyActRNGamma::build_gamma_vec(uint32_t mb, uint32_t ct_local) const {
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
                    vec[segment_base + t * H_ + h] = gamma_vals_.get(global_col);
                }
            }
        }
    }
    return vec;
}

CkksPlaintextRingt
ParUpperDiagonalPolyActRNGamma::generate_gamma_pt(CkksContext& ctx, uint32_t mb, uint32_t ct_local, uint32_t) const {
    return ctx.encode_ringt(build_gamma_vec(mb, ct_local), param_.get_q(level_));
}

void ParUpperDiagonalPolyActRNGamma::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    gamma_pt_.resize(total_cts());
    for (uint32_t mb = 0; mb < n_mb_; mb++) {
        for (uint32_t ct_local = 0; ct_local < cts_per_mb_; ct_local++) {
            gamma_pt_[ct_index(mb, ct_local)] = generate_gamma_pt(ctx, mb, ct_local);
        }
    }
}

FeatureMatEncrypted ParUpperDiagonalPolyActRNGamma::run(CkksContext& ctx, const FeatureMatEncrypted& x) {
    assert(x.level == level_);
    assert(x.shape[0] == n_prepad_ && x.shape[1] == total_cols_);
    assert(x.head_shape[0] == n_prepad_ && x.head_shape[1] == m_prepad_);
    assert(x.matmul_block_size == m_);
    assert(x.data.size() == total_cts());

    double D = param_.get_default_scale();
    FeatureMatEncrypted result(&ctx, level_ - 1);
    result.shape = x.shape;
    result.head_shape = x.head_shape;
    result.matmul_block_size = m_;
    result.data.resize(total_cts());

    parallel_for(total_cts(), th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        auto gamma_mul = ctx_copy.ringt_to_mul(gamma_pt_[idx], level_);
        auto product = ctx_copy.mult_plain_mul(x.data[idx], gamma_mul);
        result.data[idx] = ctx_copy.rescale(product, D);
    });

    result.level = level_ - 1;
    return result;  // level L-1, scale D
}

Array<double, 2> ParUpperDiagonalPolyActRNGamma::run_plaintext(const Array<double, 2>& x) const {
    auto shape = x.get_shape();
    if (shape[0] != n_prepad_ || shape[1] != total_cols_) {
        throw runtime_error("ParUpperDiagonalPolyActRNGamma plaintext input shape mismatch");
    }

    Array<double, 2> result({n_prepad_, total_cols_});
    for (uint32_t i = 0; i < n_prepad_; i++) {
        for (uint32_t j = 0; j < total_cols_; j++) {
            result.set(i, j, x.get(i, j) * gamma_vals_.get(j));
        }
    }
    return result;
}

// ============================================================
// ParUpperDiagonalPolyActRNPoly
// ============================================================

ParUpperDiagonalPolyActRNPoly::ParUpperDiagonalPolyActRNPoly(const CkksParameter& param,
                                                             Duo shape,
                                                             Duo head_shape,
                                                             uint32_t n_heads,
                                                             uint32_t init_level,
                                                             Array<double, 2>&& coeffs,
                                                             uint32_t degree)
    : Layer(param), degree_(degree), coeffs_(move(coeffs)) {
    assert(degree_ == 2 || degree_ == 4);
    assert(coeffs_.get_shape()[0] == degree_ + 1);
    level_ = init_level;
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

uint32_t ParUpperDiagonalPolyActRNPoly::ct_index(uint32_t mb, uint32_t ct_local) const {
    return mb * cts_per_mb_ + ct_local;
}

uint32_t ParUpperDiagonalPolyActRNPoly::total_cts() const {
    return n_mb_ * cts_per_mb_;
}

vector<double>
ParUpperDiagonalPolyActRNPoly::build_coeff_vec(uint32_t coeff_idx, uint32_t mb, uint32_t ct_local) const {
    assert(coeff_idx <= degree_);
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

CkksPlaintextRingt ParUpperDiagonalPolyActRNPoly::generate_coeff_pt(CkksContext& ctx,
                                                                    uint32_t coeff_idx,
                                                                    uint32_t mb,
                                                                    uint32_t ct_local,
                                                                    uint32_t) const {
    assert(coeff_idx <= degree_);

    double D = param_.get_default_scale();
    double q_L = param_.get_q(level_);
    double scale = D;

    if (coeff_idx == 1) {
        scale = q_L;
    } else if (coeff_idx == 2) {
        scale = q_L / D * param_.get_q(level_ - 1);
    } else if (coeff_idx == 3) {
        assert(degree_ == 4);
        scale = q_L / D * q_L / D * param_.get_q(level_ - 2);
    } else if (coeff_idx == 4) {
        assert(degree_ == 4);
        scale = q_L / D * q_L / D * param_.get_q(level_ - 1) / D * param_.get_q(level_ - 2);
    }

    return ctx.encode_ringt(build_coeff_vec(coeff_idx, mb, ct_local), scale);
}

void ParUpperDiagonalPolyActRNPoly::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    uint32_t n_ct = total_cts();

    c2_pt_.resize(n_ct);
    c1_pt_.resize(n_ct);
    c0_add_pt_.resize(n_ct);
    for (uint32_t mb = 0; mb < n_mb_; mb++) {
        for (uint32_t ct_local = 0; ct_local < cts_per_mb_; ct_local++) {
            uint32_t idx = ct_index(mb, ct_local);
            c2_pt_[idx] = generate_coeff_pt(ctx, 2, mb, ct_local);
            c1_pt_[idx] = generate_coeff_pt(ctx, 1, mb, ct_local);
            c0_add_pt_[idx] = generate_coeff_pt(ctx, 0, mb, ct_local);
        }
    }

    if (degree_ == 4) {
        c4_pt_.resize(n_ct);
        c3_pt_.resize(n_ct);
        for (uint32_t mb = 0; mb < n_mb_; mb++) {
            for (uint32_t ct_local = 0; ct_local < cts_per_mb_; ct_local++) {
                uint32_t idx = ct_index(mb, ct_local);
                c4_pt_[idx] = generate_coeff_pt(ctx, 4, mb, ct_local);
                c3_pt_[idx] = generate_coeff_pt(ctx, 3, mb, ct_local);
            }
        }
    }
}

FeatureMatEncrypted ParUpperDiagonalPolyActRNPoly::run(CkksContext& ctx, const FeatureMatEncrypted& x) {
    assert(x.level == level_);
    assert(x.shape[0] == n_prepad_ && x.shape[1] == total_cols_);
    assert(x.head_shape[0] == n_prepad_ && x.head_shape[1] == m_prepad_);
    assert(x.matmul_block_size == m_);
    assert(x.data.size() == total_cts());

    double D = param_.get_default_scale();
    uint32_t out_level = (degree_ == 4) ? level_ - 3 : level_ - 2;

    FeatureMatEncrypted result(&ctx, out_level);
    result.shape = x.shape;
    result.head_shape = x.head_shape;
    result.matmul_block_size = m_;
    result.data.resize(total_cts());

    if (degree_ == 2) {
        parallel_for(total_cts(), th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
            auto x_sq_raw = ctx_copy.mult(x.data[idx], x.data[idx]);
            auto x_sq = ctx_copy.rescale(ctx_copy.relinearize(x_sq_raw), D / param_.get_q(level_) * D);

            auto c2_mul = ctx_copy.ringt_to_mul(c2_pt_[idx], level_ - 1);
            auto c2x2 = ctx_copy.rescale(ctx_copy.mult_plain_mul(x_sq, c2_mul), D);  // level L-2, scale D

            auto c1_mul = ctx_copy.ringt_to_mul(c1_pt_[idx], level_);
            auto c1x = ctx_copy.rescale(ctx_copy.mult_plain_mul(x.data[idx], c1_mul), D);  // level L-1, scale D
            auto c1x_drop = ctx_copy.drop_level(c1x);

            auto y = ctx_copy.add(c1x_drop, c2x2);
            result.data[idx] = ctx_copy.add_plain_ringt(y, c0_add_pt_[idx]);  // level L-2, scale D
        });
    } else {
        double q_L2 = param_.get_q(level_ - 2);
        double S_high = param_.get_q(level_) / D * q_L2;

        parallel_for(total_cts(), th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
            auto x_sq_raw = ctx_copy.mult(x.data[idx], x.data[idx]);
            auto x_sq = ctx_copy.rescale(ctx_copy.relinearize(x_sq_raw), D / param_.get_q(level_) * D);

            auto c2_mul = ctx_copy.ringt_to_mul(c2_pt_[idx], level_ - 1);
            auto c2x2 = ctx_copy.rescale(ctx_copy.mult_plain_mul(x_sq, c2_mul), D);

            auto c1_mul = ctx_copy.ringt_to_mul(c1_pt_[idx], level_);
            auto c1x = ctx_copy.rescale(ctx_copy.mult_plain_mul(x.data[idx], c1_mul), D);
            auto c1x_drop = ctx_copy.drop_level(c1x);

            auto low = ctx_copy.add(c1x_drop, c2x2);
            low = ctx_copy.add_plain_ringt(low, c0_add_pt_[idx]);  // level L-2, scale D

            auto c4_mul = ctx_copy.ringt_to_mul(c4_pt_[idx], level_ - 1);
            auto c4x2 = ctx_copy.rescale(ctx_copy.mult_plain_mul(x_sq, c4_mul), S_high);  // level L-2, scale S_high

            auto c3_mul = ctx_copy.ringt_to_mul(c3_pt_[idx], level_);
            auto c3x =
                ctx_copy.rescale(ctx_copy.mult_plain_mul(x.data[idx], c3_mul), S_high);  // level L-1, scale S_high
            auto c3x_drop = ctx_copy.drop_level(c3x);

            auto high = ctx_copy.add(c3x_drop, c4x2);    // level L-2, scale S_high
            auto x_sq_drop = ctx_copy.drop_level(x_sq);  // level L-2, scale D/q_L *D

            auto product = ctx_copy.mult(x_sq_drop, high);
            auto x2_high = ctx_copy.rescale(ctx_copy.relinearize(product), D);  // level L-3, scale D
            auto low_drop = ctx_copy.drop_level(low);                           // level L-3, scale D

            result.data[idx] = ctx_copy.add(low_drop, x2_high);  // level L-3, scale D
        });
    }

    result.level = out_level;
    return result;  // level L-3, scale D
}

Array<double, 2> ParUpperDiagonalPolyActRNPoly::run_plaintext(const Array<double, 2>& x) const {
    auto shape = x.get_shape();
    if (shape[0] != n_prepad_ || shape[1] != total_cols_) {
        throw runtime_error("ParUpperDiagonalPolyActRNPoly plaintext input shape mismatch");
    }

    Array<double, 2> result({n_prepad_, total_cols_});
    for (uint32_t i = 0; i < n_prepad_; i++) {
        for (uint32_t j = 0; j < total_cols_; j++) {
            double v = x.get(i, j);
            double out = coeffs_.get(0, j) + coeffs_.get(1, j) * v + coeffs_.get(2, j) * v * v;
            if (degree_ == 4) {
                double v2 = v * v;
                out += coeffs_.get(3, j) * v2 * v + coeffs_.get(4, j) * v2 * v2;
            }
            result.set(i, j, out);
        }
    }
    return result;
}
