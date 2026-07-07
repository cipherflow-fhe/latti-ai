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

#include "par_upper_diagonal_softmax.h"
#include "layer_util.h"
#include <cassert>
#include <stdexcept>

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
    assert(total_cols == H_prepad * m_prepad);
    assert(n_prepad == m_prepad);

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

bool is_valid_slot(uint32_t ct_idx,
                   uint32_t local_diag,
                   uint32_t t,
                   uint32_t h,
                   uint32_t n_prepad,
                   uint32_t total_cols,
                   uint32_t m_prepad,
                   uint32_t H_prepad,
                   uint32_t m,
                   uint32_t c) {
    uint32_t diag_idx = ct_idx * c + local_diag;
    uint32_t row = t;
    uint32_t local_col = (diag_idx + t) % m;
    uint32_t global_col = h * m_prepad + local_col;
    return h < H_prepad && row < n_prepad && local_col < m_prepad && global_col < total_cols;
}

double natural_square_scale(const CkksParameter& param, uint32_t level) {
    double D = param.get_default_scale();
    return D / static_cast<double>(param.get_q(level)) * D;
}

double mask_scale_after_natural_square(const CkksParameter& param, uint32_t level) {
    double D = param.get_default_scale();
    return static_cast<double>(param.get_q(level)) / D * static_cast<double>(param.get_q(level - 1));
}

}  // namespace

// ============================================================
// ParUpperDiagonalAddPt
// ============================================================

ParUpperDiagonalAddPt::ParUpperDiagonalAddPt(const CkksParameter& param,
                                             Duo shape,
                                             Duo head_shape,
                                             uint32_t n_heads,
                                             uint32_t init_level,
                                             double value)
    : Layer(param), value_(value) {
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
    segment_len_ = H_ * n_;
    c_ = n_slot_ / segment_len_;
    n_cts_ = m_ / c_;
}

uint32_t ParUpperDiagonalAddPt::total_cts() const {
    return n_cts_;
}

vector<double> ParUpperDiagonalAddPt::build_constant_vec(uint32_t ct_idx, double value) const {
    assert(ct_idx < total_cts());
    vector<double> vec(n_slot_, 0.0);
    for (uint32_t local_diag = 0; local_diag < c_; local_diag++) {
        uint32_t segment_base = local_diag * segment_len_;
        for (uint32_t t = 0; t < n_; t++) {
            for (uint32_t h = 0; h < H_; h++) {
                if (is_valid_slot(ct_idx, local_diag, t, h, n_prepad_, total_cols_, m_prepad_, H_prepad_, m_, c_)) {
                    vec[segment_base + t * H_ + h] = value;
                }
            }
        }
    }
    return vec;
}

CkksPlaintextRingt ParUpperDiagonalAddPt::generate_pt(CkksContext& ctx, uint32_t ct_idx) const {
    return ctx.encode_ringt(build_constant_vec(ct_idx, value_), param_.get_default_scale());
}

void ParUpperDiagonalAddPt::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    pt_.resize(total_cts());
    for (uint32_t ct_idx = 0; ct_idx < total_cts(); ct_idx++) {
        pt_[ct_idx] = generate_pt(ctx, ct_idx);
    }
}

FeatureMatEncrypted ParUpperDiagonalAddPt::run(CkksContext& ctx, const FeatureMatEncrypted& x) {
    assert(x.level == level_);
    assert(x.shape[0] == n_prepad_ && x.shape[1] == total_cols_);
    assert(x.head_shape[0] == n_prepad_ && x.head_shape[1] == m_prepad_);
    assert(x.matmul_block_size == m_);
    assert(x.data.size() == total_cts());
    assert(pt_.size() == total_cts());

    FeatureMatEncrypted result(&ctx, level_);
    result.level = level_;
    result.shape = x.shape;
    result.head_shape = x.head_shape;
    result.matmul_block_size = m_;
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.data.resize(total_cts());

    parallel_for(total_cts(), th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        result.data[idx] = ctx_copy.add_plain_ringt(x.data[idx], pt_[idx]);
    });
    return result;
}

Array<double, 2> ParUpperDiagonalAddPt::run_plaintext(const Array<double, 2>& x) const {
    auto x_shape = x.get_shape();
    if (x_shape[0] != n_prepad_ || x_shape[1] != total_cols_) {
        throw runtime_error("ParUpperDiagonalAddPt plaintext input shape mismatch");
    }
    Array<double, 2> result({n_prepad_, total_cols_});
    for (uint32_t i = 0; i < n_prepad_; i++) {
        for (uint32_t j = 0; j < total_cols_; j++) {
            result.set(i, j, x.get(i, j) + value_);
        }
    }
    return result;
}

// ============================================================
// ParUpperDiagonalMultipleSquare
// ============================================================

ParUpperDiagonalMultipleSquare::ParUpperDiagonalMultipleSquare(const CkksParameter& param,
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
    segment_len_ = H_ * n_;
    c_ = n_slot_ / segment_len_;
    n_cts_ = m_ / c_;
}

uint32_t ParUpperDiagonalMultipleSquare::total_cts() const {
    return n_cts_;
}

vector<double> ParUpperDiagonalMultipleSquare::build_valid_mask(uint32_t ct_idx, double value) const {
    assert(ct_idx < total_cts());
    vector<double> mask(n_slot_, 0.0);
    for (uint32_t local_diag = 0; local_diag < c_; local_diag++) {
        uint32_t segment_base = local_diag * segment_len_;
        for (uint32_t t = 0; t < n_; t++) {
            for (uint32_t h = 0; h < H_; h++) {
                if (is_valid_slot(ct_idx, local_diag, t, h, n_prepad_, total_cols_, m_prepad_, H_prepad_, m_, c_)) {
                    mask[segment_base + t * H_ + h] = value;
                }
            }
        }
    }
    return mask;
}

CkksPlaintextRingt ParUpperDiagonalMultipleSquare::generate_mask_pt(CkksContext& ctx, uint32_t ct_idx) const {
    return ctx.encode_ringt(build_valid_mask(ct_idx, 1.0), mask_scale_after_natural_square(param_, level_));
}

void ParUpperDiagonalMultipleSquare::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    mask_pt_.resize(total_cts());
    for (uint32_t ct_idx = 0; ct_idx < total_cts(); ct_idx++) {
        mask_pt_[ct_idx] = generate_mask_pt(ctx, ct_idx);
    }
}

FeatureMatEncrypted ParUpperDiagonalMultipleSquare::run(CkksContext& ctx, const FeatureMatEncrypted& x) {
    assert(x.level == level_);
    assert(x.shape[0] == n_prepad_ && x.shape[1] == total_cols_);
    assert(x.head_shape[0] == n_prepad_ && x.head_shape[1] == m_prepad_);
    assert(x.matmul_block_size == m_);
    assert(x.data.size() == total_cts());
    assert(mask_pt_.size() == total_cts());

    double D = param_.get_default_scale();
    double square_scale = natural_square_scale(param_, level_);
    FeatureMatEncrypted result(&ctx, level_ - 2);
    result.shape = x.shape;
    result.head_shape = x.head_shape;
    result.matmul_block_size = m_;
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.data.resize(total_cts());

    parallel_for(total_cts(), th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        auto sq = ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(x.data[idx], x.data[idx])), square_scale);
        auto mask_mul = ctx_copy.ringt_to_mul(mask_pt_[idx], level_ - 1);
        result.data[idx] = ctx_copy.rescale(ctx_copy.mult_plain_mul(sq, mask_mul), D);
    });

    result.level = level_ - 2;
    return result;
}

Array<double, 2> ParUpperDiagonalMultipleSquare::run_plaintext(const Array<double, 2>& x) const {
    auto x_shape = x.get_shape();
    if (x_shape[0] != n_prepad_ || x_shape[1] != total_cols_) {
        throw runtime_error("ParUpperDiagonalMultipleSquare plaintext input shape mismatch");
    }
    Array<double, 2> result({n_prepad_, total_cols_});
    for (uint32_t i = 0; i < n_prepad_; i++) {
        for (uint32_t j = 0; j < total_cols_; j++) {
            double v = x.get(i, j);
            result.set(i, j, v * v);
        }
    }
    return result;
}

// ============================================================
// ParUpperDiagonalSum
// ============================================================

ParUpperDiagonalSum::ParUpperDiagonalSum(const CkksParameter& param,
                                         Duo shape,
                                         Duo head_shape,
                                         uint32_t n_heads,
                                         uint32_t init_level)
    : Layer(param) {
    assert(init_level >= 1);
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
    segment_len_ = H_ * n_;
    c_ = n_slot_ / segment_len_;
    n_cts_ = m_ / c_;
}

uint32_t ParUpperDiagonalSum::total_cts() const {
    return n_cts_;
}

vector<double> ParUpperDiagonalSum::build_valid_mask(uint32_t ct_idx, double value) const {
    assert(ct_idx < total_cts());
    vector<double> mask(n_slot_, 0.0);
    for (uint32_t local_diag = 0; local_diag < c_; local_diag++) {
        uint32_t segment_base = local_diag * segment_len_;
        for (uint32_t t = 0; t < n_; t++) {
            for (uint32_t h = 0; h < H_; h++) {
                if (is_valid_slot(ct_idx, local_diag, t, h, n_prepad_, total_cols_, m_prepad_, H_prepad_, m_, c_)) {
                    mask[segment_base + t * H_ + h] = value;
                }
            }
        }
    }
    return mask;
}

CkksPlaintextRingt ParUpperDiagonalSum::generate_mask_pt(CkksContext& ctx, uint32_t ct_idx) const {
    return ctx.encode_ringt(build_valid_mask(ct_idx, 1.0), param_.get_q(level_));
}

void ParUpperDiagonalSum::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    mask_pt_.resize(total_cts());
    for (uint32_t ct_idx = 0; ct_idx < total_cts(); ct_idx++) {
        mask_pt_[ct_idx] = generate_mask_pt(ctx, ct_idx);
    }
}

CkksCiphertext ParUpperDiagonalSum::reduce_local_diags(CkksContext& ctx, const CkksCiphertext& ct) const {
    CkksCiphertext result = ct.copy();
    for (uint32_t step = 1; step < c_; step <<= 1) {
        result = ctx.add(result, ctx.rotate(result, (int)(step * segment_len_)));
    }
    return result;
}

FeatureMatEncrypted ParUpperDiagonalSum::run(CkksContext& ctx, const FeatureMatEncrypted& x) {
    assert(x.level == level_);
    assert(x.shape[0] == n_prepad_ && x.shape[1] == total_cols_);
    assert(x.head_shape[0] == n_prepad_ && x.head_shape[1] == m_prepad_);
    assert(x.matmul_block_size == m_);
    assert(x.data.size() == total_cts());
    assert(mask_pt_.size() == total_cts());

    double D = param_.get_default_scale();
    vector<CkksCiphertext> partial(total_cts());
    parallel_for(total_cts(), th_nums, ctx,
                 [&](CkksContext& ctx_copy, int idx) { partial[idx] = reduce_local_diags(ctx_copy, x.data[idx]); });

    CkksCiphertext head_sum = partial[0].copy();
    for (uint32_t ct_idx = 1; ct_idx < total_cts(); ct_idx++) {
        head_sum = ctx.add(head_sum, partial[ct_idx]);
    }

    FeatureMatEncrypted result(&ctx, level_ - 1);
    result.shape = x.shape;
    result.head_shape = x.head_shape;
    result.matmul_block_size = m_;
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.data.resize(total_cts());

    parallel_for(total_cts(), th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        auto mask_mul = ctx_copy.ringt_to_mul(mask_pt_[idx], level_);
        result.data[idx] = ctx_copy.rescale(ctx_copy.mult_plain_mul(head_sum, mask_mul), D);
    });

    result.level = level_ - 1;
    return result;
}

Array<double, 2> ParUpperDiagonalSum::run_plaintext(const Array<double, 2>& x) const {
    auto x_shape = x.get_shape();
    if (x_shape[0] != n_prepad_ || x_shape[1] != total_cols_) {
        throw runtime_error("ParUpperDiagonalSum plaintext input shape mismatch");
    }
    Array<double, 2> result({n_prepad_, total_cols_});
    for (uint32_t i = 0; i < n_prepad_; i++) {
        for (uint32_t h = 0; h < H_prepad_; h++) {
            double sum = 0.0;
            uint32_t head_base = h * m_prepad_;
            for (uint32_t col = 0; col < m_prepad_; col++) {
                sum += x.get(i, head_base + col);
            }
            for (uint32_t col = 0; col < m_prepad_; col++) {
                result.set(i, head_base + col, sum);
            }
        }
    }
    return result;
}

// ============================================================
// ParUpperDiagonalInverseInit
// ============================================================

ParUpperDiagonalInverseInit::ParUpperDiagonalInverseInit(const CkksParameter& param,
                                                         Duo shape,
                                                         Duo head_shape,
                                                         uint32_t n_heads,
                                                         uint32_t init_level)
    : Layer(param) {
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
    segment_len_ = H_ * n_;
    c_ = n_slot_ / segment_len_;
    n_cts_ = m_ / c_;
}

uint32_t ParUpperDiagonalInverseInit::total_cts() const {
    return n_cts_;
}

vector<double> ParUpperDiagonalInverseInit::build_valid_mask(uint32_t ct_idx, double value) const {
    assert(ct_idx < total_cts());
    vector<double> mask(n_slot_, 0.0);
    for (uint32_t local_diag = 0; local_diag < c_; local_diag++) {
        uint32_t segment_base = local_diag * segment_len_;
        for (uint32_t t = 0; t < n_; t++) {
            for (uint32_t h = 0; h < H_; h++) {
                if (is_valid_slot(ct_idx, local_diag, t, h, n_prepad_, total_cols_, m_prepad_, H_prepad_, m_, c_)) {
                    mask[segment_base + t * H_ + h] = value;
                }
            }
        }
    }
    return mask;
}

CkksPlaintextRingt ParUpperDiagonalInverseInit::generate_two_pt(CkksContext& ctx, uint32_t ct_idx) const {
    return ctx.encode_ringt(build_valid_mask(ct_idx, 2.0), param_.get_default_scale());
}

void ParUpperDiagonalInverseInit::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    two_pt_.resize(total_cts());
    for (uint32_t ct_idx = 0; ct_idx < total_cts(); ct_idx++) {
        two_pt_[ct_idx] = generate_two_pt(ctx, ct_idx);
    }
}

FeatureMatEncrypted ParUpperDiagonalInverseInit::run(CkksContext& ctx, const FeatureMatEncrypted& b) {
    assert(b.level == level_);
    assert(b.shape[0] == n_prepad_ && b.shape[1] == total_cols_);
    assert(b.head_shape[0] == n_prepad_ && b.head_shape[1] == m_prepad_);
    assert(b.matmul_block_size == m_);
    assert(b.data.size() == total_cts());
    assert(two_pt_.size() == total_cts());

    FeatureMatEncrypted result(&ctx, level_);
    result.shape = b.shape;
    result.head_shape = b.head_shape;
    result.matmul_block_size = m_;
    result.n_channel = b.n_channel;
    result.n_channel_per_ct = b.n_channel_per_ct;
    result.data.resize(total_cts());

    parallel_for(total_cts(), th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        auto zero = ctx_copy.sub(b.data[idx], b.data[idx]);
        result.data[idx] = ctx_copy.add_plain_ringt(ctx_copy.sub(zero, b.data[idx]), two_pt_[idx]);
    });

    result.level = level_;
    return result;
}

Array<double, 2> ParUpperDiagonalInverseInit::run_plaintext(const Array<double, 2>& b) const {
    auto b_shape = b.get_shape();
    if (b_shape[0] != n_prepad_ || b_shape[1] != total_cols_) {
        throw runtime_error("ParUpperDiagonalInverseInit plaintext input shape mismatch");
    }
    Array<double, 2> result({n_prepad_, total_cols_});
    for (uint32_t i = 0; i < n_prepad_; i++) {
        for (uint32_t j = 0; j < total_cols_; j++) {
            result.set(i, j, 2.0 - b.get(i, j));
        }
    }
    return result;
}

// ============================================================
// ParUpperDiagonalInverseIter
// ============================================================

ParUpperDiagonalInverseIter::ParUpperDiagonalInverseIter(const CkksParameter& param,
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
    segment_len_ = H_ * n_;
    c_ = n_slot_ / segment_len_;
    n_cts_ = m_ / c_;
}

uint32_t ParUpperDiagonalInverseIter::total_cts() const {
    return n_cts_;
}

vector<double> ParUpperDiagonalInverseIter::build_valid_mask(uint32_t ct_idx, double value) const {
    assert(ct_idx < total_cts());
    vector<double> mask(n_slot_, 0.0);
    for (uint32_t local_diag = 0; local_diag < c_; local_diag++) {
        uint32_t segment_base = local_diag * segment_len_;
        for (uint32_t t = 0; t < n_; t++) {
            for (uint32_t h = 0; h < H_; h++) {
                if (is_valid_slot(ct_idx, local_diag, t, h, n_prepad_, total_cols_, m_prepad_, H_prepad_, m_, c_)) {
                    mask[segment_base + t * H_ + h] = value;
                }
            }
        }
    }
    return mask;
}

CkksPlaintextRingt ParUpperDiagonalInverseIter::generate_one_pt(CkksContext& ctx, uint32_t ct_idx) const {
    double D = param_.get_default_scale();
    double scale = (static_cast<double>(param_.get_q(level_)) / D) * (static_cast<double>(param_.get_q(level_)) / D) *
                   static_cast<double>(param_.get_q(level_ - 1));
    return ctx.encode_ringt(build_valid_mask(ct_idx, 1.0), scale);
}

CkksPlaintextRingt ParUpperDiagonalInverseIter::generate_two_pt(CkksContext& ctx, uint32_t ct_idx) const {
    return ctx.encode_ringt(build_valid_mask(ct_idx, 2.0), param_.get_q(level_));
}

void ParUpperDiagonalInverseIter::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    one_pt_.resize(total_cts());
    two_pt_.resize(total_cts());
    for (uint32_t ct_idx = 0; ct_idx < total_cts(); ct_idx++) {
        one_pt_[ct_idx] = generate_one_pt(ctx, ct_idx);
        two_pt_[ct_idx] = generate_two_pt(ctx, ct_idx);
    }
}

FeatureMatEncrypted
ParUpperDiagonalInverseIter::run(CkksContext& ctx, const FeatureMatEncrypted& a, const FeatureMatEncrypted& b) {
    assert(a.level == level_ && b.level == level_);
    assert(a.shape[0] == n_prepad_ && a.shape[1] == total_cols_);
    assert(b.shape == a.shape);
    assert(a.head_shape[0] == n_prepad_ && a.head_shape[1] == m_prepad_);
    assert(b.head_shape == a.head_shape);
    assert(a.matmul_block_size == m_);
    assert(b.matmul_block_size == m_);
    assert(a.data.size() == total_cts());
    assert(b.data.size() == total_cts());
    assert(one_pt_.size() == total_cts());
    assert(two_pt_.size() == total_cts());

    double D = param_.get_default_scale();
    double q_L = static_cast<double>(param_.get_q(level_));
    double q_L1 = static_cast<double>(param_.get_q(level_ - 1));
    double ba_scale = D / q_L * D;
    double one_a_scale = q_L / D * q_L1;

    FeatureMatEncrypted result(&ctx, level_ - 2);
    result.shape = a.shape;
    result.head_shape = a.head_shape;
    result.matmul_block_size = m_;
    result.n_channel = a.n_channel;
    result.n_channel_per_ct = a.n_channel_per_ct;
    result.data.resize(total_cts());

    parallel_for(total_cts(), th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        auto ba = ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(b.data[idx], a.data[idx])), ba_scale);  // L-1

        auto one_mul = ctx_copy.ringt_to_mul(one_pt_[idx], level_);
        auto one_a = ctx_copy.rescale(ctx_copy.mult_plain_mul(a.data[idx], one_mul), one_a_scale);  // L-1

        auto product = ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(one_a, ba)), D);  // level L-2, scale D

        auto two_mul = ctx_copy.ringt_to_mul(two_pt_[idx], level_);
        auto two_a = ctx_copy.rescale(ctx_copy.mult_plain_mul(a.data[idx], two_mul), D);  // L-1, scale D
        auto two_a_drop = ctx_copy.drop_level(two_a);

        result.data[idx] = ctx_copy.sub(two_a_drop, product);
    });

    result.level = level_ - 2;
    return result;
}

Array<double, 2> ParUpperDiagonalInverseIter::run_plaintext(const Array<double, 2>& a,
                                                            const Array<double, 2>& b) const {
    auto a_shape = a.get_shape();
    auto b_shape = b.get_shape();
    if (a_shape[0] != n_prepad_ || a_shape[1] != total_cols_ || b_shape != a_shape) {
        throw runtime_error("ParUpperDiagonalInverseIter plaintext input shape mismatch");
    }
    Array<double, 2> result({n_prepad_, total_cols_});
    for (uint32_t i = 0; i < n_prepad_; i++) {
        for (uint32_t j = 0; j < total_cols_; j++) {
            double av = a.get(i, j);
            result.set(i, j, av * (2.0 - b.get(i, j) * av));
        }
    }
    return result;
}

// ============================================================
// ParUpperDiagonalMultCt
// ============================================================

ParUpperDiagonalMultCt::ParUpperDiagonalMultCt(const CkksParameter& param,
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
    segment_len_ = H_ * n_;
    c_ = n_slot_ / segment_len_;
    n_cts_ = m_ / c_;
}

uint32_t ParUpperDiagonalMultCt::total_cts() const {
    return n_cts_;
}

vector<double> ParUpperDiagonalMultCt::build_valid_mask(uint32_t ct_idx, double value) const {
    assert(ct_idx < total_cts());
    vector<double> mask(n_slot_, 0.0);
    for (uint32_t local_diag = 0; local_diag < c_; local_diag++) {
        uint32_t segment_base = local_diag * segment_len_;
        for (uint32_t t = 0; t < n_; t++) {
            for (uint32_t h = 0; h < H_; h++) {
                if (is_valid_slot(ct_idx, local_diag, t, h, n_prepad_, total_cols_, m_prepad_, H_prepad_, m_, c_)) {
                    mask[segment_base + t * H_ + h] = value;
                }
            }
        }
    }
    return mask;
}

CkksPlaintextRingt ParUpperDiagonalMultCt::generate_mask_pt(CkksContext& ctx, uint32_t ct_idx) const {
    return ctx.encode_ringt(build_valid_mask(ct_idx, 1.0), mask_scale_after_natural_square(param_, level_));
}

void ParUpperDiagonalMultCt::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    mask_pt_.resize(total_cts());
    for (uint32_t ct_idx = 0; ct_idx < total_cts(); ct_idx++) {
        mask_pt_[ct_idx] = generate_mask_pt(ctx, ct_idx);
    }
}

FeatureMatEncrypted
ParUpperDiagonalMultCt::run(CkksContext& ctx, const FeatureMatEncrypted& a, const FeatureMatEncrypted& b) {
    assert(a.level == level_ && b.level == level_);
    assert(a.shape[0] == n_prepad_ && a.shape[1] == total_cols_);
    assert(b.shape == a.shape);
    assert(a.head_shape[0] == n_prepad_ && a.head_shape[1] == m_prepad_);
    assert(b.head_shape == a.head_shape);
    assert(a.matmul_block_size == m_);
    assert(b.matmul_block_size == m_);
    assert(a.data.size() == total_cts());
    assert(b.data.size() == total_cts());
    assert(mask_pt_.size() == total_cts());

    double D = param_.get_default_scale();
    double product_scale = natural_square_scale(param_, level_);
    FeatureMatEncrypted result(&ctx, level_ - 2);
    result.shape = a.shape;
    result.head_shape = a.head_shape;
    result.matmul_block_size = m_;
    result.n_channel = a.n_channel;
    result.n_channel_per_ct = a.n_channel_per_ct;
    result.data.resize(total_cts());

    parallel_for(total_cts(), th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        auto product = ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(a.data[idx], b.data[idx])), product_scale);
        auto mask_mul = ctx_copy.ringt_to_mul(mask_pt_[idx], level_ - 1);
        result.data[idx] = ctx_copy.rescale(ctx_copy.mult_plain_mul(product, mask_mul), D);
    });

    result.level = level_ - 2;
    return result;
}

Array<double, 2> ParUpperDiagonalMultCt::run_plaintext(const Array<double, 2>& a, const Array<double, 2>& b) const {
    auto a_shape = a.get_shape();
    auto b_shape = b.get_shape();
    if (a_shape[0] != n_prepad_ || a_shape[1] != total_cols_ || b_shape != a_shape) {
        throw runtime_error("ParUpperDiagonalMultCt plaintext input shape mismatch");
    }
    Array<double, 2> result({n_prepad_, total_cols_});
    for (uint32_t i = 0; i < n_prepad_; i++) {
        for (uint32_t j = 0; j < total_cols_; j++) {
            result.set(i, j, a.get(i, j) * b.get(i, j));
        }
    }
    return result;
}
