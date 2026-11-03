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

#include "par_block_col_major_softmax.h"
#include "layer_util.h"
#include <cassert>
#include <stdexcept>

using namespace std;
using namespace lattisense;

namespace {

double natural_square_scale(const CkksParameter& param, uint32_t level) {
    double D = param.get_default_scale();
    return D / static_cast<double>(param.get_q(level)) * D;
}

double mask_scale_after_natural_square(const CkksParameter& param, uint32_t level) {
    double D = param.get_default_scale();
    return static_cast<double>(param.get_q(level)) / D * static_cast<double>(param.get_q(level - 1));
}

}  // namespace

ParBlockColMajorMultipleSquare::ParBlockColMajorMultipleSquare(const CkksParameter& param,
                                                               Duo shape,
                                                               Duo head_shape,
                                                               uint32_t n_heads,
                                                               uint32_t init_level,
                                                               uint32_t block_size)
    : Layer(param), layout_(shape, head_shape, n_heads, block_size, param.get_n() / 2) {
    assert(init_level >= 2);
    level_ = init_level;
    n_prepad_ = shape[0];
    total_cols_ = shape[1];
    m_prepad_ = head_shape[1];
    H_prepad_ = n_heads;
    m_ = block_size;
}

uint32_t ParBlockColMajorMultipleSquare::total_cts() const {
    return layout_.total_cts();
}

vector<double> ParBlockColMajorMultipleSquare::build_valid_mask(uint32_t ct_idx, double value) const {
    return layout_.constant(ct_idx, value);
}

CkksPlaintextRingt ParBlockColMajorMultipleSquare::generate_mask_pt(CkksContext& ctx, uint32_t ct_idx) const {
    return ctx.encode_ringt(build_valid_mask(ct_idx, 1.0), mask_scale_after_natural_square(param_, level_));
}

void ParBlockColMajorMultipleSquare::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    mask_pt_.resize(total_cts());
    for (uint32_t ct_idx = 0; ct_idx < total_cts(); ct_idx++) {
        mask_pt_[ct_idx] = generate_mask_pt(ctx, ct_idx);
    }
}

FeatureMatEncrypted ParBlockColMajorMultipleSquare::run(CkksContext& ctx, const FeatureMatEncrypted& x) {
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

Array<double, 2> ParBlockColMajorMultipleSquare::run_plaintext(const Array<double, 2>& x) const {
    auto x_shape = x.get_shape();
    if (x_shape[0] != n_prepad_ || x_shape[1] != total_cols_) {
        throw runtime_error("ParBlockColMajorMultipleSquare plaintext input shape mismatch");
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
// ParBlockColMajorHeadColSum
// ============================================================

ParBlockColMajorHeadColSum::ParBlockColMajorHeadColSum(const CkksParameter& param,
                                                       Duo shape,
                                                       Duo head_shape,
                                                       uint32_t n_heads,
                                                       uint32_t init_level,
                                                       uint32_t block_size)
    : Layer(param), layout_(shape, head_shape, n_heads, block_size, param.get_n() / 2) {
    assert(init_level >= 2);
    level_ = init_level;
    n_prepad_ = shape[0];
    total_cols_ = shape[1];
    m_prepad_ = head_shape[1];
    H_prepad_ = n_heads;
    m_ = block_size;
}

uint32_t ParBlockColMajorHeadColSum::total_cts() const {
    return layout_.total_cts();
}

vector<double> ParBlockColMajorHeadColSum::build_valid_mask(uint32_t ct_idx, double value) const {
    return layout_.constant(ct_idx, value);
}

CkksPlaintextRingt ParBlockColMajorHeadColSum::generate_mask_pt(CkksContext& ctx, uint32_t ct_idx) const {
    return ctx.encode_ringt(build_valid_mask(ct_idx, 1.0), param_.get_q(level_ - 1));
}

CkksPlaintextRingt ParBlockColMajorHeadColSum::generate_input_mask_pt(CkksContext& ctx, uint32_t ct_idx) const {
    return ctx.encode_ringt(build_valid_mask(ct_idx, 1.0), param_.get_q(level_));
}

void ParBlockColMajorHeadColSum::prepare_weight() {
    auto ctx = CkksContext::create_empty_context(param_);
    mask_pt_.resize(total_cts());
    input_mask_pt_.resize(total_cts());
    for (uint32_t idx = 0; idx < total_cts(); idx++) {
        mask_pt_[idx] = generate_mask_pt(ctx, idx);
        input_mask_pt_[idx] = generate_input_mask_pt(ctx, idx);
    }
}

CkksCiphertext ParBlockColMajorHeadColSum::reduce_local_columns(CkksContext& ctx, const CkksCiphertext& ct) const {
    auto result = ct.copy();
    for (uint32_t step = 1; step < layout_.d; step <<= 1)
        result = ctx.add(result, ctx.rotate(result, int(step * layout_.d * layout_.u)));
    return result;
}

FeatureMatEncrypted ParBlockColMajorHeadColSum::run(CkksContext& ctx, const FeatureMatEncrypted& x) {
    assert(x.level == level_ && x.data.size() == total_cts());
    assert(x.shape == Duo({n_prepad_, total_cols_}) && x.matmul_block_size == m_);
    double S = param_.get_default_scale();
    std::vector<CkksCiphertext> partial(total_cts());
    parallel_for(total_cts(), th_nums, ctx, [&](CkksContext& c, int idx) {
        auto masked = c.rescale(c.mult_plain_mul(x.data[idx], c.ringt_to_mul(input_mask_pt_[idx], level_)), S);
        partial[idx] = reduce_local_columns(c, masked);
    });
    FeatureMatEncrypted result(&ctx, level_ - 2);
    result.shape = x.shape;
    result.head_shape = x.head_shape;
    result.matmul_block_size = m_;
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.data.resize(total_cts());
    for (uint32_t mb = 0; mb < layout_.n_mb; mb++)
        for (uint32_t bi = 0; bi < layout_.br; bi++)
            for (uint32_t g = 0; g < layout_.groups; g++) {
                uint32_t base = mb * layout_.cts_per_mb + bi * layout_.groups + g;
                auto sum = partial[base].copy();
                for (uint32_t bj = 1; bj < layout_.bc; bj++)
                    sum = ctx.add(sum, partial[base + bj * layout_.br * layout_.groups]);
                for (uint32_t bj = 0; bj < layout_.bc; bj++) {
                    uint32_t idx = base + bj * layout_.br * layout_.groups;
                    result.data[idx] =
                        ctx.rescale(ctx.mult_plain_mul(sum, ctx.ringt_to_mul(mask_pt_[idx], level_ - 1)), S);
                }
            }
    result.level = level_ - 2;
    return result;
}

Array<double, 2> ParBlockColMajorHeadColSum::run_plaintext(const Array<double, 2>& x) const {
    auto x_shape = x.get_shape();
    if (x_shape[0] != n_prepad_ || x_shape[1] != total_cols_) {
        throw runtime_error("ParBlockColMajorHeadColSum plaintext input shape mismatch");
    }
    Array<double, 2> result({n_prepad_, total_cols_});
    for (uint32_t i = 0; i < n_prepad_; i++) {
        for (uint32_t h = 0; h < div_ceil(total_cols_, m_prepad_); h++) {
            double sum = 0.0;
            uint32_t head_base = h * m_prepad_;
            for (uint32_t col = 0; col < m_prepad_ && head_base + col < total_cols_; col++) {
                sum += x.get(i, head_base + col);
            }
            for (uint32_t col = 0; col < m_prepad_ && head_base + col < total_cols_; col++) {
                result.set(i, head_base + col, sum);
            }
        }
    }
    return result;
}

// ============================================================
// ParBlockColMajorInverseInit
// ============================================================

ParBlockColMajorInverseInit::ParBlockColMajorInverseInit(const CkksParameter& param,
                                                         Duo shape,
                                                         Duo head_shape,
                                                         uint32_t n_heads,
                                                         uint32_t init_level,
                                                         uint32_t block_size)
    : Layer(param), layout_(shape, head_shape, n_heads, block_size, param.get_n() / 2) {
    level_ = init_level;
    n_prepad_ = shape[0];
    total_cols_ = shape[1];
    m_prepad_ = head_shape[1];
    H_prepad_ = n_heads;
    m_ = block_size;
}

uint32_t ParBlockColMajorInverseInit::total_cts() const {
    return layout_.total_cts();
}

vector<double> ParBlockColMajorInverseInit::build_valid_mask(uint32_t ct_idx, double value) const {
    return layout_.constant(ct_idx, value);
}

CkksPlaintextRingt ParBlockColMajorInverseInit::generate_two_pt(CkksContext& ctx, uint32_t ct_idx) const {
    return ctx.encode_ringt(build_valid_mask(ct_idx, 2.0), param_.get_default_scale());
}

void ParBlockColMajorInverseInit::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    two_pt_.resize(total_cts());
    for (uint32_t ct_idx = 0; ct_idx < total_cts(); ct_idx++) {
        two_pt_[ct_idx] = generate_two_pt(ctx, ct_idx);
    }
}

FeatureMatEncrypted ParBlockColMajorInverseInit::run(CkksContext& ctx, const FeatureMatEncrypted& b) {
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

Array<double, 2> ParBlockColMajorInverseInit::run_plaintext(const Array<double, 2>& b) const {
    auto b_shape = b.get_shape();
    if (b_shape[0] != n_prepad_ || b_shape[1] != total_cols_) {
        throw runtime_error("ParBlockColMajorInverseInit plaintext input shape mismatch");
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
// ParBlockColMajorInverseIter
// ============================================================

ParBlockColMajorInverseIter::ParBlockColMajorInverseIter(const CkksParameter& param,
                                                         Duo shape,
                                                         Duo head_shape,
                                                         uint32_t n_heads,
                                                         uint32_t init_level,
                                                         uint32_t block_size)
    : Layer(param), layout_(shape, head_shape, n_heads, block_size, param.get_n() / 2) {
    assert(init_level >= 2);
    level_ = init_level;
    n_prepad_ = shape[0];
    total_cols_ = shape[1];
    m_prepad_ = head_shape[1];
    H_prepad_ = n_heads;
    m_ = block_size;
}

uint32_t ParBlockColMajorInverseIter::total_cts() const {
    return layout_.total_cts();
}

vector<double> ParBlockColMajorInverseIter::build_valid_mask(uint32_t ct_idx, double value) const {
    return layout_.constant(ct_idx, value);
}

CkksPlaintextRingt ParBlockColMajorInverseIter::generate_one_pt(CkksContext& ctx, uint32_t ct_idx) const {
    double D = param_.get_default_scale();
    double scale = (static_cast<double>(param_.get_q(level_)) / D) * (static_cast<double>(param_.get_q(level_)) / D) *
                   static_cast<double>(param_.get_q(level_ - 1));
    return ctx.encode_ringt(build_valid_mask(ct_idx, 1.0), pcm_checked_scale(scale));
}

CkksPlaintextRingt ParBlockColMajorInverseIter::generate_two_pt(CkksContext& ctx, uint32_t ct_idx) const {
    return ctx.encode_ringt(build_valid_mask(ct_idx, 2.0), param_.get_q(level_));
}

void ParBlockColMajorInverseIter::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    one_pt_.resize(total_cts());
    two_pt_.resize(total_cts());
    for (uint32_t ct_idx = 0; ct_idx < total_cts(); ct_idx++) {
        one_pt_[ct_idx] = generate_one_pt(ctx, ct_idx);
        two_pt_[ct_idx] = generate_two_pt(ctx, ct_idx);
    }
}

FeatureMatEncrypted
ParBlockColMajorInverseIter::run(CkksContext& ctx, const FeatureMatEncrypted& a, const FeatureMatEncrypted& b) {
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

Array<double, 2> ParBlockColMajorInverseIter::run_plaintext(const Array<double, 2>& a,
                                                            const Array<double, 2>& b) const {
    auto a_shape = a.get_shape();
    auto b_shape = b.get_shape();
    if (a_shape[0] != n_prepad_ || a_shape[1] != total_cols_ || b_shape != a_shape) {
        throw runtime_error("ParBlockColMajorInverseIter plaintext input shape mismatch");
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
// ParBlockColMajorCtMul
// ============================================================

ParBlockColMajorCtMul::ParBlockColMajorCtMul(const CkksParameter& param,
                                             Duo shape,
                                             Duo head_shape,
                                             uint32_t n_heads,
                                             uint32_t init_level,
                                             uint32_t block_size)
    : Layer(param), layout_(shape, head_shape, n_heads, block_size, param.get_n() / 2) {
    assert(init_level >= 2);
    level_ = init_level;
    n_prepad_ = shape[0];
    total_cols_ = shape[1];
    m_prepad_ = head_shape[1];
    H_prepad_ = n_heads;
    m_ = block_size;
}

uint32_t ParBlockColMajorCtMul::total_cts() const {
    return layout_.total_cts();
}

vector<double> ParBlockColMajorCtMul::build_valid_mask(uint32_t ct_idx, double value) const {
    return layout_.constant(ct_idx, value);
}

CkksPlaintextRingt ParBlockColMajorCtMul::generate_mask_pt(CkksContext& ctx, uint32_t ct_idx) const {
    return ctx.encode_ringt(build_valid_mask(ct_idx, 1.0), mask_scale_after_natural_square(param_, level_));
}

void ParBlockColMajorCtMul::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    mask_pt_.resize(total_cts());
    for (uint32_t ct_idx = 0; ct_idx < total_cts(); ct_idx++) {
        mask_pt_[ct_idx] = generate_mask_pt(ctx, ct_idx);
    }
}

FeatureMatEncrypted
ParBlockColMajorCtMul::run(CkksContext& ctx, const FeatureMatEncrypted& a, const FeatureMatEncrypted& b) {
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

Array<double, 2> ParBlockColMajorCtMul::run_plaintext(const Array<double, 2>& a, const Array<double, 2>& b) const {
    auto a_shape = a.get_shape();
    auto b_shape = b.get_shape();
    if (a_shape[0] != n_prepad_ || a_shape[1] != total_cols_ || b_shape != a_shape) {
        throw runtime_error("ParBlockColMajorCtMul plaintext input shape mismatch");
    }
    Array<double, 2> result({n_prepad_, total_cols_});
    for (uint32_t i = 0; i < n_prepad_; i++) {
        for (uint32_t j = 0; j < total_cols_; j++) {
            result.set(i, j, a.get(i, j) * b.get(i, j));
        }
    }
    return result;
}
