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

#include "par_block_col_major_polyactrn.h"
#include "layer_util.h"
#include <cassert>

using namespace std;
using namespace lattisense;

// ============================================================
// ParBlockColMajorPolyActRNGamma
// ============================================================
ParBlockColMajorPolyActRNGamma::ParBlockColMajorPolyActRNGamma(const CkksParameter& param,
                                                               Duo shape,
                                                               uint32_t block_size,
                                                               uint32_t n_heads,
                                                               uint32_t K,
                                                               uint32_t init_level,
                                                               Array<double, 1>&& gamma)
    : Layer(param), gamma_vals_(move(gamma)) {
    level_ = init_level;
    m_ = shape[0];
    total_dim_ = shape[1];
    d_ = block_size;
    n_heads_ = n_heads;
    K_ = K;
    assert(K_ > 0);
    assert(total_dim_ % (K_ * n_heads_) == 0);
    assert(gamma_vals_.get_shape()[0] >= total_dim_);
    cols_per_head_ = total_dim_ / (K_ * n_heads_);
    n_h_padded_ = next_pow2(n_heads);
    n_slot_ = param_.get_n() / 2;

    if ((uint32_t)n_slot_ >= n_h_padded_ * d_ * d_) {
        S_ = n_h_padded_;
        chunk_size_ = n_h_padded_ * d_ * d_;
        n_cts_per_block_idx_ = 1;
    } else {
        S_ = n_slot_ / (d_ * d_);
        chunk_size_ = n_slot_;
        n_cts_per_block_idx_ = n_h_padded_ / S_;
    }
    num_chunks_ = n_slot_ / chunk_size_;
    num_block_rows_ = div_ceil(m_, d_);
    num_block_cols_ = div_ceil(cols_per_head_, d_);
}

CkksPlaintextRingt
ParBlockColMajorPolyActRNGamma::generate_gamma_pt(CkksContext& ctx, uint32_t mb, uint32_t bj, uint32_t g) const {
    double q_L = param_.get_q(level_);
    vector<double> gamma_vec(n_slot_, 0.0);
    for (uint32_t h_local = 0; h_local < S_; h_local++) {
        uint32_t h = g * S_ + h_local;
        for (uint32_t col = 0; col < d_; col++) {
            uint32_t actual_col = bj * d_ + col;
            for (uint32_t row = 0; row < d_; row++) {
                uint32_t base_slot = (row + d_ * col) * S_ + h_local;
                for (uint32_t ci = 0; ci < num_chunks_; ci++) {
                    uint32_t slot = ci * chunk_size_ + base_slot;
                    if (h < n_heads_ && actual_col < cols_per_head_) {
                        uint32_t global_col = mb * n_heads_ * cols_per_head_ + h * cols_per_head_ + actual_col;
                        gamma_vec[slot] = gamma_vals_.get(global_col);
                    }
                }
            }
        }
    }
    return ctx.encode_ringt(gamma_vec, q_L);
}

void ParBlockColMajorPolyActRNGamma::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);

    uint32_t n_gamma_vecs = K_ * num_block_cols_ * n_cts_per_block_idx_;
    gamma_pt_.resize(n_gamma_vecs);

    for (uint32_t mb = 0; mb < K_; mb++) {
        for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
            for (uint32_t g = 0; g < n_cts_per_block_idx_; g++) {
                uint32_t idx = (mb * num_block_cols_ + bj) * n_cts_per_block_idx_ + g;
                gamma_pt_[idx] = generate_gamma_pt(ctx, mb, bj, g);
            }
        }
    }
}

FeatureMatEncrypted ParBlockColMajorPolyActRNGamma::run(CkksContext& ctx, const FeatureMatEncrypted& x) {
    double D = param_.get_default_scale();
    uint32_t cts_per_mb = num_block_rows_ * num_block_cols_ * n_cts_per_block_idx_;
    uint32_t total_cts = K_ * cts_per_mb;
    assert(x.data.size() >= static_cast<size_t>(total_cts));

    FeatureMatEncrypted result(&ctx, level_ - 1);
    result.shape = {m_, total_dim_};
    result.matmul_block_size = d_;
    result.data.resize(total_cts);

    parallel_for(total_cts, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        uint32_t mb = ct_idx / cts_per_mb;
        uint32_t local_ct_idx = ct_idx % cts_per_mb;
        uint32_t block_idx = local_ct_idx / n_cts_per_block_idx_;
        uint32_t g = local_ct_idx % n_cts_per_block_idx_;
        uint32_t bj = block_idx / num_block_rows_;
        uint32_t gamma_idx = (mb * num_block_cols_ + bj) * n_cts_per_block_idx_ + g;

        auto gamma_mul = ctx_copy.ringt_to_mul(gamma_pt_[gamma_idx], level_);
        auto product = ctx_copy.mult_plain_mul(x.data[ct_idx], gamma_mul);
        result.data[ct_idx] = ctx_copy.rescale(product, D);
    });

    result.level = level_ - 1;
    return result;
}

Array<double, 2> ParBlockColMajorPolyActRNGamma::run_plaintext(const Array<double, 2>& x) const {
    Array<double, 2> result({m_, total_dim_});
    for (uint32_t i = 0; i < m_; i++) {
        for (uint32_t j = 0; j < total_dim_; j++) {
            result.set(i, j, x.get(i, j) * gamma_vals_.get(j));
        }
    }
    return result;
}

// ============================================================
// ParBlockColMajorPolyActRNPoly
// ============================================================

ParBlockColMajorPolyActRNPoly::ParBlockColMajorPolyActRNPoly(const CkksParameter& param,
                                                             Duo shape,
                                                             uint32_t block_size,
                                                             uint32_t n_heads,
                                                             uint32_t K,
                                                             uint32_t init_level,
                                                             Array<double, 2>&& coeffs,
                                                             uint32_t degree)
    : Layer(param), degree_(degree), coeffs_(move(coeffs)) {
    assert(degree_ == 2 || degree_ == 4);
    assert(coeffs_.get_shape()[0] == degree_ + 1);
    assert(coeffs_.get_shape()[1] >= shape[1]);
    level_ = init_level;
    m_ = shape[0];
    total_dim_ = shape[1];
    d_ = block_size;
    n_heads_ = n_heads;
    K_ = K;
    assert(K_ > 0);
    assert(total_dim_ % (K_ * n_heads_) == 0);
    cols_per_head_ = total_dim_ / (K_ * n_heads_);
    n_h_padded_ = next_pow2(n_heads);
    n_slot_ = param_.get_n() / 2;

    if ((uint32_t)n_slot_ >= n_h_padded_ * d_ * d_) {
        S_ = n_h_padded_;
        chunk_size_ = n_h_padded_ * d_ * d_;
        n_cts_per_block_idx_ = 1;
    } else {
        S_ = n_slot_ / (d_ * d_);
        chunk_size_ = n_slot_;
        n_cts_per_block_idx_ = n_h_padded_ / S_;
    }
    num_chunks_ = n_slot_ / chunk_size_;
    num_block_rows_ = div_ceil(m_, d_);
    num_block_cols_ = div_ceil(cols_per_head_, d_);
}

// Helper: encode a per-column multiplicative plaintext for coefficient row `coeff_row`
// in expanded par format, for a given (mb, bj, g) tuple.
static vector<double> build_par_coeff_vec(uint32_t n_slot,
                                          uint32_t chunk_size,
                                          uint32_t num_chunks,
                                          uint32_t d,
                                          uint32_t S,
                                          uint32_t mb,
                                          uint32_t bj,
                                          uint32_t g,
                                          uint32_t n_heads,
                                          uint32_t cols_per_head,
                                          const Array<double, 2>& coeffs,
                                          uint32_t coeff_row) {
    vector<double> vec(n_slot, 0.0);
    for (uint32_t h_local = 0; h_local < S; h_local++) {
        uint32_t h = g * S + h_local;
        for (uint32_t col = 0; col < d; col++) {
            uint32_t actual_col = bj * d + col;
            for (uint32_t row = 0; row < d; row++) {
                uint32_t base_slot = (row + d * col) * S + h_local;
                for (uint32_t ci = 0; ci < num_chunks; ci++) {
                    uint32_t slot = ci * chunk_size + base_slot;
                    if (h < n_heads && actual_col < cols_per_head) {
                        uint32_t global_col = mb * n_heads * cols_per_head + h * cols_per_head + actual_col;
                        vec[slot] = coeffs.get(coeff_row, global_col);
                    }
                }
            }
        }
    }
    return vec;
}

CkksPlaintextRingt ParBlockColMajorPolyActRNPoly::generate_coeff_pt(CkksContext& ctx,
                                                                    uint32_t coeff_idx,
                                                                    uint32_t mb,
                                                                    uint32_t bi,
                                                                    uint32_t bj,
                                                                    uint32_t g) const {
    assert(coeff_idx <= degree_);

    double D = param_.get_default_scale();
    double q_L = param_.get_q(level_);
    double scale = D;

    if (coeff_idx == 1) {
        scale = q_L;
    } else if (coeff_idx == 2) {
        double q_L1 = param_.get_q(level_ - 1);
        scale = q_L / D * q_L1;
    } else if (coeff_idx == 3) {
        assert(degree_ == 4);
        double q_L2 = param_.get_q(level_ - 2);
        scale = q_L / D * q_L / D * q_L2;
    } else if (coeff_idx == 4) {
        assert(degree_ == 4);
        double q_L1 = param_.get_q(level_ - 1);
        double q_L2 = param_.get_q(level_ - 2);
        scale = q_L / D * q_L / D * q_L1 / D * q_L2;
    }

    if (coeff_idx != 0) {
        auto vec = build_par_coeff_vec(n_slot_, chunk_size_, num_chunks_, d_, S_, mb, bj, g, n_heads_, cols_per_head_,
                                       coeffs_, coeff_idx);
        return ctx.encode_ringt(vec, scale);
    }

    // c0: special case with row-dependent zero padding
    vector<double> vec(n_slot_, 0.0);
    for (uint32_t h_local = 0; h_local < S_; h_local++) {
        uint32_t h = g * S_ + h_local;
        for (uint32_t col = 0; col < d_; col++) {
            uint32_t actual_col = bj * d_ + col;
            for (uint32_t row = 0; row < d_; row++) {
                uint32_t actual_row = bi * d_ + row;
                uint32_t base_slot = (row + d_ * col) * S_ + h_local;
                for (uint32_t ci = 0; ci < num_chunks_; ci++) {
                    uint32_t slot = ci * chunk_size_ + base_slot;
                    if (actual_row < m_ && h < n_heads_ && actual_col < cols_per_head_) {
                        uint32_t global_col = mb * n_heads_ * cols_per_head_ + h * cols_per_head_ + actual_col;
                        vec[slot] = coeffs_.get(0, global_col);
                    }
                }
            }
        }
    }
    return ctx.encode_ringt(vec, scale);
}

void ParBlockColMajorPolyActRNPoly::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    uint32_t n_coeff_vecs = K_ * num_block_cols_ * n_cts_per_block_idx_;

    c2_pt_.resize(n_coeff_vecs);
    c1_pt_.resize(n_coeff_vecs);
    for (uint32_t mb = 0; mb < K_; mb++) {
        for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
            for (uint32_t g = 0; g < n_cts_per_block_idx_; g++) {
                uint32_t idx = (mb * num_block_cols_ + bj) * n_cts_per_block_idx_ + g;
                c2_pt_[idx] = generate_coeff_pt(ctx, 2, mb, 0, bj, g);
                c1_pt_[idx] = generate_coeff_pt(ctx, 1, mb, 0, bj, g);
            }
        }
    }

    uint32_t cts_per_mb = num_block_rows_ * num_block_cols_ * n_cts_per_block_idx_;
    uint32_t total_c0 = K_ * cts_per_mb;
    c0_add_pt_.resize(total_c0);
    for (uint32_t mb = 0; mb < K_; mb++) {
        for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
            for (uint32_t bi = 0; bi < num_block_rows_; bi++) {
                for (uint32_t g = 0; g < n_cts_per_block_idx_; g++) {
                    uint32_t local_ct_idx = (bi + num_block_rows_ * bj) * n_cts_per_block_idx_ + g;
                    uint32_t c0_idx = mb * cts_per_mb + local_ct_idx;
                    c0_add_pt_[c0_idx] = generate_coeff_pt(ctx, 0, mb, bi, bj, g);
                }
            }
        }
    }

    if (degree_ == 4) {
        c4_pt_.resize(n_coeff_vecs);
        c3_pt_.resize(n_coeff_vecs);
        for (uint32_t mb = 0; mb < K_; mb++) {
            for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
                for (uint32_t g = 0; g < n_cts_per_block_idx_; g++) {
                    uint32_t idx = (mb * num_block_cols_ + bj) * n_cts_per_block_idx_ + g;
                    c4_pt_[idx] = generate_coeff_pt(ctx, 4, mb, 0, bj, g);
                    c3_pt_[idx] = generate_coeff_pt(ctx, 3, mb, 0, bj, g);
                }
            }
        }
    }
}

FeatureMatEncrypted ParBlockColMajorPolyActRNPoly::run(CkksContext& ctx, const FeatureMatEncrypted& x) {
    double D = param_.get_default_scale();
    uint32_t cts_per_mb = num_block_rows_ * num_block_cols_ * n_cts_per_block_idx_;
    uint32_t total_cts = K_ * cts_per_mb;
    assert(x.data.size() >= static_cast<size_t>(total_cts));
    uint32_t out_level = (degree_ == 4) ? level_ - 3 : level_ - 2;

    FeatureMatEncrypted result(&ctx, out_level);
    result.shape = {m_, total_dim_};
    result.matmul_block_size = d_;
    result.data.resize(total_cts);

    if (degree_ == 2) {
        parallel_for(total_cts, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
            uint32_t mb = ct_idx / cts_per_mb;
            uint32_t local_ct_idx = ct_idx % cts_per_mb;
            uint32_t block_idx = local_ct_idx / n_cts_per_block_idx_;
            uint32_t g = local_ct_idx % n_cts_per_block_idx_;
            uint32_t bj = block_idx / num_block_rows_;
            uint32_t ck_idx = (mb * num_block_cols_ + bj) * n_cts_per_block_idx_ + g;

            // x^2 = x * x -> level L-1
            auto x_sq_raw = ctx_copy.mult(x.data[ct_idx], x.data[ct_idx]);
            auto x_sq = ctx_copy.rescale(ctx_copy.relinearize(x_sq_raw), D / param_.get_q(level_) * D);

            // c2*x^2 -> level L-2, scale D
            auto c2_mul = ctx_copy.ringt_to_mul(c2_pt_[ck_idx], level_ - 1);
            auto c2x2 = ctx_copy.rescale(ctx_copy.mult_plain_mul(x_sq, c2_mul), D);

            // c1*x -> level L-1, scale D; drop to L-2
            auto c1_mul = ctx_copy.ringt_to_mul(c1_pt_[ck_idx], level_);
            auto c1x = ctx_copy.rescale(ctx_copy.mult_plain_mul(x.data[ct_idx], c1_mul), D);
            auto c1x_drop = ctx_copy.drop_level(c1x);

            // y = c0 + c1*x + c2*x^2
            auto y = ctx_copy.add(c1x_drop, c2x2);
            result.data[ct_idx] = ctx_copy.add_plain_ringt(y, c0_add_pt_[ct_idx]);
        });
    } else {
        // degree == 4
        double q_L2 = param_.get_q(level_ - 2);
        double S_high = param_.get_q(level_) / D * q_L2;

        parallel_for(total_cts, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
            uint32_t mb = ct_idx / cts_per_mb;
            uint32_t local_ct_idx = ct_idx % cts_per_mb;
            uint32_t block_idx = local_ct_idx / n_cts_per_block_idx_;
            uint32_t g = local_ct_idx % n_cts_per_block_idx_;
            uint32_t bj = block_idx / num_block_rows_;
            uint32_t ck_idx = (mb * num_block_cols_ + bj) * n_cts_per_block_idx_ + g;

            // === x^2 ===
            auto x_sq_raw = ctx_copy.mult(x.data[ct_idx], x.data[ct_idx]);
            auto x_sq = ctx_copy.rescale(ctx_copy.relinearize(x_sq_raw), D / param_.get_q(level_) * D);

            // === Low part: c0 + c1*x + c2*x^2 -> level L-2, scale D ===
            auto c2_mul = ctx_copy.ringt_to_mul(c2_pt_[ck_idx], level_ - 1);
            auto c2x2 = ctx_copy.rescale(ctx_copy.mult_plain_mul(x_sq, c2_mul), D);

            auto c1_mul = ctx_copy.ringt_to_mul(c1_pt_[ck_idx], level_);
            auto c1x = ctx_copy.rescale(ctx_copy.mult_plain_mul(x.data[ct_idx], c1_mul), D);
            auto c1x_drop = ctx_copy.drop_level(c1x);

            auto low = ctx_copy.add(c1x_drop, c2x2);
            low = ctx_copy.add_plain_ringt(low, c0_add_pt_[ct_idx]);

            // === High part: c3*x + c4*x^2 -> level L-2, scale S_high ===
            auto c4_mul = ctx_copy.ringt_to_mul(c4_pt_[ck_idx], level_ - 1);
            auto c4x2 = ctx_copy.rescale(ctx_copy.mult_plain_mul(x_sq, c4_mul), S_high);

            auto c3_mul = ctx_copy.ringt_to_mul(c3_pt_[ck_idx], level_);
            auto c3x = ctx_copy.rescale(ctx_copy.mult_plain_mul(x.data[ct_idx], c3_mul), S_high);
            auto c3x_drop = ctx_copy.drop_level(c3x);

            auto high = ctx_copy.add(c3x_drop, c4x2);

            // === Final: drop(low) + x^2 * high -> level L-3, scale D ===
            auto x_sq_drop = ctx_copy.drop_level(x_sq);

            auto product = ctx_copy.mult(x_sq_drop, high);
            auto x2_high = ctx_copy.rescale(ctx_copy.relinearize(product), D);

            auto low_drop = ctx_copy.drop_level(low);

            result.data[ct_idx] = ctx_copy.add(low_drop, x2_high);
        });
    }

    result.level = out_level;
    return result;
}

Array<double, 2> ParBlockColMajorPolyActRNPoly::run_plaintext(const Array<double, 2>& x) const {
    Array<double, 2> result({m_, total_dim_});
    for (uint32_t i = 0; i < m_; i++) {
        for (uint32_t j = 0; j < total_dim_; j++) {
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
