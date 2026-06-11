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

#include "par_block_col_major_layernorm.h"
#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace std;
using namespace lattisense;

static uint32_t next_power_of_2(uint32_t x) {
    uint32_t p = 1;
    while (p < x)
        p *= 2;
    return p;
}

// ============================================================
// ParBlockColMajorLNStats
// ============================================================

ParBlockColMajorLNStats::ParBlockColMajorLNStats(const CkksParameter& param,
                                                 Duo shape,
                                                 uint32_t block_size,
                                                 uint32_t n_heads,
                                                 uint32_t init_level,
                                                 double eps,
                                                 double inv_var)
    : Layer(param), eps_(eps), inv_var_(inv_var) {
    level_ = init_level;
    m_ = shape[0];
    total_dim_ = shape[1];
    d_ = block_size;
    n_heads_ = n_heads;
    cols_per_head_ = total_dim_ / n_heads_;
    n_h_padded_ = next_power_of_2(n_heads);
    n_slot_ = param_.get_n() / 2;
    assert(n_slot_ >= d_ * d_ && "n_slot must be at least d*d");
    assert((d_ & (d_ - 1)) == 0 && "block_size must be a power of 2");

    if ((uint32_t)n_slot_ >= n_h_padded_ * d_ * d_) {
        S_ = n_h_padded_;
        chunk_size_ = n_h_padded_ * d_ * d_;
        n_cts_per_block_idx_ = 1;
    } else {
        S_ = n_slot_ / (d_ * d_);
        chunk_size_ = n_slot_;
        if (S_ == 1) {
            n_h_padded_ = n_heads_;
        }
        n_cts_per_block_idx_ = n_h_padded_ / S_;
    }
    num_chunks_ = n_slot_ / chunk_size_;
    num_block_rows_ = div_ceil(m_, d_);
    num_block_cols_ = div_ceil(cols_per_head_, d_);
}

CkksPlaintextRingt
ParBlockColMajorLNStats::generate_pt(CkksContext& ctx, uint32_t pt_idx, uint32_t, uint32_t, uint32_t) const {
    double D = param_.get_default_scale();
    double q_L = param_.get_q(level_);
    double q_L1 = param_.get_q(level_ - 1);
    double q_L2 = param_.get_q(level_ - 2);
    double q_L3 = param_.get_q(level_ - 3);
    double norm_dim = static_cast<double>(total_dim_);

    if (pt_idx == 0) {
        vector<double> h0_mask(n_slot_, 0.0);
        for (uint32_t i = 0; i < (uint32_t)n_slot_; i++) {
            uint32_t pos_in_chunk = i % chunk_size_;
            uint32_t h_local = pos_in_chunk % S_;
            if (h_local == 0)
                h0_mask[i] = 1.0;
        }
        return ctx.encode_ringt(h0_mask, q_L);
    }
    if (pt_idx == 1) {
        vector<double> inv_n_vec(n_slot_, 1.0 / norm_dim);
        return ctx.encode_ringt(inv_n_vec, q_L1);
    }
    if (pt_idx == 2) {
        vector<double> iv_vec(n_slot_, inv_var_);
        return ctx.encode_ringt(iv_vec, q_L2 / D * q_L3);
    }
    if (pt_idx == 3) {
        vector<double> eps_vec(n_slot_, eps_ * inv_var_);
        return ctx.encode_ringt(eps_vec, D);
    }
    throw runtime_error("ParBlockColMajorLNStats: unknown pt_idx " + to_string(pt_idx));
}

void ParBlockColMajorLNStats::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    h0_mask_pt_ = generate_pt(ctx, 0);
    inv_n_pt_ = generate_pt(ctx, 1);
    iv_pt_ = generate_pt(ctx, 2);
    eps_add_pt_ = generate_pt(ctx, 3);
}

CkksCiphertext ParBlockColMajorLNStats::intra_block_col_sum(CkksContext& ctx, const CkksCiphertext& ct) const {
    // Par format slot: (row + d*col)*S + h_local
    // Column sum only — rotate by d*S, 2*d*S, ..., (d/2)*d*S
    // output's each block have same value in each row, which is the column sum of the input's corresponding block
    CkksCiphertext result = ct.copy();
    for (uint32_t step = 1; step < d_; step *= 2) {
        int rot = step * d_ * S_;
        auto rotated = ctx.rotate(result, rot);
        result = ctx.add(result, rotated);
    }
    return result;
}

vector<CkksCiphertext> ParBlockColMajorLNStats::run(CkksContext& ctx, const FeatureMatEncrypted& x) {
    double D = param_.get_default_scale();
    uint32_t total_cts = num_block_rows_ * num_block_cols_ * n_cts_per_block_idx_;

    // Step 1: Column sum + x_sq for all ciphertexts
    vector<CkksCiphertext> col_sum_per_block(total_cts);
    vector<CkksCiphertext> x_sq(total_cts);

    parallel_for(total_cts, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        col_sum_per_block[ct_idx] = intra_block_col_sum(ctx_copy, x.data[ct_idx]);
        auto prod = ctx_copy.mult(x.data[ct_idx], x.data[ct_idx]);
        auto relin = ctx_copy.relinearize(prod);
        x_sq[ct_idx] = ctx_copy.rescale(relin, D / param_.get_q(level_) * D);
    });

    // Step 2: Cross-block-col and cross-group sums of column-summed data
    vector<CkksCiphertext> col_sum_x(num_block_rows_);
    for (uint32_t bi = 0; bi < num_block_rows_; bi++) {
        bool first = true;
        for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
            for (uint32_t g = 0; g < n_cts_per_block_idx_; g++) {
                uint32_t ct_idx = (bi + num_block_rows_ * bj) * n_cts_per_block_idx_ + g;
                if (first) {
                    col_sum_x[bi] = col_sum_per_block[ct_idx].copy();
                    first = false;
                } else {
                    col_sum_x[bi] = ctx.add(col_sum_x[bi], col_sum_per_block[ct_idx]);
                }
            }
        }
    }

    // Step 3: Cross-head sum + mask + replicate for sum_x
    vector<CkksCiphertext> sum_x(num_block_rows_);
    parallel_for(num_block_rows_, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        CkksCiphertext summed = col_sum_x[bi].copy();
        for (uint32_t step = 1; step < S_; step *= 2) {
            auto rotated = ctx_copy.rotate(summed, (int)step);
            summed = ctx_copy.add(summed, rotated);
        }
        auto mask_mul = ctx_copy.ringt_to_mul(h0_mask_pt_, summed.get_level());
        auto masked = ctx_copy.mult_plain_mul(summed, mask_mul);
        auto masked_rescaled = ctx_copy.rescale(masked, D);
        CkksCiphertext replicated = masked_rescaled.copy();
        for (uint32_t step = 1; step < S_; step *= 2) {
            auto rotated = ctx_copy.rotate(replicated, -(int)step);
            replicated = ctx_copy.add(replicated, rotated);
        }
        sum_x[bi] = move(replicated);
    });
    // sum_x at level L-1, scale D

    // Step 4: Column sum + cross-block/cross-group sum of x_sq
    vector<CkksCiphertext> col_sum_x_sq_per_block(total_cts);
    parallel_for(total_cts, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        col_sum_x_sq_per_block[ct_idx] = intra_block_col_sum(ctx_copy, x_sq[ct_idx]);
    });

    vector<CkksCiphertext> col_sum_x_sq(num_block_rows_);
    for (uint32_t bi = 0; bi < num_block_rows_; bi++) {
        bool first = true;
        for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
            for (uint32_t g = 0; g < n_cts_per_block_idx_; g++) {
                uint32_t ct_idx = (bi + num_block_rows_ * bj) * n_cts_per_block_idx_ + g;
                if (first) {
                    col_sum_x_sq[bi] = col_sum_x_sq_per_block[ct_idx].copy();
                    first = false;
                } else {
                    col_sum_x_sq[bi] = ctx.add(col_sum_x_sq[bi], col_sum_x_sq_per_block[ct_idx]);
                }
            }
        }
    }

    // Step 5: Cross-head sum + mask + replicate for sum_x_sq
    vector<CkksCiphertext> sum_x_sq_row(num_block_rows_);
    parallel_for(num_block_rows_, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        CkksCiphertext summed = col_sum_x_sq[bi].copy();
        for (uint32_t step = 1; step < S_; step *= 2) {
            auto rotated = ctx_copy.rotate(summed, (int)step);
            summed = ctx_copy.add(summed, rotated);
        }
        auto mask_mul = ctx_copy.ringt_to_mul(h0_mask_pt_, summed.get_level());
        auto masked = ctx_copy.mult_plain_mul(summed, mask_mul);
        auto masked_rescaled = ctx_copy.rescale(masked, D / param_.get_q(level_ - 1) * D);
        CkksCiphertext replicated = masked_rescaled.copy();
        for (uint32_t step = 1; step < S_; step *= 2) {
            auto rotated = ctx_copy.rotate(replicated, -(int)step);
            replicated = ctx_copy.add(replicated, rotated);
        }
        sum_x_sq_row[bi] = move(replicated);
    });
    // sum_x_sq_row at level L-2, scale D^2/q_{L-1}

    // Step 6: mean = sum_x * pt(1/norm_dim) -> L-2, exact D
    vector<CkksCiphertext> mean_cts(num_block_rows_);
    parallel_for(num_block_rows_, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        auto pt_inv_n = ctx_copy.ringt_to_mul(inv_n_pt_, level_ - 1);
        auto p = ctx_copy.mult_plain_mul(sum_x[bi], pt_inv_n);
        mean_cts[bi] = ctx_copy.rescale(p, D);
    });
    // mean at L-2, scale D

    // Step 7: mean_sq and E_x_sq -> L-3
    vector<CkksCiphertext> E_x_sq(num_block_rows_);
    vector<CkksCiphertext> mean_sq(num_block_rows_);
    parallel_for(num_block_rows_, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        auto prod = ctx_copy.mult(mean_cts[bi], mean_cts[bi]);
        mean_sq[bi] = ctx_copy.rescale(ctx_copy.relinearize(prod), D / param_.get_q(level_ - 2) * D);

        auto pt_inv_n = ctx_copy.ringt_to_mul(inv_n_pt_, level_ - 2);
        auto p = ctx_copy.mult_plain_mul(sum_x_sq_row[bi], pt_inv_n);
        E_x_sq[bi] = ctx_copy.rescale(p, D / param_.get_q(level_ - 2) * D);
    });

    // Step 8: var = E_x_sq - mean_sq
    vector<CkksCiphertext> var_cts(num_block_rows_);
    for (uint32_t bi = 0; bi < num_block_rows_; bi++) {
        var_cts[bi] = ctx.sub(E_x_sq[bi], mean_sq[bi]);
    }

    // Step 9: a = var * inv_var + eps*inv_var
    vector<CkksCiphertext> a_cts(num_block_rows_);
    parallel_for(num_block_rows_, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        auto pt_iv = ctx_copy.ringt_to_mul(iv_pt_, level_ - 3);
        auto p = ctx_copy.mult_plain_mul(var_cts[bi], pt_iv);
        a_cts[bi] = ctx_copy.rescale(p, D);
        a_cts[bi] = ctx_copy.add_plain_ringt(a_cts[bi], eps_add_pt_);
    });

    return move(a_cts);
}

Array<double, 2> ParBlockColMajorLNStats::run_plaintext(const Array<double, 2>& x) const {
    Array<double, 2> result({m_, 1});
    for (uint32_t i = 0; i < m_; i++) {
        double sum_x = 0.0;
        double sum_x2 = 0.0;
        for (uint32_t j = 0; j < total_dim_; j++) {
            double v = x.get(i, j);
            sum_x += v;
            sum_x2 += v * v;
        }
        double mean = sum_x / total_dim_;
        double var = sum_x2 / total_dim_ - mean * mean;
        result.set(i, 0, (var + eps_) * inv_var_);
    }
    return result;
}

// ============================================================
// ParBlockColMajorLNXCentered
// ============================================================

ParBlockColMajorLNXCentered::ParBlockColMajorLNXCentered(const CkksParameter& param,
                                                         Duo shape,
                                                         uint32_t block_size,
                                                         uint32_t n_heads,
                                                         uint32_t init_level)
    : Layer(param) {
    level_ = init_level;
    m_ = shape[0];
    total_dim_ = shape[1];
    d_ = block_size;
    n_heads_ = n_heads;
    cols_per_head_ = total_dim_ / n_heads_;
    n_h_padded_ = next_power_of_2(n_heads);
    n_slot_ = param_.get_n() / 2;
    assert(n_slot_ >= d_ * d_ && "n_slot must be at least d*d");
    assert((d_ & (d_ - 1)) == 0 && "block_size must be a power of 2");

    if ((uint32_t)n_slot_ >= n_h_padded_ * d_ * d_) {
        S_ = n_h_padded_;
        chunk_size_ = n_h_padded_ * d_ * d_;
        n_cts_per_block_idx_ = 1;
    } else {
        S_ = n_slot_ / (d_ * d_);
        chunk_size_ = n_slot_;
        if (S_ == 1) {
            n_h_padded_ = n_heads_;
        }
        n_cts_per_block_idx_ = n_h_padded_ / S_;
    }
    num_chunks_ = n_slot_ / chunk_size_;
    num_block_rows_ = div_ceil(m_, d_);
    num_block_cols_ = div_ceil(cols_per_head_, d_);
}

CkksPlaintextRingt
ParBlockColMajorLNXCentered::generate_pt(CkksContext& ctx, uint32_t pt_idx, uint32_t, uint32_t, uint32_t) const {
    double q_L = param_.get_q(level_);
    double q_L1 = param_.get_q(level_ - 1);
    double norm_dim = static_cast<double>(total_dim_);

    if (pt_idx == 0) {
        vector<double> h0_mask(n_slot_, 0.0);
        for (uint32_t i = 0; i < (uint32_t)n_slot_; i++) {
            uint32_t pos_in_chunk = i % chunk_size_;
            uint32_t h_local = pos_in_chunk % S_;
            if (h_local == 0)
                h0_mask[i] = 1.0;
        }
        return ctx.encode_ringt(h0_mask, q_L);
    }
    if (pt_idx == 1) {
        vector<double> inv_n_vec(n_slot_, 1.0 / norm_dim);
        return ctx.encode_ringt(inv_n_vec, q_L1);
    }
    throw runtime_error("ParBlockColMajorLNXCentered: unknown pt_idx " + to_string(pt_idx));
}

void ParBlockColMajorLNXCentered::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    h0_mask_pt_ = generate_pt(ctx, 0);
    inv_n_pt_ = generate_pt(ctx, 1);
}

CkksCiphertext ParBlockColMajorLNXCentered::intra_block_col_sum(CkksContext& ctx, const CkksCiphertext& ct) const {
    CkksCiphertext result = ct.copy();
    for (uint32_t step = 1; step < d_; step *= 2) {
        int rot = step * d_ * S_;
        auto rotated = ctx.rotate(result, rot);
        result = ctx.add(result, rotated);
    }
    return result;
}

vector<CkksCiphertext> ParBlockColMajorLNXCentered::run(CkksContext& ctx, const FeatureMatEncrypted& x) {
    double D = param_.get_default_scale();
    uint32_t total_cts = num_block_rows_ * num_block_cols_ * n_cts_per_block_idx_;

    // Step 1: Column sum for all ciphertexts
    vector<CkksCiphertext> col_sum_per_block(total_cts);
    parallel_for(total_cts, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        col_sum_per_block[ct_idx] = intra_block_col_sum(ctx_copy, x.data[ct_idx]);
    });

    // Step 2: Cross-block-col and cross-group sums
    vector<CkksCiphertext> col_sum_x(num_block_rows_);
    for (uint32_t bi = 0; bi < num_block_rows_; bi++) {
        bool first = true;
        for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
            for (uint32_t g = 0; g < n_cts_per_block_idx_; g++) {
                uint32_t ct_idx = (bi + num_block_rows_ * bj) * n_cts_per_block_idx_ + g;
                if (first) {
                    col_sum_x[bi] = col_sum_per_block[ct_idx].copy();
                    first = false;
                } else {
                    col_sum_x[bi] = ctx.add(col_sum_x[bi], col_sum_per_block[ct_idx]);
                }
            }
        }
    }

    // Step 3: Cross-head sum + mask + replicate for sum_x
    vector<CkksCiphertext> sum_x(num_block_rows_);
    parallel_for(num_block_rows_, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        CkksCiphertext summed = col_sum_x[bi].copy();
        for (uint32_t step = 1; step < S_; step *= 2) {
            auto rotated = ctx_copy.rotate(summed, (int)step);
            summed = ctx_copy.add(summed, rotated);
        }
        auto mask_mul = ctx_copy.ringt_to_mul(h0_mask_pt_, summed.get_level());
        auto masked = ctx_copy.mult_plain_mul(summed, mask_mul);
        auto masked_rescaled = ctx_copy.rescale(masked, D);
        CkksCiphertext replicated = masked_rescaled.copy();
        for (uint32_t step = 1; step < S_; step *= 2) {
            auto rotated = ctx_copy.rotate(replicated, -(int)step);
            replicated = ctx_copy.add(replicated, rotated);
        }
        sum_x[bi] = move(replicated);
    });
    // sum_x at level L-1, scale D

    // Step 4: mean = sum_x * pt(1/norm_dim) -> L-2, exact D
    vector<CkksCiphertext> mean_cts(num_block_rows_);
    parallel_for(num_block_rows_, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        auto pt_inv_n = ctx_copy.ringt_to_mul(inv_n_pt_, level_ - 1);
        auto p = ctx_copy.mult_plain_mul(sum_x[bi], pt_inv_n);
        mean_cts[bi] = ctx_copy.rescale(p, D);
    });
    // mean at L-2, scale D

    // Step 5: x_centered = drop(x,2) - mean, per ct -> L-2, scale D
    vector<CkksCiphertext> x_centered(total_cts);
    parallel_for(total_cts, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        uint32_t block_idx = ct_idx / n_cts_per_block_idx_;
        uint32_t bi = block_idx % num_block_rows_;
        auto x_drop = ctx_copy.drop_level(x.data[ct_idx], 2);
        x_centered[ct_idx] = ctx_copy.sub(x_drop, mean_cts[bi]);
    });

    return move(x_centered);
}

Array<double, 2> ParBlockColMajorLNXCentered::run_plaintext(const Array<double, 2>& x) const {
    Array<double, 2> result({m_, total_dim_});
    for (uint32_t i = 0; i < m_; i++) {
        double sum_x = 0.0;
        for (uint32_t j = 0; j < total_dim_; j++) {
            sum_x += x.get(i, j);
        }
        double mean = sum_x / total_dim_;
        for (uint32_t j = 0; j < total_dim_; j++) {
            result.set(i, j, x.get(i, j) - mean);
        }
    }
    return result;
}

// ============================================================
// ParBlockColMajorLNMinimaxInit (same logic as block version)
// ============================================================

ParBlockColMajorLNMinimaxInit::ParBlockColMajorLNMinimaxInit(const CkksParameter& param,
                                                             uint32_t block_size,
                                                             uint32_t input_level,
                                                             double c0,
                                                             double c1,
                                                             double c2)
    : Layer(param), c0_(c0), c1_(c1), c2_(c2) {
    level_ = input_level;
    d_ = block_size;
    n_slot_ = param_.get_n() / 2;
    chunk_size_ = d_ * d_;
}

CkksPlaintextRingt
ParBlockColMajorLNMinimaxInit::generate_pt(CkksContext& ctx, uint32_t pt_idx, uint32_t, uint32_t, uint32_t) const {
    double D = param_.get_default_scale();
    double q_L = param_.get_q(level_);

    if (pt_idx == 0) {
        vector<double> c0_vec(n_slot_, c0_);
        return ctx.encode_ringt(c0_vec, D);
    }
    if (pt_idx == 1) {
        vector<double> c1_vec(n_slot_, c1_);
        return ctx.encode_ringt(c1_vec, q_L);
    }
    if (pt_idx == 2) {
        double q_L1 = param_.get_q(level_ - 1);
        vector<double> c2_vec(n_slot_, c2_);
        return ctx.encode_ringt(c2_vec, q_L / D * q_L1);
    }
    throw runtime_error("ParBlockColMajorLNMinimaxInit: unknown pt_idx " + to_string(pt_idx));
}

void ParBlockColMajorLNMinimaxInit::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    c0_add_pt_ = generate_pt(ctx, 0);
    c1_pt_ = generate_pt(ctx, 1);
    c2_norm_pt_ = generate_pt(ctx, 2);
}

vector<CkksCiphertext> ParBlockColMajorLNMinimaxInit::run(CkksContext& ctx, const vector<CkksCiphertext>& a_cts) {
    double D = param_.get_default_scale();
    uint32_t n_rows = a_cts.size();
    vector<CkksCiphertext> y_cts(n_rows);

    parallel_for(n_rows, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        auto a_sq_raw = ctx_copy.mult(a_cts[bi], a_cts[bi]);
        auto a_sq = ctx_copy.rescale(ctx_copy.relinearize(a_sq_raw), D / param_.get_q(level_) * D);

        auto c2_mul = ctx_copy.ringt_to_mul(c2_norm_pt_, level_ - 1);
        auto c2a2 = ctx_copy.rescale(ctx_copy.mult_plain_mul(a_sq, c2_mul), D);

        auto c1_mul = ctx_copy.ringt_to_mul(c1_pt_, level_);
        auto c1a = ctx_copy.rescale(ctx_copy.mult_plain_mul(a_cts[bi], c1_mul), D);

        auto c1a_drop = ctx_copy.drop_level(c1a);

        auto y0 = ctx_copy.add(c1a_drop, c2a2);
        y0 = ctx_copy.add_plain_ringt(y0, c0_add_pt_);

        y_cts[bi] = move(y0);
    });

    return y_cts;
}

Array<double, 2> ParBlockColMajorLNMinimaxInit::run_plaintext(const Array<double, 2>& a) const {
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
// ParBlockColMajorLNGoldschmidt (same logic as block version)
// ============================================================

ParBlockColMajorLNGoldschmidt::ParBlockColMajorLNGoldschmidt(const CkksParameter& param,
                                                             uint32_t block_size,
                                                             uint32_t input_level)
    : Layer(param) {
    level_ = input_level;
    d_ = block_size;
    n_slot_ = param_.get_n() / 2;
    chunk_size_ = d_ * d_;
}

CkksPlaintextRingt
ParBlockColMajorLNGoldschmidt::generate_pt(CkksContext& ctx, uint32_t pt_idx, uint32_t, uint32_t, uint32_t) const {
    double D = param_.get_default_scale();
    double q_L = param_.get_q(level_);
    double q_L1 = param_.get_q(level_ - 1);
    double q_L2 = param_.get_q(level_ - 2);

    if (pt_idx == 0) {
        vector<double> three_vec(n_slot_, 3.0);
        return ctx.encode_ringt(three_vec, D / q_L * D / q_L1 * D);
    }
    if (pt_idx == 1) {
        vector<double> half_vec(n_slot_, 0.5);
        return ctx.encode_ringt(half_vec, q_L / D * q_L / D * q_L1 / D * q_L2);
    }
    throw runtime_error("ParBlockColMajorLNGoldschmidt: unknown pt_idx " + to_string(pt_idx));
}

void ParBlockColMajorLNGoldschmidt::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    three_pt_ = generate_pt(ctx, 0);
    half_norm_pt_ = generate_pt(ctx, 1);
}

vector<CkksCiphertext> ParBlockColMajorLNGoldschmidt::run(CkksContext& ctx,
                                                          const vector<CkksCiphertext>& y_cts,
                                                          const vector<CkksCiphertext>& a_cts) {
    double D = param_.get_default_scale();
    uint32_t n_rows = y_cts.size();
    vector<CkksCiphertext> y_new(n_rows);

    parallel_for(n_rows, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        // Drop a to level_ to match y
        auto a_drop = a_cts[bi].copy();
        if (a_drop.get_level() > (int)level_) {
            a_drop = ctx_copy.drop_level(a_drop, a_drop.get_level() - (int)level_);
        }

        // Step 1 (parallel): y*a and y*y -> level_-1, scale S1 = D²/q_{L_y}
        auto ya_raw = ctx_copy.mult(y_cts[bi], a_drop);
        auto ya = ctx_copy.rescale(ctx_copy.relinearize(ya_raw), D / param_.get_q(level_) * D);

        auto yy_raw = ctx_copy.mult(y_cts[bi], y_cts[bi]);
        auto yy = ctx_copy.rescale(ctx_copy.relinearize(yy_raw), D / param_.get_q(level_) * D);

        // Step 2: (y*a)*(y*y) -> level_-2, scale S_prod = D⁴/(q_L²·q_{L-1})
        auto ya_yy_raw = ctx_copy.mult(ya, yy);
        double S_prod = D / param_.get_q(level_) * D / param_.get_q(level_) * D / param_.get_q(level_ - 1) * D;
        auto ya_yy = ctx_copy.rescale(ctx_copy.relinearize(ya_yy_raw), S_prod);

        // Step 3: 3*y (pt*ct) -> level_-1, scale S_prod
        auto three_mul = ctx_copy.ringt_to_mul(three_pt_, level_);
        auto three_y_raw = ctx_copy.mult_plain_mul(y_cts[bi], three_mul);
        auto three_y = ctx_copy.rescale(three_y_raw, S_prod);
        // Drop 3*y from level_-1 to level_-2 to match (y*a)*(y*y)
        auto three_y_drop = ctx_copy.drop_level(three_y);

        // Step 4: 3*y - (y*a)*(y*y) -> level_-2, scale S_prod
        auto diff = ctx_copy.sub(three_y_drop, ya_yy);

        // Step 5: 0.5 * diff (normalizing pt*ct -> exact D at level_-3)
        auto half_mul = ctx_copy.ringt_to_mul(half_norm_pt_, level_ - 2);
        auto half_raw = ctx_copy.mult_plain_mul(diff, half_mul);
        y_new[bi] = ctx_copy.rescale(half_raw, D);
    });

    return y_new;
}

Array<double, 2> ParBlockColMajorLNGoldschmidt::run_plaintext(const Array<double, 2>& y,
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
// ParBlockColMajorLNAffine
// ============================================================

ParBlockColMajorLNAffine::ParBlockColMajorLNAffine(const CkksParameter& param,
                                                   Duo shape,
                                                   uint32_t block_size,
                                                   uint32_t n_heads,
                                                   uint32_t y_level,
                                                   double inv_std,
                                                   Array<double, 1>&& gamma,
                                                   Array<double, 1>&& beta)
    : Layer(param), y_level_(y_level), inv_std_(inv_std), gamma_vals_(move(gamma)), beta_vals_(move(beta)) {
    level_ = y_level;
    m_ = shape[0];
    d_ = block_size;
    n_heads_ = n_heads;
    cols_per_head_ = shape[1] / n_heads_;
    n_h_padded_ = next_power_of_2(n_heads);
    n_slot_ = param_.get_n() / 2;
    assert(n_slot_ >= d_ * d_ && "n_slot must be at least d*d");
    assert((d_ & (d_ - 1)) == 0 && "block_size must be a power of 2");

    if ((uint32_t)n_slot_ >= n_h_padded_ * d_ * d_) {
        S_ = n_h_padded_;
        chunk_size_ = n_h_padded_ * d_ * d_;
        n_cts_per_block_idx_ = 1;
    } else {
        S_ = n_slot_ / (d_ * d_);
        chunk_size_ = n_slot_;
        if (S_ == 1) {
            n_h_padded_ = n_heads_;
        }
        n_cts_per_block_idx_ = n_h_padded_ / S_;
    }
    num_chunks_ = n_slot_ / chunk_size_;
    num_block_rows_ = div_ceil(m_, d_);
    num_block_cols_ = div_ceil(cols_per_head_, d_);
}

CkksPlaintextRingt
ParBlockColMajorLNAffine::generate_pt(CkksContext& ctx, uint32_t pt_idx, uint32_t bi, uint32_t bj, uint32_t g) const {
    double D = param_.get_default_scale();

    if (pt_idx == 0) {
        double q_L = param_.get_q(y_level_);
        double q_L1 = param_.get_q(y_level_ - 1);
        vector<double> gamma_vec(n_slot_, 0.0);
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
                            gamma_vec[slot] = inv_std_ * gamma_vals_.get(h * cols_per_head_ + actual_col);
                        }
                    }
                }
            }
        }
        return ctx.encode_ringt(gamma_vec, q_L / D * q_L1);
    }

    if (pt_idx == 1) {
        vector<double> beta_vec(n_slot_, 0.0);
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
                            beta_vec[slot] = beta_vals_.get(h * cols_per_head_ + actual_col);
                        }
                    }
                }
            }
        }
        return ctx.encode_ringt(beta_vec, D);
    }

    throw runtime_error("ParBlockColMajorLNAffine: unknown pt_idx " + to_string(pt_idx));
}

void ParBlockColMajorLNAffine::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);

    uint32_t n_gamma_vecs = num_block_rows_ * num_block_cols_ * n_cts_per_block_idx_;
    gamma_pt_.resize(n_gamma_vecs);
    for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
        for (uint32_t bi = 0; bi < num_block_rows_; bi++) {
            for (uint32_t g = 0; g < n_cts_per_block_idx_; g++) {
                uint32_t idx = (bi + num_block_rows_ * bj) * n_cts_per_block_idx_ + g;
                gamma_pt_[idx] = generate_pt(ctx, 0, bi, bj, g);
            }
        }
    }

    uint32_t total_beta = num_block_rows_ * num_block_cols_ * n_cts_per_block_idx_;
    beta_add_pt_.resize(total_beta);
    for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
        for (uint32_t bi = 0; bi < num_block_rows_; bi++) {
            for (uint32_t g = 0; g < n_cts_per_block_idx_; g++) {
                uint32_t idx = (bi + num_block_rows_ * bj) * n_cts_per_block_idx_ + g;
                beta_add_pt_[idx] = generate_pt(ctx, 1, bi, bj, g);
            }
        }
    }
}

FeatureMatEncrypted ParBlockColMajorLNAffine::run(CkksContext& ctx,
                                                  const vector<CkksCiphertext>& x_centered,
                                                  const vector<CkksCiphertext>& y_cts) {
    double D = param_.get_default_scale();
    uint32_t total_cts = num_block_rows_ * num_block_cols_ * n_cts_per_block_idx_;

    FeatureMatEncrypted result(&ctx, y_level_ - 2);
    result.head_shape = {m_, cols_per_head_};
    result.shape = {m_, n_heads_ * cols_per_head_};
    result.matmul_block_size = d_;
    result.data.resize(total_cts);

    parallel_for(total_cts, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        uint32_t block_idx = ct_idx / n_cts_per_block_idx_;
        uint32_t bi = block_idx % num_block_rows_;

        uint32_t gamma_idx = ct_idx;
        auto gamma_mul = ctx_copy.ringt_to_mul(gamma_pt_[gamma_idx], y_level_);
        auto yw = ctx_copy.rescale(ctx_copy.mult_plain_mul(y_cts[bi], gamma_mul), param_.get_q(y_level_ - 1));

        auto xc = x_centered[ct_idx].copy();
        if (xc.get_level() > (int)(y_level_ - 1)) {
            xc = ctx_copy.drop_level(xc, xc.get_level() - (int)(y_level_ - 1));
        }

        auto out = ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(xc, yw)), D);

        result.data[ct_idx] = ctx_copy.add_plain_ringt(out, beta_add_pt_[ct_idx]);
    });

    result.level = y_level_ - 2;
    return result;
}

Array<double, 2> ParBlockColMajorLNAffine::run_plaintext(const Array<double, 2>& x_centered,
                                                         const Array<double, 2>& y) const {
    uint32_t total_dim = n_heads_ * cols_per_head_;
    Array<double, 2> result({m_, total_dim});
    for (uint32_t i = 0; i < m_; i++) {
        double yi = y.get(i, 0);
        for (uint32_t j = 0; j < total_dim; j++) {
            double out = x_centered.get(i, j) * yi * inv_std_ * gamma_vals_.get(j) + beta_vals_.get(j);
            result.set(i, j, out);
        }
    }
    return result;
}
