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

#include "par_block_col_major_add_pt.h"
#include "layer_util.h"
#include <cassert>

using namespace std;
using namespace lattisense;

ParBlockColMajorAddPt::ParBlockColMajorAddPt(const CkksParameter& param_in,
                                             Duo shape,
                                             uint32_t block_size,
                                             uint32_t n_heads,
                                             uint32_t level,
                                             Array<double, 2>&& B)
    : Layer(param_in), B_vals_(move(B)) {
    level_ = level;
    m_ = shape[0];
    uint32_t total_dim = shape[1];
    d_ = block_size;
    n_heads_ = n_heads;

    assert(total_dim % n_heads_ == 0);
    cols_per_head_ = total_dim / n_heads_;

    assert(B_vals_.get_shape()[0] == m_ && B_vals_.get_shape()[1] == total_dim);

    n_h_padded_ = next_pow2(n_heads);
    n_slot_ = param_.get_n() / 2;

    if (n_slot_ >= n_h_padded_ * d_ * d_) {
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

std::vector<double> ParBlockColMajorAddPt::build_pt_vec(uint32_t bi, uint32_t bj, uint32_t g) const {
    vector<double> pt_vec(n_slot_, 0.0);
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
                        uint32_t global_col = h * cols_per_head_ + actual_col;
                        pt_vec[slot] = B_vals_.get(actual_row, global_col);
                    }
                }
            }
        }
    }
    return pt_vec;
}

CkksPlaintextRingt ParBlockColMajorAddPt::generate_pt(CkksContext& ctx, uint32_t bi, uint32_t bj, uint32_t g) const {
    return ctx.encode_ringt(build_pt_vec(bi, bj, g), param_.get_default_scale());
}

void ParBlockColMajorAddPt::precompute_pts() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    double D = param_.get_default_scale();

    uint32_t total = num_block_rows_ * num_block_cols_ * n_cts_per_block_idx_;
    pt_.resize(total);
    for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
        for (uint32_t bi = 0; bi < num_block_rows_; bi++) {
            for (uint32_t g = 0; g < n_cts_per_block_idx_; g++) {
                uint32_t idx = (bi + num_block_rows_ * bj) * n_cts_per_block_idx_ + g;
                pt_[idx] = ctx.encode_ringt(build_pt_vec(bi, bj, g), D);
            }
        }
    }
}

FeatureMatEncrypted ParBlockColMajorAddPt::run(CkksContext& ctx, const FeatureMatEncrypted& A) {
    uint32_t total_cts = num_block_rows_ * num_block_cols_ * n_cts_per_block_idx_;
    assert(A.data.size() == total_cts);

    FeatureMatEncrypted result(&ctx, A.level);
    result.level = A.level;  // no level consumed
    result.shape = A.shape;
    result.matmul_block_size = d_;
    result.data.resize(total_cts);

    parallel_for(total_cts, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        result.data[ct_idx] = ctx_copy.add_plain_ringt(A.data[ct_idx], pt_[ct_idx]);
    });

    return result;
}

Array<double, 2> ParBlockColMajorAddPt::run_plaintext(const Array<double, 2>& A) const {
    uint32_t total_dim = n_heads_ * cols_per_head_;
    assert(A.get_shape()[0] == m_ && A.get_shape()[1] == total_dim);
    Array<double, 2> result({m_, total_dim});
    for (uint32_t i = 0; i < m_; i++) {
        for (uint32_t j = 0; j < total_dim; j++) {
            result.set(i, j, A.get(i, j) + B_vals_.get(i, j));
        }
    }
    return result;
}
