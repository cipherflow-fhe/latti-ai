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

#include "par_lower_diagonal_add_pt.h"
#include "layer_util.h"
#include <cassert>
#include <utility>

using namespace std;
using namespace lattisense;

namespace {

bool is_power_of_two(uint32_t x) {
    return x > 0 && (x & (x - 1)) == 0;
}

void validate_lower_diag_geometry(uint32_t total_rows,
                                  uint32_t n_prepad,
                                  uint32_t m_prepad,
                                  uint32_t H_prepad,
                                  uint32_t n_slot) {
    assert(total_rows > 0);
    assert(n_prepad > 0);
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

}  // namespace

ParLowerDiagonalAddPt::ParLowerDiagonalAddPt(const CkksParameter& param_in,
                                             Duo shape,
                                             Duo head_shape,
                                             uint32_t n_heads,
                                             uint32_t level,
                                             Array<double, 2>&& B)
    : Layer(param_in), B_vals_(move(B)) {
    level_ = level;
    total_rows_ = shape[0];
    n_prepad_ = shape[1];
    m_prepad_ = head_shape[0];
    H_prepad_ = n_heads;

    assert(head_shape[1] == n_prepad_);
    assert(B_vals_.get_shape()[0] == total_rows_ && B_vals_.get_shape()[1] == n_prepad_);

    n_slot_ = param_.get_n() / 2;
    validate_lower_diag_geometry(total_rows_, n_prepad_, m_prepad_, H_prepad_, n_slot_);

    H_ = next_pow2(H_prepad_);
    m_ = next_pow2(m_prepad_);
    n_ = next_pow2(n_prepad_);
    packed_extent_ = H_prepad_ * m_prepad_;
    segment_len_ = H_ * n_;
    c_ = n_slot_ / segment_len_;
    cts_per_mb_ = m_ / c_;
    n_mb_ = div_ceil(total_rows_, packed_extent_);
}

std::vector<double> ParLowerDiagonalAddPt::build_pt_vec(uint32_t mb, uint32_t ct_local) const {
    assert(mb < n_mb_);
    assert(ct_local < cts_per_mb_);

    vector<double> pt_vec(n_slot_, 0.0);
    for (uint32_t local_diag = 0; local_diag < c_; local_diag++) {
        uint32_t diag_idx = ct_local * c_ + local_diag;
        uint32_t segment_base = local_diag * segment_len_;
        for (uint32_t t = 0; t < n_; t++) {
            uint32_t col = t;
            uint32_t local_row = (diag_idx + t) % m_;
            for (uint32_t h = 0; h < H_; h++) {
                uint32_t global_row = mb * packed_extent_ + h * m_prepad_ + local_row;
                if (h < H_prepad_ && local_row < m_prepad_ && col < n_prepad_ && global_row < total_rows_) {
                    pt_vec[segment_base + t * H_ + h] = B_vals_.get(global_row, col);
                }
            }
        }
    }
    return pt_vec;
}

CkksPlaintextRingt ParLowerDiagonalAddPt::generate_pt(CkksContext& ctx, uint32_t mb, uint32_t ct_local) const {
    return ctx.encode_ringt(build_pt_vec(mb, ct_local), param_.get_default_scale());
}

void ParLowerDiagonalAddPt::precompute_pts() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    pt_.resize(n_mb_ * cts_per_mb_);
    for (uint32_t mb = 0; mb < n_mb_; mb++) {
        for (uint32_t ct_local = 0; ct_local < cts_per_mb_; ct_local++) {
            pt_[mb * cts_per_mb_ + ct_local] = generate_pt(ctx, mb, ct_local);
        }
    }
}

FeatureMatEncrypted ParLowerDiagonalAddPt::run(CkksContext& ctx, const FeatureMatEncrypted& A) {
    assert(A.level == level_);
    assert(A.shape[0] == total_rows_ && A.shape[1] == n_prepad_);
    assert(A.head_shape[0] == m_prepad_ && A.head_shape[1] == n_prepad_);
    assert(A.matmul_block_size == m_);
    assert(A.data.size() == pt_.size());

    FeatureMatEncrypted result(&ctx, A.level);
    result.level = A.level;
    result.shape = A.shape;
    result.head_shape = A.head_shape;
    result.matmul_block_size = m_;
    result.data.resize(pt_.size());

    parallel_for(pt_.size(), th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        result.data[ct_idx] = ctx_copy.add_plain_ringt(A.data[ct_idx], pt_[ct_idx]);
    });

    return result;
}

Array<double, 2> ParLowerDiagonalAddPt::run_plaintext(const Array<double, 2>& A) const {
    assert(A.get_shape()[0] == total_rows_ && A.get_shape()[1] == n_prepad_);

    Array<double, 2> result({total_rows_, n_prepad_});
    for (uint32_t i = 0; i < total_rows_; i++) {
        for (uint32_t j = 0; j < n_prepad_; j++) {
            result.set(i, j, A.get(i, j) + B_vals_.get(i, j));
        }
    }
    return result;
}
