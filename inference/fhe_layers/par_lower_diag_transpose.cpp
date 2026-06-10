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

#include "par_lower_diag_transpose.h"
#include "layer_util.h"
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <utility>

using namespace std;
using namespace lattisense;

namespace {

bool is_power_of_two(uint32_t x) {
    return x > 0 && (x & (x - 1)) == 0;
}

}  // namespace

ParLowerDiagTranspose::ParLowerDiagTranspose(const CkksParameter& param_in,
                                             const Duo& shape,
                                             uint32_t n_heads,
                                             uint32_t head_dim,
                                             uint32_t level)
    : Layer(param_in) {
    assert(level >= 1);
    level_ = level;
    shape_ = shape;
    H_prepad_ = n_heads;
    m_ = head_dim;
    n_prepad_ = shape_[1];

    assert(H_prepad_ > 0);
    assert(m_ > 0 && is_power_of_two(m_));
    assert(shape_[0] == m_);
    assert(n_prepad_ > 0);

    H_ = next_pow2(H_prepad_);
    n_ = next_pow2(n_prepad_);
    assert(n_ % m_ == 0);

    n_slot_ = param_.get_n() / 2;
    segment_len_ = H_ * n_;
    assert(segment_len_ > 0);
    assert(n_slot_ % segment_len_ == 0);
    c_ = n_slot_ / segment_len_;
    assert(c_ > 0 && is_power_of_two(c_));
    assert(m_ % c_ == 0);
    m_c_ = m_ / c_;
}

uint32_t ParLowerDiagTranspose::expected_ct_count() const {
    return m_c_;
}

std::vector<double> ParLowerDiagTranspose::build_transpose_mask(uint32_t out_diag_idx, uint32_t mask_idx) const {
    assert(out_diag_idx < m_);
    assert(mask_idx < 2);
    vector<double> mask(n_slot_, 0.0);
    uint32_t out_local_idx = out_diag_idx % c_;
    uint32_t base = out_local_idx * segment_len_;

    if (mask_idx == 0) {
        for (uint32_t t = 0; t < n_ - out_diag_idx; t++) {
            uint32_t start = base + t * H_;
            for (uint32_t h = 0; h < H_; h++) {
                mask[start + h] = 1.0;
            }
        }
    } else {
        for (uint32_t t = 0; t < out_diag_idx; t++) {
            uint32_t start = (base + n_slot_ - out_diag_idx * H_ + t * H_) % n_slot_;
            for (uint32_t h = 0; h < H_; h++) {
                mask[start + h] = 1.0;
            }
        }
    }
    return mask;
}

void ParLowerDiagTranspose::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    double mask_scale = param_.get_q(level_);

    transpose_mask_pt_.clear();
    transpose_mask_pt_.resize(m_);
    for (uint32_t out_diag_idx = 0; out_diag_idx < m_; out_diag_idx++) {
        transpose_mask_pt_[out_diag_idx].resize(2);
        for (uint32_t mask_idx = 0; mask_idx < 2; mask_idx++) {
            transpose_mask_pt_[out_diag_idx][mask_idx] =
                ctx.encode_ringt(build_transpose_mask(out_diag_idx, mask_idx), mask_scale);
        }
    }
}

CkksCiphertext ParLowerDiagTranspose::apply_mask(CkksContext& ctx,
                                                 const CkksCiphertext& input_level_l,
                                                 const CkksPlaintextRingt& mask_pt) const {
    double default_scale = param_.get_default_scale();
    auto mask_mul = ctx.ringt_to_mul(mask_pt, level_);
    return ctx.rescale(ctx.mult_plain_mul(input_level_l, mask_mul), default_scale);
}

std::vector<CkksCiphertext> ParLowerDiagTranspose::run_core(CkksContext& ctx,
                                                            const std::vector<CkksCiphertext>& input_cts) const {
    assert(input_cts.size() == m_c_);
    assert(transpose_mask_pt_.size() == m_);

    vector<CkksCiphertext> ct_ell_0(m_c_);
    vector<CkksCiphertext> ct_ell_1(m_c_);
    vector<bool> init_0(m_c_, false);
    vector<bool> init_1(m_c_, false);

    for (uint32_t j = 0; j < m_c_; j++) {
        for (uint32_t k = 0; k < c_; k++) {
            uint32_t source_diag_idx = c_ * j + k;
            uint32_t out_diag_idx = (m_ - (source_diag_idx % m_)) % m_;
            uint32_t ell = out_diag_idx / c_;
            uint32_t out_local_idx = out_diag_idx % c_;
            int rot = ((int)k - (int)out_local_idx) * (int)segment_len_ + (int)out_diag_idx * (int)H_;
            CkksCiphertext ct_rot = (rot == 0) ? input_cts[j].copy() : ctx.rotate(input_cts[j], rot);

            auto term0 = apply_mask(ctx, ct_rot, transpose_mask_pt_[out_diag_idx][0]);
            if (!init_0[ell]) {
                ct_ell_0[ell] = std::move(term0);
                init_0[ell] = true;
            } else {
                ct_ell_0[ell] = ctx.add(ct_ell_0[ell], term0);
            }

            auto term1 = apply_mask(ctx, ct_rot, transpose_mask_pt_[out_diag_idx][1]);
            if (!init_1[ell]) {
                ct_ell_1[ell] = std::move(term1);
                init_1[ell] = true;
            } else {
                ct_ell_1[ell] = ctx.add(ct_ell_1[ell], term1);
            }
        }
    }

    vector<CkksCiphertext> result;
    result.reserve(m_c_);
    for (uint32_t ell = 0; ell < m_c_; ell++) {
        assert(init_0[ell]);
        CkksCiphertext ct = std::move(ct_ell_0[ell]);
        if (init_1[ell]) {
            ct = ctx.add(ct, ctx.rotate(ct_ell_1[ell], -(int)segment_len_));
        }
        result.push_back(std::move(ct));
    }
    return result;
}

FeatureMatEncrypted ParLowerDiagTranspose::run(CkksContext& ctx, const FeatureMatEncrypted& input) {
    assert(input.level == level_);
    assert(input.head_shape[0] == m_ && input.head_shape[1] == n_prepad_);
    assert(input.shape[0] == H_prepad_ * input.head_shape[0] && input.shape[1] == input.head_shape[1]);
    assert(input.matmul_block_size == m_);
    assert(input.data.size() == expected_ct_count());

    FeatureMatEncrypted result(&ctx, input.level);
    Duo output_head_shape = {n_prepad_, m_};
    result.level = input.level - 1;
    result.shape = {output_head_shape[0], H_prepad_ * output_head_shape[1]};
    result.head_shape = output_head_shape;
    result.matmul_block_size = m_;
    result.data = run_core(ctx, input.data);
    return result;
}

Array<double, 2> ParLowerDiagTranspose::run_plaintext(const Array<double, 2>& A) const {
    assert(A.get_shape()[0] == H_prepad_ * m_);
    assert(A.get_shape()[1] == n_prepad_);

    Array<double, 2> T({n_prepad_, H_prepad_ * m_});
    for (uint32_t h = 0; h < H_prepad_; h++) {
        for (uint32_t i = 0; i < m_; i++) {
            for (uint32_t j = 0; j < n_prepad_; j++) {
                T.set(j, h * m_ + i, A.get(h * m_ + i, j));
            }
        }
    }
    return T;
}
