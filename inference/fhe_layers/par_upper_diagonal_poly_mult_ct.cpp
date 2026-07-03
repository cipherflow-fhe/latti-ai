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

#include "par_upper_diagonal_poly_mult_ct.h"
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

ParUpperDiagonalPolyMultCt::ParUpperDiagonalPolyMultCt(const CkksParameter& param,
                                                       Duo shape,
                                                       Duo head_shape,
                                                       uint32_t n_heads,
                                                       uint32_t init_level)
    : Layer(param) {
    level_ = init_level;
    assert(level_ >= 2);
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

uint32_t ParUpperDiagonalPolyMultCt::ct_index(uint32_t mb, uint32_t ct_local) const {
    return mb * cts_per_mb_ + ct_local;
}

uint32_t ParUpperDiagonalPolyMultCt::total_cts() const {
    return n_mb_ * cts_per_mb_;
}

vector<double> ParUpperDiagonalPolyMultCt::build_constant_vec(double value, uint32_t mb, uint32_t ct_local) const {
    assert(ct_local < cts_per_mb_);
    vector<double> vec(n_slot_, 0.0);
    for (uint32_t local_diag = 0; local_diag < c_; local_diag++) {
        uint32_t segment_base = local_diag * segment_len_;
        for (uint32_t t = 0; t < n_; t++) {
            for (uint32_t h = 0; h < H_; h++) {
                if (is_valid_slot(mb, ct_local, local_diag, t, h, n_prepad_, total_cols_, m_prepad_, H_prepad_, m_, c_,
                                  packed_extent_)) {
                    vec[segment_base + t * H_ + h] = value;
                }
            }
        }
    }
    return vec;
}

CkksPlaintextRingt
ParUpperDiagonalPolyMultCt::generate_one_pt(CkksContext& ctx, uint32_t mb, uint32_t ct_local, uint32_t) const {
    double D = param_.get_default_scale();
    double scale = static_cast<double>(param_.get_q(level_ - 1)) / D * static_cast<double>(param_.get_q(level_));
    return ctx.encode_ringt(build_constant_vec(1.0, mb, ct_local), scale);
}

CkksPlaintextRingt
ParUpperDiagonalPolyMultCt::generate_half_pt(CkksContext& ctx, uint32_t mb, uint32_t ct_local, uint32_t) const {
    return ctx.encode_ringt(build_constant_vec(0.5, mb, ct_local), param_.get_default_scale());
}

void ParUpperDiagonalPolyMultCt::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    uint32_t n_ct = total_cts();
    one_pt_.resize(n_ct);
    half_pt_.resize(n_ct);
    for (uint32_t mb = 0; mb < n_mb_; mb++) {
        for (uint32_t ct_local = 0; ct_local < cts_per_mb_; ct_local++) {
            uint32_t idx = ct_index(mb, ct_local);
            one_pt_[idx] = generate_one_pt(ctx, mb, ct_local);
            half_pt_[idx] = generate_half_pt(ctx, mb, ct_local);
        }
    }
}

FeatureMatEncrypted
ParUpperDiagonalPolyMultCt::run(CkksContext& ctx, const FeatureMatEncrypted& half_tanh, const FeatureMatEncrypted& x) {
    assert(x.level == level_);
    assert(half_tanh.level == level_);
    assert(x.shape[0] == n_prepad_ && x.shape[1] == total_cols_);
    assert(half_tanh.shape == x.shape);
    assert(x.head_shape[0] == n_prepad_ && x.head_shape[1] == m_prepad_);
    assert(half_tanh.head_shape == x.head_shape);
    assert(x.matmul_block_size == m_);
    assert(half_tanh.matmul_block_size == x.matmul_block_size);
    assert(x.data.size() == total_cts());
    assert(half_tanh.data.size() == x.data.size());

    double D = param_.get_default_scale();
    FeatureMatEncrypted result(&ctx, level_ - 2);
    result.shape = x.shape;
    result.head_shape = x.head_shape;
    result.matmul_block_size = m_;
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.data.resize(total_cts());

    parallel_for(total_cts(), th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        auto one_mul = ctx_copy.ringt_to_mul(one_pt_[idx], level_);
        auto x_scaled = ctx_copy.rescale(ctx_copy.mult_plain_mul(x.data[idx], one_mul), D);  // level L-1, scale q_{L-1}
        auto half_tanh_drop = ctx_copy.drop_level(half_tanh.data[idx]);
        auto half_plus = ctx_copy.add_plain_ringt(half_tanh_drop, half_pt_[idx]);  // level L-1, scale D
        auto product = ctx_copy.mult(x_scaled, half_plus);
        result.data[idx] = ctx_copy.rescale(ctx_copy.relinearize(product), D);  // level L-2, scale D
    });

    result.level = level_ - 2;
    return result;
}

Array<double, 2> ParUpperDiagonalPolyMultCt::run_plaintext(const Array<double, 2>& half_tanh,
                                                           const Array<double, 2>& x) const {
    auto x_shape = x.get_shape();
    auto half_tanh_shape = half_tanh.get_shape();
    if (x_shape[0] != n_prepad_ || x_shape[1] != total_cols_ || half_tanh_shape != x_shape) {
        throw runtime_error("ParUpperDiagonalPolyMultCt plaintext input shape mismatch");
    }

    Array<double, 2> result({n_prepad_, total_cols_});
    for (uint32_t i = 0; i < n_prepad_; i++) {
        for (uint32_t j = 0; j < total_cols_; j++) {
            result.set(i, j, x.get(i, j) * (0.5 + half_tanh.get(i, j)));
        }
    }
    return result;
}
