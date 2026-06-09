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

#include "par_lower_diag_ccmm.h"
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

ParLowerDiagCCMM::ParLowerDiagCCMM(const CkksParameter& param_in,
                                   const Duo& shape_A,
                                   const Duo& shape_B,
                                   uint32_t n_heads,
                                   uint32_t head_dim,
                                   uint32_t level)
    : Layer(param_in) {
    assert(level >= 3);
    level_ = level;
    shape_A_ = shape_A;
    shape_B_ = shape_B;
    H_prepad_ = n_heads;
    m_ = head_dim;
    n_prepad_ = shape_B_[1];

    assert(H_prepad_ > 0);
    assert(m_ > 0 && is_power_of_two(m_));
    assert(n_prepad_ > 0);

    H_ = next_pow2(H_prepad_);
    n_ = next_pow2(n_prepad_);
    assert(n_ >= m_);
    assert(n_ % m_ == 0);

    n_slot_ = param_.get_n() / 2;
    segment_len_ = H_ * n_;
    assert(segment_len_ > 0);
    assert(n_slot_ % segment_len_ == 0);
    c_ = n_slot_ / segment_len_;
    assert(c_ > 0 && is_power_of_two(c_));
    assert(m_ % c_ == 0);
    assert(n_ % c_ == 0);
    m_c_ = m_ / c_;
    n_c_ = n_ / c_;

    bool matches_kqt = shape_A_[0] == n_prepad_ && shape_A_[1] == m_ && shape_B_[0] == m_ && shape_B_[1] == n_prepad_;
    bool matches_ordinary =
        shape_A_[0] == m_ && shape_A_[1] == n_prepad_ && shape_B_[0] == n_prepad_ && shape_B_[1] == n_prepad_;
    if (matches_kqt && matches_ordinary) {
        is_kqt_ = false;
        output_shape_ = {m_, n_prepad_};
    } else if (matches_kqt) {
        is_kqt_ = true;
        output_shape_ = {n_prepad_, n_prepad_};
    } else if (matches_ordinary) {
        is_kqt_ = false;
        output_shape_ = {m_, n_prepad_};
    } else {
        throw runtime_error("ParLowerDiagCCMM supports only A(n,m)@B(m,n) and A(m,n)@B(n,n)");
    }
}

Duo ParLowerDiagCCMM::logical_to_full_shape(const Duo& logical_shape) const {
    return {logical_shape[1], H_prepad_ * logical_shape[0]};
}

uint32_t ParLowerDiagCCMM::expected_ct_count(const Duo& logical_shape) const {
    if (logical_shape[0] == n_prepad_ && logical_shape[1] == n_prepad_) {
        return n_c_;
    }
    assert(std::min(logical_shape[0], logical_shape[1]) == m_);
    return m_c_;
}

std::vector<double> ParLowerDiagCCMM::rotate_plain(const std::vector<double>& values, int step) const {
    assert(values.size() == n_slot_);
    vector<double> rotated(n_slot_, 0.0);
    int normalized = step % (int)n_slot_;
    if (normalized < 0) {
        normalized += (int)n_slot_;
    }
    for (uint32_t idx = 0; idx < n_slot_; idx++) {
        rotated[idx] = values[(idx + (uint32_t)normalized) % n_slot_];
    }
    return rotated;
}

std::vector<double> ParLowerDiagCCMM::build_replication_mask(uint32_t ell) const {
    vector<double> mask(n_slot_, 0.0);
    uint32_t local = ell % c_;
    uint32_t segment_base = local * segment_len_;
    for (uint32_t idx = 0; idx < segment_len_; idx++) {
        mask[segment_base + idx] = 1.0;
    }
    return mask;
}

std::vector<double> ParLowerDiagCCMM::build_ordinary_route_masks(uint32_t ell, uint32_t mask_idx) const {
    assert(mask_idx < 4);
    vector<double> mask(n_slot_, 0.0);
    uint32_t ell_c = ell % c_;
    uint32_t first_len = n_ - ell;
    for (uint32_t r = 0; r < c_; r++) {
        bool uses_prev = r < ell_c;
        for (uint32_t t = 0; t < n_; t++) {
            bool first_entries = t < first_len;
            uint32_t selected_mask_idx = 0;
            if (uses_prev && first_entries) {
                selected_mask_idx = 0;
            } else if (!uses_prev && first_entries) {
                selected_mask_idx = 1;
            } else if (uses_prev) {
                selected_mask_idx = 2;
            } else {
                selected_mask_idx = 3;
            }
            if (selected_mask_idx == mask_idx) {
                uint32_t start = r * segment_len_ + t * H_;
                for (uint32_t h = 0; h < H_; h++) {
                    mask[start + h] = 1.0;
                }
            }
        }
    }
    return mask_idx >= 2 ? rotate_plain(mask, (int)segment_len_) : mask;
}

std::vector<double> ParLowerDiagCCMM::build_kqt_route_masks(uint32_t j, uint32_t ell, uint32_t mask_idx) const {
    assert(mask_idx < 4);
    uint32_t a_ell = ell / c_;
    uint32_t b_ell = ell % c_;
    uint32_t p_prev = (j + n_c_ - 1 - a_ell) % n_c_;
    uint32_t p_curr = (j + n_c_ - a_ell) % n_c_;
    uint32_t q_prev = p_prev / m_c_;
    uint32_t q_curr = p_curr / m_c_;
    uint32_t R_prev = q_prev * m_ + ell;
    uint32_t R_curr = q_curr * m_ + ell;
    assert(R_prev < n_);
    assert(R_curr < n_);

    vector<double> mask(n_slot_, 0.0);
    for (uint32_t r = 0; r < c_; r++) {
        bool uses_prev = r < b_ell;
        uint32_t split = uses_prev ? (n_ - R_prev) : (n_ - R_curr);
        uint32_t first_mask = uses_prev ? 0 : 1;
        uint32_t wrap_mask = uses_prev ? 2 : 3;
        uint32_t selected_mask_idx = 0;
        uint32_t base = r * segment_len_;
        for (uint32_t t = 0; t < n_; t++) {
            selected_mask_idx = (t < split) ? first_mask : wrap_mask;
            if (selected_mask_idx == mask_idx) {
                uint32_t start = base + t * H_;
                for (uint32_t h = 0; h < H_; h++) {
                    mask[start + h] = 1.0;
                }
            }
        }
    }
    return mask_idx >= 2 ? rotate_plain(mask, (int)segment_len_) : mask;
}

void ParLowerDiagCCMM::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    double replication_scale = param_.get_q(level_);
    double route_scale = param_.get_q(level_ - 2);

    uint32_t replication_count = is_kqt_ ? m_ : n_;
    uint32_t replication_mask_count = std::min(c_, replication_count);
    replication_mask_pt_.clear();
    replication_mask_pt_.reserve(replication_mask_count);
    for (uint32_t ell = 0; ell < replication_mask_count; ell++) {
        replication_mask_pt_.push_back(ctx.encode_ringt(build_replication_mask(ell), replication_scale));
    }

    ordinary_route_pt_.clear();
    kqt_route_pt_.clear();
    if (is_kqt_) {
        kqt_route_pt_.resize(n_c_);
        for (uint32_t j = 0; j < n_c_; j++) {
            kqt_route_pt_[j].resize(m_);
            for (uint32_t ell = 0; ell < m_; ell++) {
                kqt_route_pt_[j][ell].resize(4);
                for (uint32_t mask_idx = 0; mask_idx < 4; mask_idx++) {
                    kqt_route_pt_[j][ell][mask_idx] =
                        ctx.encode_ringt(build_kqt_route_masks(j, ell, mask_idx), route_scale);
                }
            }
        }
    } else {
        ordinary_route_pt_.resize(n_);
        for (uint32_t ell = 0; ell < n_; ell++) {
            ordinary_route_pt_[ell].resize(4);
            for (uint32_t mask_idx = 0; mask_idx < 4; mask_idx++) {
                ordinary_route_pt_[ell][mask_idx] =
                    ctx.encode_ringt(build_ordinary_route_masks(ell, mask_idx), route_scale);
            }
        }
    }
}

CkksCiphertext
ParLowerDiagCCMM::replicate_lower_diag(CkksContext& ctx, const std::vector<CkksCiphertext>& B_cts, uint32_t ell) const {
    assert(!replication_mask_pt_.empty());
    uint32_t ct_idx = ell / c_;
    uint32_t mask_idx = ell % c_;
    assert(ct_idx < B_cts.size());
    assert(mask_idx < replication_mask_pt_.size());

    double default_scale = param_.get_default_scale();
    auto mask_mul = ctx.ringt_to_mul(replication_mask_pt_[mask_idx], level_);
    auto replicated = ctx.rescale(ctx.mult_plain_mul(B_cts[ct_idx], mask_mul), default_scale);
    for (uint32_t step = 1; step < c_; step <<= 1) {
        replicated = ctx.add(replicated, ctx.rotate(replicated, (int)(segment_len_ * step)));
    }
    return replicated;
}

CkksCiphertext ParLowerDiagCCMM::multiply_cts(CkksContext& ctx,
                                              const CkksCiphertext& a_level_l,
                                              const CkksCiphertext& b_level_l_minus_1) const {
    double default_scale = param_.get_default_scale();
    auto a = ctx.drop_level(a_level_l.copy());
    auto product = ctx.mult(a, b_level_l_minus_1);
    return ctx.rescale(ctx.relinearize(product), default_scale);
}

CkksCiphertext ParLowerDiagCCMM::apply_route_mask(CkksContext& ctx,
                                                  const CkksCiphertext& product_level_l_minus_2,
                                                  const CkksPlaintextRingt& mask_pt) const {
    double default_scale = param_.get_default_scale();
    auto mask_mul = ctx.ringt_to_mul(mask_pt, level_ - 2);
    return ctx.rescale(ctx.mult_plain_mul(product_level_l_minus_2, mask_mul), default_scale);
}

std::vector<CkksCiphertext> ParLowerDiagCCMM::run_core(CkksContext& ctx,
                                                       const std::vector<CkksCiphertext>& A_cts,
                                                       const std::vector<CkksCiphertext>& B_cts) const {
    return is_kqt_ ? run_core_kqt(ctx, A_cts, B_cts) : run_core_ordinary(ctx, A_cts, B_cts);
}

std::vector<CkksCiphertext> ParLowerDiagCCMM::run_core_ordinary(CkksContext& ctx,
                                                                const std::vector<CkksCiphertext>& A_cts,
                                                                const std::vector<CkksCiphertext>& B_cts) const {
    assert(A_cts.size() == m_c_);
    assert(B_cts.size() == n_c_);
    assert(replication_mask_pt_.size() == std::min(c_, n_));
    assert(ordinary_route_pt_.size() == n_);

    vector<CkksCiphertext> replicated_B;
    replicated_B.reserve(n_);
    for (uint32_t ell = 0; ell < n_; ell++) {
        replicated_B.push_back(replicate_lower_diag(ctx, B_cts, ell));
    }

    vector<vector<CkksCiphertext>> ct_C_j_ell(m_c_);
    for (uint32_t j = 0; j < m_c_; j++) {
        ct_C_j_ell[j].reserve(n_);
        ct_C_j_ell[j].push_back(multiply_cts(ctx, A_cts[j], replicated_B[0]));
        for (uint32_t ell = 1; ell < n_; ell++) {
            uint32_t ell_m = ell % m_;
            uint32_t ell_c = ell_m % c_;
            uint32_t source_shift = (ell_m - ell_c) / c_;
            uint32_t source_j = (j + m_c_ - source_shift) % m_c_;
            int rot = ((int)ell - (int)(n_ * ell_c)) * (int)H_;
            CkksCiphertext A_rot = (rot == 0) ? A_cts[source_j].copy() : ctx.rotate(A_cts[source_j], rot);
            ct_C_j_ell[j].push_back(multiply_cts(ctx, A_rot, replicated_B[ell]));
        }
    }

    vector<CkksCiphertext> C_cts;
    C_cts.reserve(m_c_);
    for (uint32_t j = 0; j < m_c_; j++) {
        CkksCiphertext ct_C_prime(0);
        CkksCiphertext ct_C_double_prime(0);
        bool prime_init = false;
        bool double_prime_init = false;
        uint32_t prev_j = (j + m_c_ - 1) % m_c_;

        for (uint32_t ell = 1; ell < n_; ell++) {
            auto term0 = apply_route_mask(ctx, ct_C_j_ell[prev_j][ell], ordinary_route_pt_[ell][0]);
            auto term1 = apply_route_mask(ctx, ct_C_j_ell[j][ell], ordinary_route_pt_[ell][1]);
            auto term2 = apply_route_mask(ctx, ct_C_j_ell[prev_j][ell], ordinary_route_pt_[ell][2]);
            auto term3 = apply_route_mask(ctx, ct_C_j_ell[j][ell], ordinary_route_pt_[ell][3]);

            auto prime_term = ctx.add(term0, term1);
            if (!prime_init) {
                ct_C_prime = std::move(prime_term);
                prime_init = true;
            } else {
                ct_C_prime = ctx.add(ct_C_prime, prime_term);
            }

            auto double_prime_term = ctx.add(term2, term3);
            if (!double_prime_init) {
                ct_C_double_prime = std::move(double_prime_term);
                double_prime_init = true;
            } else {
                ct_C_double_prime = ctx.add(ct_C_double_prime, double_prime_term);
            }
        }

        CkksCiphertext ct = ctx.drop_level(ct_C_j_ell[j][0]);
        if (prime_init) {
            ct = ctx.add(ct, ct_C_prime);
        }
        if (double_prime_init) {
            ct = ctx.add(ct, ctx.rotate(ct_C_double_prime, -(int)segment_len_));
        }
        C_cts.push_back(std::move(ct));
    }
    return C_cts;
}

std::vector<CkksCiphertext> ParLowerDiagCCMM::run_core_kqt(CkksContext& ctx,
                                                           const std::vector<CkksCiphertext>& A_cts,
                                                           const std::vector<CkksCiphertext>& B_cts) const {
    assert(A_cts.size() == m_c_);
    assert(B_cts.size() == m_c_);
    assert(replication_mask_pt_.size() == std::min(c_, m_));
    assert(kqt_route_pt_.size() == n_c_);

    vector<CkksCiphertext> replicated_B;
    replicated_B.reserve(m_);
    for (uint32_t ell = 0; ell < m_; ell++) {
        replicated_B.push_back(replicate_lower_diag(ctx, B_cts, ell));
    }

    vector<vector<CkksCiphertext>> ct_C_p_ell(n_c_);
    for (uint32_t p = 0; p < n_c_; p++) {
        uint32_t q_p = p / m_c_;
        uint32_t u_p = p % m_c_;
        ct_C_p_ell[p].reserve(m_);
        for (uint32_t ell = 0; ell < m_; ell++) {
            uint32_t b_ell = ell % c_;
            uint32_t R_p_ell = q_p * m_ + ell;
            assert(R_p_ell < n_);
            int rot = ((int)R_p_ell - (int)(n_ * b_ell)) * (int)H_;
            CkksCiphertext A_rot = (rot == 0) ? A_cts[u_p].copy() : ctx.rotate(A_cts[u_p], rot);
            ct_C_p_ell[p].push_back(multiply_cts(ctx, A_rot, replicated_B[ell]));
        }
    }

    vector<CkksCiphertext> C_cts;
    C_cts.reserve(n_c_);
    for (uint32_t j = 0; j < n_c_; j++) {
        assert(kqt_route_pt_[j].size() == m_);
        CkksCiphertext ct_C_prime(0);
        CkksCiphertext ct_C_double_prime(0);
        bool prime_init = false;
        bool double_prime_init = false;

        for (uint32_t ell = 0; ell < m_; ell++) {
            uint32_t a_ell = ell / c_;
            uint32_t p_prev = (j + n_c_ - 1 - a_ell) % n_c_;
            uint32_t p_curr = (j + n_c_ - a_ell) % n_c_;

            auto term0 = apply_route_mask(ctx, ct_C_p_ell[p_prev][ell], kqt_route_pt_[j][ell][0]);
            auto term1 = apply_route_mask(ctx, ct_C_p_ell[p_curr][ell], kqt_route_pt_[j][ell][1]);
            auto term2 = apply_route_mask(ctx, ct_C_p_ell[p_prev][ell], kqt_route_pt_[j][ell][2]);
            auto term3 = apply_route_mask(ctx, ct_C_p_ell[p_curr][ell], kqt_route_pt_[j][ell][3]);

            auto prime_term = ctx.add(term0, term1);
            if (!prime_init) {
                ct_C_prime = std::move(prime_term);
                prime_init = true;
            } else {
                ct_C_prime = ctx.add(ct_C_prime, prime_term);
            }

            auto double_prime_term = ctx.add(term2, term3);
            if (!double_prime_init) {
                ct_C_double_prime = std::move(double_prime_term);
                double_prime_init = true;
            } else {
                ct_C_double_prime = ctx.add(ct_C_double_prime, double_prime_term);
            }
        }

        assert(prime_init);
        CkksCiphertext ct = std::move(ct_C_prime);
        if (double_prime_init) {
            ct = ctx.add(ct, ctx.rotate(ct_C_double_prime, -(int)segment_len_));
        }
        C_cts.push_back(std::move(ct));
    }
    return C_cts;
}

FeatureMatEncrypted
ParLowerDiagCCMM::run(CkksContext& ctx, const FeatureMatEncrypted& A, const FeatureMatEncrypted& B) {
    assert(A.level == level_);
    assert(B.level == level_);

    Duo expected_A_shape = logical_to_full_shape(shape_A_);
    Duo expected_B_shape = logical_to_full_shape(shape_B_);
    assert(A.shape[0] == expected_A_shape[0] && A.shape[1] == expected_A_shape[1]);
    assert(B.shape[0] == expected_B_shape[0] && B.shape[1] == expected_B_shape[1]);
    assert(A.head_shape[0] == shape_A_[1] && A.head_shape[1] == shape_A_[0]);
    assert(B.head_shape[0] == shape_B_[1] && B.head_shape[1] == shape_B_[0]);
    assert(A.matmul_block_size == m_);
    assert(B.matmul_block_size == m_);
    assert(A.data.size() == expected_ct_count(shape_A_));
    assert(B.data.size() == expected_ct_count(shape_B_));

    FeatureMatEncrypted result(&ctx, A.level);
    result.level = A.level - 3;
    result.shape = logical_to_full_shape(output_shape_);
    result.head_shape = {output_shape_[1], output_shape_[0]};
    result.matmul_block_size = m_;
    result.data = run_core(ctx, A.data, B.data);
    return result;
}
