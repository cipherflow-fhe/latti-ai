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

#include "par_block_col_major_ccmm.h"
#include "layer_util.h"
#include <cassert>
#include <cmath>

using namespace std;
using namespace lattisense;

ParBlockColMajorCCMM::ParBlockColMajorCCMM(const CkksParameter& param_in,
                                           const Duo& shape_A,
                                           const Duo& shape_B,
                                           uint32_t block_size,
                                           uint32_t n_heads,
                                           uint32_t level)
    : Layer(param_in) {
    assert(shape_A[1] == shape_B[0] && "inner dimensions must match: shape_A[1] != shape_B[0]");

    level_ = level;
    d_ = block_size;
    n_heads_ = n_heads;

    uint32_t m = shape_A[0];
    uint32_t n = shape_A[1];
    uint32_t p = shape_B[1];
    m_ = m;
    n_ = n;
    p_ = p;

    n_slot_ = param_.get_n() / 2;
    assert(n_slot_ >= d_ * d_ && "n_slot must be at least d*d");
    assert((d_ & (d_ - 1)) == 0 && "block_size must be a power of 2");
    n_h_padded_ = next_pow2(n_heads);

    // Determine chunk sizing
    if (n_slot_ >= n_h_padded_ * d_ * d_) {
        n_blocks_per_chunk_ = n_h_padded_;
        chunk_size_ = n_h_padded_ * d_ * d_;
        n_cts_per_block_idx_ = 1;
    } else {
        n_blocks_per_chunk_ = n_slot_ / (d_ * d_);
        chunk_size_ = n_slot_;
        if (n_blocks_per_chunk_ == 1) {
            n_h_padded_ = n_heads_;
        }
        n_cts_per_block_idx_ = n_h_padded_ / n_blocks_per_chunk_;
    }

    assert(n_slot_ % chunk_size_ == 0 && "n_slot must be divisible by chunk_size");
    num_chunks_ = n_slot_ / chunk_size_;

    num_block_rows_A_ = div_ceil(m, d_);
    num_block_cols_A_ = div_ceil(n, d_);
    num_block_rows_B_ = div_ceil(n, d_);
    num_block_cols_B_ = div_ceil(p, d_);

    bsgs_bs_sigma_ = (uint32_t)ceil(sqrt((double)d_));
    bsgs_gs_sigma_ = div_ceil(d_, bsgs_bs_sigma_);
    uint32_t n_tau = 2 * d_ - 1;
    bsgs_bs_tau_ = (uint32_t)ceil(sqrt((double)n_tau));
    bsgs_gs_tau_ = div_ceil(n_tau, bsgs_bs_tau_);
}

int ParBlockColMajorCCMM::get_block_index(int bi, int bj, int num_block_rows) {
    return bi + num_block_rows * bj;
}

// Build sigma diagonal: base on d² elements, then expand by S
std::vector<double> ParBlockColMajorCCMM::build_sigma_diagonal(int k_idx) const {
    uint32_t d_sq = d_ * d_;
    uint32_t S = n_blocks_per_chunk_;

    vector<double> u_base(d_sq, 0.0);
    for (uint32_t j = 0; j < d_; j++) {
        for (uint32_t i = 0; i < d_; i++) {
            int idx = i + d_ * j;
            int col_from_diag = (idx + (int)(d_ * k_idx)) % (int)d_sq;
            int col_required = i + d_ * ((i + j) % d_);
            if (col_from_diag == col_required) {
                u_base[idx] = 1.0;
            }
        }
    }

    // Expand by S
    vector<double> expanded_chunk(chunk_size_, 0.0);
    for (uint32_t idx = 0; idx < d_sq; idx++) {
        for (uint32_t h = 0; h < S; h++) {
            expanded_chunk[idx * S + h] = u_base[idx];
        }
    }

    // Tile to n_slot_
    vector<double> u(n_slot_, 0.0);
    for (uint32_t c = 0; c < num_chunks_; c++) {
        for (uint32_t s = 0; s < chunk_size_; s++) {
            u[c * chunk_size_ + s] = expanded_chunk[s];
        }
    }
    return u;
}

// Build tau diagonal: base on d² elements, then expand by S
// BSGS: shift base vector to compensate for giant-step rotation
std::vector<double> ParBlockColMajorCCMM::build_tau_diagonal(int offset) const {
    uint32_t d_sq = d_ * d_;
    uint32_t S = n_blocks_per_chunk_;

    vector<double> u_base(d_sq, 0.0);
    for (uint32_t j = 0; j < d_; j++) {
        for (uint32_t i = 0; i < d_; i++) {
            int idx = i + d_ * j;
            int col_from_diag = ((idx + offset) % (int)d_sq + (int)d_sq) % (int)d_sq;
            int col_required = ((i + j) % d_) + d_ * j;
            if (col_from_diag == col_required) {
                u_base[idx] = 1.0;
            }
        }
    }

    // BSGS shift: rotate base by -giant_rot/S to compensate
    uint32_t j_idx = (uint32_t)(offset + (int)(d_ - 1));
    uint32_t g_bsgs = j_idx / bsgs_bs_tau_;
    int shift = (int)(g_bsgs * bsgs_bs_tau_) - (int)(d_ - 1);
    shift = ((shift % (int)d_sq) + (int)d_sq) % (int)d_sq;
    if (shift != 0) {
        vector<double> u_shifted(d_sq, 0.0);
        for (uint32_t idx = 0; idx < d_sq; idx++) {
            u_shifted[idx] = u_base[(idx + d_sq - shift) % d_sq];
        }
        u_base = move(u_shifted);
    }

    // Expand by S
    vector<double> expanded_chunk(chunk_size_, 0.0);
    for (uint32_t idx = 0; idx < d_sq; idx++) {
        for (uint32_t h = 0; h < S; h++) {
            expanded_chunk[idx * S + h] = u_base[idx];
        }
    }

    // Tile to n_slot_
    vector<double> u(n_slot_, 0.0);
    for (uint32_t c = 0; c < num_chunks_; c++) {
        for (uint32_t s = 0; s < chunk_size_; s++) {
            u[c * chunk_size_ + s] = expanded_chunk[s];
        }
    }
    return u;
}

// Build psi diagonal pair (w_k, w_{k-d}): base on d² elements, then expand by S
std::pair<std::vector<double>, std::vector<double>> ParBlockColMajorCCMM::build_psi_diagonals(int k_val) const {
    uint32_t d_sq = d_ * d_;
    uint32_t S = n_blocks_per_chunk_;

    vector<double> w_k_base(d_sq, 0.0);
    vector<double> w_k_minus_d_base(d_sq, 0.0);

    for (uint32_t j = 0; j < d_; j++) {
        for (uint32_t i = 0; i < d_; i++) {
            int idx = i + d_ * j;
            int col_required = ((i + k_val) % (int)d_ + (int)d_) % (int)d_ + d_ * j;

            int col_from_diag_k = ((idx + k_val) % (int)d_sq + (int)d_sq) % (int)d_sq;
            if (col_from_diag_k == col_required) {
                w_k_base[idx] = 1.0;
            }

            int col_from_diag_kd = ((idx + k_val - (int)d_) % (int)d_sq + (int)d_sq) % (int)d_sq;
            if (col_from_diag_kd == col_required) {
                w_k_minus_d_base[idx] = 1.0;
            }
        }
    }

    // Expand both by S
    vector<double> w_k_chunk(chunk_size_, 0.0);
    vector<double> w_kd_chunk(chunk_size_, 0.0);
    for (uint32_t idx = 0; idx < d_sq; idx++) {
        for (uint32_t h = 0; h < S; h++) {
            w_k_chunk[idx * S + h] = w_k_base[idx];
            w_kd_chunk[idx * S + h] = w_k_minus_d_base[idx];
        }
    }

    // Tile both to n_slot_
    vector<double> w_k(n_slot_, 0.0);
    vector<double> w_kd(n_slot_, 0.0);
    for (uint32_t c = 0; c < num_chunks_; c++) {
        for (uint32_t s = 0; s < chunk_size_; s++) {
            w_k[c * chunk_size_ + s] = w_k_chunk[s];
            w_kd[c * chunk_size_ + s] = w_kd_chunk[s];
        }
    }
    return {w_k, w_kd};
}

// Build all-ones vector for psi when i=0 (identity transform)
std::vector<double> ParBlockColMajorCCMM::build_psi_k_equal_0_diagonals() const {
    return vector<double>(n_slot_, 1.0);
}

void ParBlockColMajorCCMM::precompute_diagonals() {
    CkksContext ctx = CkksContext::create_empty_context(param_);

    double default_scale = param_.get_default_scale();
    double sigma_tau_scale = param_.get_q(level_);
    double psi_scale = param_.get_q(level_ - 2) / default_scale * param_.get_q(level_ - 1);

    // Sigma: d diagonal vectors
    sigma_diag_pt_.clear();
    sigma_diag_pt_.reserve(d_);
    for (uint32_t k_idx = 0; k_idx < d_; k_idx++) {
        auto diag_vec = build_sigma_diagonal(k_idx);
        sigma_diag_pt_.push_back(ctx.encode_ringt(diag_vec, sigma_tau_scale));
    }

    // Tau: 2d-1 diagonal vectors, offsets -(d-1) to (d-1)
    tau_diag_pt_.clear();
    tau_diag_pt_.reserve(2 * d_ - 1);
    for (int offset = -(int)(d_ - 1); offset <= (int)(d_ - 1); offset++) {
        auto diag_vec = build_tau_diagonal(offset);
        tau_diag_pt_.push_back(ctx.encode_ringt(diag_vec, sigma_tau_scale));
    }

    // Psi i=0: all-ones vector (identity)
    auto k0_vec = build_psi_k_equal_0_diagonals();
    psi_k0_pt_ = ctx.encode_ringt(k0_vec, psi_scale);

    // Psi i=1..d-1: pairs of diagonal vectors
    psi_w_k_pt_.clear();
    psi_w_k_minus_d_pt_.clear();
    psi_w_k_pt_.reserve(d_ - 1);
    psi_w_k_minus_d_pt_.reserve(d_ - 1);
    for (uint32_t i = 1; i < d_; i++) {
        auto [w_k, w_kd] = build_psi_diagonals(i);
        psi_w_k_pt_.push_back(ctx.encode_ringt(w_k, psi_scale));
        psi_w_k_minus_d_pt_.push_back(ctx.encode_ringt(w_kd, psi_scale));
    }
}

// ── Generate methods for encode_pt executor ───────────────────────────────────

CkksPlaintextRingt ParBlockColMajorCCMM::generate_sigma_pt(CkksContext& ctx, uint32_t k) const {
    return ctx.encode_ringt(generate_sigma_values(k), param_.get_q(level_));
}

std::vector<double> ParBlockColMajorCCMM::generate_sigma_values(uint32_t k) const {
    return build_sigma_diagonal(k);
}

CkksPlaintextRingt ParBlockColMajorCCMM::generate_tau_pt(CkksContext& ctx, uint32_t offset_idx) const {
    return ctx.encode_ringt(generate_tau_values(offset_idx), param_.get_q(level_));
}

std::vector<double> ParBlockColMajorCCMM::generate_tau_values(uint32_t offset_idx) const {
    int offset = (int)offset_idx - (int)(d_ - 1);
    return build_tau_diagonal(offset);
}

CkksPlaintextRingt ParBlockColMajorCCMM::generate_psi_k0_pt(CkksContext& ctx) const {
    double psi_scale = param_.get_q(level_ - 2) / param_.get_default_scale() * param_.get_q(level_ - 1);
    return ctx.encode_ringt(generate_psi_k0_values(), psi_scale);
}

std::vector<double> ParBlockColMajorCCMM::generate_psi_k0_values() const {
    return build_psi_k_equal_0_diagonals();
}

CkksPlaintextRingt ParBlockColMajorCCMM::generate_psi_wk_pt(CkksContext& ctx, uint32_t i) const {
    double psi_scale = param_.get_q(level_ - 2) / param_.get_default_scale() * param_.get_q(level_ - 1);
    return ctx.encode_ringt(generate_psi_wk_values(i), psi_scale);
}

std::vector<double> ParBlockColMajorCCMM::generate_psi_wk_values(uint32_t i) const {
    return build_psi_diagonals(i).first;
}

CkksPlaintextRingt ParBlockColMajorCCMM::generate_psi_wkd_pt(CkksContext& ctx, uint32_t i) const {
    double psi_scale = param_.get_q(level_ - 2) / param_.get_default_scale() * param_.get_q(level_ - 1);
    return ctx.encode_ringt(generate_psi_wkd_values(i), psi_scale);
}

std::vector<double> ParBlockColMajorCCMM::generate_psi_wkd_values(uint32_t i) const {
    return build_psi_diagonals(i).second;
}

// sigma with BSGS: (bsgs_bs-1) baby + (bsgs_gs-1) giant rotations
// + d pt_muls + (d-1) adds + 1 rescale.  Level L -> L-1.
CkksCiphertext ParBlockColMajorCCMM::sigma_on_ct(CkksContext& ctx, const CkksCiphertext& a) const {
    double default_scale = param_.get_default_scale();
    uint32_t S = n_blocks_per_chunk_;
    int unit = (int)(d_ * S);

    auto baby_rots = populate_rotations_1_side(ctx, a, bsgs_bs_sigma_ - 1, unit);

    CkksCiphertext result(0);
    bool result_init = false;

    for (uint32_t g = 0; g < bsgs_gs_sigma_; g++) {
        CkksCiphertext inner(0);
        bool inner_init = false;
        uint32_t b_end = std::min(bsgs_bs_sigma_, d_ - g * bsgs_bs_sigma_);

        for (uint32_t b = 0; b < b_end; b++) {
            uint32_t k = g * bsgs_bs_sigma_ + b;
            auto diag_mul = ctx.ringt_to_mul(sigma_diag_pt_[k], level_);
            auto product = ctx.mult_plain_mul(baby_rots[b], diag_mul);

            if (!inner_init) {
                inner = move(product);
                inner_init = true;
            } else {
                inner = ctx.add(inner, product);
            }
        }

        int giant_rot = (int)(g * bsgs_bs_sigma_) * unit;
        if (giant_rot != 0)
            inner = ctx.rotate(inner, giant_rot);

        if (!result_init) {
            result = move(inner);
            result_init = true;
        } else {
            result = ctx.add(result, inner);
        }
    }
    return ctx.rescale(result, default_scale);
}

// tau with BSGS: (bsgs_bs-1) baby + up to bsgs_gs giant rotations
// + (2d-1) pt_muls + (2d-2) adds + 1 rescale.  Level L -> L-1.
CkksCiphertext ParBlockColMajorCCMM::tau_on_ct(CkksContext& ctx, const CkksCiphertext& b) const {
    double default_scale = param_.get_default_scale();
    uint32_t S = n_blocks_per_chunk_;
    int unit = (int)S;
    uint32_t n_tau = 2 * d_ - 1;

    auto baby_rots = populate_rotations_1_side(ctx, b, bsgs_bs_tau_ - 1, unit);

    CkksCiphertext result(0);
    bool result_init = false;

    for (uint32_t g = 0; g < bsgs_gs_tau_; g++) {
        CkksCiphertext inner(0);
        bool inner_init = false;
        uint32_t b_end = std::min(bsgs_bs_tau_, n_tau - g * bsgs_bs_tau_);

        for (uint32_t b_step = 0; b_step < b_end; b_step++) {
            uint32_t j_idx = g * bsgs_bs_tau_ + b_step;
            auto diag_mul = ctx.ringt_to_mul(tau_diag_pt_[j_idx], level_);
            auto product = ctx.mult_plain_mul(baby_rots[b_step], diag_mul);

            if (!inner_init) {
                inner = move(product);
                inner_init = true;
            } else {
                inner = ctx.add(inner, product);
            }
        }

        int giant_rot = ((int)(g * bsgs_bs_tau_) - (int)(d_ - 1)) * (int)S;
        giant_rot = ((giant_rot % (int)chunk_size_) + (int)chunk_size_) % (int)chunk_size_;
        if (giant_rot != 0)
            inner = ctx.rotate(inner, giant_rot);

        if (!result_init) {
            result = move(inner);
            result_init = true;
        } else {
            result = ctx.add(result, inner);
        }
    }
    return ctx.rescale(result, default_scale);
}

// phi^i: rotation by d*i*S positions (rotation only, no level consumed)
// Input level L-1 -> Output level L-1
CkksCiphertext ParBlockColMajorCCMM::phi_on_ct(CkksContext& ctx, const CkksCiphertext& a_sigma, int i) const {
    uint32_t S = n_blocks_per_chunk_;
    int rot = ((int)(d_ * i) * (int)S) % (int)chunk_size_;
    return (rot == 0) ? a_sigma.copy() : ctx.rotate(a_sigma, rot);
}

// psi^i: 2 rotations + 2 pt_muls + 1 add + 1 rescale
// Input level L-1 -> Output level L-2
// Rotation amounts scaled by n_blocks_per_chunk_
CkksCiphertext ParBlockColMajorCCMM::psi_on_ct(CkksContext& ctx, const CkksCiphertext& b_tau, int i) const {
    double default_scale = param_.get_default_scale();
    uint32_t S = n_blocks_per_chunk_;
    int psi_idx = i - 1;  // psi_w_k_pt_ is 0-indexed for i=1..d-1

    // Scaled rotation amounts
    int rot_k = ((i * (int)S) % (int)chunk_size_ + (int)chunk_size_) % (int)chunk_size_;
    int rot_kd = (((i - (int)d_) * (int)S) % (int)chunk_size_ + (int)chunk_size_) % (int)chunk_size_;

    CkksCiphertext rotated_k = (rot_k == 0) ? b_tau.copy() : ctx.rotate(b_tau, rot_k);
    CkksCiphertext rotated_kd = (rot_kd == 0) ? b_tau.copy() : ctx.rotate(b_tau, rot_kd);

    auto w_k_mul = ctx.ringt_to_mul(psi_w_k_pt_[psi_idx], level_ - 1);
    auto w_kd_mul = ctx.ringt_to_mul(psi_w_k_minus_d_pt_[psi_idx], level_ - 1);

    auto term1 = ctx.mult_plain_mul(rotated_k, w_k_mul);
    auto term2 = ctx.mult_plain_mul(rotated_kd, w_kd_mul);

    return ctx.rescale(ctx.add(term1, term2), default_scale);
}

// block_mult: parallel block multiply for n_blocks_per_chunk pairs of corresponding blocks
// of two interleaved block ciphertexts
// Input level L -> Output level L-3
CkksCiphertext
ParBlockColMajorCCMM::block_mult_ct(CkksContext& ctx, const CkksCiphertext& a, const CkksCiphertext& b) const {
    double default_scale = param_.get_default_scale();

    // Step 1: Apply sigma/tau transforms (L -> L-1)
    CkksCiphertext a_sigma = sigma_on_ct(ctx, a);
    CkksCiphertext b_tau = tau_on_ct(ctx, b);

    // Step 2: i=0, use all-ones psi_k0 plaintext to scale b_tau
    auto psi_k0_mul = ctx.ringt_to_mul(psi_k0_pt_, level_ - 1);
    auto b_0 = ctx.rescale(ctx.mult_plain_mul(b_tau, psi_k0_mul), default_scale);  // L-1 -> L-2
    auto a_0 = ctx.drop_level(a_sigma.copy());                                     // L-1 -> L-2
    auto ct3_0 = ctx.mult(a_0, b_0);
    auto result = ctx.rescale(ctx.relinearize(ct3_0), default_scale);  // L-2 -> L-3

    // Step 2: i=1..d-1, use psi_on_ct for general case
    for (uint32_t i = 1; i < d_; i++) {
        CkksCiphertext a_i = phi_on_ct(ctx, a_sigma, i);  // L-1 (rotation only)
        CkksCiphertext b_i = psi_on_ct(ctx, b_tau, i);    // L-1 -> L-2

        auto a_i_dropped = ctx.drop_level(a_i);  // L-1 -> L-2

        auto ct3_i = ctx.mult(a_i_dropped, b_i);
        auto prod_i = ctx.rescale(ctx.relinearize(ct3_i), default_scale);  // L-2 -> L-3

        result = ctx.add(result, prod_i);
    }

    return result;  // at L-3
}

// perform block-based matrix multiplication for each head parallelly in the multi-head attention module
std::vector<CkksCiphertext> ParBlockColMajorCCMM::run_core(CkksContext& ctx,
                                                           const std::vector<CkksCiphertext>& A_cts,
                                                           const std::vector<CkksCiphertext>& B_cts) {
    uint32_t num_result_blocks = num_block_rows_A_ * num_block_cols_B_;
    uint32_t num_result_vecs = num_result_blocks * n_cts_per_block_idx_;
    vector<CkksCiphertext> C_cts;
    C_cts.resize(num_result_vecs);

    parallel_for(num_result_vecs, th_nums, ctx, [&](CkksContext& ctx_copy, int c_vec_idx) {
        uint32_t c_block_idx = c_vec_idx / n_cts_per_block_idx_;
        uint32_t g = c_vec_idx % n_cts_per_block_idx_;

        // Column-major index: c_block_idx = bi + num_block_rows_A_ * bp
        int bi = c_block_idx % num_block_rows_A_;
        int bp = c_block_idx / num_block_rows_A_;

        for (uint32_t bj = 0; bj < num_block_cols_A_; bj++) {
            int a_block_idx = get_block_index(bi, bj, num_block_rows_A_);
            int b_block_idx = get_block_index(bj, bp, num_block_rows_B_);

            int a_vec_idx = a_block_idx * n_cts_per_block_idx_ + g;
            int b_vec_idx = b_block_idx * n_cts_per_block_idx_ + g;

            auto product = block_mult_ct(ctx_copy, A_cts[a_vec_idx], B_cts[b_vec_idx]);

            if (bj == 0) {
                C_cts[c_vec_idx] = move(product);
            } else {
                C_cts[c_vec_idx] = ctx_copy.add(C_cts[c_vec_idx], product);
            }
        }
    });

    return C_cts;
}

FeatureMatEncrypted
ParBlockColMajorCCMM::run(CkksContext& ctx, const FeatureMatEncrypted& A, const FeatureMatEncrypted& B) {
    FeatureMatEncrypted result(&ctx, A.level);
    result.data = run_core(ctx, A.data, B.data);
    result.level = A.level - 3;  // block_mult consumes 3 levels
    result.head_shape = {m_, p_};
    result.shape = {m_, p_ * n_heads_};
    result.matmul_block_size = d_;
    return result;
}

Array<double, 2> ParBlockColMajorCCMM::run_plaintext(const Array<double, 2>& A, const Array<double, 2>& B) const {
    Array<double, 2> C({m_, p_ * n_heads_});
    for (uint32_t h = 0; h < n_heads_; h++)
        for (uint32_t i = 0; i < m_; i++)
            for (uint32_t j = 0; j < p_; j++) {
                double s = 0;
                for (uint32_t k = 0; k < n_; k++)
                    s += A.get(i, h * n_ + k) * B.get(k, h * p_ + j);
                C.set(i, h * p_ + j, s);
            }
    return C;
}
