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

#include "par_lower_diag_pcmm.h"
#include "layer_util.h"
#include <algorithm>
#include <cassert>
#include <utility>

using namespace std;
using namespace lattisense;

ParLowerDiagPCMM::ParLowerDiagPCMM(const CkksParameter& param_in,
                                   const Duo& shape_X_T,
                                   uint32_t n_heads,
                                   uint32_t head_dim,
                                   const Array<double, 2>& W_mat,
                                   uint32_t level_X,
                                   Array<double, 1>&& bias)
    : Layer(param_in) {
    assert(level_X >= 2);
    level_ = level_X;
    H_prepad_ = n_heads;
    m_ = head_dim;
    assert(H_prepad_ > 0);
    assert(m_ > 0 && (m_ & (m_ - 1)) == 0);

    d_prepad_ = H_prepad_ * m_;
    in_rows_ = shape_X_T[0];
    n_prepad_ = shape_X_T[1];

    H_ = next_pow2(H_prepad_);
    n_ = next_pow2(n_prepad_);
    d_ = H_ * m_;
    assert(n_ >= m_);
    assert(n_ % m_ == 0);

    n_slot_ = param_.get_n() / 2;
    segment_len_ = H_ * n_;
    assert(n_slot_ % segment_len_ == 0);
    c_ = n_slot_ / segment_len_;
    assert(m_ % c_ == 0);
    m_c_ = m_ / c_;

    W_T_rows_ = W_mat.get_shape()[1];
    W_T_cols_ = W_mat.get_shape()[0];
    assert(in_rows_ == W_T_cols_);
    out_cols_ = W_T_rows_;

    K_row_ = div_ceil(W_T_rows_, d_prepad_);
    K_col_ = div_ceil(W_T_cols_, d_prepad_);
    if (K_row_ == K_col_) {
        mode_ = Mode::SQUARE;
        assert(K_row_ == 1);
    } else if (K_row_ > K_col_) {
        mode_ = Mode::EXPAND;
        assert(K_col_ == 1);
    } else {
        mode_ = Mode::REDUCE;
        assert(K_row_ == 1);
    }
    K_ = std::max(K_row_, K_col_);

    // weight matrix is tranposed here
    W_padded_.resize(K_);
    for (uint32_t mb = 0; mb < K_; mb++) {
        Array<double, 2> W_sub({d_, d_});
        for (uint32_t row = 0; row < d_prepad_; row++) {
            for (uint32_t col = 0; col < d_prepad_; col++) {
                uint32_t src_row = (mode_ == Mode::EXPAND) ? mb * d_prepad_ + row : row;
                uint32_t src_col = (mode_ == Mode::REDUCE) ? mb * d_prepad_ + col : col;
                double val = 0.0;
                if (src_row < W_T_rows_ && src_col < W_T_cols_) {
                    val = W_mat.get(src_col, src_row);
                }
                W_sub.set(row, col, val);
            }
        }
        W_padded_[mb] = std::move(W_sub);
    }

    if (bias.get_size() > 0) {
        has_bias_ = true;
        assert(bias.get_size() == out_cols_);
        bias_vals_ = std::move(bias);
    }
}

void ParLowerDiagPCMM::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    double pt_scale = param_.get_q(level_);
    double mask_scale = param_.get_q(level_ - 1);
    double bias_scale = param_.get_default_scale();

    pt_A_.clear();
    pt_A_.resize(K_);
    for (uint32_t mb = 0; mb < K_; mb++) {
        pt_A_[mb].resize(H_);
        for (uint32_t i = 0; i < H_; i++) {
            pt_A_[mb][i].reserve(m_c_ * c_ * m_c_);
            for (uint32_t j = 0; j < m_c_; j++) {
                for (uint32_t ell = 0; ell < c_; ell++) {
                    for (uint32_t r = 0; r < m_c_; r++) {
                        vector<double> plaintext(n_slot_, 0.0);
                        for (uint32_t tau = 0; tau < c_; tau++) {
                            uint32_t b_diag_idx = c_ * j + ((tau + ell) % c_);
                            uint32_t out_diag_idx = c_ * r + tau;
                            uint32_t segment_base = tau * segment_len_;
                            uint32_t rotate_step = (out_diag_idx * H_) % segment_len_;
                            uint32_t diag_idx = (b_diag_idx + m_ - (out_diag_idx % m_)) % m_;
                            for (uint32_t t = 0; t < n_; t++) {
                                for (uint32_t h = 0; h < H_; h++) {
                                    uint32_t block_row_base = ((i + h) % H_) * m_;
                                    uint32_t block_col_base = h * m_;
                                    uint32_t row = t % m_;
                                    uint32_t col = (diag_idx + row) % m_;
                                    uint32_t src_row = block_row_base + row;
                                    uint32_t src_col = block_col_base + col;
                                    uint32_t slot_in_segment = t * H_ + h;
                                    uint32_t rotated_slot =
                                        (slot_in_segment + segment_len_ - rotate_step) % segment_len_;
                                    plaintext[segment_base + rotated_slot] = W_padded_[mb].get(src_row, src_col);
                                }
                            }
                        }
                        pt_A_[mb][i].push_back(ctx.encode_ringt(plaintext, pt_scale));
                    }
                }
            }
        }
    }

    mask_wrap_pt_.clear();
    mask_wrap_pt_.resize(H_);
    for (uint32_t i = 1; i < H_; i++) {
        vector<double> mask(n_slot_, 0.0);
        for (uint32_t segment_start = 0; segment_start < n_slot_; segment_start += segment_len_) {
            for (uint32_t t = 0; t < n_; t++) {
                uint32_t group_start = segment_start + t * H_;
                for (uint32_t h = H_ - i; h < H_; h++) {
                    mask[group_start + h] = 1.0;
                }
            }
        }
        mask_wrap_pt_[i] = ctx.encode_ringt(mask, mask_scale);
    }

    bias_pt_.clear();
    if (has_bias_) {
        uint32_t output_mbs = (mode_ == Mode::EXPAND) ? K_ : 1;
        bias_pt_.resize(output_mbs * m_c_);
        for (uint32_t mb = 0; mb < output_mbs; mb++) {
            for (uint32_t r = 0; r < m_c_; r++) {
                vector<double> bias_vec(n_slot_, 0.0);
                for (uint32_t local_diag = 0; local_diag < c_; local_diag++) {
                    uint32_t diag_idx = r * c_ + local_diag;
                    uint32_t segment_base = local_diag * segment_len_;
                    for (uint32_t t = 0; t < n_; t++) {
                        for (uint32_t h = 0; h < H_; h++) {
                            uint32_t out_row = mb * d_prepad_ + h * m_ + ((diag_idx + t) % m_);
                            if (h < H_prepad_ && out_row < out_cols_) {
                                bias_vec[segment_base + t * H_ + h] = bias_vals_.get(out_row);
                            }
                        }
                    }
                }
                bias_pt_[mb * m_c_ + r] = ctx.encode_ringt(bias_vec, bias_scale);
            }
        }
    }
}

std::vector<CkksCiphertext> ParLowerDiagPCMM::run_core(CkksContext& ctx,
                                                       const std::vector<CkksCiphertext>& input_cts,
                                                       const std::vector<uint32_t>& mb_indices) const {
    double default_scale = param_.get_default_scale();
    vector<CkksCiphertext> reduced(m_c_);
    bool reduced_init = false;

    for (uint32_t local_mb = 0; local_mb < mb_indices.size(); local_mb++) {
        uint32_t weight_mb = mb_indices[local_mb];
        uint32_t input_offset = local_mb * m_c_;

        vector<vector<CkksCiphertext>> ct_Br(m_c_);
        for (uint32_t j = 0; j < m_c_; j++) {
            ct_Br[j].resize(c_);
            ct_Br[j][0] = input_cts[input_offset + j].copy();
            for (uint32_t ell = 1; ell < c_; ell++) {
                ct_Br[j][ell] = ctx.rotate(input_cts[input_offset + j], (int)(segment_len_ * ell));
            }
        }

        vector<vector<CkksCiphertext>> ct_ir(H_);
        for (uint32_t i = 0; i < H_; i++) {
            ct_ir[i].resize(m_c_);
            for (uint32_t r = 0; r < m_c_; r++) {
                CkksCiphertext acc(0);
                bool init = false;
                for (uint32_t j = 0; j < m_c_; j++) {
                    for (uint32_t ell = 0; ell < c_; ell++) {
                        uint32_t pt_idx = (j * c_ + ell) * m_c_ + r;
                        auto pt_mul = ctx.ringt_to_mul(pt_A_[weight_mb][i][pt_idx], level_);
                        auto product = ctx.mult_plain_mul(ct_Br[j][ell], pt_mul);
                        if (!init) {
                            acc = std::move(product);
                            init = true;
                        } else {
                            acc = ctx.add(acc, product);
                        }
                    }
                }
                ct_ir[i][r] = ctx.rescale(acc, default_scale);
            }
        }

        vector<CkksCiphertext> ct_C(m_c_);
        for (uint32_t r = 0; r < m_c_; r++) {
            CkksCiphertext acc = ctx.drop_level(ct_ir[0][r]);
            for (uint32_t i = 1; i < H_; i++) {
                auto mask_mul = ctx.ringt_to_mul(mask_wrap_pt_[i], level_ - 1);
                auto ct_R = ctx.rescale(ctx.mult_plain_mul(ct_ir[i][r], mask_mul), default_scale);
                auto ct_i_drop = ctx.drop_level(ct_ir[i][r]);
                auto ct_L = ctx.sub(ct_i_drop, ct_R);
                auto ct_R_rot = ctx.rotate(ct_R, (int)(H_ - i));
                auto ct_L_rot = ctx.rotate(ct_L, -(int)i);
                auto ct_prime = ctx.add(ct_R_rot, ct_L_rot);
                acc = ctx.add(acc, ct_prime);
            }
            ct_C[r] = std::move(acc);
        }

        if (!reduced_init) {
            reduced = std::move(ct_C);
            reduced_init = true;
        } else {
            for (uint32_t r = 0; r < m_c_; r++) {
                reduced[r] = ctx.add(reduced[r], ct_C[r]);
            }
        }
    }

    return reduced;
}

FeatureMatEncrypted ParLowerDiagPCMM::run(CkksContext& ctx, const FeatureMatEncrypted& X_T) {
    assert(X_T.level == level_);
    assert(X_T.shape[0] == n_prepad_);
    assert(X_T.shape[1] == in_rows_);
    assert(X_T.head_shape[0] == n_prepad_);
    assert(X_T.head_shape[1] == m_);
    assert(X_T.matmul_block_size == m_);
    assert(X_T.data.size() == K_col_ * m_c_);

    FeatureMatEncrypted result(&ctx, X_T.level);
    result.level = X_T.level - 2;
    result.shape = {n_prepad_, out_cols_};
    result.head_shape = {n_prepad_, m_};
    result.matmul_block_size = m_;

    if (mode_ == Mode::EXPAND) {
        for (uint32_t mb = 0; mb < K_; mb++) {
            auto mb_cts = run_core(ctx, X_T.data, {mb});
            if (has_bias_) {
                for (uint32_t r = 0; r < m_c_; r++) {
                    mb_cts[r] = ctx.add_plain_ringt(mb_cts[r], bias_pt_[mb * m_c_ + r]);
                }
            }
            for (auto& ct : mb_cts) {
                result.data.push_back(std::move(ct));
            }
        }
    } else {
        vector<uint32_t> all_mbs(K_);
        for (uint32_t i = 0; i < K_; i++) {
            all_mbs[i] = i;
        }
        result.data = run_core(ctx, X_T.data, all_mbs);
        if (has_bias_) {
            for (uint32_t r = 0; r < m_c_; r++) {
                result.data[r] = ctx.add_plain_ringt(result.data[r], bias_pt_[r]);
            }
        }
    }

    return result;
}
