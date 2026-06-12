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

#pragma once
#include "layer.h"
#include "../data_structs/feature_mat.h"

// LayerNorm for ciphertexts packed by FeatureMatEncrypted::par_upper_diagonal_pack.
// Matrix shape is {rows, cols}; normalization is row-wise across cols.

class ParUpperDiagonalLNStats : public Layer {
public:
    ParUpperDiagonalLNStats(const ls::CkksParameter& param,
                            Duo shape,
                            uint32_t n_heads,
                            uint32_t head_dim,
                            uint32_t init_level,
                            double eps,
                            double inv_var);
    void prepare_weight() override;

    std::vector<ls::CkksCiphertext> run(ls::CkksContext& ctx, const FeatureMatEncrypted& x);
    Array<double, 2> run_plaintext(const Array<double, 2>& x) const;

    ls::CkksPlaintextRingt
    generate_pt(ls::CkksContext& ctx, uint32_t pt_idx, uint32_t mb = 0, uint32_t ct_local = 0, uint32_t g = 0) const;

private:
    uint32_t total_rows_ = 0, n_prepad_ = 0;
    uint32_t H_prepad_ = 0, H_ = 0, m_ = 0, n_ = 0, d_prepad_ = 0;
    uint32_t n_slot_ = 0, segment_len_ = 0, c_ = 0, cts_per_mb_ = 0, n_mb_ = 0, row_cycle_len_ = 0;
    double eps_ = 0.0, inv_var_ = 0.0;

    uint32_t ct_index(uint32_t mb, uint32_t ct_local) const;
    uint32_t total_cts() const;
    std::vector<double> build_valid_mask(uint32_t mb, uint32_t ct_local, double value = 1.0) const;
    ls::CkksCiphertext intra_row_sum(ls::CkksContext& ctx, const ls::CkksCiphertext& ct) const;

    ls::CkksPlaintextRingt inv_n_pt_;
    ls::CkksPlaintextRingt iv_pt_;
    ls::CkksPlaintextRingt eps_add_pt_;
    std::vector<ls::CkksPlaintextRingt> valid_mask_pt_;
};

class ParUpperDiagonalLNXCentered : public Layer {
public:
    ParUpperDiagonalLNXCentered(const ls::CkksParameter& param,
                                Duo shape,
                                uint32_t n_heads,
                                uint32_t head_dim,
                                uint32_t init_level);
    void prepare_weight() override;

    std::vector<ls::CkksCiphertext> run(ls::CkksContext& ctx, const FeatureMatEncrypted& x);
    Array<double, 2> run_plaintext(const Array<double, 2>& x) const;

    ls::CkksPlaintextRingt
    generate_pt(ls::CkksContext& ctx, uint32_t pt_idx, uint32_t mb = 0, uint32_t ct_local = 0, uint32_t g = 0) const;

private:
    uint32_t total_rows_ = 0, n_prepad_ = 0;
    uint32_t H_prepad_ = 0, H_ = 0, m_ = 0, n_ = 0, d_prepad_ = 0;
    uint32_t n_slot_ = 0, segment_len_ = 0, c_ = 0, cts_per_mb_ = 0, n_mb_ = 0, row_cycle_len_ = 0;

    uint32_t ct_index(uint32_t mb, uint32_t ct_local) const;
    uint32_t total_cts() const;
    std::vector<double> build_valid_mask(uint32_t mb, uint32_t ct_local, double value = 1.0) const;
    ls::CkksCiphertext intra_row_sum(ls::CkksContext& ctx, const ls::CkksCiphertext& ct) const;

    ls::CkksPlaintextRingt inv_n_pt_;
    std::vector<ls::CkksPlaintextRingt> valid_mask_pt_;
};

class ParUpperDiagonalLNMinimaxInit : public Layer {
public:
    ParUpperDiagonalLNMinimaxInit(const ls::CkksParameter& param,
                                  Duo shape,
                                  uint32_t n_heads,
                                  uint32_t head_dim,
                                  uint32_t input_level,
                                  double c0,
                                  double c1,
                                  double c2);
    void prepare_weight() override;

    std::vector<ls::CkksCiphertext> run(ls::CkksContext& ctx, const std::vector<ls::CkksCiphertext>& a_cts);
    Array<double, 2> run_plaintext(const Array<double, 2>& a) const;

    ls::CkksPlaintextRingt
    generate_pt(ls::CkksContext& ctx, uint32_t pt_idx, uint32_t mb = 0, uint32_t ct_local = 0, uint32_t g = 0) const;

private:
    uint32_t total_rows_ = 0, n_prepad_ = 0;
    uint32_t H_prepad_ = 0, H_ = 0, m_ = 0, n_ = 0, d_prepad_ = 0;
    uint32_t n_slot_ = 0, segment_len_ = 0, c_ = 0, cts_per_mb_ = 0, n_mb_ = 0;
    double c0_ = 0.0, c1_ = 0.0, c2_ = 0.0;

    uint32_t ct_index(uint32_t mb, uint32_t ct_local) const;
    uint32_t total_cts() const;
    std::vector<double> build_valid_mask(uint32_t mb, uint32_t ct_local, double value) const;

    std::vector<ls::CkksPlaintextRingt> c2_norm_pt_;
    std::vector<ls::CkksPlaintextRingt> c1_pt_;
    std::vector<ls::CkksPlaintextRingt> c0_add_pt_;
};

class ParUpperDiagonalLNGoldschmidt : public Layer {
public:
    ParUpperDiagonalLNGoldschmidt(const ls::CkksParameter& param, uint32_t input_level);
    void prepare_weight() override;

    std::vector<ls::CkksCiphertext> run(ls::CkksContext& ctx,
                                        const std::vector<ls::CkksCiphertext>& y_cts,
                                        const std::vector<ls::CkksCiphertext>& a_cts);
    Array<double, 2> run_plaintext(const Array<double, 2>& y, const Array<double, 2>& a) const;

    ls::CkksPlaintextRingt
    generate_pt(ls::CkksContext& ctx, uint32_t pt_idx, uint32_t mb = 0, uint32_t ct_local = 0, uint32_t g = 0) const;

private:
    uint32_t n_slot_ = 0;

    ls::CkksPlaintextRingt three_pt_;
    ls::CkksPlaintextRingt half_norm_pt_;
};

class ParUpperDiagonalLNAffine : public Layer {
public:
    ParUpperDiagonalLNAffine(const ls::CkksParameter& param,
                             Duo shape,
                             uint32_t n_heads,
                             uint32_t head_dim,
                             uint32_t y_level,
                             double inv_std,
                             Array<double, 1>&& gamma,
                             Array<double, 1>&& beta);
    void prepare_weight() override;

    FeatureMatEncrypted run(ls::CkksContext& ctx,
                            const std::vector<ls::CkksCiphertext>& x_centered,
                            const std::vector<ls::CkksCiphertext>& y_cts);
    Array<double, 2> run_plaintext(const Array<double, 2>& x_centered, const Array<double, 2>& y) const;

    ls::CkksPlaintextRingt
    generate_pt(ls::CkksContext& ctx, uint32_t pt_idx, uint32_t mb = 0, uint32_t ct_local = 0, uint32_t g = 0) const;

private:
    uint32_t total_rows_ = 0, n_prepad_ = 0;
    uint32_t H_prepad_ = 0, H_ = 0, m_ = 0, n_ = 0, d_prepad_ = 0;
    uint32_t n_slot_ = 0, segment_len_ = 0, c_ = 0, cts_per_mb_ = 0, n_mb_ = 0;
    uint32_t y_level_ = 0;
    double inv_std_ = 0.0;
    Array<double, 1> gamma_vals_, beta_vals_;

    uint32_t ct_index(uint32_t mb, uint32_t ct_local) const;
    uint32_t total_cts() const;
    std::vector<double>
    build_valid_weight(uint32_t mb, uint32_t ct_local, const Array<double, 1>& values, double factor) const;

    std::vector<ls::CkksPlaintextRingt> gamma_pt_;
    std::vector<ls::CkksPlaintextRingt> beta_add_pt_;
};
