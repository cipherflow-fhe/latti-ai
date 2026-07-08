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
#include <vector>

// Softmax helpers for ciphertexts packed by
// FeatureMatEncrypted::par_diagonal_pack(..., is_lower=false, is_transposed=false).
// They are intended for attention matrices with shape {n_prepad, n_prepad * n_heads}
// and head_shape {n_prepad, n_prepad}. Head-wise reductions never cross heads.

class ParUpperDiagonalAddPt : public Layer {
public:
    ParUpperDiagonalAddPt(const ls::CkksParameter& param,
                          Duo shape,
                          Duo head_shape,
                          uint32_t n_heads,
                          uint32_t init_level,
                          double value);
    void prepare_weight() override;
    std::vector<ls::CkksPlaintextRingt> pt_;

    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& x);
    Array<double, 2> run_plaintext(const Array<double, 2>& x) const;

    ls::CkksPlaintextRingt generate_pt(ls::CkksContext& ctx, uint32_t ct_idx) const;

private:
    uint32_t n_prepad_ = 0, total_cols_ = 0, m_prepad_ = 0;
    uint32_t H_prepad_ = 0, H_ = 0, m_ = 0, n_ = 0;
    uint32_t n_slot_ = 0, segment_len_ = 0, c_ = 0, n_cts_ = 0;
    double value_ = 0.0;

    uint32_t total_cts() const;
    std::vector<double> build_constant_vec(uint32_t ct_idx, double value) const;
};

class ParUpperDiagonalMultipleSquare : public Layer {
public:
    ParUpperDiagonalMultipleSquare(const ls::CkksParameter& param,
                                   Duo shape,
                                   Duo head_shape,
                                   uint32_t n_heads,
                                   uint32_t init_level);
    void prepare_weight() override;
    std::vector<ls::CkksPlaintextRingt> mask_pt_;

    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& x);
    Array<double, 2> run_plaintext(const Array<double, 2>& x) const;

    ls::CkksPlaintextRingt generate_mask_pt(ls::CkksContext& ctx, uint32_t ct_idx) const;

private:
    uint32_t n_prepad_ = 0, total_cols_ = 0, m_prepad_ = 0;
    uint32_t H_prepad_ = 0, H_ = 0, m_ = 0, n_ = 0;
    uint32_t n_slot_ = 0, segment_len_ = 0, c_ = 0, n_cts_ = 0;

    uint32_t total_cts() const;
    std::vector<double> build_valid_mask(uint32_t ct_idx, double value) const;
};

class ParUpperDiagonalHeadColSum : public Layer {
public:
    ParUpperDiagonalHeadColSum(const ls::CkksParameter& param,
                               Duo shape,
                               Duo head_shape,
                               uint32_t n_heads,
                               uint32_t init_level);
    void prepare_weight() override;
    std::vector<ls::CkksPlaintextRingt> mask_pt_;

    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& x);
    Array<double, 2> run_plaintext(const Array<double, 2>& x) const;

    ls::CkksPlaintextRingt generate_mask_pt(ls::CkksContext& ctx, uint32_t ct_idx) const;

private:
    uint32_t n_prepad_ = 0, total_cols_ = 0, m_prepad_ = 0;
    uint32_t H_prepad_ = 0, H_ = 0, m_ = 0, n_ = 0;
    uint32_t n_slot_ = 0, segment_len_ = 0, c_ = 0, n_cts_ = 0;

    uint32_t total_cts() const;
    std::vector<double> build_valid_mask(uint32_t ct_idx, double value) const;
    ls::CkksCiphertext reduce_local_diags(ls::CkksContext& ctx, const ls::CkksCiphertext& ct) const;
};

class ParUpperDiagonalInverseInit : public Layer {
public:
    ParUpperDiagonalInverseInit(const ls::CkksParameter& param,
                                Duo shape,
                                Duo head_shape,
                                uint32_t n_heads,
                                uint32_t init_level);
    void prepare_weight() override;
    std::vector<ls::CkksPlaintextRingt> two_pt_;

    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& b);
    Array<double, 2> run_plaintext(const Array<double, 2>& b) const;

    ls::CkksPlaintextRingt generate_two_pt(ls::CkksContext& ctx, uint32_t ct_idx) const;

private:
    uint32_t n_prepad_ = 0, total_cols_ = 0, m_prepad_ = 0;
    uint32_t H_prepad_ = 0, H_ = 0, m_ = 0, n_ = 0;
    uint32_t n_slot_ = 0, segment_len_ = 0, c_ = 0, n_cts_ = 0;

    uint32_t total_cts() const;
    std::vector<double> build_valid_mask(uint32_t ct_idx, double value) const;
};

class ParUpperDiagonalInverseIter : public Layer {
public:
    ParUpperDiagonalInverseIter(const ls::CkksParameter& param,
                                Duo shape,
                                Duo head_shape,
                                uint32_t n_heads,
                                uint32_t init_level);
    void prepare_weight() override;
    std::vector<ls::CkksPlaintextRingt> one_pt_;
    std::vector<ls::CkksPlaintextRingt> two_pt_;

    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& a, const FeatureMatEncrypted& b);
    Array<double, 2> run_plaintext(const Array<double, 2>& a, const Array<double, 2>& b) const;

    ls::CkksPlaintextRingt generate_one_pt(ls::CkksContext& ctx, uint32_t ct_idx) const;
    ls::CkksPlaintextRingt generate_two_pt(ls::CkksContext& ctx, uint32_t ct_idx) const;

private:
    uint32_t n_prepad_ = 0, total_cols_ = 0, m_prepad_ = 0;
    uint32_t H_prepad_ = 0, H_ = 0, m_ = 0, n_ = 0;
    uint32_t n_slot_ = 0, segment_len_ = 0, c_ = 0, n_cts_ = 0;

    uint32_t total_cts() const;
    std::vector<double> build_valid_mask(uint32_t ct_idx, double value) const;
};

class ParUpperDiagonalGELU : public Layer {
public:
    ParUpperDiagonalGELU(const ls::CkksParameter& param,
                         Duo shape,
                         Duo head_shape,
                         uint32_t n_heads,
                         uint32_t init_level);
    void prepare_weight() override;
    std::vector<ls::CkksPlaintextRingt> mask_pt_;

    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& a, const FeatureMatEncrypted& b);
    Array<double, 2> run_plaintext(const Array<double, 2>& a, const Array<double, 2>& b) const;

    ls::CkksPlaintextRingt generate_mask_pt(ls::CkksContext& ctx, uint32_t ct_idx) const;

private:
    uint32_t n_prepad_ = 0, total_cols_ = 0, m_prepad_ = 0;
    uint32_t H_prepad_ = 0, H_ = 0, m_ = 0, n_ = 0;
    uint32_t n_slot_ = 0, segment_len_ = 0, c_ = 0, n_cts_ = 0;

    uint32_t total_cts() const;
    std::vector<double> build_valid_mask(uint32_t ct_idx, double value) const;
};
