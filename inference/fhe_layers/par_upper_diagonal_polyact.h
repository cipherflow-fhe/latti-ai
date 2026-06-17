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

// Polynomial activation helpers for ciphertexts packed by
// FeatureMatEncrypted::par_diagonal_pack(..., is_lower=false, is_transposed=false).
// Matrix shape is {n_prepad, total_cols}; head_shape is {n_prepad, m_prepad}.

class ParUpperDiagonalPolyActRNGamma : public Layer {
public:
    ParUpperDiagonalPolyActRNGamma(const ls::CkksParameter& param,
                                   Duo shape,
                                   Duo head_shape,
                                   uint32_t n_heads,
                                   uint32_t init_level,
                                   Array<double, 1>&& gamma);
    void prepare_weight() override;
    std::vector<ls::CkksPlaintextRingt> gamma_pt_;

    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& x);
    Array<double, 2> run_plaintext(const Array<double, 2>& x) const;

    ls::CkksPlaintextRingt
    generate_gamma_pt(ls::CkksContext& ctx, uint32_t mb, uint32_t ct_local, uint32_t g = 0) const;

private:
    uint32_t n_prepad_ = 0, total_cols_ = 0, m_prepad_ = 0;
    uint32_t H_prepad_ = 0, H_ = 0, m_ = 0, n_ = 0, packed_extent_ = 0;
    uint32_t n_slot_ = 0, segment_len_ = 0, c_ = 0, cts_per_mb_ = 0, n_mb_ = 0;
    Array<double, 1> gamma_vals_;

    uint32_t ct_index(uint32_t mb, uint32_t ct_local) const;
    uint32_t total_cts() const;
    std::vector<double> build_gamma_vec(uint32_t mb, uint32_t ct_local) const;
};

class ParUpperDiagonalPolyActRNPoly : public Layer {
public:
    ParUpperDiagonalPolyActRNPoly(const ls::CkksParameter& param,
                                  Duo shape,
                                  Duo head_shape,
                                  uint32_t n_heads,
                                  uint32_t init_level,
                                  Array<double, 2>&& coeffs,
                                  uint32_t degree);
    void prepare_weight() override;
    std::vector<ls::CkksPlaintextRingt> c2_pt_;
    std::vector<ls::CkksPlaintextRingt> c1_pt_;
    std::vector<ls::CkksPlaintextRingt> c0_add_pt_;
    std::vector<ls::CkksPlaintextRingt> c4_pt_;
    std::vector<ls::CkksPlaintextRingt> c3_pt_;

    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& x);
    Array<double, 2> run_plaintext(const Array<double, 2>& x) const;

    ls::CkksPlaintextRingt
    generate_coeff_pt(ls::CkksContext& ctx, uint32_t coeff_idx, uint32_t mb, uint32_t ct_local, uint32_t g = 0) const;

private:
    uint32_t n_prepad_ = 0, total_cols_ = 0, m_prepad_ = 0;
    uint32_t H_prepad_ = 0, H_ = 0, m_ = 0, n_ = 0, packed_extent_ = 0;
    uint32_t n_slot_ = 0, segment_len_ = 0, c_ = 0, cts_per_mb_ = 0, n_mb_ = 0;
    uint32_t degree_ = 0;
    Array<double, 2> coeffs_;

    uint32_t ct_index(uint32_t mb, uint32_t ct_local) const;
    uint32_t total_cts() const;
    std::vector<double> build_coeff_vec(uint32_t coeff_idx, uint32_t mb, uint32_t ct_local) const;
};

using ParUpperDiagonalPolyActGamma = ParUpperDiagonalPolyActRNGamma;
using ParUpperDiagonalPolyActPoly = ParUpperDiagonalPolyActRNPoly;
