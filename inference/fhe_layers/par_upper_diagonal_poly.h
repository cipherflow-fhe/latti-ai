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

#include "mat_poly_base.h"
#include "../data_structs/feature_mat.h"
#include <vector>

// Polynomial evaluation for ciphertexts packed by
// FeatureMatEncrypted::par_diagonal_pack(..., is_lower=false, is_transposed=false).
// Coefficients are column-wise: coeffs[k, j] is the coefficient of x^k for column j.
class ParUpperDiagonalPoly : public MatPolyBase {
public:
    ParUpperDiagonalPoly(const ls::CkksParameter& param,
                         Duo shape,
                         Duo head_shape,
                         uint32_t n_heads,
                         uint32_t init_level,
                         Array<double, 2>&& coeffs,
                         uint32_t order);

    void prepare_weight() override;
    void prepare_weight_lazy() override;

    ls::CkksPlaintextRingt generate_weight_pt_for_stockmeyer(ls::CkksContext& ctx, int coeff_idx, int ct_idx) const;

    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& x);
    FeatureMatEncrypted run_stockmeyer(ls::CkksContext& ctx, const FeatureMatEncrypted& x);
    std::vector<ls::CkksCiphertext> run_core(ls::CkksContext& ctx, const std::vector<ls::CkksCiphertext>& x);
    std::vector<ls::CkksCiphertext> run_core_stockmeyer(ls::CkksContext& ctx, const std::vector<ls::CkksCiphertext>& x);
    void prepare_weight_stockmeyer();
    void prepare_weight_stockmeyer_lazy();
    Array<double, 2> run_plaintext(const Array<double, 2>& x) const;

private:
    uint32_t n_prepad_ = 0, total_cols_ = 0, m_prepad_ = 0;
    uint32_t H_prepad_ = 0, H_ = 0, m_ = 0, n_ = 0, packed_extent_ = 0;
    uint32_t n_slot_ = 0, segment_len_ = 0, c_ = 0, cts_per_mb_ = 0, n_mb_ = 0;
    std::vector<std::vector<ls::CkksPlaintextRingt>> stockmeyer_weight_pt_;

    uint32_t ct_index(uint32_t mb, uint32_t ct_local) const;
    uint32_t total_cts() const;
    std::vector<double> build_coeff_vec(uint32_t coeff_idx, uint32_t mb, uint32_t ct_local) const;
};
