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
#include <array>
#include <vector>

class ParLowerDiagCCMM : public Layer {
public:
    ParLowerDiagCCMM(const ls::CkksParameter& param_in,
                     const Duo& shape_A,
                     const Duo& shape_B,
                     uint32_t n_heads,
                     uint32_t head_dim,
                     uint32_t level);

    void prepare_weight() override;
    std::vector<ls::CkksPlaintextRingt> replication_mask_pt_;
    std::vector<std::vector<ls::CkksPlaintextRingt>> ordinary_route_pt_;
    std::vector<std::vector<std::vector<ls::CkksPlaintextRingt>>> kqt_route_pt_;
    ls::CkksPlaintextRingt generate_replication_mask_pt(ls::CkksContext& ctx, uint32_t ell) const;
    ls::CkksPlaintextRingt generate_ordinary_route_pt(ls::CkksContext& ctx, uint32_t ell, uint32_t mask_idx) const;
    ls::CkksPlaintextRingt
    generate_kqt_route_pt(ls::CkksContext& ctx, uint32_t j, uint32_t ell, uint32_t mask_idx) const;
    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& A, const FeatureMatEncrypted& B);
    Array<double, 2> run_plaintext(const Array<double, 2>& A, const Array<double, 2>& B) const;

private:
    std::vector<ls::CkksCiphertext> run_core(ls::CkksContext& ctx,
                                             const std::vector<ls::CkksCiphertext>& A_cts,
                                             const std::vector<ls::CkksCiphertext>& B_cts) const;
    std::vector<ls::CkksCiphertext> run_core_kqt(ls::CkksContext& ctx,
                                                 const std::vector<ls::CkksCiphertext>& A_cts,
                                                 const std::vector<ls::CkksCiphertext>& B_cts) const;
    std::vector<ls::CkksCiphertext> run_core_ordinary(ls::CkksContext& ctx,
                                                      const std::vector<ls::CkksCiphertext>& A_cts,
                                                      const std::vector<ls::CkksCiphertext>& B_cts) const;

    std::vector<double> build_replication_mask(uint32_t ell) const;
    std::vector<double> build_ordinary_route_masks(uint32_t ell, uint32_t mask_idx) const;
    std::vector<double> build_kqt_route_masks(uint32_t j, uint32_t ell, uint32_t mask_idx) const;
    std::vector<double> rotate_plain(const std::vector<double>& values, int step) const;

    ls::CkksCiphertext
    replicate_lower_diag(ls::CkksContext& ctx, const std::vector<ls::CkksCiphertext>& B_cts, uint32_t ell) const;
    ls::CkksCiphertext multiply_cts(ls::CkksContext& ctx,
                                    const ls::CkksCiphertext& a_level_l,
                                    const ls::CkksCiphertext& b_level_l_minus_1) const;
    ls::CkksCiphertext apply_route_mask(ls::CkksContext& ctx,
                                        const ls::CkksCiphertext& product_level_l_minus_2,
                                        const ls::CkksPlaintextRingt& mask_pt) const;

    uint32_t expected_ct_count_for_head_shape(const Duo& head_shape) const;

    Duo shape_A_ = {0, 0};
    Duo shape_B_ = {0, 0};
    Duo output_shape_ = {0, 0};
    bool is_kqt_ = false;

    uint32_t H_prepad_ = 0;
    uint32_t H_ = 0;
    uint32_t m_ = 0;
    uint32_t n_prepad_ = 0;
    uint32_t n_ = 0;
    uint32_t n_slot_ = 0;
    uint32_t segment_len_ = 0;
    uint32_t c_ = 0;
    uint32_t m_c_ = 0;
    uint32_t n_c_ = 0;
};
