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

class ParLowerDiagTranspose : public Layer {
public:
    ParLowerDiagTranspose(const ls::CkksParameter& param_in,
                          const Duo& shape,
                          uint32_t n_heads,
                          uint32_t head_dim,
                          uint32_t level);

    void prepare_weight() override;
    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& input);

private:
    std::vector<ls::CkksCiphertext> run_core(ls::CkksContext& ctx,
                                             const std::vector<ls::CkksCiphertext>& input_cts) const;

    std::vector<double> build_transpose_mask(uint32_t out_diag_idx, uint32_t mask_idx) const;
    ls::CkksCiphertext apply_mask(ls::CkksContext& ctx,
                                  const ls::CkksCiphertext& input_level_l,
                                  const ls::CkksPlaintextRingt& mask_pt) const;

    uint32_t expected_ct_count() const;

    Duo shape_ = {0, 0};
    uint32_t H_prepad_ = 0;
    uint32_t H_ = 0;
    uint32_t m_ = 0;
    uint32_t n_prepad_ = 0;
    uint32_t n_ = 0;
    uint32_t n_slot_ = 0;
    uint32_t segment_len_ = 0;
    uint32_t c_ = 0;
    uint32_t m_c_ = 0;

    std::vector<std::vector<ls::CkksPlaintextRingt>> transpose_mask_pt_;
};
