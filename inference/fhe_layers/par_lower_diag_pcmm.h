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

class ParLowerDiagPCMM : public Layer {
public:
    ParLowerDiagPCMM(const ls::CkksParameter& param_in,
                     const Duo& shape_X_T,
                     uint32_t n_heads,
                     uint32_t head_dim,
                     const Array<double, 2>& W_mat,
                     uint32_t level_X,
                     Array<double, 1>&& bias = Array<double, 1>());

    void prepare_weight() override;
    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& X_T);

private:
    enum class Mode { SQUARE, EXPAND, REDUCE };

    std::vector<ls::CkksCiphertext> run_core(ls::CkksContext& ctx,
                                             const std::vector<ls::CkksCiphertext>& input_cts,
                                             const std::vector<uint32_t>& mb_indices) const;

    Mode mode_;
    uint32_t K_ = 0;
    uint32_t K_row_ = 0;
    uint32_t K_col_ = 0;

    uint32_t H_prepad_ = 0;
    uint32_t H_ = 0;
    uint32_t m_ = 0;
    uint32_t n_prepad_ = 0;
    uint32_t n_ = 0;
    uint32_t d_prepad_ = 0;
    uint32_t d_ = 0;
    uint32_t n_slot_ = 0;
    uint32_t segment_len_ = 0;
    uint32_t c_ = 0;
    uint32_t m_c_ = 0;
    uint32_t in_rows_ = 0;
    uint32_t out_cols_ = 0;
    uint32_t W_T_rows_ = 0;
    uint32_t W_T_cols_ = 0;

    std::vector<Array<double, 2>> W_padded_;
    std::vector<std::vector<std::vector<ls::CkksPlaintextRingt>>> pt_A_;
    std::vector<ls::CkksPlaintextRingt> mask_wrap_pt_;

    bool has_bias_ = false;
    Array<double, 1> bias_vals_;
    std::vector<ls::CkksPlaintextRingt> bias_pt_;
};
