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
#include "../data_structs/feature.h"
#include "data_structs/constants.h"

#include <array>
#include <cstdint>
#include <vector>

class ParMultiplexedConv1DPackedLayer : public Layer {
public:
    ParMultiplexedConv1DPackedLayer(const ls::CkksParameter& param_in,
                                    uint32_t input_shape_in,
                                    const Array<double, 3>& weight_in,
                                    const Array<double, 1>& bias_in,
                                    uint32_t stride_in,
                                    uint32_t skip_in,
                                    uint32_t n_channel_per_ct_in,
                                    uint32_t level_in,
                                    double residual_scale = 1.0);
    void prepare_weight();
    void prepare_weight_for_lazy();

    // Helper functions to generate weights/bias on-demand (for lazy mode)
    ls::CkksPlaintextRingt
    generate_weight_pt_for_indices(ls::CkksContext& ctx, int wg, int w_idx, int kernel_idx) const;
    ls::CkksPlaintextRingt generate_bias_pt_for_index(ls::CkksContext& ctx, int idx) const;
    ls::CkksPlaintext generate_select_tensor_pt_for_index(ls::CkksContext& ctx, int t) const;

    Feature1DEncrypted run(ls::CkksContext& ctx, Feature1DEncrypted& x);
    virtual std::vector<double> select_tensor(int num) const;

    Array<double, 2> plaintext_call(const Array<double, 2>& x);

    std::vector<std::vector<std::vector<ls::CkksPlaintextRingt>>> weight_pt;
    std::vector<ls::CkksPlaintextRingt> bias_pt;
    std::vector<ls::CkksPlaintextRingt> block_select_pt;

    uint32_t input_shape;
    uint32_t skip;
    uint32_t stride;
    uint32_t kernel_shape;
    uint32_t n_channel_in;
    uint32_t n_channel_out;
    uint32_t n_channel_per_ct;
    double weight_scale;

    Array<double, 3> weight;
    Array<double, 1> bias;

private:
    std::vector<ls::CkksCiphertext> run_core(ls::CkksContext& ctx, std::vector<ls::CkksCiphertext>& x);

    uint32_t n_packed_in_channel;
    uint32_t n_packed_out_channel;
    uint32_t n_mult_pack_per_ct;

    uint32_t cached_input_block_size;
    std::vector<std::vector<double>> kernel_masks_;
};
