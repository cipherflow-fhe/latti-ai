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
#include <stdio.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include "layer.h"
#include "util.h"
#include "data_structs/feature2d.h"

class ConcatLayer : public Layer {
public:
    ConcatLayer();
    Feature2DEncrypted run(ls::CkksContext& ctx, const Feature2DEncrypted& x1, const Feature2DEncrypted& x2);
    Feature2DEncrypted run_multiple_inputs(ls::CkksContext& ctx, const std::vector<Feature2DEncrypted>& inputs);
    Array<double, 3> concatenate_channels(const Array<double, 3>& x1, const Array<double, 3>& x2);
    Array<double, 3> concatenate_channels_multiple_inputs(const std::vector<Array<double, 3>>& inputs);
    // Plaintext concat for dim=1 features: Array<double, 2> shape (C, L); requires all inputs to share L.
    Array<double, 2> concatenate_channels_multiple_inputs_1d(const std::vector<Array<double, 2>>& inputs);
    // Plaintext concat for dim=0 features: flat (length = n_channel) vectors; just flat-appends.
    std::vector<double> concatenate_channels_multiple_inputs_0d(const std::vector<std::vector<double>>& inputs);

    void prepare_mask_data(const ls::CkksParameter& param,
                           const std::vector<uint32_t>& input_n_channels,
                           uint32_t n_channel_per_ct,
                           Duo shape,
                           Duo skip,
                           int level);

    // dim=0 variant that accepts per-input pack_num and slot stride. Needed when
    // concat inputs have different pack_num (e.g. backbone pack=2 and MLP pack=128).
    // For each global channel, marks exactly ONE slot at position
    //   (local_ch % pack_i) * skip_i
    // in the source CT, mirroring the Python call_multiple_inputs_mixed_pack logic.
    void prepare_mask_data_0d(const ls::CkksParameter& param,
                              const std::vector<uint32_t>& input_n_channels,
                              const std::vector<uint32_t>& input_packs,
                              const std::vector<uint32_t>& input_skip_scalars,
                              int level);

    // dim=1 variant: each channel occupies L_i contiguous-strided slots in its CT.
    // For each global channel, marks L_i slots at positions
    //   slot(l) = block_idx * (L_i * skip_i) + l * skip_i + sub_pos   for l in [0, L_i)
    // following the Feature1DEncrypted::pack_multiplexed layout (invalid_fill is
    // assumed to be 1 by default; can be extended if needed).
    void prepare_mask_data_1d(const ls::CkksParameter& param,
                              const std::vector<uint32_t>& input_n_channels,
                              const std::vector<uint32_t>& input_packs,
                              const std::vector<uint32_t>& input_lengths,
                              const std::vector<uint32_t>& input_skips,
                              const std::vector<uint32_t>& input_invalid_fills,
                              int level);

    std::vector<ls::CkksPlaintextRingt> mask_pt;

private:
    Feature2DEncrypted run_multiple_inputs_uneven(ls::CkksContext& ctx, const std::vector<Feature2DEncrypted>& inputs);
};
