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

/*
 * Encoding Formats for Feature1DEncrypted
 * ========================================
 * Feature1DEncrypted supports two ciphertext packing layouts for a 2-D feature
 * tensor (n_channel × shape).
 *
 * ----------------------------------------------------------------------------
 * MultChannelPacking  (pack / unpack)
 * ----------------------------------------------------------------------------
 * Channels are packed sequentially. Within each channel the elements are stored
 * with stride skip (skip-1 zero slots between consecutive elements).
 *
 *   n_channel_per_ct = n_slot / (shape * skip)
 *   n_ct             = ceil(n_channel / n_channel_per_ct)
 *
 * Pack mapping — for channel k (0-indexed within a ciphertext), element i:
 *   slot = k * shape * skip + i * skip
 *
 * ----------------------------------------------------------------------------
 * MultiplexedPacking  (pack_multiplexed / unpack_multiplexed)
 * ----------------------------------------------------------------------------
 * skip channels are interleaved within each block. invalid_fill controls how
 * many slots per spatial position are reserved:
 *
 *   block_stride = skip * invalid_fill    (slots per spatial position)
 *   block_size   = shape * block_stride   (slots per block)
 *
 *   invalid_fill == 1 : every slot in a block is valid (pure interleaved).
 *   invalid_fill >  1 : the first skip slots per spatial position are valid;
 *                       the remaining skip*(invalid_fill-1) slots are zero.
 *
 *   n_channel_per_ct = n_slot / (shape * invalid_fill)
 *                    = n_slot / block_stride * skip   (must be a multiple of skip)
 *   n_block_per_ct   = n_channel_per_ct / skip
 *   n_ct             = ceil(n_channel / n_channel_per_ct)
 *
 * Pack mapping — for channel j (0-indexed within a ciphertext), element data_idx:
 *   block_idx = j / skip
 *   sub_pos   = j % skip
 *   slot      = block_idx * block_size + data_idx * block_stride + sub_pos
 */

#pragma once
#include <cstdint>
#include <vector>
#include "feature.h"

class Feature1DEncrypted : public FeatureEncrypted {
public:
    Feature1DEncrypted(ls::CkksContext* context_in, int ct_level, uint32_t skip_in = 1, uint32_t invalid_fill_in = 1);
    virtual void pack(Array<double, 2>& feature_mg, bool is_symmetric = false, double scale_in = DEFAULT_SCALE);
    virtual Array<double, 2> unpack() const;
    virtual void
    pack_multiplexed(const Array<double, 2>& feature_mg, bool is_symmetric = false, double scale_in = DEFAULT_SCALE);
    virtual Array<double, 2> unpack_multiplexed() const;
    uint32_t shape = 0;
    uint32_t skip = 0;
    uint32_t invalid_fill = 1;
    std::vector<ls::CkksCiphertext> data;
    std::vector<ls::CkksCompressedCiphertext> data_compress;

    Bytes serialize() const;
    void deserialize(const Bytes& bytes) override;
    Feature1DEncrypted copy() const;
};
