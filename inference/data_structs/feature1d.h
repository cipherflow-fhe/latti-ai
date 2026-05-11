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
 *   n_channel_per_ct = n_slot / shape
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
 *   block_stride = skip                    (slots per spatial position, no invalid_fill)
 *   block_size   = shape * block_stride   (slots per block = shape * skip)
 *
 *   invalid_fill == 1 : every sub_pos in a block carries a channel (pure interleaved).
 *   invalid_fill >  1 : only sub_pos in [0, skip/invalid_fill) carry channels;
 *                       sub_pos in [skip/invalid_fill, skip) are left as zero.
 *
 *   n_channel_per_ct = n_slot / shape
 *                    = n_block_per_ct * skip   (must be a multiple of skip)
 *   n_block_per_ct   = n_slot / (shape * skip)
 *   n_valid_per_ct   = n_block_per_ct * (skip / invalid_fill)   (channels with actual data)
 *   n_ct             = ceil(n_channel / n_valid_per_ct)
 *
 * Pack mapping — for valid channel j (0-indexed within a ciphertext), element data_idx:
 *   valid_sub        = skip / invalid_fill
 *   block_idx        = j / valid_sub
 *   sub_pos          = j % valid_sub          (in [0, valid_sub))
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

    Bytes serialize() const override;
    void deserialize(const Bytes& bytes) override;
    Feature1DEncrypted copy() const;
    Feature1DEncrypted drop_level(int n_level_to_drop) const;
};
