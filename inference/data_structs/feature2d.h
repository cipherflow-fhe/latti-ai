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
 * Encoding Formats for Feature2DEncrypted
 * ========================================
 * Feature2DEncrypted supports several ciphertext packing layouts. Each layout
 * defines how a 3-D feature tensor (n_channel × shape[0] × shape[1]) is mapped
 * to a flat vector of CKKS slots.
 *
 * ----------------------------------------------------------------------------
 * MultChannelPacking  (pack / unpack)
 * ----------------------------------------------------------------------------
 * Multiple channels are concatenated sequentially into each ciphertext,
 * stored in row-major order within each channel.
 *
 *   n_channel_per_ct = n_slot / (shape[0] * shape[1])
 *   n_ct             = ceil(n_channel / n_channel_per_ct)
 *
 * Pack mapping — for all channel_idx in [0, n_channel),
 *                x in [0, shape[0]), y in [0, shape[1]):
 *
 *   ct_idx           = channel_idx / n_channel_per_ct
 *   channel_idx_in_ct = channel_idx % n_channel_per_ct
 *
 *   ct[ct_idx][ channel_idx_in_ct * shape[0]*shape[1] + x * shape[1] + y ]
 *       = feature[channel_idx, x, y]
 *
 * ----------------------------------------------------------------------------
 * MultiplexedPacking  (pack_multiplexed / unpack_multiplexed)
 * ----------------------------------------------------------------------------
 * Channels are packed with a spatial multiplexed pattern that enables
 * downsampling and upsampling operations with few rotations and minimal
 * wasted slots.
 *
 * invalid_fill[d] controls the slot spacing along spatial dimension d:
 *   - skip[d] consecutive valid slots are followed by
 *     skip[d] * (invalid_fill[d] - 1) invalid slots.
 *   - Example: skip[0]=4, invalid_fill[0]=2 → 4 valid, then 4 invalid.
 *
 * Structural layout (within one ciphertext):
 *   mini-block — a 2-D region of size skip[0] × skip[1]. The top-left
 *                sub-region of size (skip[0]/invalid_fill[0]) ×
 *                (skip[1]/invalid_fill[1]) holds valid channel values;
 *                remaining slots are invalid.
 *   block      — a 2-D region of size (shape[0]*skip[0]) × (shape[1]*skip[1]),
 *                composed of a shape[0] × shape[1] array of mini-blocks,
 *                one per spatial position (x, y).
 *   ciphertext — n_block_per_ct blocks concatenated sequentially.
 *
 *   n_channel_per_ct      = n_slot / prod(shape) / prod(invalid_fill)
 *   n_channel_per_block   = prod(skip) / prod(invalid_fill)
 *   n_block_per_ct        = n_channel_per_ct / n_channel_per_block
 *   n_channel_per_block_col = skip[1] / invalid_fill[1]
 *   n_ct                  = ceil(n_channel / n_channel_per_ct)
 *
 * Pack mapping — for all channel_idx in [0, n_channel),
 *                x in [0, shape[0]), y in [0, shape[1]):
 *
 *   ct_idx               = channel_idx / n_channel_per_ct
 *   channel_idx_in_ct    = channel_idx % n_channel_per_ct
 *   block_idx            = channel_idx_in_ct / n_channel_per_block
 *   channel_idx_in_block = channel_idx_in_ct % n_channel_per_block
 *   channel_x_offset     = channel_idx_in_block / n_channel_per_block_col
 *   channel_y_offset     = channel_idx_in_block % n_channel_per_block_col
 *
 *   x_in_block = x * skip[0] + channel_x_offset
 *   y_in_block = y * skip[1] + channel_y_offset
 *
 *   ct[ct_idx][ block_idx * (shape[0]*skip[0]) * (shape[1]*skip[1])
 *             + x_in_block * (shape[1]*skip[1]) + y_in_block ]
 *       = feature[channel_idx, x, y]
 *
 * ----------------------------------------------------------------------------
 * InterleavedPacking  (pack_interleaved / unpack_interleaved)
 * ----------------------------------------------------------------------------
 * Each channel's 2-D feature map is decomposed into stride[0]*stride[1]
 * interleaved sub-grids, one per ciphertext. Sub-grid (grid_x, grid_y)
 * contains the elements at positions (x, y) where x%stride[0]==grid_x and
 * y%stride[1]==grid_y, stored in row-major order over the reduced grid.
 *
 *   n_ct             = n_channel * stride[0] * stride[1]
 *
 * Pack mapping — for all channel_idx in [0, n_channel),
 *                x in [0, shape[0]), y in [0, shape[1]):
 *
 *   grid_idx    = (x % stride[0]) * stride[1] + (y % stride[1])
 *   ct_idx      = channel_idx * stride[0] * stride[1] + grid_idx
 *   x_in_ct     = x / stride[0]
 *   y_in_ct     = y / stride[1]
 *
 *   ct[ct_idx][ x_in_ct * (shape[1]/stride[1]) + y_in_ct ]
 *       = feature[channel_idx, x, y]
 *
 */

#pragma once
#include <cstdint>
#include <vector>
#include "feature.h"
#include <iostream>

class Feature2DShare;

class Feature2DEncrypted : public FeatureEncrypted {
public:
    Duo shape;
    Duo skip;
    Duo invalid_fill;
    PackType packing_type;

    std::vector<ls::CkksCiphertext> data;
    std::vector<ls::CkksCompressedCiphertext> data_compress;

    Feature2DEncrypted(ls::CkksContext* context_in,
                       int ct_level,
                       const Duo& skip_in = {1, 1},
                       const Duo& invalid_fill_in = {1, 1},
                       PackType packing_type_in = PackType::MultiplexedPacking);

    virtual std::vector<ls::CkksPlaintext> encode_multiple_channel(const Array<double, 3>& feature_mg,
                                                                   double scale_in = DEFAULT_SCALE);
    virtual std::vector<ls::CkksPlaintext> encode_interleaved(const Array<double, 3>& feature_mg,
                                                              const Duo& block_shape,
                                                              const Duo& stride,
                                                              double scale_in = DEFAULT_SCALE);
    virtual std::vector<ls::CkksPlaintext> encode_multiplexed(const Array<double, 3>& feature_mg,
                                                              double scale_in = DEFAULT_SCALE);

    virtual void
    column_pack(const Array<double, 2>& feature_mg, bool is_symmetric = false, double scale_in = DEFAULT_SCALE);
    virtual void
    row_pack(const Array<double, 2>& feature_mg, bool is_symmetric = false, double scale_in = DEFAULT_SCALE);
    virtual void pack_multiple_channel(const Array<double, 3>& feature_mg,
                                       bool is_symmetric = false,
                                       double scale_in = DEFAULT_SCALE);
    virtual void
    pack_multiplexed(const Array<double, 3>& feature_mg, bool is_symmetric = false, double scale_in = DEFAULT_SCALE);
    virtual void pack_interleaved(const Array<double, 3>& feature_mg,
                                  const Duo& block_shape,
                                  const Duo& stride,
                                  bool is_symmetric = false,
                                  double scale_in = DEFAULT_SCALE);

    virtual Array<double, 3> unpack_multiplexed() const;
    virtual Array<double, 3> unpack_interleaved(const Duo& block_shape, const Duo& stride) const;
    Feature2DEncrypted refresh_ciphertext() const;
    virtual Array<double, 3> unpack_multiple_channel() const;
    virtual Array<double, 2> unpack_column() const;
    virtual Array<double, 2> unpack_row() const;

    void split_to_shares(Feature2DEncrypted* share0, Feature2DShare* share1) const;
    void split_to_shares_for_multi_channel_pack(Feature2DEncrypted* share0, Feature2DShare* share1) const;
    Feature2DEncrypted combine_with_share(const Feature2DShare& share) const;
    Feature2DEncrypted
    combine_with_share_new_protocol(const Feature2DShare& share, const Feature2DEncrypted& f2d, const Bytes& b1) const;
    Feature2DEncrypted combine_with_share_new_protocol_for_multi_pack(const Feature2DShare& share,
                                                                      const Feature2DEncrypted& f2d,
                                                                      const Bytes& b1) const;
    void decrypt_to_share(Feature2DShare* share, PackType pack_type = PackType::MultiplexedPacking) const;
    Array<uint64_t, 1> encrypt_from_share(const Feature2DShare& share,
                                          int n_channel,
                                          const Duo& input_shape,
                                          PackType pack_type = PackType::MultiplexedPacking);
    void decompress();

    Bytes serialize() const;
    void deserialize(const Bytes& bytes) override;
    Feature2DEncrypted drop_level(int drop_level_num) const;
    Feature2DEncrypted copy() const;
};

class Feature2DShare : public FeatureShare {
public:
    Feature2DShare(uint64_t q, int s);

    Duo shape = {0, 0};
};

inline void
set_shape(Feature2DEncrypted& f2d, uint32_t n_channel, uint32_t n_channel_per_ct, const Duo& shape, const Duo& skip) {
    f2d.n_channel = n_channel;
    f2d.shape = shape;
    f2d.skip = skip;
    f2d.n_channel_per_ct = n_channel_per_ct;
}
