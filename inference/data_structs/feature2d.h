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
 * MultChannelPack  (pack / unpack)
 * ----------------------------------------------------------------------------
 * Multiple channels are concatenated sequentially into each ciphertext.
 *
 *   n_channel_per_ct = n_slot / (shape[0] * shape[1])
 *   n_ct             = ceil(n_channel / n_channel_per_ct)
 *
 *   ct[ct_idx][ k * shape[0]*shape[1] + h * shape[1] + w ]
 *       = feature[ ct_idx*n_channel_per_ct + k, h, w ]
 *
 * ----------------------------------------------------------------------------
 * MultiplexedPack  (mult_pack / mult_unpack)
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
 * InterleavedDecompositionPack  (split_with_stride_pack / split_with_stride_unpack)
 * ----------------------------------------------------------------------------
 * The spatial grid is decomposed into stride[0]*stride[1] interleaved
 * sub-grids, each stored in a separate ciphertext. Used to implement
 * strided convolutions without data movement.
 *
 *   n_channel_per_ct = 1
 *   n_ct             = n_channel * stride[0] * stride[1]
 *
 *   For ciphertext i = channel_idx * stride[0]*stride[1] + seg_idx,
 *   where seg_idx = row_seg*stride[1] + col_seg:
 *
 *   ct[i][ block_row * block_shape[1] + block_col ]
 *       = feature[ channel_idx,
 *                  block_row * stride[0] + row_seg,
 *                  block_col * stride[1] + col_seg ]
 *
 */

#pragma once
#include <cstdint>
#include <vector>
#include "feature.h"
#include <iostream>

class Feature2DShare;
class Feature3DShare;

class Feature2DEncrypted : public FeatureEncrypted {
public:
    Duo shape;
    Duo skip;
    Duo invalid_fill;

    std::vector<std::vector<int>> segment_valid_range;
    Duo n_segment;
    std::vector<CkksCiphertext> data;
    std::vector<CkksCompressedCiphertext> data_compress;

    Feature2DEncrypted(CkksContext* context_in, int ct_level, Duo skip_in = {1, 1}, Duo invalid_fill_in = {1, 1});

    virtual vector<vector<double>>
    pack_feature(PackType& packtype, const Array<double, 3>& feature_mg, const Duo& block_shape, const Duo& stride);

    virtual void pack(const Array<double, 3>& feature_mg, bool is_symmetric = false, double scale_in = DEFAULT_SCALE);
    virtual void
    column_pack(const Array<double, 2>& feature_mg, bool is_symmetric = false, double scale_in = DEFAULT_SCALE);
    virtual void
    row_pack(const Array<double, 2>& feature_mg, bool is_symmetric = false, double scale_in = DEFAULT_SCALE);

    virtual void
    single_pack(const Array<double, 3>& feature_mg, bool is_sysmmetric = false, double scale_in = DEFAULT_SCALE);
    virtual Array<double, 3> single_unpack() const;

    virtual void
    mult_pack(const Array<double, 3>& feature_mg, bool is_sysmmetric = false, double scale_in = DEFAULT_SCALE);

    virtual void split_with_overlap_pack(const Array<double, 3>& feature_mg,
                                         const Duo& block_shape,
                                         const Duo& n_overlap,
                                         bool is_sysmmetric = false,
                                         double scale_in = DEFAULT_SCALE);
    virtual void split_with_stride_pack(const Array<double, 3>& feature_mg,
                                        const Duo& block_shape,
                                        const Duo& stride,
                                        bool is_sysmmetric = false,
                                        double scale_in = DEFAULT_SCALE);
    virtual void zero_inserted_mult_pack(const Array<double, 3>& feature_mg,
                                         const Duo stride,
                                         bool is_sysmmetric = false,
                                         double scale_in = DEFAULT_SCALE);
    virtual Array<double, 3> zero_inserted_mult_unpack(const Duo stride_next) const;
    virtual void
    par_mult_pack(const Array<double, 3>& feature_mg, bool is_sysmmetric = false, double scale_in = DEFAULT_SCALE);

    virtual Array<double, 3> par_mult_unpack() const;
    virtual Array<double, 3> mult_unpack() const;
    virtual Array<double, 3> split_with_overlap_unpack(const Duo& block_shape) const;
    virtual Array<double, 3> split_with_stride_unpack(const Duo& block_shape, const Duo& stride) const;
    Feature2DEncrypted refresh_ciphertext() const;
    virtual Array<double, 3> unpack() const;
    virtual Array<double, 2> unpack_column() const;
    virtual Array<double, 2> unpack_row() const;

    // Block column-major packing: each d*d block -> one ciphertext
    virtual void block_col_major_pack(const Array<double, 2>& matrix,
                                      uint32_t d,
                                      bool is_symmetric = false,
                                      double scale_in = DEFAULT_SCALE);
    virtual Array<double, 2> block_col_major_unpack(uint32_t m, uint32_t n, uint32_t d) const;

    // Parallel (interleaved) block column-major packing: interleave blocks from
    // multiple heads at the same block position into a single ciphertext.
    // matrix shape: m × (n_heads * cols_per_head), block_size d = head_dim.
    virtual void par_block_col_major_pack(const Array<double, 2>& matrix,
                                          uint32_t d,
                                          uint32_t n_heads,
                                          bool is_symmetric = false,
                                          double scale_in = DEFAULT_SCALE);
    virtual Array<double, 2>
    par_block_col_major_unpack(uint32_t m, uint32_t n_per_head, uint32_t d, uint32_t n_heads) const;

    void split_to_shares(Feature2DEncrypted* share0, Feature2DShare* share1) const;
    void split_to_shares_for_multi_channel_pack(Feature2DEncrypted* share0,
                                                Feature2DShare* share1,
                                                PackType pack_type_in = PackType::ParMultiplexedPack) const;
    Feature2DEncrypted combine_with_share(const Feature2DShare& share) const;
    Feature2DEncrypted
    combine_with_share_new_protocol(const Feature2DShare& share, const Feature2DEncrypted& f2d, const Bytes& b1) const;
    Feature2DEncrypted
    combine_with_share_new_protocol_for_multi_pack(const Feature2DShare& share,
                                                   const Feature2DEncrypted& f2d,
                                                   const Bytes& b1,
                                                   PackType pack_type = PackType::ParMultiplexedPack) const;
    void decrypt_to_share(Feature2DShare* share, PackType pack_type = PackType::SinglePack) const;
    Array<uint64_t, 1> encrypt_from_share(const Feature2DShare& share,
                                          int n_channel,
                                          const Duo& input_shape,
                                          PackType pack_type = PackType::SinglePack);
    void decompress();

    Bytes serialize() const;
    void deserialize(const Bytes& bytes) override;
    Feature2DEncrypted drop_level(int drop_level_num) const;
    Feature2DEncrypted copy() const;
};

class Feature2DShare : public FeatureShare {
public:
    Feature2DShare(uint64_t q, int s);

    Duo shape;
};

class Feature3DShare : public FeatureShare {
public:
    Feature3DShare(uint64_t q, int s);

    Duo shape;
};

inline void
set_shape(Feature2DEncrypted& f2d, uint32_t n_channel, uint32_t n_channel_per_ct, const Duo& shape, const Duo& skip) {
    f2d.n_channel = n_channel;
    f2d.shape[0] = shape[0];
    f2d.shape[1] = shape[1];
    f2d.skip[0] = skip[0];
    f2d.skip[1] = skip[1];
    f2d.n_channel_per_ct = n_channel_per_ct;
}
