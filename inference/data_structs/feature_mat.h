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
#include <cstdint>
#include <vector>
#include "feature.h"

class FeatureMatEncrypted : public FeatureEncrypted {
public:
    Duo shape = {0, 0};  // {rows (L), cols (C)}
    uint32_t matmul_block_size = 0;
    std::vector<ls::CkksCiphertext> data;
    std::vector<ls::CkksCompressedCiphertext> data_compress;

    FeatureMatEncrypted(ls::CkksContext* context_in, int ct_level);

    // Block column-major packing: each d*d block -> one ciphertext
    void block_col_major_pack(const Array<double, 2>& matrix,
                              uint32_t d,
                              bool is_symmetric = false,
                              double scale_in = DEFAULT_SCALE);
    Array<double, 2> block_col_major_unpack(uint32_t m, uint32_t n, uint32_t d) const;

    // Parallel (interleaved) block column-major packing: interleave blocks from
    // multiple heads at the same block position into a single ciphertext.
    // head_dim: columns per head within each megablock.
    void par_block_col_major_pack(const Array<double, 2>& matrix,
                                  uint32_t d,
                                  uint32_t n_heads,
                                  uint32_t head_dim,
                                  bool is_symmetric = false,
                                  double scale_in = DEFAULT_SCALE);
    // Unpack: K_col is inferred from data.size(). n_per_head = head_dim within each megablock.
    // total_cols is taken from shape[1].
    Array<double, 2> par_block_col_major_unpack(uint32_t m, uint32_t n_per_head, uint32_t d, uint32_t n_heads) const;

    void decompress();

    Bytes serialize() const override;
    void deserialize(const Bytes& bytes) override;
    FeatureMatEncrypted drop_level(int n_level_to_drop) const;
};
