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
#include "feature.h"

class Feature0DShare;
class Feature2DShare;

class Feature0DEncrypted : public FeatureEncrypted {
public:
    uint32_t pack_type = 0;
    uint32_t skip = 0;
    std::vector<ls::CkksCiphertext> data;
    std::vector<ls::CkksCompressedCiphertext> data_compressed;

    Feature0DEncrypted(ls::CkksContext* context_in, int ct_level);
    void pack(const Array<double, 1>& feature_mg,
              bool is_symmetric = false,
              double scale_in = DEFAULT_SCALE,
              uint32_t skip_in = 1);
    void pack_cyclic(const std::vector<double>& feature_mg, bool is_symmetric = false, double scale_in = DEFAULT_SCALE);
    Array<double, 1> unpack() const;

    void to_share(Feature0DEncrypted* share0, Feature0DShare* share1) const;
    Array<uint64_t, 1> encrypt_from_share(const Feature0DShare& share, int n_channel);
    void split_to_shares(Feature0DEncrypted* share0, Feature0DShare* share1) const;
    void split_to_shares_reshape(Feature0DEncrypted* share0, Feature0DShare* share1) const;
    Bytes serialize() const;
    void deserialize(const Bytes& bytes) override;
    void decompress();
    Feature0DEncrypted combine_with_share(const Feature0DShare& share) const;
    Feature0DEncrypted
    combine_with_share_new_protocol(const Feature0DShare& share, const Feature0DEncrypted& f2d, const Bytes& b1) const;
    void decrypt_to_share(Feature0DShare* share, int n_channel);
    Feature0DEncrypted refresh_ciphertext() const;
    Feature0DEncrypted drop_level(int n_level_to_drop) const;
    Feature0DEncrypted copy() const;
};

class Feature0DShare : public FeatureShare {
public:
    Feature0DShare(uint64_t q, int s);
    void to_encrypted(Feature0DEncrypted* encrypted_share, Feature0DEncrypted* encrypted, int level);
    void encrypt_from_share(const Feature2DShare& share, int n_channel, const Duo& input_shape);
};

inline void set_shape_0D(Feature0DEncrypted& f0d, uint32_t n_channel, uint32_t n_channel_per_ct, uint32_t skip) {
    f0d.n_channel = n_channel;
    f0d.skip = skip;
    f0d.n_channel_per_ct = n_channel_per_ct;
}
