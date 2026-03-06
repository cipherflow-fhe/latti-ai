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
#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <cxx_sdk_v2/cxx_fhe_task.h>
#include "../common.h"

using namespace std;
using namespace cxx_sdk_v2;

enum class PackType { MultChannelPack, SinglePack, MultiplexedPack, ParMultiplexedPack, InterleavedDecompositionPack };

enum class DecryptType { RESHAPE, SPARSE };
enum class ExecuteType { FPGA, SDK };

class FeatureEncrypted {
public:
    CkksContext* context = nullptr;
    uint32_t dim = 0;
    uint32_t n_channel = 0;
    uint32_t n_channel_per_ct = 0;
    uint32_t level = 0;
    uint32_t matmul_block_size = 0;
    double ckks_scale = 0;
    double multiplier = 0;

    FeatureEncrypted();
    virtual ~FeatureEncrypted();

    virtual void deserialize(const Bytes& bytes) {};
};

class Feature0DShare;
class Feature2DShare;

class Feature0DEncrypted : public FeatureEncrypted {
public:
    uint32_t pack_type = 0;
    uint32_t skip = 0;
    std::vector<CkksCiphertext> data;
    std::vector<CkksCompressedCiphertext> data_compressed;

    Feature0DEncrypted(CkksContext* context_in, int ct_level);
    void pack(const Array<double, 1>& feature_mg, bool is_symmetric = false, double scale_in = DEFAULT_SCALE);
    void pack_cyclic(const std::vector<double>& feature_mg, bool is_symmetric = false, double scale_in = DEFAULT_SCALE);
    void pack_skip(const Array<double, 1>& feature_mg, bool is_symmetric = false);
    Array<double, 1> unpack(DecryptType dec_type) const;

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

class FeatureShare {
public:
    FeatureShare(uint64_t q, int s);

    uint64_t ring_mod;
    int scale_ord;
    Array<uint64_t, 1> data;
};

class Feature0DShare : public FeatureShare {
public:
    Feature0DShare(uint64_t q, int s);
    void to_encrypted(Feature0DEncrypted* encrypted_share, Feature0DEncrypted* encrypted, int level);
    void encrypt_from_share(const Feature2DShare& share, int n_channel, const Duo& input_shape);
};

class Feature2DShare;
class Feature3DShare;

class Feature2DEncrypted : public FeatureEncrypted {
public:
    Duo shape;
    Duo skip;
    std::vector<std::vector<int>> segment_valid_range;
    Duo n_segment;
    std::vector<CkksCiphertext> data;
    std::vector<CkksCompressedCiphertext> data_compress;

    Feature2DEncrypted(CkksContext* context_in, int ct_level, Duo skip_in = {1, 1});

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

    // Block column-major packing: each k*k block -> one ciphertext
    virtual void block_col_major_pack(const Array<double, 2>& matrix,
                                      uint32_t k,
                                      bool is_symmetric = false,
                                      double scale_in = DEFAULT_SCALE);
    virtual Array<double, 2> block_col_major_unpack(uint32_t m, uint32_t n, uint32_t k) const;

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

int64_t gen_random_for_share(int r_bitlength);
double uint64_to_double(uint64_t input, double scale, uint64_t ring_mod);
uint64_t double_to_uint64(double input, double scale, uint64_t ring_mod);

inline void
set_shape(Feature2DEncrypted& f2d, uint32_t n_channel, uint32_t n_channel_per_ct, const Duo& shape, const Duo& skip) {
    f2d.n_channel = n_channel;
    f2d.shape[0] = shape[0];
    f2d.shape[1] = shape[1];
    f2d.skip[0] = skip[0];
    f2d.skip[1] = skip[1];
    f2d.n_channel_per_ct = n_channel_per_ct;
}

inline void set_shape_0D(Feature0DEncrypted& f0d, uint32_t n_channel, uint32_t n_channel_per_ct, uint32_t skip) {
    f0d.n_channel = n_channel;
    f0d.skip = skip;
    f0d.n_channel_per_ct = n_channel_per_ct;
}

void parallel_for(int n, int n_thread, CkksContext& context, const function<void(CkksContext&, int)>& fn);

void parallel_for(int n, int n_thread, CkksBtpContext& context, const function<void(CkksBtpContext&, int)>& fn);

class Feature1DEncrypted : public FeatureEncrypted {
public:
    Feature1DEncrypted(CkksContext* context_in, int ct_level);
    virtual void pack(Array<double, 2>& feature_mg, bool is_symmetric = false, double scale_in = DEFAULT_SCALE);
    virtual Array<double, 2> unpack() const;
    uint32_t shape = 0;
    uint32_t skip = 0;
    std::vector<CkksCiphertext> data;
    std::vector<CkksCompressedCiphertext> data_compress;
};

inline CkksCiphertext drop_level_to(const CkksCiphertext& x, CkksContext& ctx, int level_pre, int level_next) {
    int level_diff = level_pre - level_next;
    assert(level_diff > 0);
    CkksCiphertext res = ctx.drop_level(x);
    for (int i = 1; i < level_diff; i++) {
        res = ctx.drop_level(res);
    }
    return res;
}
