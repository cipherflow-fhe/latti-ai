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

#include "feature.h"
#include "util.h"
#include <sstream>

using namespace std;
using namespace cxx_sdk_v2;

int32_t bitlength = RING_MOD_BIT;

class Feature2DShare;
Bytes save_ct(const CkksCiphertext& ct, const CkksParameter& param_h) {
    auto vec = ct.serialize(param_h);
    return vec;
}
CkksCiphertext load_ct(Bytes& vec) {
    auto y_ct = CkksCiphertext::deserialize(vec);
    return y_ct;
}

int64_t gen_random_for_share(int r_bitlength) {
    const int rand_bitlength = 16;
    int n_rand = div_ceil(r_bitlength, rand_bitlength);
    uint64_t mask = (1 << (r_bitlength % rand_bitlength)) - 1;
    int64_t result = rand() & mask;
    mask = (1 << rand_bitlength) - 1;
    for (int i = 1; i < n_rand; i++) {
        result = (result << rand_bitlength) + (rand() & mask);
    }
    if (rand() % 2 == 1) {
        result = -result;
    }
    return result;
}

void parallel_for(int n, int n_thread, CkksContext& context, const function<void(CkksContext&, int)>& fn) {
    int n_group = div_ceil(n, n_thread);
    context.resize_copies(n_thread);
#pragma omp parallel for num_threads(n_thread)
    for (int thread_idx = 0; thread_idx < n_thread; thread_idx++) {
        CkksContext& context_copy = context.get_copy(thread_idx);
        for (int group_idx = 0; group_idx < n_group; group_idx++) {
            int i = group_idx * n_thread + thread_idx;
            if (i >= n) {
                continue;
            }
            fn(context_copy, i);
        }
    }
}

void parallel_for(int n, int n_thread, CkksBtpContext& context, const function<void(CkksBtpContext&, int)>& fn) {
    int n_group = div_ceil(n, n_thread);
    context.resize_copies(n_thread);
#pragma omp parallel for num_threads(n_thread)
    for (int thread_idx = 0; thread_idx < n_thread; thread_idx++) {
        CkksBtpContext& context_copy = context.get_copy(thread_idx);
        for (int group_idx = 0; group_idx < n_group; group_idx++) {
            int i = group_idx * n_thread + thread_idx;
            if (i >= n) {
                continue;
            }
            fn(context_copy, i);
        }
    }
}

void parallel_for_with_extra_level_context(int n,
                                           int n_thread,
                                           CkksContext& context,
                                           const function<void(CkksContext&, CkksContext&, int)>& fn) {
    int n_group = div_ceil(n, n_thread);
    context.resize_copies(n_thread);
    CkksContext& extra_level_context = context.get_extra_level_context();
    extra_level_context.resize_copies(n_thread);
#pragma omp parallel for num_threads(n_thread)
    for (int thread_idx = 0; thread_idx < n_thread; thread_idx++) {
        CkksContext& context_copy = context.get_copy(thread_idx);
        CkksContext& extra_level_context_copy = extra_level_context.get_copy(thread_idx);
        for (int group_idx = 0; group_idx < n_group; group_idx++) {
            int i = group_idx * n_thread + thread_idx;
            if (i >= n) {
                continue;
            }
            fn(context_copy, extra_level_context_copy, i);
        }
    }
}

FeatureEncrypted::FeatureEncrypted() : ckks_scale{DEFAULT_SCALE}, multiplier{1.0} {}

FeatureEncrypted::~FeatureEncrypted() {}

FeatureShare::FeatureShare(uint64_t q, int s) {
    ring_mod = q;
    scale_ord = s;
}

Feature0DShare::Feature0DShare(uint64_t q, int s) : FeatureShare{q, s} {}

void Feature0DEncrypted::to_share(Feature0DEncrypted* share0, Feature0DShare* share1) const {
    int n_slot = context->get_parameter().get_n() / 2;
    share1->data.resize({data.size() * n_slot});
    share0->n_channel = n_channel;
    share0->n_channel_per_ct = n_channel_per_ct;
    share0->skip = skip;
    share0->data.clear();
    double share_scale = pow(2, share1->scale_ord);

    for (uint32_t i = 0; i < data.size(); i++) {
        std::vector<uint64_t> mask_i(n_slot);
        std::vector<double> mask_d(n_slot);
        for (int j = 0; j < n_slot; j++) {
            mask_i[j] = gen_random_for_share(40);
            mask_d[j] = mask_i[j] / share_scale;
            share1->data[i * n_slot + j] = double_to_uint64(-mask_d[j], share_scale, share1->ring_mod);
        }
        auto mask_pt = context->encode(mask_d, level, data[i].get_scale());
        auto share0_c = context->add_plain(data[i], mask_pt);
        share0->data.push_back(move(share0_c));
    }
}

Bytes Feature0DEncrypted::serialize() const {
    stringstream ss;
    ss_write(ss, dim);
    ss_write(ss, n_channel);
    ss_write(ss, n_channel_per_ct);
    ss_write(ss, level);
    for (int i = 0; i < 2; i++) {
        ss_write(ss, skip);
    }
    uint32_t n_ct = data.size();
    ss_write(ss, n_ct);
    for (const CkksCiphertext& ct : data) {
        Bytes ct_data = ct.serialize(context->get_parameter());
        ss_write_vector(ss, ct_data);
    }
    uint32_t n_cct = data_compressed.size();
    ss_write(ss, n_cct);
    for (const CkksCompressedCiphertext& cct : data_compressed) {
        Bytes cct_data = cct.serialize(context->get_parameter());
        ss_write_vector(ss, cct_data);
    }

    Bytes bytes = ss_to_bytes(ss);
    return bytes;
}

void Feature0DEncrypted::deserialize(const Bytes& bytes) {
    stringstream ss;
    bytes_to_ss(bytes, ss);
    ss_read(ss, &dim);
    ss_read(ss, &n_channel);
    ss_read(ss, &n_channel_per_ct);
    ss_read(ss, &level);
    for (int i = 0; i < 2; i++) {
        ss_read(ss, &skip);
    }
    uint32_t n_ct;
    ss_read(ss, &n_ct);
    for (int i = 0; i < n_ct; i++) {
        Bytes ct_data;
        ss_read_vector(ss, &ct_data);
        auto y_ct = CkksCiphertext::deserialize(ct_data);
        data.push_back(move(y_ct));
    }
    uint32_t n_cct;
    ss_read(ss, &n_cct);
    for (int i = 0; i < n_cct; i++) {
        Bytes cct_data;
        ss_read_vector(ss, &cct_data);
        auto y_ct = CkksCompressedCiphertext::deserialize(cct_data);
        data_compressed.push_back(move(y_ct));
    }
}

void Feature0DShare::to_encrypted(Feature0DEncrypted* encrypted_share, Feature0DEncrypted* encrypted, int level) {
    int n_slot = encrypted_share->context->get_parameter().get_n() / 2;
    int n_ct = data.get_size() / n_slot;
    encrypted->data.clear();
    encrypted->n_channel = encrypted_share->n_channel;
    encrypted->n_channel_per_ct = encrypted_share->n_channel_per_ct;
    encrypted->skip = 1;
    double scale = pow(2, scale_ord);

    for (int i = 0; i < n_ct; i++) {
        std::vector<double> mask_d(n_slot);
        for (int j = 0; j < n_slot; j++) {
            mask_d[j] = uint64_to_double(data[i * n_slot + j], scale, ring_mod);
        }
        auto mask_pt = encrypted_share->context->encode(mask_d, level,
                                                        encrypted_share->context->get_parameter().get_default_scale());
        encrypted->data.push_back(encrypted_share->context->add_plain(encrypted_share->data[i], mask_pt));
    }
}

Array<uint64_t, 1> Feature0DEncrypted::encrypt_from_share(const Feature0DShare& share, int n_channel) {
    int n_slot = context->get_parameter().get_n() / 2;
    uint32_t skip = 1;
    this->skip = skip;

    Array<double, 1> out_data_mg(share.data.get_shape());
    Array<uint64_t, 1> data_add(share.data.get_shape());
    double scale = ENC_TO_SHARE_SCALE;
    for (int i = 0; i < share.data.get_size(); i++) {
        uint64_t data_add_value = (share.data[i] + (share.ring_mod / 2)) % share.ring_mod;
        data_add.set(i, data_add_value);
        double out_data_value = double(int64_t(data_add_value) - int64_t(share.ring_mod / 2)) / scale;
        out_data_mg.set(i, out_data_value);
    }

    double encode_scale = pow(2, DEFAULT_SCALE_BIT);
    this->pack_cyclic(out_data_mg.to_array_1d(), true, encode_scale);
    this->n_channel = n_channel;
    this->n_channel_per_ct = n_slot;
    return data_add;
}

void Feature0DEncrypted::decrypt_to_share(Feature0DShare* share, int n_channel) {
    Array<double, 1> x_double_vec = this->unpack(DecryptType::SPARSE);
    share->data = array_double_to_uint64(x_double_vec, share->scale_ord, share->ring_mod);
}

void Feature0DEncrypted::decompress() {
    assert(data.size() == 0);
    assert(data_compressed.size() > 0);
    size_t n_ct = data_compressed.size();
    for (int i = 0; i < n_ct; i++) {
        data.push_back(context->compressed_ciphertext_to_ciphertext(data_compressed[i]));
    }
    data_compressed.clear();
}

Feature0DEncrypted Feature0DEncrypted::combine_with_share(const Feature0DShare& share) const {
    int n_slot = context->get_parameter().get_n() / 2;
    Feature0DEncrypted result(this->context, this->level);
    result.n_channel = this->n_channel;
    result.n_channel_per_ct = this->n_channel_per_ct;
    result.skip = this->skip;
    double scale = pow(2, share.scale_ord);

    for (int i = 0; i < this->data.size(); i++) {
        vector<double> mask_d(n_slot, 0.0);
        for (int j = 0; j < n_slot; j++) {
            if (i * n_slot + j >= share.data.get_size()) {
                mask_d[j] =
                    uint64_to_double(share.data.get((i * n_slot + j) % share.data.get_size()), scale, share.ring_mod);
            } else {
                mask_d[j] = uint64_to_double(share.data.get(i * n_slot + j), scale, share.ring_mod);
            }
        }
        CkksPlaintext mask_pt = context->encode(mask_d, level, context->get_parameter().get_default_scale());
        result.data.push_back(context->add_plain(data[i], mask_pt));
    }
    return result;
}

Feature0DEncrypted Feature0DEncrypted::combine_with_share_new_protocol(const Feature0DShare& share,
                                                                       const Feature0DEncrypted& f2d,
                                                                       const Bytes& b1) const {
    int n_slot = context->get_parameter().get_n() / 2;
    Feature0DEncrypted result(this->context, this->level);
    result.n_channel = this->n_channel;
    result.n_channel_per_ct = this->n_channel_per_ct;
    result.skip = this->skip;
    double scale = ENC_TO_SHARE_SCALE;
    double encode_scale = pow(2, DEFAULT_SCALE_BIT);

    for (int i = 0; i < this->data.size(); i++) {
        vector<double> b1_value(n_slot, 0);
        vector<double> mask_d(n_slot, 0.0);
        for (int j = 0; j < n_slot; j++) {
            int64_t mask_value;
            if (i * n_slot + j >= share.data.get_size()) {
                b1_value[j] = b1[(i * n_slot + j) % share.data.get_size()];
                mask_value = int64_t(share.data.get((i * n_slot + j) % share.data.get_size())) -
                             int64_t(b1_value[j] * share.ring_mod);
            } else {
                b1_value[j] = b1[i * n_slot + j];
                mask_value = int64_t(share.data.get(i * n_slot + j)) - int64_t(b1[i * n_slot + j] * share.ring_mod);
            }
            b1_value[j] = 2 * b1_value[j] - 1;
            mask_d[j] = double(mask_value) / scale;
        }
        CkksPlaintext mask_pt = context->encode(mask_d, level, encode_scale);
        result.data.push_back(context->add_plain(data[i], mask_pt));

        CkksContext& ctx_extra = context->get_extra_level_context();
        CkksPlaintext b1_pt = ctx_extra.encode(b1_value, level + 1, ctx_extra.get_parameter().get_q(level + 1));
        auto f2d_mult = ctx_extra.mult_plain(f2d.data[i], b1_pt);
        f2d_mult = ctx_extra.rescale(f2d_mult, encode_scale);

        result.data[i] = context->add(result.data[i], f2d_mult);
    }
    return result;
}

void Feature0DEncrypted::split_to_shares(Feature0DEncrypted* share0, Feature0DShare* share1) const {
    int n_slot = context->get_parameter().get_n() / 2;
    double share_scale = pow(2, share1->scale_ord);
    int r_bitlength = 40;
    int feature_bitlength = ENC_TO_SHARE_SCALE_BIT + 1;
    int sigma = SIGMA;
    share0->n_channel = n_channel;
    share0->n_channel_per_ct = n_channel_per_ct;
    share0->skip = skip;
    share0->level = level;
    share0->data.clear();
    vector<vector<double>> mask_d_mat;
    vector<vector<int64_t>> r_mat;
    for (int i = 0; i < data.size(); i++) {
        vector<double> mask_d(n_slot);
        vector<int64_t> r(n_slot);
        for (int j = 0; j < n_slot; j++) {
            r[j] =
                int64_t(gen_random_uint(feature_bitlength + sigma)) - int64_t(1ull << (feature_bitlength + sigma - 1));
            mask_d[j] = double(r[j]) / share_scale;
        }
        mask_d_mat.push_back(mask_d);
        r_mat.push_back(r);
        CkksPlaintext mask_pt = context->encode(mask_d, level, ENC_TO_SHARE_SCALE);
        CkksCiphertext share0_ct = context->add_plain(data[i], mask_pt);

        share0->data.push_back(move(share0_ct));
    }
    double scale = pow(2, share1->scale_ord);
    share1->data.resize({n_channel});
    int T = 0;
    for (int i = 0; i < mask_d_mat.size(); i++) {
        for (int j = 0; j < n_channel_per_ct; j++) {
            if (T >= n_channel) {
                break;
            }
            uint64_t neg_r = (-r_mat[i][j * skip] % share1->ring_mod + share1->ring_mod) % share1->ring_mod;
            share1->data.set(T, neg_r);
            T = T + 1;
        }
    }
}

void Feature0DEncrypted::split_to_shares_reshape(Feature0DEncrypted* share0, Feature0DShare* share1) const {
    int n_slot = context->get_parameter().get_n() / 2;
    double share_scale = pow(2, share1->scale_ord);
    int r_bitlength = 40;
    int feature_bitlength = ENC_TO_SHARE_SCALE_BIT + 1;
    int sigma = SIGMA;
    share0->n_channel = n_channel;
    share0->n_channel_per_ct = n_channel_per_ct;
    share0->skip = skip;
    share0->level = level;
    share0->data.clear();
    vector<vector<double>> mask_d_mat;
    vector<vector<int64_t>> r_mat;
    for (int i = 0; i < data.size(); i++) {
        vector<double> mask_d(n_slot);
        vector<int64_t> r(n_slot);
        for (int j = 0; j < n_slot; j++) {
            r[j] =
                int64_t(gen_random_uint(feature_bitlength + sigma)) - int64_t(1ull << (feature_bitlength + sigma - 1));
            mask_d[j] = double(r[j]) / share_scale;
        }
        r_mat.push_back(r);
        mask_d_mat.push_back(mask_d);
        CkksPlaintext mask_pt = context->encode(mask_d, level, ENC_TO_SHARE_SCALE);
        CkksCiphertext share0_ct = context->add_plain(data[i], mask_pt);
        share0->data.push_back(move(share0_ct));
    }

    share1->data.resize({n_channel});
    int T = 0;
    double scale = pow(2, share1->scale_ord);

    for (int i = 0; i < mask_d_mat.size(); i++) {
        for (int j = 0; j < div_ceil(n_channel, data.size()); j++) {
            if (T >= n_channel) {
                break;
            }
            uint64_t neg_r = (-r_mat[i][j * skip] % share1->ring_mod + share1->ring_mod) % share1->ring_mod;
            share1->data.set(T, neg_r);
            T += 1;
        }
    }
}

void Feature2DEncrypted::split_to_shares(Feature2DEncrypted* share0, Feature2DShare* share1) const {
    int n_slot = context->get_parameter().get_n() / 2;
    double share_scale = ENC_TO_SHARE_SCALE;
    int feature_bitlength = ENC_TO_SHARE_SCALE_BIT + 1;
    int sigma = SIGMA;

    Duo pre_skip_shape = {shape[0] * skip[0], shape[1] * skip[1]};
    size_t n_share_feature = n_channel * shape[0] * shape[1];
    size_t n_mask = n_channel * pre_skip_shape[0] * pre_skip_shape[1];

    vector<double> mask_d(n_mask);
    vector<int64_t> r(n_mask);
    for (int i = 0; i < n_mask; i++) {
        r[i] = int64_t(gen_random_uint(feature_bitlength + sigma)) - int64_t(1ull << (feature_bitlength + sigma - 1));
        mask_d[i] = double(r[i]) / share_scale;
    }

    share0->n_channel = n_channel;
    share0->n_channel_per_ct = n_channel_per_ct;
    share0->shape = shape;
    share0->skip = skip;
    share0->level = level;
    share0->data.clear();
    vector<double> mask_d_span(mask_d);
    for (int i = 0; i < data.size(); i++) {
        size_t start = i * n_slot;
        size_t length = i == data.size() - 1 ? (mask_d_span.size() - start) : n_slot;
        std::vector<double> mask_mg_vec(mask_d_span.begin() + start, mask_d_span.begin() + start + length);
        CkksPlaintext mask_pt = context->encode(mask_mg_vec, level, ENC_TO_SHARE_SCALE);
        CkksCiphertext share0_ct = context->add_plain(data[i], mask_pt);
        share0->data.push_back(move(share0_ct));
    }

    share1->shape = shape;
    share1->data.resize({n_share_feature});
    double scale = pow(2, share1->scale_ord);
    for (int i = 0; i < n_channel; i++) {
        for (int j = 0; j < shape[0]; j++) {
            for (int k = 0; k < shape[1]; k++) {
                int skipped_index = i * shape[0] * shape[1] + j * shape[1] + k;
                int pre_skip_index =
                    i * pre_skip_shape[0] * pre_skip_shape[1] + j * pre_skip_shape[1] * skip[0] + k * skip[1];
                share1->data[skipped_index] =
                    (-int64_t(r[pre_skip_index]) % share1->ring_mod + share1->ring_mod) % share1->ring_mod;
            }
        }
    }
}

vector<CkksPlaintext> multi_pack_to_pt(const Array<double, 3>& feature_mg,
                                       Feature2DEncrypted& f2d,
                                       int n_channel,
                                       Duo shape,
                                       Duo skip,
                                       CkksContext& context,
                                       int level,
                                       double scale_in = DEFAULT_SCALE,
                                       PackType pack_type = PackType::ParMultiplexedPack) {
    vector<vector<double>> packed;
    Duo block_expansion = {(uint32_t)ceil(shape[0] / (double)BLOCK_SHAPE[0]),
                           (uint32_t)ceil(shape[1] / (double)BLOCK_SHAPE[1])};
    packed = f2d.pack_feature(pack_type, feature_mg, BLOCK_SHAPE, block_expansion);

    vector<CkksPlaintext> pt_vec;
    for (auto& vec : packed) {
        pt_vec.push_back(context.encode(vec, level, scale_in));
    }
    return pt_vec;
}

void Feature2DEncrypted::split_to_shares_for_multi_channel_pack(Feature2DEncrypted* share0,
                                                                Feature2DShare* share1,
                                                                PackType pack_type_in) const {
    int n_slot = context->get_parameter().get_n() / 2;
    double share_scale = ENC_TO_SHARE_SCALE;

    int feature_bitlength = ENC_TO_SHARE_SCALE_BIT + 1;

    int sigma = SIGMA;
    Duo pre_skip_shape = {shape[0] * skip[0], shape[1] * skip[1]};
    // cppcheck-suppress duplicateAssignExpression
    size_t n_share_feature = n_channel * shape[0] * shape[1];
    size_t n_mask = n_channel * shape[0] * shape[1];

    vector<double> mask_d(n_mask);
    vector<int64_t> r(n_mask);
    for (int i = 0; i < n_mask; i++) {
        r[i] = int64_t(gen_random_uint(feature_bitlength + sigma)) - int64_t(1ull << (feature_bitlength + sigma - 1));
        mask_d[i] = double(r[i]) / share_scale;
    }
    share0->n_channel = n_channel;
    share0->n_channel_per_ct = n_channel_per_ct;
    share0->shape = shape;
    share0->skip = skip;
    share0->level = level;
    share0->data.clear();
    auto mask_d_array = Array<double, 1>::from_array_1d(mask_d).reshape<3>({n_channel, shape[0], shape[1]});
    auto mask_pt =
        multi_pack_to_pt(mask_d_array, *share0, n_channel, shape, skip, *context, level, DEFAULT_SCALE, pack_type_in);
    for (int i = 0; i < data.size(); i++) {
        CkksCiphertext share0_ct = context->add_plain(data[i], mask_pt[i]);
        share0->data.push_back(move(share0_ct));
    }

    share1->shape = shape;
    share1->data.resize({n_mask});
    double scale = pow(2, share1->scale_ord);
    for (int i = 0; i < n_mask; i++) {
        share1->data[i] = (-int64_t(r[i]) % share1->ring_mod + share1->ring_mod) % share1->ring_mod;
    }
}

Feature2DEncrypted Feature2DEncrypted::combine_with_share(const Feature2DShare& share) const {
    const int N_THREAD = 4;
    int n_slot = context->get_parameter().get_n() / 2;
    Feature2DEncrypted result(this->context, this->level);
    result.n_channel = this->n_channel;
    result.n_channel_per_ct = this->n_channel_per_ct;
    result.shape = this->shape;
    result.skip = this->skip;
    double scale = pow(2, share.scale_ord);
    int n_ct = this->data.size();

    result.data.clear();
    result.data.resize(n_ct);
    parallel_for(n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, int i) {
        vector<double> mask_d(n_slot);
        for (int j = 0; j < n_slot; j++) {
            uint64_t v;
            if (i * n_slot + j >= share.data.get_size()) {
                v = share.data.get((i * n_slot + j) % share.data.get_size());
            } else {
                v = share.data.get(i * n_slot + j);
            }
            mask_d[j] = uint64_to_double(v, scale, share.ring_mod);
        }
        CkksPlaintext mask_pt = ctx_copy.encode(mask_d, level, ctx_copy.get_parameter().get_default_scale());
        result.data[i] = ctx_copy.add_plain(data[i], mask_pt);
    });
    return result;
}

Feature2DEncrypted Feature2DEncrypted::combine_with_share_new_protocol(const Feature2DShare& share,
                                                                       const Feature2DEncrypted& f2d,
                                                                       const Bytes& b1) const {
    const int N_THREAD = 8;
    int n_slot = context->get_parameter().get_n() / 2;
    Feature2DEncrypted result(this->context, this->level);
    result.n_channel = this->n_channel;
    result.n_channel_per_ct = this->n_channel_per_ct;
    result.shape = this->shape;
    result.skip = this->skip;
    double scale = ENC_TO_SHARE_SCALE;
    double encode_scale = pow(2, DEFAULT_SCALE_BIT);
    int n_ct = this->data.size();

    result.data.clear();
    result.data.resize(n_ct);

    parallel_for_with_extra_level_context(
        n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, CkksContext& extra_level_ctx_copy, int i) {
            vector<double> mask_d(n_slot, 0);
            vector<double> b1_value(n_slot, 0);
            for (int j = 0; j < n_slot; j++) {
                int mg_idx = (i * n_slot + j) % share.data.get_size();
                b1_value[j] = 2 * b1[mg_idx] - 1;
                int64_t mask_value = int64_t(share.data.get(mg_idx)) - int64_t(b1[mg_idx] * share.ring_mod);
                mask_d[j] = double(mask_value) / scale;
            }
            CkksPlaintext mask_pt = ctx_copy.encode(mask_d, level, encode_scale);
            result.data[i] = ctx_copy.add_plain(data[i], mask_pt);

            CkksPlaintext b1_pt =
                extra_level_ctx_copy.encode(b1_value, level + 1, extra_level_ctx_copy.get_parameter().get_q(level + 1));

            auto f2d_mult = extra_level_ctx_copy.mult_plain(f2d.data[i], b1_pt);
            f2d_mult = extra_level_ctx_copy.rescale(f2d_mult, encode_scale);

            result.data[i] = ctx_copy.add(result.data[i], f2d_mult);
        });
    return result;
}

Feature2DEncrypted Feature2DEncrypted::combine_with_share_new_protocol_for_multi_pack(const Feature2DShare& share,
                                                                                      const Feature2DEncrypted& f2d,
                                                                                      const Bytes& b1,
                                                                                      PackType pack_type) const {
    const int N_THREAD = 8;
    int n_slot = context->get_parameter().get_n() / 2;
    Feature2DEncrypted result(this->context, this->level);
    result.n_channel = this->n_channel;
    result.n_channel_per_ct = this->n_channel_per_ct;
    result.shape = this->shape;
    result.skip = this->skip;
    double scale = ENC_TO_SHARE_SCALE;
    double encode_scale = pow(2, DEFAULT_SCALE_BIT);
    int n_ct = this->data.size();

    result.data.clear();
    result.data.resize(n_ct);

    Array<double, 1> mask_d({share.data.get_size()});
    for (int i = 0; i < share.data.get_size(); i++) {
        int64_t mask_value = int64_t(share.data.get(i)) - int64_t(b1[i] * share.ring_mod);
        mask_d.set(i, (mask_value) / scale);
    }
    auto f2d_copy = f2d.copy();
    Array<double, 3> mask_d_3d = mask_d.reshape<3>({this->n_channel, this->shape[0], this->shape[1]});
    auto mask_pt = multi_pack_to_pt(mask_d_3d, f2d_copy, this->n_channel, this->shape, this->skip, *context, level,
                                    DEFAULT_SCALE, pack_type);
    Array<double, 1> b1_value({b1.size()});
    for (int i = 0; i < b1.size(); i++) {
        b1_value.set(i, 2 * b1[i] - 1);
    }
    Array<double, 3> b1_value_3d = b1_value.reshape<3>({this->n_channel, this->shape[0], this->shape[1]});
    CkksContext& extra_level_context = context->get_extra_level_context();
    auto mask_b1 =
        multi_pack_to_pt(b1_value_3d, f2d_copy, this->n_channel, this->shape, this->skip, extra_level_context,
                         level + 1, extra_level_context.get_parameter().get_q(level + 1), pack_type);
    for (int i = 0; i < data.size(); i++) {
        auto f2d_mult = extra_level_context.mult_plain(f2d.data[i], mask_b1[i]);
        f2d_mult = extra_level_context.rescale(f2d_mult, encode_scale);
        result.data[i] = (*context).add_plain(data[i], mask_pt[i]);
        result.data[i] = (*context).add(result.data[i], f2d_mult);
    }
    return result;
}

void Feature2DEncrypted::decrypt_to_share(Feature2DShare* share, PackType pack_type) const {
    uint64_t ring_mod = RING_MOD;
    int n_slot = context->get_parameter().get_n() / 2;
    share->shape = shape;
    Array<double, 3> x_double_matrix;
    if (pack_type == PackType::ParMultiplexedPack) {
        x_double_matrix = this->par_mult_unpack();
    } else if (pack_type == PackType::SinglePack) {
        x_double_matrix = this->unpack();
    } else if (pack_type == PackType::InterleavedDecompositionPack) {
        Duo block_expansion = {(uint32_t)ceil(shape[0] / (double)BLOCK_SHAPE[0]),
                               (uint32_t)ceil(shape[1] / (double)BLOCK_SHAPE[1])};
        x_double_matrix = this->split_with_stride_unpack(BLOCK_SHAPE, block_expansion);
    }

    share->data = array_double_to_uint64(x_double_matrix, share->scale_ord, share->ring_mod).reshape<1>({0});
}

Array<uint64_t, 1> Feature2DEncrypted::encrypt_from_share(const Feature2DShare& share,
                                                          int n_channel,
                                                          const Duo& input_shape,
                                                          PackType pack_type) {
    int n_slot = context->get_parameter().get_n() / 2;
    if (pack_type == PackType::SinglePack) {
        this->skip = {1, 1};
    }

    this->shape = input_shape;
    Array<double, 1> y0_sub_mod_div_s(share.data.get_shape());
    Array<uint64_t, 1> y0_add_mod(share.data.get_shape());
    double scale = ENC_TO_SHARE_SCALE;
    for (int i = 0; i < share.data.get_size(); i++) {
        uint64_t y0_add_mod_value = (share.data[i] + (share.ring_mod / 2)) % share.ring_mod;
        y0_add_mod.set(i, y0_add_mod_value);
        double y0_sub = double(int64_t(y0_add_mod_value) - int64_t(share.ring_mod / 2)) / scale;
        y0_sub_mod_div_s.set(i, y0_sub);
    }

    Array<double, 3> y3 = y0_sub_mod_div_s.reshape<3>({uint64_t(n_channel), input_shape[0], input_shape[1]});
    if (pack_type == PackType::ParMultiplexedPack) {
        this->par_mult_pack(y3, true, DEFAULT_SCALE);
    } else if (pack_type == PackType::SinglePack) {
        this->pack(y3, true, DEFAULT_SCALE);
    } else if (pack_type == PackType::InterleavedDecompositionPack) {
        Duo block_expansion = {(uint32_t)ceil(input_shape[0] / (double)BLOCK_SHAPE[0]),
                               (uint32_t)ceil(input_shape[1] / (double)BLOCK_SHAPE[1])};
        this->split_with_stride_pack(y3, BLOCK_SHAPE, block_expansion, true);
    }

    return y0_add_mod;
}

void Feature2DEncrypted::decompress() {
    assert(data.size() == 0 && data_compress.size() > 0);
    size_t n_ct = data_compress.size();
    for (int i = 0; i < n_ct; i++) {
        data.push_back(context->compressed_ciphertext_to_ciphertext(data_compress[i]));
    }
    data_compress.clear();
}

Bytes Feature2DEncrypted::serialize() const {
    stringstream ss;
    ss_write(ss, dim);
    ss_write(ss, n_channel);
    ss_write(ss, n_channel_per_ct);
    ss_write(ss, level);
    for (int i = 0; i < 2; i++) {
        ss_write(ss, shape[i]);
    }
    for (int i = 0; i < 2; i++) {
        ss_write(ss, skip[i]);
    }
    uint32_t n_ct = data.size();
    ss_write(ss, n_ct);
    for (const CkksCiphertext& ct : data) {
        Bytes ct_data = ct.serialize(context->get_parameter());
        ss_write_vector(ss, ct_data);
    }
    uint32_t n_cct = data_compress.size();
    ss_write(ss, n_cct);
    for (const CkksCompressedCiphertext& cct : data_compress) {
        Bytes cct_data = cct.serialize(context->get_parameter());
        ss_write_vector(ss, cct_data);
    }

    Bytes bytes = ss_to_bytes(ss);
    return bytes;
}

void Feature2DEncrypted::deserialize(const Bytes& bytes) {
    stringstream ss;
    bytes_to_ss(bytes, ss);
    ss_read(ss, &dim);
    ss_read(ss, &n_channel);
    ss_read(ss, &n_channel_per_ct);
    ss_read(ss, &level);
    for (int i = 0; i < 2; i++) {
        ss_read(ss, &shape[i]);
    }
    for (int i = 0; i < 2; i++) {
        ss_read(ss, &skip[i]);
    }
    uint32_t n_ct;
    ss_read(ss, &n_ct);
    for (int i = 0; i < n_ct; i++) {
        Bytes ct_data;
        ss_read_vector(ss, &ct_data);
        auto y_ct = CkksCiphertext::deserialize(ct_data);
        data.push_back(move(y_ct));
    }
    uint32_t n_cct;
    ss_read(ss, &n_cct);
    for (int i = 0; i < n_cct; i++) {
        Bytes cct_data;
        ss_read_vector(ss, &cct_data);
        auto y_ct = CkksCompressedCiphertext::deserialize(cct_data);
        data_compress.push_back(move(y_ct));
    }
}

Feature2DShare::Feature2DShare(uint64_t q, int s) : FeatureShare{q, s} {}
