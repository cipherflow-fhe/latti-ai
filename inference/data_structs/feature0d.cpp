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

#include <sstream>
#include "feature0d.h"
#include "feature2d.h"
#include "../util.h"

using namespace std;
using namespace lattisense;

Feature0DEncrypted::Feature0DEncrypted(CkksContext* context_in, int ct_level) {
    dim = 0;
    context = context_in;
    level = ct_level;
}

void Feature0DEncrypted::pack(const Array<double, 1>& feature_mg,
                              bool is_symmetric,
                              double scale_in,
                              uint32_t skip_in) {
    uint32_t n_in_features = feature_mg.get_size();
    uint32_t n_slots = context->get_parameter().get_n() / 2;
    n_channel = n_in_features;
    skip = skip_in;
    n_channel_per_ct = n_slots / skip;

    for (int pack_ct_idx = 0; pack_ct_idx < div_ceil(n_in_features, n_channel_per_ct); pack_ct_idx++) {
        vector<double> feature_flat((int)n_slots, 0.0);
        for (int i = 0; i < (int)n_channel_per_ct; i++) {
            int src_idx = pack_ct_idx * (int)n_channel_per_ct + i;
            if (src_idx < (int)n_in_features) {
                feature_flat[i * skip] = feature_mg[src_idx];
            }
        }
        auto feature_flat_pt = context->encode(feature_flat, level, scale_in);
        if (!is_symmetric) {
            auto feature_flat_ct = context->encrypt_asymmetric(feature_flat_pt);
            data.push_back(move(feature_flat_ct));
        } else {
            auto feature_flat_ct = context->encrypt_symmetric_compressed(feature_flat_pt);
            data_compressed.push_back(move(feature_flat_ct));
        }
    }
}

void Feature0DEncrypted::pack_cyclic(const std::vector<double>& feature_mg, bool is_symmetric, double scale_in) {
    uint32_t n_in_features = feature_mg.size();
    uint32_t n_slots = context->get_parameter().get_n() / 2;
    n_channel_per_ct = n_slots / skip;
    for (int pack_ct_idx = 0; pack_ct_idx < div_ceil(n_in_features, n_slots); pack_ct_idx++) {
        vector<double> feature_flat;
        feature_flat.reserve((int)n_slots);
        for (int i = pack_ct_idx * (int)n_slots; i < (pack_ct_idx + 1) * n_slots; i++) {
            if (i >= 0 && i < n_in_features) {
                feature_flat.push_back(feature_mg[i]);
            } else {
                // cppcheck-suppress signConversionCond
                feature_flat.push_back(feature_mg[i % n_in_features]);
            }
        }
        auto feature_flat_pt = context->encode(feature_flat, level, scale_in);
        if (!is_symmetric) {
            auto feature_flat_ct = context->encrypt_symmetric(feature_flat_pt);
            data.push_back(move(feature_flat_ct));
        } else {
            auto feature_flat_ct = context->encrypt_symmetric_compressed(feature_flat_pt);
            data_compressed.push_back(move(feature_flat_ct));
        }
    }
}

Feature0DEncrypted Feature0DEncrypted::refresh_ciphertext() const {
    CkksBtpContext* ctx = dynamic_cast<CkksBtpContext*>(context);
    int new_level = 9;
    Feature0DEncrypted result(ctx, new_level);
    for (int i = 0; i < data.size(); i++) {
        result.data.push_back(ctx->bootstrap(data[i]));
    }
    return result;
}

Array<double, 1> Feature0DEncrypted::unpack() const {
    Array<double, 1> result({n_channel});
    int T = 0;
    for (int ct_idx = 0; ct_idx < data.size(); ct_idx++) {
        auto c_pt = context->decrypt(data[ct_idx]);
        auto c = context->decode(c_pt);
        for (int j = 0; j < n_channel_per_ct; j++) {
            if (T >= n_channel) {
                break;
            }
            result.set(T, (c)[j * skip]);
            T += 1;
        }
    }
    return result;
}

Feature0DEncrypted Feature0DEncrypted::drop_level(int n_level_to_drop) const {
    int new_level = level - n_level_to_drop;
    Feature0DEncrypted result(context, new_level);
    result.n_channel = n_channel;
    result.n_channel_per_ct = n_channel_per_ct;
    result.skip = skip;
    result.data.resize(data.size());
    parallel_for(data.size(), th_nums, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        auto ct_tmp = data[ct_idx].copy();
        for (int j = 0; j < n_level_to_drop; j++) {
            ct_tmp = ctx_copy.drop_level(ct_tmp);
        }
        result.data[ct_idx] = move(ct_tmp);
        assert(new_level == result.data[ct_idx].get_level());
    });
    return result;
}

Feature0DEncrypted Feature0DEncrypted::copy() const {
    Feature0DEncrypted result(context, level);
    result.dim = dim;
    result.n_channel = n_channel;
    result.n_channel_per_ct = n_channel_per_ct;
    result.skip = skip;
    for (int i = 0; i < data.size(); i++) {
        result.data.push_back(data[i].copy());
    }
    return result;
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

void Feature0DEncrypted::decompress() {
    assert(data.size() == 0);
    assert(data_compressed.size() > 0);
    size_t n_ct = data_compressed.size();
    for (int i = 0; i < n_ct; i++) {
        data.push_back(context->compressed_ciphertext_to_ciphertext(data_compressed[i]));
    }
    data_compressed.clear();
}
