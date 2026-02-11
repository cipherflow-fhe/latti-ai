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

#include "feature0d.h"
#include "util.h"

using namespace std;

Feature0DEncrypted::Feature0DEncrypted(CkksContext* context_in, int ct_level) {
    dim = 0;
    context = context_in;
    level = ct_level;
}

void Feature0DEncrypted::pack(const Array<double, 1>& feature_mg, bool is_symmetric, double scale_in) {
    uint32_t n_in_features = feature_mg.get_size();
    uint32_t n_slots = context->get_parameter().get_n() / 2;
    n_channel_per_ct = n_slots;

    for (int pack_ct_idx = 0; pack_ct_idx < div_ceil(n_in_features, n_slots); pack_ct_idx++) {
        vector<double> feature_flat;
        feature_flat.reserve((int)n_slots);
        for (int i = pack_ct_idx * int(n_slots); i < (pack_ct_idx + 1) * n_slots; i++) {
            if (i >= 0 && i < n_in_features) {
                feature_flat.push_back(feature_mg[i]);
            } else {
                feature_flat.push_back(0);
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

void Feature0DEncrypted::pack_skip(const Array<double, 1>& feature_mg, bool is_symmetric) {
    uint32_t n_in_features = feature_mg.get_size();
    uint32_t n_slots = context->get_parameter().get_n() / 2;
    n_channel_per_ct = n_slots / skip;
    for (int pack_ct_idx = 0; pack_ct_idx < div_ceil(n_in_features, n_channel_per_ct); pack_ct_idx++) {
        vector<double> feature_flat((int)n_slots);
        feature_flat.reserve((int)n_slots);
        for (int i = 0; i < n_channel_per_ct; i++) {
            if ((pack_ct_idx * n_channel_per_ct + i) < n_in_features) {
                feature_flat[i * skip] = feature_mg[pack_ct_idx * n_channel_per_ct + i];
            } else {
                feature_flat[i * skip] = 0;
            }
        }
        auto feature_flat_pt = context->encode(feature_flat, level, context->get_parameter().get_default_scale());
        auto feature_flat_ct = context->encrypt_asymmetric(feature_flat_pt);
        data.push_back(move(feature_flat_ct));
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

Array<double, 1> Feature0DEncrypted::unpack(DecryptType dec_type) const {
    Array<double, 1> result({n_channel});
    if (dec_type == DecryptType::RESHAPE) {
        for (int ct_idx = 0; ct_idx < data.size(); ct_idx++) {
            auto c_pt = context->decrypt(data[ct_idx]);
            auto c = context->decode(c_pt);
            // auto c_vec = mg_to_vec(c);
            int T = 0;
            for (int j = 0; j < div_ceil(n_channel, data.size()); j++) {
                if (T > n_channel) {
                    break;
                }
                result.set(T, (c)[j * skip]);
                T += 1;
            }
        }
    } else {
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
