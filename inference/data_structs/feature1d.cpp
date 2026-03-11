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

#include "feature1d.h"

using namespace std;

Feature1DEncrypted::Feature1DEncrypted(CkksContext* context_in, int ct_level, uint32_t skip_in) {
    dim = 1;
    context = context_in;
    level = ct_level;
    skip = skip_in;
}

void Feature1DEncrypted::pack(Array<double, 2>& feature_mg, bool is_symmetric, double scale_in) {
    const int N_THREAD = 4;
    n_channel = feature_mg.get_shape()[0];
    shape = feature_mg.get_shape()[1];

    uint32_t shape_with_skip = shape * skip;
    n_channel_per_ct = context->get_parameter().get_n() / 2 / shape_with_skip;
    uint32_t n_ct = div_ceil(n_channel, n_channel_per_ct);

    data.clear();
    data_compress.clear();
    if (is_symmetric) {
        data_compress.resize(n_ct);
    } else {
        data.resize(n_ct);
    }

    parallel_for(n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        vector<double> image_flat;
        image_flat.reserve(n_channel_per_ct * shape_with_skip);

        for (int k = 0; k < n_channel_per_ct; k++) {
            int channel_idx = ct_idx * n_channel_per_ct + k;

            if (channel_idx < n_channel) {
                for (int i = 0; i < shape; i++) {
                    image_flat.push_back(feature_mg.get(channel_idx, i));
                    for (int s = 1; s < skip; s++) {
                        image_flat.push_back(0.0);
                    }
                }
            } else {
                int wrap_channel_idx = channel_idx % n_channel;
                for (int i = 0; i < shape; i++) {
                    image_flat.push_back(feature_mg.get(wrap_channel_idx, i));
                    for (int s = 1; s < skip; s++) {
                        image_flat.push_back(0.0);
                    }
                }
            }
        }

        auto image_flat_pt = ctx_copy.encode(image_flat, level, scale_in);
        if (is_symmetric) {
            auto image_flat_ct = ctx_copy.encrypt_symmetric_compressed(image_flat_pt);
            data_compress[ct_idx] = move(image_flat_ct);
        } else {
            auto image_flat_ct = ctx_copy.encrypt_symmetric(image_flat_pt);
            data[ct_idx] = move(image_flat_ct);
        }
    });
}

void Feature1DEncrypted::par_mult_pack(const Array<double, 2>& feature_mg, bool is_symmetric, double scale_in) {
    const int N_THREAD = 4;
    n_channel = feature_mg.get_shape()[0];
    shape = feature_mg.get_shape()[1];

    int n_slot = context->get_parameter().get_n() / 2;

    uint32_t shape_with_skip = shape * skip;
    n_channel_per_ct = (n_slot / shape_with_skip) * skip;

    int n_mult_pack_per_ct = std::min((int)n_channel_per_ct, (int)n_channel);

    if (n_channel > n_channel_per_ct) {
        throw std::runtime_error("over slot!");
    }

    int f_ct_num = div_ceil(n_channel, n_mult_pack_per_ct);

    data.clear();
    data_compress.clear();
    if (is_symmetric) {
        data_compress.resize(f_ct_num);
    } else {
        data.resize(f_ct_num);
    }

    parallel_for(f_ct_num, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        vector<double> image_flat;
        image_flat.resize(n_slot, 0.0);

        for (int j = 0; j < n_mult_pack_per_ct; j++) {
            int channel = ct_idx * n_mult_pack_per_ct + j;

            if (channel >= n_channel) {
                continue;
            }

            for (int data_idx = 0; data_idx < shape; data_idx++) {
                int slot_idx = (j / skip) * shape_with_skip + data_idx * skip + (j % skip);
                image_flat[slot_idx] = feature_mg.get(channel, data_idx);
            }
        }

        auto image_flat_pt = ctx_copy.encode(image_flat, level, scale_in);
        if (is_symmetric) {
            auto image_flat_ct = ctx_copy.encrypt_symmetric_compressed(image_flat_pt);
            data_compress[ct_idx] = move(image_flat_ct);
        } else {
            auto image_flat_ct = ctx_copy.encrypt_symmetric(image_flat_pt);
            data[ct_idx] = move(image_flat_ct);
        }
    });
}

Array<double, 2> Feature1DEncrypted::par_mult_unpack() const {
    const int N_THREAD = 4;
    int n_ct = data.size();
    int n_slot = context->get_parameter().get_n() / 2;

    uint32_t shape_with_skip = shape * skip;
    int n_mult_pack_per_ct = std::min((int)n_channel_per_ct, (int)n_channel);

    Array<double, 2> result({n_channel, shape});

    parallel_for(n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        CkksPlaintext x_pt = ctx_copy.decrypt(data[ct_idx]);
        Array1D x_mg = ctx_copy.decode(x_pt);

        for (int j = 0; j < n_mult_pack_per_ct; j++) {
            int channel = ct_idx * n_mult_pack_per_ct + j;

            if (channel >= n_channel) {
                continue;
            }

            for (int data_idx = 0; data_idx < shape; data_idx++) {
                int slot_idx = (j / skip) * shape_with_skip + data_idx * skip + (j % skip);
                result.set(channel, data_idx, x_mg[slot_idx]);
            }
        }
    });

    return result;
}

Array<double, 2> Feature1DEncrypted::unpack() const {
    const int N_THREAD = 4;
    int n_ct = data.size();
    int pre_skip_shape = shape * skip;

    Array<double, 2> result({n_channel, shape});
    parallel_for(n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        CkksPlaintext x_pt = ctx_copy.decrypt(data[ct_idx]);
        Array1D x_mg = ctx_copy.decode(x_pt);
        for (int i = 0; i < n_channel_per_ct; i++) {
            int channel_idx = ct_idx * n_channel_per_ct + i;
            if (channel_idx >= n_channel) {
                continue;
            }
            for (int j = 0; j < shape; j++) {
                result.set(channel_idx, j, x_mg[i * pre_skip_shape + j * skip]);
            }
        }
    });
    return result;
}
