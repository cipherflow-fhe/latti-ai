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

Feature1DEncrypted::Feature1DEncrypted(CkksContext* context_in, int ct_level) {
    dim = 1;
    context = context_in;
    level = ct_level;
}

void Feature1DEncrypted::pack(Array<double, 2>& feature_mg, bool is_symmetric, double scale_in) {
    const int N_THREAD = 4;
    n_channel = feature_mg.get_shape()[0];
    shape = feature_mg.get_shape()[1];
    n_channel_per_ct = context->get_parameter().get_n() / 2 / shape;
    uint32_t n_ct = div_ceil(n_channel, n_channel_per_ct);
    vector<vector<double>> feature_data;

    data.clear();
    data_compress.clear();
    if (is_symmetric) {
        data_compress.resize(n_ct);
    } else {
        data.resize(n_ct);
    }

    parallel_for(n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        vector<double> image_flat;
        image_flat.reserve(n_channel_per_ct * shape);
        for (int k = 0; k < n_channel_per_ct; k++) {
            if (ct_idx * n_channel_per_ct + k < n_channel) {
                for (int i = 0; i < shape; i++) {
                    image_flat.push_back(feature_mg.get(ct_idx * n_channel_per_ct + k, i));
                }
            } else {
                for (int i = 0; i < shape; i++) {
                    image_flat.push_back(feature_mg.get((ct_idx * n_channel_per_ct + k) % n_channel, i));
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
