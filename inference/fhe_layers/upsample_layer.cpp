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

#include "upsample_layer.h"

using namespace std;
using namespace cxx_sdk_v2;

UpsampleLayer::UpsampleLayer(const CkksParameter& param_in,
                             const Duo& stride_in,
                             const Duo& upsample_factor_in,
                             const int& level_in,
                             const int& n_channel_in,
                             const int& n_channel_per_ct_in)
    : Layer(param_in) {
    stride[0] = stride_in[0];
    stride[1] = stride_in[1];
    level_ = level_in;
    n_channel_per_ct = n_channel_per_ct_in;
    upsample_factor[0] = upsample_factor_in[0];
    upsample_factor[1] = upsample_factor_in[1];
    n_channel = n_channel_in;
}

void UpsampleLayer::prepare_data() {
    CkksContext ctx = CkksContext::create_empty_context(this->param_);
    vector<double> zero_vector;
    zero_vector.resize(ctx.get_parameter().get_n() / 2);
    zero_vector_encoded = ctx.encode(zero_vector, level_, ctx.get_parameter().get_default_scale());
}

Feature2DEncrypted UpsampleLayer::run(CkksContext& ctx, const Feature2DEncrypted& x) {
    zero_vector_encrypted = ctx.encrypt_asymmetric(zero_vector_encoded);

    if (n_channel_per_ct != 1) {
        throw std::runtime_error("invalid n_channel_per_ct value in UpsampleLayer");
    }

    Feature2DEncrypted result(&ctx, x.level);
    result.data.clear();
    for (int channel_idx = 0; channel_idx < n_channel; channel_idx++) {
        int base_ct_idx = channel_idx * stride[0] * stride[1];
        for (int i = 0; i < stride[0] * upsample_factor[0]; i++) {
            for (int j = 0; j < stride[1] * upsample_factor[1]; j++) {
                if (i % upsample_factor[0] == 0 && j % upsample_factor[1] == 0) {
                    int idx = (i / upsample_factor[0]) * stride[1] + j / upsample_factor[1];
                    result.data.push_back(x.data[idx + base_ct_idx].copy());
                } else {
                    result.data.push_back(zero_vector_encrypted.copy());
                }
            }
        }
    }

    result.n_channel = x.n_channel;
    result.level = x.level;
    result.n_channel_per_ct = 1;
    result.shape[0] = x.shape[0] * upsample_factor[0];
    result.shape[1] = x.shape[1] * upsample_factor[1];
    result.skip[0] = 1;
    result.skip[1] = 1;
    return result;
}

Array<double, 3> UpsampleLayer::upsample_with_zero(const Array<double, 3>& x) {
    std::array<uint64_t, 3UL> x_shape = x.get_shape();
    uint32_t C = x_shape[0];
    int H = x_shape[1];
    int W = x_shape[2];

    uint32_t H_new = H * upsample_factor[0];
    uint32_t W_new = W * upsample_factor[1];

    Array<double, 3> out({C, H_new, W_new});

    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H_new; h++) {
            for (int w = 0; w < W_new; w++) {
                out.set(c, h, w, 0.0);
            }
        }
    }
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                out.set(c, h * upsample_factor[0], w * upsample_factor[1], x.get(c, h, w));
            }
        }
    }
    return out;
}
