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

#include "mult_scaler.h"

MultScalarLayer::MultScalarLayer(const CkksParameter& param_in,
                                 const Duo& input_shape_in,
                                 const Array<double, 1>& weight_in,
                                 const Duo& skip_in,
                                 uint32_t n_channel_per_ct_in,
                                 uint32_t level_in,
                                 const Duo& upsample_factor_in,
                                 const Duo& block_expansion_in)
    : param(param_in.copy()), input_shape(input_shape_in), weight(weight_in.copy()), skip(skip_in) {
    if ((input_shape[0] & (input_shape[0] - 1)) != 0 || (input_shape[1] & (input_shape[1] - 1)) != 0) {
        throw std::invalid_argument("input_shape must be powers of 2, got: ["
                                    + std::to_string(input_shape[0]) + ", " + std::to_string(input_shape[1]) + "]");
    }
    if ((skip[0] & (skip[0] - 1)) != 0 || (skip[1] & (skip[1] - 1)) != 0) {
        throw std::invalid_argument("skip must be powers of 2, got: ["
                                    + std::to_string(skip[0]) + ", " + std::to_string(skip[1]) + "]");
    }

    n_channel_per_ct = n_channel_per_ct_in;
    level = level_in;
    n_block_per_ct = std::floor(n_channel_per_ct / (skip[0] * skip[1]));
    upsample_factor[0] = upsample_factor_in[0];
    upsample_factor[1] = upsample_factor_in[1];
    pre_skip[0] = skip[0] * upsample_factor[0];
    pre_skip[1] = skip[1] * upsample_factor[1];
    block_expansion[0] = block_expansion_in[0];
    block_expansion[1] = block_expansion_in[1];
    block_shape[0] = input_shape[0] / block_expansion[0] * skip[0];
    block_shape[1] = input_shape[1] / block_expansion[1] * skip[1];
    if ((block_shape[0] & (block_shape[0] - 1)) != 0 || (block_shape[1] & (block_shape[1] - 1)) != 0) {
        throw std::invalid_argument("block_shape must be powers of 2, got: ["
                                    + std::to_string(block_shape[0]) + ", " + std::to_string(block_shape[1]) + "]");
    }
}

MultScalarLayer::~MultScalarLayer() {}

void MultScalarLayer::prepare_weight() {
    int skip_prod = skip[0] * skip[1];
    int channel = weight.get_shape()[0];
    int n_packed_out_channel = div_ceil(channel, n_channel_per_ct) * block_expansion[0] * block_expansion[1];
    weight_pt.clear();

    CkksContext ctx = CkksContext::create_empty_context(this->param);
    ctx.resize_copies(n_packed_out_channel);
    weight_pt.resize(n_packed_out_channel);
    double pack_scale = ctx.get_parameter().get_q(level);
    parallel_for(n_packed_out_channel, th_nums, ctx, [&](CkksContext& ctx_copy, int n_packed_out_channel_idx) {
        const int total_block_size = n_block_per_ct * block_shape[0] * block_shape[1];
        vector<double> feature_tmp_pack(ctx_copy.get_parameter().get_n() / 2);
        for (int linear_idx = 0; linear_idx < total_block_size; ++linear_idx) {
            int block_i = linear_idx / (block_shape[0] * block_shape[1]);
            int residual = linear_idx % (block_shape[0] * block_shape[1]);
            int shape_i = residual / block_shape[1];
            int shape_j = residual % block_shape[1];

            int channel_idx = n_packed_out_channel_idx * n_channel_per_ct / block_expansion[0] / block_expansion[1] +
                              block_i * skip_prod + (skip[1] * (shape_i % skip[0]) + shape_j % skip[1]);
            if (channel_idx >= channel || (shape_i % pre_skip[0]) >= skip[0] || (shape_j % pre_skip[1]) >= skip[1])
                continue;

            int index = block_i * block_shape[0] * block_shape[1] + shape_i * block_shape[1] + shape_j;
            feature_tmp_pack[index] = weight.get(channel_idx);
        }
        weight_pt[n_packed_out_channel_idx] = move(ctx_copy.encode_ringt(feature_tmp_pack, pack_scale));
    });
}

std::vector<CkksCiphertext> MultScalarLayer::run_core(CkksContext& ctx, const std::vector<CkksCiphertext>& x) {
    vector<CkksCiphertext> res(x.size());
    parallel_for(x.size(), th_nums, ctx, [&](CkksContext& ctx_copy, int x_idx) {
        auto w_pt = ctx_copy.ringt_to_mul(weight_pt[x_idx], x[x_idx].get_level());
        res[x_idx] =
            ctx_copy.rescale(ctx_copy.mult_plain_mul(x[x_idx], w_pt), ctx_copy.get_parameter().get_default_scale());
    });
    return res;
}

Feature2DEncrypted MultScalarLayer::run(CkksContext& ctx, const Feature2DEncrypted& x) {
    Feature2DEncrypted result(&ctx, x.level - 1);

    result.data = move(run_core(ctx, x.data));
    result.skip = x.skip;
    result.shape = x.shape;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.dim = x.dim;
    result.n_channel = x.n_channel;
    result.level = x.level - 1;

    return result;
}

Array<double, 3> MultScalarLayer::run_plaintext(const Array<double, 3>& x) {
    int n_out_channel = x.get_shape()[0];
    Array<double, 3> result({x.get_shape()[0], x.get_shape()[1], x.get_shape()[2]});
    for (int in_channel_idx = 0; in_channel_idx < n_out_channel; in_channel_idx++) {
        for (int i = 0; i < input_shape[0]; i++) {
            for (int j = 0; j < input_shape[1]; j++) {
                auto p = x.get(in_channel_idx, i, j) * weight.get(in_channel_idx);
                result.set(in_channel_idx, i, j, p);
            }
        }
    }
    return result;
}
