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

#include "activation_layer.h"
#include <iostream>
#include "util.h"

using namespace std;
using namespace cxx_sdk_v2;

SquareLayer::SquareLayer(const CkksParameter& param_in) : Layer(param_in) {}

vector<CkksCiphertext> SquareLayer::call(CkksContext& ctx, const vector<CkksCiphertext>& x) {
    int x_size = x.size();
    vector<CkksCiphertext> result;
    result.resize(x_size);
    parallel_for(x_size, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        auto d = ctx_copy.mult(x[ct_idx], x[ct_idx]);
        auto d_res = ctx_copy.relinearize(d);
        d_res = ctx_copy.rescale(d_res, param_.get_default_scale() * param_.get_default_scale() /
                                            ctx.get_parameter().get_q(x[0].get_level()));
        result[ct_idx] = move(d_res);
        result[ct_idx].set_scale(param_.get_default_scale());
    });
    return result;
}

Feature2DEncrypted SquareLayer::call(CkksContext& ctx, const Feature2DEncrypted& x) {
    Feature2DEncrypted result(x.context, x.level);
    result.data = move(call(ctx, x.data));
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.shape[0] = x.shape[0];
    result.shape[1] = x.shape[1];
    result.skip[0] = x.skip[0];
    result.skip[1] = x.skip[1];
    result.level = x.level - 1;
    return result;
}

Feature0DEncrypted SquareLayer::call(CkksContext& ctx, const Feature0DEncrypted& x) {
    Feature0DEncrypted result(x.context, x.level);
    result.data = move(call(ctx, x.data));
    result.skip = x.skip;
    result.pack_type = 0;
    result.n_channel = x.n_channel;
    result.dim = x.dim;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.level = x.level - 1;
    return result;
}
