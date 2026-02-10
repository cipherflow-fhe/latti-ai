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

#include "add_layer.h"

using namespace std;

void copy_param(const Feature2DEncrypted& x1, Feature2DEncrypted& result) {
    result.dim = x1.dim;
    result.shape[0] = x1.shape[0];
    result.shape[1] = x1.shape[1];
    result.skip[0] = x1.skip[0];
    result.skip[1] = x1.skip[1];
    result.level = x1.data[0].get_level();
    result.n_channel = x1.n_channel;
    result.n_channel_per_ct = x1.n_channel_per_ct;
}

void AddLayer::add(CkksContext* ctx,
                   const Feature2DEncrypted& x0,
                   const Feature2DEncrypted& x1,
                   Feature2DEncrypted& result) {
    int n_ct = x0.data.size();
    result.data.resize(n_ct);

    parallel_for(n_ct, th_nums, *ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        CkksCiphertext res = ctx_copy.add(x0.data[ct_idx], x1.data[ct_idx]);
        result.data[ct_idx] = move(res);
    });
}

AddLayer::AddLayer(const CkksParameter& param_in) : param(param_in.copy()) {}

Feature2DEncrypted AddLayer::run(CkksContext& ctx, const Feature2DEncrypted& x0, const Feature2DEncrypted& x1) {
    Feature2DEncrypted result(&ctx, x0.level);
    add(&ctx, x0, x1, result);
    result.dim = x1.dim;
    result.shape[0] = x1.shape[0];
    result.shape[1] = x1.shape[1];
    result.skip[0] = x1.skip[0];
    result.skip[1] = x1.skip[1];
    result.level = result.data[0].get_level();
    result.n_channel = x1.n_channel;
    result.n_channel_per_ct = x1.n_channel_per_ct;
    return result;
}

Array<double, 3> AddLayer::run_plaintext(const Array<double, 3>& x0, const Array<double, 3>& x1) {
    auto shape1 = x0.get_shape();
    Array<double, 3> y(shape1);
    for (int i = 0; i < shape1[0]; i++) {
        for (int j = 0; j < shape1[1]; j++) {
            for (int z = 0; z < shape1[2]; z++) {
                y.set(i, j, z, x0.get(i, j, z) + x1.get(i, j, z));
            }
        }
    }
    return y;
}

DropLevelLayer::DropLevelLayer() {}

DropLevelLayer::~DropLevelLayer() {}

void DropLevelLayer::run(CkksContext& ctx,
                         const Feature2DEncrypted& x0,
                         Feature2DEncrypted& result0,
                         int level_in,
                         int level_out) {
    result0 = x0.drop_level(level_in - level_out);
}
