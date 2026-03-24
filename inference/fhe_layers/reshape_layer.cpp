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

#include "reshape_layer.h"
#include "util.h"

using namespace std;
using namespace cxx_sdk_v2;

ReshapeLayer::ReshapeLayer(const CkksParameter& param_in) : Layer(param_in) {}

Feature0DEncrypted ReshapeLayer::call(CkksContext& ctx, const Feature2DEncrypted& x) {
    Feature0DEncrypted result(&ctx, x.level);
    for (int i = 0; i < x.data.size(); i++) {
        result.data.push_back(x.data[i].copy());
    }
    result.dim = 0;
    result.skip = (x.shape[0] * x.skip[0]) * (x.shape[1] * x.skip[1]);
    result.level = x.level;
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = div_ceil(param_.get_n() / 2, (x.shape[0] * x.skip[0]) * (x.shape[1] * x.skip[1]));
    return result;
}
