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

#include "concat_layer.h"

using namespace std;
using namespace cxx_sdk_v2;

ConcatLayer::ConcatLayer() {}

Feature2DEncrypted ConcatLayer::run(CkksContext& ctx, const Feature2DEncrypted& x1, const Feature2DEncrypted& x2) {
    if (x1.n_channel_per_ct != x2.n_channel_per_ct) {
        throw std::runtime_error("channel per ct mismatch in ConcatLayer");
    }
    if (x1.level != x2.level) {
        throw std::runtime_error("level mismatch in ConcatLayer");
    }
    if (x1.shape[0] != x2.shape[0] || x1.shape[1] != x2.shape[1]) {
        throw std::runtime_error("shape mismatch in ConcatLayer");
    }
    if (x1.skip[0] != x2.skip[0] || x1.skip[1] != x2.skip[1]) {
        throw std::runtime_error("skip mismatch in ConcatLayer");
    }
    if ((x2.n_channel % x2.n_channel_per_ct) != 0) {
        throw std::runtime_error("n_channel is not divisible by n_channel_per_ct in ConcatLayer");
    }
    Feature2DEncrypted result(&ctx, x1.level);
    result.data.clear();

    for (const auto& elem : x1.data) {
        result.data.push_back(elem.copy());
    }
    for (const auto& elem : x2.data) {
        result.data.push_back(elem.copy());
    }

    result.n_channel = x1.n_channel + x2.n_channel;
    result.level = x1.level;
    result.n_channel_per_ct = x1.n_channel_per_ct;
    result.shape[0] = x1.shape[0];
    result.shape[1] = x1.shape[1];
    result.skip[0] = x1.skip[0];
    result.skip[1] = x1.skip[1];
    return result;
}

Feature2DEncrypted ConcatLayer::run_multiple_inputs(CkksContext& ctx, const std::vector<Feature2DEncrypted>& inputs) {
    if (inputs.empty()) {
        throw std::runtime_error("Empty input vector in ConcatLayer::run_multiple_inputs");
    }
    if (inputs.size() == 1) {
        return inputs[0].copy();
    }

    const Feature2DEncrypted& first = inputs[0];
    if ((first.n_channel % first.n_channel_per_ct) != 0) {
        throw std::runtime_error("n_channel is not divisible by n_channel_per_ct in ConcatLayer::run_multiple_inputs");
    }
    for (size_t i = 1; i < inputs.size(); ++i) {
        const Feature2DEncrypted& current = inputs[i];
        if (current.n_channel_per_ct != first.n_channel_per_ct) {
            throw std::runtime_error("channel per ct mismatch in ConcatLayer::run_multiple_inputs");
        }
        if (current.level != first.level) {
            throw std::runtime_error("level mismatch in ConcatLayer::run_multiple_inputs");
        }
        if (current.shape[0] != first.shape[0] || current.shape[1] != first.shape[1]) {
            throw std::runtime_error("shape mismatch in ConcatLayer::run_multiple_inputs");
        }
        if (current.skip[0] != first.skip[0] || current.skip[1] != first.skip[1]) {
            throw std::runtime_error("skip mismatch in ConcatLayer::run_multiple_inputs");
        }
        if ((current.n_channel % current.n_channel_per_ct) != 0) {
            throw std::runtime_error(
                "n_channel is not divisible by n_channel_per_ct in ConcatLayer::run_multiple_inputs");
        }
    }

    Feature2DEncrypted result(&ctx, first.level);
    result.data.clear();

    uint64_t total_channels = 0;
    for (const auto& input : inputs) {
        for (const auto& elem : input.data) {
            result.data.push_back(elem.copy());
        }
        total_channels += input.n_channel;
    }

    result.n_channel = total_channels;
    result.level = first.level;
    result.n_channel_per_ct = first.n_channel_per_ct;
    result.shape[0] = first.shape[0];
    result.shape[1] = first.shape[1];
    result.skip[0] = first.skip[0];
    result.skip[1] = first.skip[1];
    return result;
}

Array<double, 3> ConcatLayer::concatenate_channels(const Array<double, 3>& x1, const Array<double, 3>& x2) {
    auto shape_x1 = x1.get_shape();
    uint64_t C1 = shape_x1[0];
    uint64_t H1 = shape_x1[1];
    uint64_t W1 = shape_x1[2];

    auto shape_x2 = x2.get_shape();
    uint64_t C2 = shape_x2[0];
    uint64_t H2 = shape_x2[1];
    uint64_t W2 = shape_x2[2];
    if (H1 != H2 || W1 != W2) {
        throw std::invalid_argument("Arrays must have same height and width dimensions");
    }
    Array<double, 3> result({C1 + C2, H1, W1});
    for (int c = 0; c < C1; ++c) {
        for (int h = 0; h < H1; ++h) {
            for (int w = 0; w < W1; ++w) {
                result.set(c, h, w, x1.get(c, h, w));
            }
        }
    }
    for (int c = 0; c < C2; ++c) {
        for (int h = 0; h < H2; ++h) {
            for (int w = 0; w < W2; ++w) {
                result.set(C1 + c, h, w, x2.get(c, h, w));
            }
        }
    }
    return result;
}

Array<double, 3> ConcatLayer::concatenate_channels_multiple_inputs(const std::vector<Array<double, 3>>& inputs) {
    if (inputs.empty()) {
        throw std::invalid_argument("Empty input vector in concatenate_channels_multiple_inputs");
    }
    if (inputs.size() == 1) {
        auto shape = inputs[0].get_shape();
        Array<double, 3> result(shape);
        for (uint64_t c = 0; c < shape[0]; ++c) {
            for (uint64_t h = 0; h < shape[1]; ++h) {
                for (uint64_t w = 0; w < shape[2]; ++w) {
                    result.set(c, h, w, inputs[0].get(c, h, w));
                }
            }
        }
        return result;
    }

    auto first_shape = inputs[0].get_shape();
    uint64_t H = first_shape[1];
    uint64_t W = first_shape[2];

    uint64_t total_channels = 0;
    for (const auto& input : inputs) {
        auto shape = input.get_shape();
        if (shape[1] != H || shape[2] != W) {
            throw std::invalid_argument("Arrays must have same height and width dimensions");
        }
        total_channels += shape[0];
    }

    Array<double, 3> result({total_channels, H, W});
    uint64_t channel_offset = 0;

    for (const auto& input : inputs) {
        auto shape = input.get_shape();
        uint64_t C = shape[0];
        for (uint64_t c = 0; c < C; ++c) {
            for (uint64_t h = 0; h < H; ++h) {
                for (uint64_t w = 0; w < W; ++w) {
                    result.set(channel_offset + c, h, w, input.get(c, h, w));
                }
            }
        }
        channel_offset += C;
    }

    return result;
}
