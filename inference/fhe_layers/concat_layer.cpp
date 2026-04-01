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
    }

    // Check if any input has uneven channels
    bool has_uneven = false;
    for (const auto& input : inputs) {
        if ((input.n_channel % input.n_channel_per_ct) != 0) {
            has_uneven = true;
            break;
        }
    }

    if (has_uneven) {
        return run_multiple_inputs_uneven(ctx, inputs);
    }

    // Fast path: all inputs have n_channel divisible by n_channel_per_ct
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

Feature2DEncrypted ConcatLayer::run_multiple_inputs_uneven(CkksContext& ctx,
                                                           const std::vector<Feature2DEncrypted>& inputs) {
    const Feature2DEncrypted& first = inputs[0];
    const Duo shape = first.shape;
    const Duo skip = first.skip;
    uint32_t n_channel_per_ct = first.n_channel_per_ct;
    uint32_t n_channel_per_block = prod(skip);
    uint32_t block_size = prod(shape * skip);  // slots per block
    uint32_t level = first.level;

    // Compute channel offsets and total channels
    vector<uint64_t> channel_offsets(inputs.size());
    uint64_t total_channels = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        channel_offsets[i] = total_channels;
        total_channels += inputs[i].n_channel;
    }
    uint32_t n_out_ct = div_ceil((uint32_t)total_channels, n_channel_per_ct);

    // Step 1: For each global channel, mask the source CT using pre-computed mask_pt
    vector<CkksCiphertext> masked_cts(total_channels);
    for (uint64_t global_ch = 0; global_ch < total_channels; ++global_ch) {
        // global_ch channle is the local_ch channel of the input_idx Feature2D input
        size_t input_idx = 0;
        uint64_t local_ch = global_ch;
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (local_ch < inputs[i].n_channel) {
                input_idx = i;
                break;
            }
            local_ch -= inputs[i].n_channel;
        }

        // Source position in source CT
        uint32_t src_ct_idx = local_ch / n_channel_per_ct;

        // Use pre-computed mask plaintext
        CkksPlaintextMul mask_mul = ctx.ringt_to_mul(mask_pt[global_ch], level);
        masked_cts[global_ch] = ctx.mult_plain_mul(inputs[input_idx].data[src_ct_idx], mask_mul);
    }

    // Step 2: Rotate and accumulate into output CTs
    vector<CkksCiphertext> out_data(n_out_ct);
    for (uint32_t out_ct_idx = 0; out_ct_idx < n_out_ct; ++out_ct_idx) {
        CkksCiphertext sum(0);
        bool first_in_ct = true;

        for (uint32_t ch_in_ct = 0; ch_in_ct < n_channel_per_ct; ++ch_in_ct) {
            uint64_t global_ch = (uint64_t)out_ct_idx * n_channel_per_ct + ch_in_ct;
            if (global_ch >= total_channels)
                break;

            // Source position (where the channel currently sits after masking)
            size_t input_idx = 0;
            uint64_t local_ch = global_ch;
            for (size_t i = 0; i < inputs.size(); ++i) {
                if (local_ch < inputs[i].n_channel) {
                    input_idx = i;
                    break;
                }
                local_ch -= inputs[i].n_channel;
            }
            uint32_t src_channel_in_ct = local_ch % n_channel_per_ct;
            uint32_t src_block = src_channel_in_ct / n_channel_per_block;
            uint32_t src_offset = src_channel_in_ct % n_channel_per_block;
            uint32_t src_cx = src_offset / skip[1];
            uint32_t src_cy = src_offset % skip[1];
            long src_slot_base = (long)src_block * block_size + (long)src_cx * (shape[1] * skip[1]) + src_cy;

            // Target position in output CT
            uint32_t dst_channel_in_ct = ch_in_ct;
            uint32_t dst_block = dst_channel_in_ct / n_channel_per_block;
            uint32_t dst_offset = dst_channel_in_ct % n_channel_per_block;
            uint32_t dst_cx = dst_offset / skip[1];
            uint32_t dst_cy = dst_offset % skip[1];
            long dst_slot_base = (long)dst_block * block_size + (long)dst_cx * (shape[1] * skip[1]) + dst_cy;

            long rot_step = -(dst_slot_base - src_slot_base);

            CkksCiphertext rotated;
            if (rot_step == 0) {
                rotated = masked_cts[global_ch].copy();
            } else {
                rotated = ctx.rotate(masked_cts[global_ch], rot_step);
            }

            if (first_in_ct) {
                sum = move(rotated);
                first_in_ct = false;
            } else {
                sum = ctx.add(sum, rotated);
            }
        }
        out_data[out_ct_idx] = ctx.rescale(sum, ctx.get_parameter().get_default_scale());
    }

    Feature2DEncrypted result(&ctx, level - 1);
    result.data = move(out_data);
    result.n_channel = total_channels;
    result.n_channel_per_ct = n_channel_per_ct;
    result.shape[0] = shape[0];
    result.shape[1] = shape[1];
    result.skip[0] = skip[0];
    result.skip[1] = skip[1];
    return result;
}

void ConcatLayer::prepare_mask_data(const CkksParameter& param,
                                    const vector<uint32_t>& input_n_channels,
                                    uint32_t n_channel_per_ct,
                                    Duo shape,
                                    Duo skip,
                                    int level) {
    CkksContext ctx = CkksContext::create_empty_context(param);
    uint32_t n_channel_per_block = prod(skip);
    uint32_t block_size = prod(shape * skip);
    uint32_t N = param.get_n();

    uint64_t total_channels = 0;
    for (auto n_ch : input_n_channels)
        total_channels += n_ch;

    mask_pt.clear();
    mask_pt.resize(total_channels);

    for (uint64_t global_ch = 0; global_ch < total_channels; ++global_ch) {
        uint64_t local_ch = global_ch;
        for (auto n_ch : input_n_channels) {
            if (local_ch < n_ch)
                break;
            local_ch -= n_ch;
        }

        uint32_t src_channel_in_ct = local_ch % n_channel_per_ct;
        uint32_t src_block = src_channel_in_ct / n_channel_per_block;
        uint32_t src_offset = src_channel_in_ct % n_channel_per_block;
        uint32_t src_cx = src_offset / skip[1];
        uint32_t src_cy = src_offset % skip[1];

        vector<double> mask_vec(N / 2, 0.0);
        for (uint32_t x = 0; x < shape[0]; ++x) {
            for (uint32_t y = 0; y < shape[1]; ++y) {
                uint32_t slot =
                    src_block * block_size + (x * skip[0] + src_cx) * (shape[1] * skip[1]) + (y * skip[1] + src_cy);
                mask_vec[slot] = 1.0;
            }
        }

        mask_pt[global_ch] = ctx.encode_ringt(mask_vec, ctx.get_parameter().get_q(level));
    }
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
