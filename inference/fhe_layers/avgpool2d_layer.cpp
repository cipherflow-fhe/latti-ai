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

#include "avgpool2d_layer.h"

using namespace std;
using namespace cxx_sdk_v2;

Avgpool2DLayer::Avgpool2DLayer(const Duo& shape_in, const Duo& stride_in) : n_block_per_ct(0) {
    shape[0] = shape_in[0];
    shape[1] = shape_in[1];
    stride[0] = stride_in[0];
    stride[1] = stride_in[1];

    if ((shape[0] & (shape[0] - 1)) != 0 || (shape[1] & (shape[1] - 1)) != 0) {
        throw std::invalid_argument("shape must be powers of 2, got: [" + std::to_string(shape[0]) + ", " +
                                    std::to_string(shape[1]) + "]");
    }
    if ((stride[0] & (stride[0] - 1)) != 0 || (stride[1] & (stride[1] - 1)) != 0) {
        throw std::invalid_argument("stride must be powers of 2, got: [" + std::to_string(stride[0]) + ", " +
                                    std::to_string(stride[1]) + "]");
    }
}

Feature2DEncrypted Avgpool2DLayer::run(CkksContext& ctx, const Feature2DEncrypted& x) {
    Feature2DEncrypted result(&ctx, x.level);
    int x_size = x.data.size();
    result.data.resize(x_size);
    parallel_for(x_size, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        CkksCiphertext y = x.data[ct_idx].copy();
        for (int j = 1; j < stride[0]; j++) {
            y = ctx_copy.add(y, ctx_copy.rotate(x.data[ct_idx], j * shape[0]));
        }
        int step = stride[0];
        CkksCiphertext r;
        while (step > 1) {
            r = ctx_copy.rotate(y, step / 2);
            y = ctx_copy.add(y, r);
            step = step / 2;
        }
        result.data[ct_idx] = move(y);
    });

    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.shape[0] = x.shape[0] / stride[0];
    result.shape[1] = x.shape[1] / stride[1];
    result.skip[0] = x.skip[0] * stride[0];
    result.skip[1] = x.skip[1] * stride[1];
    result.level = x.level;
    return result;
}

Feature2DEncrypted Avgpool2DLayer::run_adaptive_avgpool(CkksContext& ctx, const Feature2DEncrypted& x) {
    Feature2DEncrypted result(&ctx, x.level);
    int x_size = x.data.size();
    result.data.resize(x_size);
    Duo skip = x.skip;
    Duo shape = x.shape;
    int n_rot = (ctx.get_parameter().get_n() / 2) / (x.n_channel * x.shape[0] * x.shape[1]);

    int log2_stride_0 = static_cast<int>(std::ceil(std::log2(stride[0])));
    int log2_stride_1 = static_cast<int>(std::ceil(std::log2(stride[1])));
    parallel_for(x_size, th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        result.data[idx] = x.data[idx].copy();
        for (int i = log2_stride_0 - 1; i >= 0; --i) {
            auto ct_tmp = ctx_copy.rotate(result.data[idx], pow(2, i) * shape[0] * skip[0] * skip[1]);
            result.data[idx] = ctx_copy.add(result.data[idx], ct_tmp);
        }
        for (int j = log2_stride_1 - 1; j >= 0; --j) {
            auto ct_tmp = ctx_copy.rotate(result.data[idx], pow(2, j) * skip[1]);
            result.data[idx] = ctx_copy.add(result.data[idx], ct_tmp);
        }
        int n_rot_iters = (n_rot > 1) ? static_cast<int>(std::floor(std::log2(n_rot))) : 0;
        for (int r = 0; r < n_rot_iters; r++) {
            result.data[idx] = ctx_copy.add(
                result.data[idx], ctx_copy.rotate(result.data[idx], pow(2, r) * x.n_channel * x.shape[0] * x.shape[1]));
        }
    });
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.skip[0] = x.skip[0] * stride[0];
    result.skip[1] = x.skip[1] * stride[1];
    result.invalid_fill[0] = x.invalid_fill[0] * stride[0];
    result.invalid_fill[1] = x.invalid_fill[1] * stride[1];
    result.shape[0] = x.shape[0] / stride[0];
    result.shape[1] = x.shape[1] / stride[1];
    result.level = x.level;
    return result;
}

vector<double> Avgpool2DLayer::select_tensor(int num) {
    vector<double> tensor;
    for (int k = 0; k < n_block_per_ct; k++) {
        for (int i = 0; i < shape[0] * skip[0]; i++) {
            for (int j = 0; j < shape[1] * skip[1]; j++) {
                if (k * skip[0] * skip[1] * stride[0] * stride[1] + stride[1] * skip[1] * (i % (skip[0] * stride[0])) +
                        (j % (stride[1] * skip[1])) ==
                    num) {
                    tensor.push_back(1.0 / (static_cast<double>(stride[0] * stride[1])));
                } else {
                    tensor.push_back(0.0);
                }
            }
        }
    }
    return tensor;
}

void Avgpool2DLayer::prepare_weight(const CkksParameter& param_in,
                                    int n_channel_per_ct,
                                    int n_channel,
                                    int level,
                                    const Duo& skip_in,
                                    const Duo& shape_in) {
    CkksContext ctx = CkksContext::create_empty_context(param_in);
    skip = skip_in;
    n_block_per_ct = div_ceil(n_channel_per_ct, (skip[0] * skip[1]));
    shape = shape_in;
    level_ = level;
    uint32_t out_channels_per_ct = n_channel_per_ct * stride[0] * stride[1];
    uint32_t n_select_pt = std::min((uint32_t)n_channel, out_channels_per_ct);
    select_tensor_pt.clear();
    select_tensor_pt.resize(n_select_pt);
    for (uint32_t i = 0; i < n_select_pt; i++) {
        vector<double> si = select_tensor(i);
        CkksPlaintextRingt p_st = ctx.encode_ringt(si, ctx.get_parameter().get_q(level));
        select_tensor_pt[i] = move(p_st);
    }
}

Feature2DEncrypted Avgpool2DLayer::run_multiplexed_avgpool(CkksContext& ctx, const Feature2DEncrypted& x) {
    uint32_t x_size = x.data.size();
    vector<CkksCiphertext> result_ct;
    result_ct.resize(x_size);
    vector<CkksCiphertext> result_tmp;
    result_tmp.resize(x.n_channel);

    uint32_t n_packed_out_channel = div_ceil(x.n_channel, x.n_channel_per_ct * stride[0] * stride[1]);
    uint32_t log2_stride_0 = static_cast<int>(std::ceil(std::log2(stride[0])));
    uint32_t log2_stride_1 = static_cast<int>(std::ceil(std::log2(stride[1])));

    parallel_for(x_size, th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        result_ct[idx] = x.data[idx].copy();
        for (int i = log2_stride_0 - 1; i >= 0; --i) {
            cxx_sdk_v2::CkksCiphertext ct_tmp =
                ctx_copy.rotate(result_ct[idx], pow(2, i) * shape[1] * skip[0] * skip[1]);
            result_ct[idx] = ctx_copy.add(result_ct[idx], ct_tmp);
        }
        for (int j = log2_stride_1 - 1; j >= 0; --j) {
            cxx_sdk_v2::CkksCiphertext ct_tmp = ctx_copy.rotate(result_ct[idx], pow(2, j) * skip[1]);
            result_ct[idx] = ctx_copy.add(result_ct[idx], ct_tmp);
        }
        vector<int32_t> steps;
        uint32_t n_valid = std::min(x.n_channel_per_ct, x.n_channel - idx * x.n_channel_per_ct);
        for (uint32_t i = 0; i < n_valid; i++) {
            int32_t rp = (idx * x.n_channel_per_ct + i) % (x.n_channel_per_ct * stride[0] * stride[1]);
            int32_t r_num0 =
                floor(rp / (skip[0] * skip[1] * stride[0] * stride[1])) * skip[0] * skip[1] * shape[0] * shape[1];
            int32_t r_num1 =
                floor((rp % (skip[0] * skip[1] * stride[0] * stride[1])) / (stride[1] * skip[1])) * shape[1] * skip[1];
            int32_t r_num2 = rp % (skip[1] * stride[1]);

            int32_t lp = (idx * x.n_channel_per_ct + i) % (x.n_channel_per_ct);
            int32_t l_num0 = floor(lp / (skip[0] * skip[1])) * skip[0] * skip[1] * shape[0] * shape[1];
            int32_t l_num1 = floor((lp % (skip[0] * skip[1])) / skip[1]) * shape[1] * skip[1];
            int32_t l_num2 = lp % skip[1];

            int32_t r_num = -r_num0 - r_num1 - r_num2 + l_num0 + l_num1 + l_num2;
            steps.push_back(r_num);
        }
        std::map<int32_t, cxx_sdk_v2::CkksCiphertext> s_rots = ctx_copy.rotate(result_ct[idx], steps);
        for (uint32_t i = 0; i < n_valid; i++) {
            int out_channel_pos = (idx * x.n_channel_per_ct + i) % (x.n_channel_per_ct * stride[0] * stride[1]);
            auto& pt_ringt = select_tensor_pt[out_channel_pos];
            auto pt = ctx_copy.ringt_to_mul(pt_ringt, level_);
            cxx_sdk_v2::CkksCiphertext c_m_s = ctx_copy.mult_plain_mul(s_rots[steps[i]], pt);
            result_tmp[idx * x.n_channel_per_ct + i] =
                move(ctx_copy.rescale(c_m_s, ctx_copy.get_parameter().get_default_scale()));
        }
    });
    vector<CkksCiphertext> res;
    res.reserve(n_packed_out_channel);
    CkksCiphertext sp;
    for (int i = 0; i < x.n_channel; i++) {
        int p = i % (stride[0] * stride[1] * x.n_channel_per_ct);
        cxx_sdk_v2::CkksCiphertext c_m_s = result_tmp[i].copy();
        if (p == 0) {
            sp = move(c_m_s);
        } else {
            sp = ctx.add(sp, c_m_s);
        }
        if ((i + 1) % (stride[0] * stride[1] * x.n_channel_per_ct) == 0 || i == result_tmp.size() - 1) {
            res.push_back(move(sp));
        }
    }
    Feature2DEncrypted result(&ctx, x.level);
    result.data = move(res);
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct * stride[0] * stride[1];
    result.skip[0] = x.skip[0] * stride[0];
    result.skip[1] = x.skip[1] * stride[1];
    result.shape[0] = x.shape[0] / stride[0];
    result.shape[1] = x.shape[1] / stride[1];
    result.level = x.level - 1;
    return result;
}

Feature2DEncrypted
Avgpool2DLayer::run_split_avgpool(CkksContext& ctx, const Feature2DEncrypted& x, const Duo block_expansion) {
    uint32_t x_size = x.data.size();
    uint32_t out_size = x_size / (stride[0] * stride[1]);
    vector<CkksCiphertext> res;
    res.resize(out_size);

    parallel_for(x.n_channel, th_nums, ctx, [&](CkksContext& ctx_copy, int channel_idx) {
        int base_idx = channel_idx * (block_expansion[0] / stride[0]) * (block_expansion[1] / stride[1]);
        for (int row_idx = 0; row_idx < block_expansion[0]; row_idx++) {
            for (int col_idx = 0; col_idx < block_expansion[1]; col_idx++) {
                int ct_idx =
                    channel_idx * block_expansion[0] * block_expansion[1] + row_idx * block_expansion[1] + col_idx;
                int out_idx = base_idx + (row_idx / stride[0]) * (block_expansion[1] / stride[1]) + col_idx / stride[1];
                if (row_idx % stride[0] == 0 && col_idx % stride[1] == 0) {
                    res[out_idx] = x.data[ct_idx].copy();
                } else {
                    res[out_idx] = ctx_copy.add(res[out_idx], x.data[ct_idx]);
                }
            }
        }
    });

    int N = ctx.get_parameter().get_n();
    int output_h = shape[0] / stride[0];
    int output_w = shape[1] / stride[1];

    int n_channel_per_ct_out;
    if (2 * output_h * output_w < N) {
        n_channel_per_ct_out = N / (2 * output_h * output_w);
    } else {
        n_channel_per_ct_out = 1;
    }

    vector<CkksCiphertext> packed_res(out_size / n_channel_per_ct_out);
    if (n_channel_per_ct_out == 1) {
        packed_res = move(res);
    } else {
        for (uint32_t out_ct_idx = 0; out_ct_idx < out_size; out_ct_idx++) {
            int pack_out_ct_idx = out_ct_idx / n_channel_per_ct_out;
            int channel_idx_in_ct = out_ct_idx % n_channel_per_ct_out;
            if (channel_idx_in_ct == 0) {
                packed_res[pack_out_ct_idx] = move(res[out_ct_idx]);
            } else {
                long step = -1 * channel_idx_in_ct * output_h * output_w;
                auto s_rot = ctx.rotate(res[out_ct_idx], step);
                packed_res[pack_out_ct_idx] = ctx.add(packed_res[pack_out_ct_idx], move(s_rot));
            }
        }
    }

    Feature2DEncrypted result(&ctx, x.level);
    result.data = move(packed_res);
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = n_channel_per_ct_out;
    result.skip[0] = 1;
    result.skip[1] = 1;
    result.shape[0] = output_h;
    result.shape[1] = output_w;
    result.level = x.level;
    return result;
}

Array<double, 3> Avgpool2DLayer::plaintext_call(const Array<double, 3>& x) {
    std::array<uint64_t, 3UL> input_shape = x.get_shape();
    uint64_t output_height = input_shape[1] / stride[0];
    uint64_t output_width = input_shape[2] / stride[1];
    Array<double, 3> result({input_shape[0], output_height, output_width});
    for (int idx = 0; idx < input_shape[0]; idx++) {
        vector<vector<double>> output(output_height, vector<double>(output_width, 0.0));
        for (int i = 0; i < output_height; i++) {
            for (int j = 0; j < output_width; j++) {
                double sum = 0.0;
                for (int m = i * stride[0]; m < (i + 1) * stride[0]; m++) {
                    for (int n = j * stride[1]; n < (j + 1) * stride[1]; n++) {
                        sum += x.get(idx, m, n);
                    }
                }
                result.set(idx, i, j, sum);
            }
        }
    }
    return result;
}

Array<double, 3> Avgpool2DLayer::plaintext_call_multiplexed(const Array<double, 3>& x) {
    std::array<uint64_t, 3UL> input_shape = x.get_shape();
    uint64_t output_height = input_shape[1] / stride[0];
    uint64_t output_width = input_shape[2] / stride[1];
    Array<double, 3> result({input_shape[0], output_height, output_width});
    for (int idx = 0; idx < input_shape[0]; idx++) {
        vector<vector<double>> output(output_height, vector<double>(output_width, 0.0));
        for (int i = 0; i < output_height; i++) {
            for (int j = 0; j < output_width; j++) {
                double sum = 0.0;
                for (int m = i * stride[0]; m < (i + 1) * stride[0]; m++) {
                    for (int n = j * stride[1]; n < (j + 1) * stride[1]; n++) {
                        sum += x.get(idx, m, n) / (static_cast<double>(stride[0] * stride[1]));
                    }
                }
                result.set(idx, i, j, sum);
            }
        }
    }
    return result;
}
