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

#include "avgpool1d_layer.h"

using namespace std;
using namespace lattisense;

Avgpool1DLayer::Avgpool1DLayer(uint32_t shape_in, uint32_t stride_in) : n_block_per_ct(0), skip(0) {
    shape = shape_in;
    stride = stride_in;

    if ((shape & (shape - 1)) != 0) {
        throw std::invalid_argument("shape must be a power of 2, got: " + std::to_string(shape));
    }
    if ((stride & (stride - 1)) != 0) {
        throw std::invalid_argument("stride must be a power of 2, got: " + std::to_string(stride));
    }
}

Feature1DEncrypted Avgpool1DLayer::run_adaptive_avgpool(CkksContext& ctx, const Feature1DEncrypted& x) {
    Feature1DEncrypted result(&ctx, x.level);
    int x_size = x.data.size();
    result.data.resize(x_size);
    uint32_t skip = x.skip;
    uint32_t shape = x.shape;
    int n_rot = (ctx.get_parameter().get_n() / 2) / (x.n_channel * x.shape);

    int log2_stride = static_cast<int>(std::ceil(std::log2(stride)));
    parallel_for(x_size, th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        result.data[idx] = x.data[idx].copy();
        for (int i = log2_stride - 1; i >= 0; --i) {
            auto ct_tmp = ctx_copy.rotate(result.data[idx], pow(2, i) * skip);
            result.data[idx] = ctx_copy.add(result.data[idx], ct_tmp);
        }
        int n_rot_iters = (n_rot > 1) ? static_cast<int>(std::floor(std::log2(n_rot))) : 0;
        for (int r = 0; r < n_rot_iters; r++) {
            result.data[idx] =
                ctx_copy.add(result.data[idx], ctx_copy.rotate(result.data[idx], pow(2, r) * x.n_channel * x.shape));
        }
    });
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.skip = x.skip * stride;
    result.invalid_fill = x.invalid_fill * stride;
    result.shape = x.shape / stride;
    result.level = x.level;
    return result;
}

vector<double> Avgpool1DLayer::select_tensor(int num) const {
    vector<double> tensor;
    for (int k = 0; k < n_block_per_ct; k++) {
        for (int i = 0; i < shape * skip; i++) {
            if (k * skip * stride + (i % (stride * skip)) == num) {
                tensor.push_back(1.0 / (static_cast<double>(stride)));
            } else {
                tensor.push_back(0.0);
            }
        }
    }
    return tensor;
}

void Avgpool1DLayer::prepare_weight_lazy(const CkksParameter& param_in,
                                         int n_channel_per_ct,
                                         int n_channel,
                                         int level,
                                         uint32_t skip_in,
                                         uint32_t shape_in) {
    skip = skip_in;
    n_channel_per_ct_ = n_channel_per_ct;
    n_block_per_ct = div_ceil(n_channel_per_ct, skip);
    shape = shape_in;
    level_ = level;
    n_channel_ = n_channel;
}

CkksPlaintextRingt Avgpool1DLayer::generate_select_tensor_pt_for_index(CkksContext& ctx, int i) const {
    vector<double> si = select_tensor(i);
    return ctx.encode_ringt(si, ctx.get_parameter().get_q(level_));
}

uint32_t Avgpool1DLayer::num_select_tensors() const {
    return std::min(n_channel_, n_channel_per_ct_ * stride);
}

void Avgpool1DLayer::prepare_weight(const CkksParameter& param_in,
                                    int n_channel_per_ct,
                                    int n_channel,
                                    int level,
                                    uint32_t skip_in,
                                    uint32_t shape_in) {
    CkksContext ctx = CkksContext::create_empty_context(param_in);
    skip = skip_in;
    n_channel_per_ct_ = n_channel_per_ct;
    n_block_per_ct = div_ceil(n_channel_per_ct, skip);
    shape = shape_in;
    level_ = level;
    n_channel_ = n_channel;
    uint32_t out_channels_per_ct = n_channel_per_ct * stride;
    uint32_t n_select_pt = std::min((uint32_t)n_channel, out_channels_per_ct);
    select_tensor_pt.clear();
    select_tensor_pt.resize(n_select_pt);
    for (uint32_t i = 0; i < n_select_pt; i++) {
        vector<double> si = select_tensor(i);
        CkksPlaintextRingt p_st = ctx.encode_ringt(si, ctx.get_parameter().get_q(level));
        select_tensor_pt[i] = move(p_st);
    }
}

Feature1DEncrypted Avgpool1DLayer::run_multiplexed_avgpool(CkksContext& ctx, const Feature1DEncrypted& x) {
    uint32_t x_size = x.data.size();
    vector<CkksCiphertext> result_ct;
    result_ct.resize(x_size);
    vector<CkksCiphertext> result_tmp;
    result_tmp.resize(x.n_channel);

    uint32_t n_packed_out_channel = div_ceil(x.n_channel, x.n_channel_per_ct * stride);
    uint32_t log2_stride = static_cast<int>(std::ceil(std::log2(stride)));

    parallel_for(x_size, th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        result_ct[idx] = x.data[idx].copy();
        for (int i = log2_stride - 1; i >= 0; --i) {
            lattisense::CkksCiphertext ct_tmp = ctx_copy.rotate(result_ct[idx], pow(2, i) * skip);
            result_ct[idx] = ctx_copy.add(result_ct[idx], ct_tmp);
        }
        vector<int32_t> steps;
        uint32_t n_valid = std::min(x.n_channel_per_ct, x.n_channel - idx * x.n_channel_per_ct);
        for (uint32_t i = 0; i < n_valid; i++) {
            int32_t rp = (idx * x.n_channel_per_ct + i) % (x.n_channel_per_ct * stride);
            int32_t r_num0 = floor(rp / (skip * stride)) * skip * shape;
            int32_t r_num1 = rp % (skip * stride);

            int32_t lp = (idx * x.n_channel_per_ct + i) % (x.n_channel_per_ct);
            int32_t l_num0 = floor(lp / skip) * skip * shape;
            int32_t l_num1 = lp % skip;

            int32_t r_num = -r_num0 - r_num1 + l_num0 + l_num1;
            steps.push_back(r_num);
        }
        std::map<int32_t, lattisense::CkksCiphertext> s_rots = ctx_copy.rotate(result_ct[idx], steps);
        for (uint32_t i = 0; i < n_valid; i++) {
            int out_channel_pos = (idx * x.n_channel_per_ct + i) % (x.n_channel_per_ct * stride);
            auto& pt_ringt = select_tensor_pt[out_channel_pos];
            auto pt = ctx_copy.ringt_to_mul(pt_ringt, level_);
            lattisense::CkksCiphertext c_m_s = ctx_copy.mult_plain_mul(s_rots[steps[i]], pt);
            result_tmp[idx * x.n_channel_per_ct + i] =
                move(ctx_copy.rescale(c_m_s, ctx_copy.get_parameter().get_default_scale()));
        }
    });
    vector<CkksCiphertext> res;
    res.reserve(n_packed_out_channel);
    CkksCiphertext sp;
    for (int i = 0; i < x.n_channel; i++) {
        int p = i % (stride * x.n_channel_per_ct);
        lattisense::CkksCiphertext c_m_s = result_tmp[i].copy();
        if (p == 0) {
            sp = move(c_m_s);
        } else {
            sp = ctx.add(sp, c_m_s);
        }
        if ((i + 1) % (stride * x.n_channel_per_ct) == 0 || i == result_tmp.size() - 1) {
            res.push_back(move(sp));
        }
    }
    Feature1DEncrypted result(&ctx, x.level);
    result.data = move(res);
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct * stride;
    result.skip = x.skip * stride;
    result.shape = x.shape / stride;
    result.level = x.level - 1;
    return result;
}

Array<double, 2> Avgpool1DLayer::run_plaintext(const Array<double, 2>& x) {
    std::array<uint64_t, 2UL> input_shape = x.get_shape();
    uint64_t output_length = input_shape[1] / stride;
    Array<double, 2> result({input_shape[0], output_length});
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
    for (int idx = 0; idx < input_shape[0]; idx++) {
        for (int i = 0; i < output_length; i++) {
            double sum = 0.0;
            for (int m = i * stride; m < (i + 1) * stride; m++) {
                sum += x.get(idx, m);
            }
            result.set(idx, i, sum);
        }
    }
    return result;
}

Array<double, 2> Avgpool1DLayer::run_plaintext_multiplexed(const Array<double, 2>& x) {
    std::array<uint64_t, 2UL> input_shape = x.get_shape();
    uint64_t output_length = input_shape[1] / stride;
    Array<double, 2> result({input_shape[0], output_length});
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
    for (int idx = 0; idx < input_shape[0]; idx++) {
        for (int i = 0; i < output_length; i++) {
            double sum = 0.0;
            for (int m = i * stride; m < (i + 1) * stride; m++) {
                sum += x.get(idx, m) / (static_cast<double>(stride));
            }
            result.set(idx, i, sum);
        }
    }
    return result;
}
