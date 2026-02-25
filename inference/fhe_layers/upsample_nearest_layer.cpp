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

#include "upsample_nearest_layer.h"

using namespace std;

UpsampleNearestLayer::UpsampleNearestLayer(const CkksParameter& param_in,
                                           const Duo& shape_in,
                                           const Duo& skip_in,
                                           const Duo& upsample_factor_in,
                                           const uint32_t& n_channel_per_ct_in,
                                           const uint32_t& level_in)
    : param(param_in.copy()), cached_block_size(0), cached_skip_div_upsample_0(0), cached_skip_div_upsample_1(0) {
    upsample_factor[0] = upsample_factor_in[0];
    upsample_factor[1] = upsample_factor_in[1];
    shape[0] = shape_in[0];
    shape[1] = shape_in[1];
    skip[0] = skip_in[0];
    skip[1] = skip_in[1];

    if ((shape[0] & (shape[0] - 1)) != 0 || (shape[1] & (shape[1] - 1)) != 0) {
        throw std::invalid_argument("shape must be powers of 2, got: ["
                                    + std::to_string(shape[0]) + ", " + std::to_string(shape[1]) + "]");
    }
    if ((skip[0] & (skip[0] - 1)) != 0 || (skip[1] & (skip[1] - 1)) != 0) {
        throw std::invalid_argument("skip must be powers of 2, got: ["
                                    + std::to_string(skip[0]) + ", " + std::to_string(skip[1]) + "]");
    }

    n_channel_per_ct = n_channel_per_ct_in;
    level = level_in;
    n_block_per_ct = div_ceil(n_channel_per_ct, (skip[0] * skip[1]));
}

vector<double> UpsampleNearestLayer::select_tensor(int num) const {
    unsigned int block_size = shape[0] * shape[1] * skip[0] * skip[1];
    vector<double> tensor(n_block_per_ct * block_size, 0.0);
    for (int k = 0; k < n_block_per_ct; k++) {
        for (int i = 0; i < shape[0] * skip[0]; i++) {
            for (int j = 0; j < shape[1] * skip[1]; j++) {
                if ((i % skip[0]) < skip[0] / upsample_factor[0] && (j % skip[1]) < skip[1] / upsample_factor[1] &&
                    k * skip[0] / upsample_factor[0] * skip[1] / upsample_factor[1] +
                            skip[1] / upsample_factor[1] * (i % (skip[0] / upsample_factor[0])) +
                            j % (skip[1] / upsample_factor[1]) ==
                        num) {
                    int index = k * block_size + i * shape[1] * skip[1] + j;
                    tensor[index] = 1.0;
                }
            }
        }
    }
    return tensor;
}

void UpsampleNearestLayer::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(this->param);
    select_tensor_pt.clear();
    select_tensor_pt.resize(n_channel_per_ct / (upsample_factor[0] * upsample_factor[1]));
    for (int i = 0; i < n_channel_per_ct / (upsample_factor[0] * upsample_factor[1]); i++) {
        vector<double> si = select_tensor(i);
        CkksPlaintextRingt p_si = ctx.encode_ringt(si, ctx.get_parameter().get_q(level));
        select_tensor_pt[i] = move(p_si);
    }
}

void UpsampleNearestLayer::prepare_weight_lazy() {
    // Cache commonly used values for on-demand generation
    cached_block_size = shape[0] * shape[1] * skip[0] * skip[1];
    cached_skip_div_upsample_0 = skip[0] / upsample_factor[0];
    cached_skip_div_upsample_1 = skip[1] / upsample_factor[1];

    // Clear the select_tensor_pt to indicate lazy mode
    select_tensor_pt.clear();
}

CkksPlaintextRingt UpsampleNearestLayer::generate_select_tensor_pt_for_index(CkksContext& ctx, int idx) const {
    // Generate select_tensor on-demand
    auto si = select_tensor(idx);
    return ctx.encode_ringt(si, ctx.get_parameter().get_q(level));
}

Feature2DEncrypted UpsampleNearestLayer::run(CkksContext& ctx, const Feature2DEncrypted& x) {
    uint32_t x_size = x.data.size();
    vector<CkksCiphertext> x_data_cpy(x_size);
    vector<CkksCiphertext> result_tmp(x.n_channel);

    uint32_t n_packed_out_channel =
        div_ceil(x.n_channel, x.n_channel_per_ct / (upsample_factor[0] * upsample_factor[1]));

    for (uint32_t idx = 0; idx < x_size; idx++) {
        x_data_cpy[idx] = x.data[idx].copy();
    }

    parallel_for(x_size, th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        vector<int32_t> steps(x.n_channel_per_ct);
        for (int i = 0; i < x.n_channel_per_ct; i++) {
            int32_t rp =
                (idx * x.n_channel_per_ct + i) % (x.n_channel_per_ct / (upsample_factor[0] * upsample_factor[1]));
            int32_t r_num0 = floor(rp / (skip[0] * skip[1] / (upsample_factor[0] * upsample_factor[1]))) * skip[0] *
                             skip[1] * shape[0] * shape[1];
            int32_t r_num1 = floor((rp % (skip[0] * skip[1] / (upsample_factor[0] * upsample_factor[1]))) /
                                   (skip[1] / upsample_factor[1])) *
                             shape[1] * skip[1];
            int32_t r_num2 = rp % (skip[1] / upsample_factor[1]);

            int32_t lp = (idx * x.n_channel_per_ct + i) % (x.n_channel_per_ct);
            int32_t l_num0 = floor(lp / (skip[0] * skip[1])) * skip[0] * skip[1] * shape[0] * shape[1];
            int32_t l_num1 = floor((lp % (skip[0] * skip[1])) / skip[1]) * shape[1] * skip[1];
            int32_t l_num2 = lp % skip[1];

            int32_t r_num = -r_num0 - r_num1 - r_num2 + l_num0 + l_num1 + l_num2;
            steps[i] = r_num;
        }
        std::map<int32_t, cxx_sdk_v2::CkksCiphertext> s_rots = ctx_copy.rotate(x_data_cpy[idx], steps);

        for (int i = 0; i < x.n_channel_per_ct; i++) {
            int out_channel_pos =
                (idx * x.n_channel_per_ct + i) % (x.n_channel_per_ct / (upsample_factor[0] * upsample_factor[1]));

            cxx_sdk_v2::CkksCiphertext c_m_s;
            // Lazy mode: generate plaintext on-demand if select_tensor_pt is empty
            if (select_tensor_pt.empty()) {
                auto pt_ringt = generate_select_tensor_pt_for_index(ctx_copy, out_channel_pos);
                auto pt = ctx_copy.ringt_to_mul(pt_ringt, level);
                c_m_s = ctx_copy.mult_plain_mul(s_rots[steps[i]], pt);
            } else {
                auto& pt_ringt = select_tensor_pt[out_channel_pos];
                auto pt = ctx_copy.ringt_to_mul(pt_ringt, level);
                c_m_s = ctx_copy.mult_plain_mul(s_rots[steps[i]], pt);
            }

            if ((idx * x.n_channel_per_ct + i) < x.n_channel) {
                result_tmp[idx * x.n_channel_per_ct + i] =
                    move(ctx_copy.rescale(c_m_s, ctx_copy.get_parameter().get_default_scale()));
            }
        }
    });
    vector<CkksCiphertext> res;
    res.reserve(n_packed_out_channel);
    CkksCiphertext sp;
    for (int i = 0; i < x.n_channel; i++) {
        int p = i % (x.n_channel_per_ct / (upsample_factor[0] * upsample_factor[1]));
        cxx_sdk_v2::CkksCiphertext c_m_s = result_tmp[i].copy();
        if (p == 0) {
            sp = move(c_m_s);
        } else {
            sp = ctx.add(sp, c_m_s);
        }
        if ((i + 1) % (x.n_channel_per_ct / (upsample_factor[0] * upsample_factor[1])) == 0 ||
            i == result_tmp.size() - 1) {
            res.push_back(move(sp));
        }
    }

    uint32_t res_size = res.size();
    vector<CkksCiphertext> result_ct(res_size);

    for (uint32_t idx = 0; idx < res_size; idx++) {
        result_ct[idx] = res[idx].copy();
    }

    uint32_t log2_upsample_0 = static_cast<int>(std::ceil(std::log2(upsample_factor[0])));
    uint32_t log2_upsample_1 = static_cast<int>(std::ceil(std::log2(upsample_factor[1])));

    parallel_for(res_size, th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        for (int i = 0; i < log2_upsample_0; i++) {
            cxx_sdk_v2::CkksCiphertext ct_tmp =
                ctx_copy.rotate(result_ct[idx], pow(2, i) * shape[1] * skip[1] * skip[0] / upsample_factor[0] * -1);
            result_ct[idx] = ctx_copy.add(result_ct[idx], move(ct_tmp));
        }
        for (int j = 0; j < log2_upsample_1; j++) {
            cxx_sdk_v2::CkksCiphertext ct_tmp =
                ctx_copy.rotate(result_ct[idx], pow(2, j) * skip[1] / upsample_factor[1] * -1);
            result_ct[idx] = ctx_copy.add(result_ct[idx], move(ct_tmp));
        }
    });
    Feature2DEncrypted result(&ctx, x.level - 1);
    result.data = move(result_ct);
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct / (upsample_factor[0] * upsample_factor[1]);
    result.skip[0] = x.skip[0] / upsample_factor[0];
    result.skip[1] = x.skip[1] / upsample_factor[1];
    result.shape[0] = x.shape[0] * upsample_factor[0];
    result.shape[1] = x.shape[1] * upsample_factor[1];
    result.level = x.level - 1;
    return result;
}

Array<double, 3> UpsampleNearestLayer::run_plaintext(const Array<double, 3>& x) {
    std::array<uint64_t, 3UL> input_shape = x.get_shape();
    uint64_t output_height = input_shape[1] * upsample_factor[0];
    uint64_t output_width = input_shape[2] * upsample_factor[1];
    Array<double, 3> result({input_shape[0], output_height, output_width});
    for (uint64_t idx = 0; idx < input_shape[0]; idx++) {
        for (uint64_t i = 0; i < input_shape[1]; i++) {
            for (uint64_t j = 0; j < input_shape[2]; j++) {
                for (uint64_t i_t = 0; i_t < upsample_factor[0]; i_t++) {
                    for (uint64_t j_t = 0; j_t < upsample_factor[1]; j_t++) {
                        result.set(idx, i * upsample_factor[0] + i_t, j * upsample_factor[1] + j_t, x.get(idx, i, j));
                    }
                }
            }
        }
    }
    return result;
}
