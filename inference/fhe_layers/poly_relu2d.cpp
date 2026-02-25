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

#include "poly_relu2d.h"

PolyRelu::PolyRelu(const CkksParameter& param_in,
                   const Duo& input_shape_in,
                   const int order_in,
                   const Array<double, 2>& weight_in,
                   const Duo& skip_in,
                   uint32_t n_channel_per_ct_in,
                   uint32_t level_in,
                   const Duo& upsample_factor_in,
                   const Duo& block_expansion_in,
                   bool is_ordinary_pack_in)
    : param(param_in.copy()), input_shape(input_shape_in), weight(weight_in.copy()), skip(skip_in) {
    upsample_factor[0] = upsample_factor_in[0];
    upsample_factor[1] = upsample_factor_in[1];
    order = order_in;
    level = level_in;
    n_channel_per_ct = n_channel_per_ct_in;
    n_block_per_ct = std::ceil(n_channel_per_ct / (skip[0] * skip[1]));
    pre_skip[0] = skip[0] * upsample_factor[0];
    pre_skip[1] = skip[1] * upsample_factor[1];
    block_expansion[0] = block_expansion_in[0];
    block_expansion[1] = block_expansion_in[1];
    block_shape[0] = input_shape[0] * skip[0] / block_expansion[0];
    block_shape[1] = input_shape[1] * skip[1] / block_expansion[1];

    N = param_in.get_n();
    cached_skip_prod = skip[0] * skip[1];
    cached_channel = weight.get_shape()[1];
    cached_n_packed_out_channel = div_ceil(cached_channel, n_channel_per_ct) * block_expansion[0] * block_expansion[1];
    cached_total_block_size = n_block_per_ct * block_shape[0] * block_shape[1];
    is_ordinary_pack = is_ordinary_pack_in;
}

PolyRelu::~PolyRelu() {}

void PolyRelu::prepare_weight() {
    int skip_prod = skip[0] * skip[1];
    int channel = weight.get_shape()[1];
    int n_packed_out_channel = div_ceil(channel, n_channel_per_ct) * block_expansion[0] * block_expansion[1];
    weight_pt.resize(order);

    CkksContext ctx = CkksContext::create_empty_context(this->param);
    ctx.resize_copies(order);
    parallel_for(order, th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        for (int n_packed_out_channel_idx = 0; n_packed_out_channel_idx < n_packed_out_channel;
             n_packed_out_channel_idx++) {
            const int total_block_size = n_block_per_ct * block_shape[0] * block_shape[1];
            vector<double> feature_tmp_pack(ctx_copy.get_parameter().get_n() / 2);
            for (int linear_idx = 0; linear_idx < total_block_size; ++linear_idx) {
                int block_i = linear_idx / (block_shape[0] * block_shape[1]);
                int residual = linear_idx % (block_shape[0] * block_shape[1]);
                int shape_i = residual / block_shape[1];
                int shape_j = residual % block_shape[1];

                int channel_idx =
                    n_packed_out_channel_idx * n_channel_per_ct / block_expansion[0] / block_expansion[1] +
                    block_i * skip_prod + (skip[0] * (shape_i % skip[0]) + shape_j % skip[0]);
                if (channel_idx >= channel || (shape_i % pre_skip[0]) >= skip[0] || (shape_j % pre_skip[1]) >= skip[1])
                    continue;

                int index = block_i * block_shape[0] * block_shape[1] + shape_i * block_shape[1] + shape_j;
                feature_tmp_pack[index] = weight.get(idx, channel_idx);
            }

            double pack_scale = param.get_default_scale();
            if (idx == order - 1) {
                pack_scale = param.get_default_scale();
            } else {
                for (int k = 0; k < order - idx - 1; k++) {
                    pack_scale = pack_scale * param.get_default_scale() / param.get_q(level - k);
                }
            }
            weight_pt[idx].push_back(ctx_copy.encode_ringt(feature_tmp_pack, pack_scale));
        }
    });
}

void PolyRelu::prepare_weight_for_non_absorb_case() {
    int skip_prod = skip[0] * skip[1];
    int channel = weight.get_shape()[1];
    int n_packed_out_channel = div_ceil(channel, n_channel_per_ct) * block_expansion[0] * block_expansion[1];
    weight_pt.resize(order + 1);
    CkksContext ctx = CkksContext::create_empty_context(this->param);
    if (order != 4) {
        parallel_for(order + 1, th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
            for (int n_packed_out_channel_idx = 0; n_packed_out_channel_idx < n_packed_out_channel;
                 n_packed_out_channel_idx++) {
                const int total_block_size = n_block_per_ct * block_shape[0] * block_shape[1];
                vector<double> feature_tmp_pack(ctx_copy.get_parameter().get_n() / 2);
                if (is_ordinary_pack) {
                    for (int ch = 0; ch < (int)n_channel_per_ct; ch++) {
                        int channel_idx = n_packed_out_channel_idx * n_channel_per_ct + ch;
                        if (channel_idx >= channel)
                            continue;
                        for (int j = 0; j < input_shape[0]; j++) {
                            for (int k = 0; k < input_shape[1]; k++) {
                                int index =
                                    ch * block_shape[0] * block_shape[1] + j * block_shape[1] * skip[0] + k * skip[1];
                                feature_tmp_pack[index] = weight.get(idx, channel_idx);
                            }
                        }
                    }
                } else {
                    for (int linear_idx = 0; linear_idx < total_block_size; ++linear_idx) {
                        int block_i = linear_idx / (block_shape[0] * block_shape[1]);
                        int residual = linear_idx % (block_shape[0] * block_shape[1]);
                        int shape_i = residual / block_shape[1];
                        int shape_j = residual % block_shape[1];

                        int channel_idx =
                            n_packed_out_channel_idx * n_channel_per_ct / block_expansion[0] / block_expansion[1] +
                            block_i * skip_prod + (skip[0] * (shape_i % skip[0]) + shape_j % skip[0]);
                        if (channel_idx >= channel || (shape_i % pre_skip[0]) >= skip[0] ||
                            (shape_j % pre_skip[1]) >= skip[1])
                            continue;

                        int index = block_i * block_shape[0] * block_shape[1] + shape_i * block_shape[1] + shape_j;
                        feature_tmp_pack[index] = weight.get(idx, channel_idx);
                    }
                }

                double pack_scale = 1;
                if (idx == 0) {
                    pack_scale = pack_scale * param.get_default_scale();
                } else {
                    for (int k = 0; k < idx; k++) {
                        if (k == idx - 1) {
                            pack_scale = pack_scale * param.get_q(level - (order - k - 1));
                        } else {
                            pack_scale = param.get_q(level - (order - k - 1)) / param.get_default_scale() * pack_scale;
                        }
                    }
                }
                weight_pt[idx].push_back(ctx_copy.encode_ringt(feature_tmp_pack, pack_scale));
            }
        });
    } else {
        map<int, double> coeff_scale;
        coeff_scale[0] = param.get_default_scale();
        coeff_scale[1] = param.get_q(level - 2);
        coeff_scale[2] = param.get_q(level - 2) / coeff_scale[0] * param.get_q(level);
        coeff_scale[3] =
            param.get_q(level - 2) / coeff_scale[0] * param.get_q(level - 1) * param.get_q(level) / coeff_scale[0];
        coeff_scale[4] = param.get_q(level - 2) / coeff_scale[0] * param.get_q(level - 1) / coeff_scale[0] *
                         param.get_q(level) / coeff_scale[0] * param.get_q(level - 1);

        map<int, int> level_order;
        level_order[0] = level - 3;
        level_order[1] = level - 2;
        level_order[2] = level - 2;
        level_order[3] = level - 2;
        level_order[4] = level - 1;

        int baby_steps = (int)ceil(sqrt(order + 1));
        int giant_steps = (int)ceil((double)(order + 1) / baby_steps);
        parallel_for(order + 1, 1, ctx, [&](CkksContext& ctx_copy, int idx) {
            for (int n_packed_out_channel_idx = 0; n_packed_out_channel_idx < n_packed_out_channel;
                 n_packed_out_channel_idx++) {
                const int total_block_size = n_block_per_ct * block_shape[0] * block_shape[1];
                vector<double> feature_tmp_pack(ctx_copy.get_parameter().get_n() / 2);
                if (is_ordinary_pack) {
                    for (int ch = 0; ch < (int)n_channel_per_ct; ch++) {
                        int channel_idx = n_packed_out_channel_idx * n_channel_per_ct + ch;
                        if (channel_idx >= channel)
                            continue;
                        for (int j = 0; j < input_shape[0]; j++) {
                            for (int k = 0; k < input_shape[1]; k++) {
                                int index =
                                    ch * block_shape[0] * block_shape[1] + j * block_shape[1] * skip[0] + k * skip[1];
                                feature_tmp_pack[index] = weight.get(idx, channel_idx);
                            }
                        }
                    }
                } else {
                    for (int linear_idx = 0; linear_idx < total_block_size; ++linear_idx) {
                        int block_i = linear_idx / (block_shape[0] * block_shape[1]);
                        int residual = linear_idx % (block_shape[0] * block_shape[1]);
                        int shape_i = residual / block_shape[1];
                        int shape_j = residual % block_shape[1];

                        int channel_idx =
                            n_packed_out_channel_idx * n_channel_per_ct / block_expansion[0] / block_expansion[1] +
                            block_i * skip_prod + (skip[0] * (shape_i % skip[0]) + shape_j % skip[0]);
                        if (channel_idx >= channel || (shape_i % pre_skip[0]) >= skip[0] ||
                            (shape_j % pre_skip[1]) >= skip[1])
                            continue;

                        int index = block_i * block_shape[0] * block_shape[1] + shape_i * block_shape[1] + shape_j;
                        feature_tmp_pack[index] = weight.get(idx, channel_idx);
                    }
                }
                weight_pt[idx].push_back(ctx_copy.encode_ringt(feature_tmp_pack, coeff_scale[idx]));
            }
        });
    }
}

std::vector<CkksCiphertext> PolyRelu::run_core(CkksContext& ctx, const std::vector<CkksCiphertext>& x) {
    vector<CkksCiphertext> result(x.size());

    parallel_for(x.size(), th_nums, ctx, [&](CkksContext& ctx_copy, int x_idx) {
        result[x_idx] = x[x_idx].copy();
        for (int order_idx = order - 1; order_idx > 0; --order_idx) {
            if (weight_pt.empty()) {
                auto w_pt = generate_weight_pt_for_indices(ctx_copy, order_idx, x_idx);
                result[x_idx] = ctx_copy.add_plain_ringt(result[x_idx], w_pt);
                result[x_idx] = ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(result[x_idx], x[x_idx])),
                                                 ctx_copy.get_parameter().get_default_scale());
            } else {
                result[x_idx] = ctx_copy.add_plain_ringt(result[x_idx], weight_pt[order_idx][x_idx]);
                result[x_idx] = ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(result[x_idx], x[x_idx])),
                                                 ctx_copy.get_parameter().get_default_scale());
            }
        }
        if (weight_pt.empty()) {
            auto w_pt = generate_weight_pt_for_indices(ctx_copy, 0, x_idx);
            result[x_idx] = ctx_copy.add_plain_ringt(result[x_idx], w_pt);
        } else {
            result[x_idx] = ctx_copy.add_plain_ringt(result[x_idx], weight_pt[0][x_idx]);
        }
        result[x_idx].set_scale(ctx_copy.get_parameter().get_default_scale());
    });
    return result;
}

std::vector<CkksCiphertext> PolyRelu::run_core_for_non_absorb_case(CkksContext& ctx,
                                                                   const std::vector<CkksCiphertext>& x) {
    vector<CkksCiphertext> result(x.size());
    if (order != 4) {
        parallel_for(x.size(), th_nums, ctx, [&](CkksContext& ctx_copy, int x_idx) {
            if (weight_pt.empty()) {
                auto w_pt_order_rt = generate_weight_pt_for_non_absorb_indices(ctx_copy, order, x_idx);
                auto w_pt_order = ctx_copy.ringt_to_mul(w_pt_order_rt, x[x_idx].get_level());
                result[x_idx] = ctx_copy.rescale(ctx_copy.mult_plain_mul(x[x_idx], w_pt_order),
                                                 ctx_copy.get_parameter().get_default_scale());
            } else {
                auto w_pt = ctx_copy.ringt_to_mul(weight_pt[order][x_idx], x[x_idx].get_level());
                result[x_idx] = ctx_copy.rescale(ctx_copy.mult_plain_mul(x[x_idx], w_pt),
                                                 ctx_copy.get_parameter().get_default_scale());
            }

            for (int order_idx = order - 1; order_idx > 0; --order_idx) {
                if (weight_pt.empty()) {
                    auto w_pt = generate_weight_pt_for_non_absorb_indices(ctx_copy, order_idx, x_idx);
                    result[x_idx] = ctx_copy.add_plain_ringt(result[x_idx], w_pt);
                } else {
                    result[x_idx] = ctx_copy.add_plain_ringt(result[x_idx], weight_pt[order_idx][x_idx]);
                }
                result[x_idx] = ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(result[x_idx], x[x_idx])),
                                                 ctx_copy.get_parameter().get_default_scale());
            }
            if (weight_pt.empty()) {
                auto w0_pt = generate_weight_pt_for_non_absorb_indices(ctx_copy, 0, x_idx);
                result[x_idx] = ctx_copy.add_plain_ringt(result[x_idx], w0_pt);
            } else {
                result[x_idx] = ctx_copy.add_plain_ringt(result[x_idx], weight_pt[0][x_idx]);
            }
        });
    } else {
        parallel_for(x.size(), th_nums, ctx, [&](CkksContext& ctx_copy, int x_idx) {
            if (order <= 1) {
                if (weight_pt.empty()) {
                    auto w1_pt_rt = generate_weight_pt_for_non_absorb_indices(ctx_copy, 1, x_idx);
                    auto w1_pt = ctx_copy.ringt_to_mul(w1_pt_rt, x[x_idx].get_level());
                    result[x_idx] = ctx_copy.rescale(ctx_copy.mult_plain_mul(x[x_idx], w1_pt),
                                                     ctx_copy.get_parameter().get_default_scale());
                    auto w0_pt = generate_weight_pt_for_non_absorb_indices(ctx_copy, 0, x_idx);
                    result[x_idx] = ctx_copy.add_plain_ringt(result[x_idx], w0_pt);
                } else {
                    auto w_pt = ctx_copy.ringt_to_mul(weight_pt[1][x_idx], x[x_idx].get_level());
                    result[x_idx] = ctx_copy.rescale(ctx_copy.mult_plain_mul(x[x_idx], w_pt),
                                                     ctx_copy.get_parameter().get_default_scale());
                    result[x_idx] = ctx_copy.add_plain_ringt(result[x_idx], weight_pt[0][x_idx]);
                }
            } else {
                int baby_steps = (int)ceil(sqrt(order + 1));
                int giant_steps = (int)ceil((double)(order + 1) / baby_steps);

                vector<CkksCiphertext> x_powers(baby_steps + 1);
                x_powers[1] = x[x_idx].copy();

                for (int i = 2; i <= baby_steps; i++) {
                    if (i % 2 == 0) {
                        int half = i / 2;
                        x_powers[i] =
                            ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(x_powers[half], x_powers[half])),
                                             ctx_copy.get_parameter().get_default_scale());
                    } else {
                        x_powers[i] =
                            ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(x_powers[i - 1], x_powers[1])),
                                             ctx_copy.get_parameter().get_default_scale());
                    }
                }

                CkksCiphertext x_giant = x_powers[baby_steps].copy();

                auto current_giant_power = x_giant.copy();
                for (int giant_step = 0; giant_step < giant_steps; giant_step++) {
                    CkksCiphertext baby_poly;
                    CkksCiphertext term(0);
                    bool has_coeff0 = false;

                    for (int baby_step = 0; baby_step < baby_steps; baby_step++) {
                        int coeff_idx = giant_step * baby_steps + baby_step;
                        if (coeff_idx > order) {
                            continue;
                        }
                        if (baby_step == 0) {
                            has_coeff0 = true;
                        } else {
                            int lv = x_powers[baby_step].get_level();
                            auto x_copy = x_powers[baby_step].copy();
                            while (x_copy.get_level() > level - 2 + giant_step) {
                                x_copy = ctx_copy.drop_level(x_copy);
                            }

                            if (weight_pt.empty()) {
                                auto coeff_pt_rt =
                                    generate_weight_pt_for_non_absorb_indices(ctx_copy, coeff_idx, x_idx);
                                auto coeff_pt = ctx_copy.ringt_to_mul(coeff_pt_rt, x_copy.get_level());
                                term = ctx_copy.rescale(ctx_copy.mult_plain_mul(x_copy, coeff_pt),
                                                        ctx_copy.get_parameter().get_default_scale());
                            } else {
                                auto w_pt = ctx_copy.ringt_to_mul(weight_pt[coeff_idx][x_idx], x_copy.get_level());
                                term = ctx_copy.rescale(ctx_copy.mult_plain_mul(x_copy, w_pt),
                                                        ctx_copy.get_parameter().get_default_scale());
                            }
                        }

                        if (baby_step == 0) {
                            continue;
                        } else if (baby_step == 1) {
                            int coeff0_idx = giant_step * baby_steps;
                            if (weight_pt.empty()) {
                                auto coeff0_pt = generate_weight_pt_for_non_absorb_indices(ctx_copy, coeff0_idx, x_idx);
                                baby_poly = ctx_copy.add_plain_ringt(term, coeff0_pt);
                            } else {
                                baby_poly = ctx_copy.add_plain_ringt(term, weight_pt[coeff0_idx][x_idx]);
                            }
                        } else {
                            baby_poly = ctx_copy.add(baby_poly, term);
                        }
                    }

                    if (giant_step > 0) {
                        baby_poly =
                            ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(baby_poly, current_giant_power)),
                                             ctx_copy.get_parameter().get_default_scale());
                    }
                    if (giant_step == 0) {
                        result[x_idx] = baby_poly.copy();
                    } else {
                        result[x_idx] = ctx_copy.add(result[x_idx], baby_poly);
                    }
                    if (giant_step < giant_steps - 1 && giant_step > 1) {
                        current_giant_power =
                            ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(current_giant_power, x_giant)),
                                             ctx_copy.get_parameter().get_default_scale());
                    }
                }
            }
            result[x_idx].set_scale(ctx_copy.get_parameter().get_default_scale());
        });
    }
    return result;
}

Feature2DEncrypted PolyRelu::run(CkksContext& ctx, const Feature2DEncrypted& x) {
    Feature2DEncrypted result(&ctx, x.level);
    result.shape[0] = x.shape[0];
    result.shape[1] = x.shape[1];
    result.skip[0] = x.skip[0];
    result.skip[1] = x.skip[1];
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct;
    if (is_ordinary_pack) {
        result.level = x.level - order;
    } else {
        result.level = x.level - order + 1;
    }
    result.data = run_core(ctx, x.data);
    return result;
}

Feature2DEncrypted PolyRelu::run_for_non_absorb_case(CkksContext& ctx, const Feature2DEncrypted& x) {
    Feature2DEncrypted result(&ctx, x.level);
    result.shape[0] = x.shape[0];
    result.shape[1] = x.shape[1];
    result.skip[0] = x.skip[0];
    result.skip[1] = x.skip[1];
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.data = run_core_for_non_absorb_case(ctx, x.data);
    result.level = result.data[0].get_level();
    return result;
}

Array<double, 3> PolyRelu::run_plaintext(const Array<double, 3>& x) {
    int n_out_channel = x.get_shape()[0];
    Array<double, 3> result({x.get_shape()[0], x.get_shape()[1], x.get_shape()[2]});
    for (int in_channel_idx = 0; in_channel_idx < n_out_channel; in_channel_idx++) {
        for (int i = 0; i < input_shape[0]; i++) {
            for (int j = 0; j < input_shape[1]; j++) {
                auto p = weight.get(0, in_channel_idx);

                for (int k = 1; k < order; k++) {
                    p += weight.get(k, in_channel_idx) * pow(x.get(in_channel_idx, i, j), k);
                }
                p += pow(x.get(in_channel_idx, i, j), order);
                result.set(in_channel_idx, i, j, p);
            }
        }
    }
    return result;
}

Array<double, 3> PolyRelu::run_plaintext_for_non_absorb_case(const Array<double, 3>& x) {
    int n_out_channel = x.get_shape()[0];
    Array<double, 3> result({x.get_shape()[0], x.get_shape()[1], x.get_shape()[2]});
    for (int in_channel_idx = 0; in_channel_idx < n_out_channel; in_channel_idx++) {
        for (int i = 0; i < input_shape[0]; i++) {
            for (int j = 0; j < input_shape[1]; j++) {
                if (i % upsample_factor[0] == 0 && j % upsample_factor[1] == 0) {
                    auto p = weight.get(0, in_channel_idx);

                    for (int k = 1; k < order + 1; k++) {
                        p += weight.get(k, in_channel_idx) * pow(x.get(in_channel_idx, i, j), k);
                    }
                    result.set(in_channel_idx, i, j, p);
                }
            }
        }
    }
    return result;
}

// ======================== Lazy Mode Implementations ========================

void PolyRelu::prepare_weight_lazy() {
    // Don't pre-generate weight_pt, just resize the container
    weight_pt.clear();
    // Empty containers - weights will be generated on-demand
}

void PolyRelu::prepare_weight_for_non_absorb_case_lazy() {
    // Don't pre-generate weight_pt, just resize and cache scale values
    weight_pt.clear();

    // Cache scale/level values for order==4 case
    if (order == 4) {
        cached_coeff_scale[0] = param.get_default_scale();
        cached_coeff_scale[1] = param.get_q(level - 2);
        cached_coeff_scale[2] = param.get_q(level - 2) / cached_coeff_scale[0] * param.get_q(level);
        cached_coeff_scale[3] = param.get_q(level - 2) / cached_coeff_scale[0] * param.get_q(level - 1) *
                                param.get_q(level) / cached_coeff_scale[0];
        cached_coeff_scale[4] = param.get_q(level - 2) / cached_coeff_scale[0] * param.get_q(level - 1) /
                                cached_coeff_scale[0] * param.get_q(level) / cached_coeff_scale[0] *
                                param.get_q(level - 1);

        cached_level_order[0] = level - 3;
        cached_level_order[1] = level - 2;
        cached_level_order[2] = level - 2;
        cached_level_order[3] = level - 2;
        cached_level_order[4] = level - 1;
    }
}

CkksPlaintextRingt
PolyRelu::generate_weight_pt_for_indices(CkksContext& ctx, int idx, int n_packed_out_channel_idx) const {
    vector<double> feature_tmp_pack(N / 2, 0.0);

    for (int linear_idx = 0; linear_idx < cached_total_block_size; ++linear_idx) {
        int block_i = linear_idx / (block_shape[0] * block_shape[1]);
        int residual = linear_idx % (block_shape[0] * block_shape[1]);
        int shape_i = residual / block_shape[1];
        int shape_j = residual % block_shape[1];

        int channel_idx = n_packed_out_channel_idx * n_channel_per_ct / block_expansion[0] / block_expansion[1] +
                          block_i * cached_skip_prod + (skip[0] * (shape_i % skip[0]) + shape_j % skip[0]);
        if (channel_idx >= cached_channel || (shape_i % pre_skip[0]) >= skip[0] || (shape_j % pre_skip[1]) >= skip[1])
            continue;

        int index = block_i * block_shape[0] * block_shape[1] + shape_i * block_shape[1] + shape_j;
        feature_tmp_pack[index] = weight.get(idx, channel_idx);
    }

    double pack_scale = param.get_default_scale();
    if (idx == order - 1) {
        pack_scale = param.get_default_scale();
    } else {
        for (int k = 0; k < order - idx - 1; k++) {
            pack_scale = pack_scale * param.get_default_scale() / param.get_q(level - k);
        }
    }

    return ctx.encode_ringt(feature_tmp_pack, pack_scale);
}

CkksPlaintextRingt
PolyRelu::generate_weight_pt_for_non_absorb_indices(CkksContext& ctx, int idx, int n_packed_out_channel_idx) const {
    vector<double> feature_tmp_pack(N / 2, 0.0);
    if (is_ordinary_pack) {
        for (int ch = 0; ch < (int)n_channel_per_ct; ch++) {
            int channel_idx = n_packed_out_channel_idx * n_channel_per_ct + ch;
            if (channel_idx >= cached_channel)
                continue;
            for (int j = 0; j < input_shape[0]; j++) {
                for (int k = 0; k < input_shape[1]; k++) {
                    int index = ch * block_shape[0] * block_shape[1] + j * block_shape[1] * skip[0] + k * skip[1];
                    feature_tmp_pack[index] = weight.get(idx, channel_idx);
                }
            }
        }
    } else {
        for (int linear_idx = 0; linear_idx < cached_total_block_size; ++linear_idx) {
            int block_i = linear_idx / (block_shape[0] * block_shape[1]);
            int residual = linear_idx % (block_shape[0] * block_shape[1]);
            int shape_i = residual / block_shape[1];
            int shape_j = residual % block_shape[1];

            int channel_idx = n_packed_out_channel_idx * n_channel_per_ct / block_expansion[0] / block_expansion[1] +
                              block_i * cached_skip_prod + (skip[0] * (shape_i % skip[0]) + shape_j % skip[0]);
            if (channel_idx >= cached_channel || (shape_i % pre_skip[0]) >= skip[0] ||
                (shape_j % pre_skip[1]) >= skip[1])
                continue;

            int index = block_i * block_shape[0] * block_shape[1] + shape_i * block_shape[1] + shape_j;
            feature_tmp_pack[index] = weight.get(idx, channel_idx);
        }
    }

    double pack_scale = 1.0;
    int target_level = level - (order - idx);

    if (order != 4) {
        // General case
        if (idx == 0) {
            pack_scale = pack_scale * param.get_default_scale();
        } else {
            for (int k = 0; k < idx; k++) {
                if (k == idx - 1) {
                    pack_scale = pack_scale * param.get_q(level - (order - k - 1));
                } else {
                    pack_scale = param.get_q(level - (order - k - 1)) / param.get_default_scale() * pack_scale;
                }
            }
        }
    } else {
        // Order == 4 case, use cached values
        pack_scale = cached_coeff_scale.at(idx);
        target_level = cached_level_order.at(idx);
    }
    auto result = ctx.encode_ringt(feature_tmp_pack, pack_scale);
    return result;
}
