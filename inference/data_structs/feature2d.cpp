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

#include <sstream>
#include "feature2d.h"
#include "util.h"

using namespace std;
using namespace cxx_sdk_v2;

Feature2DEncrypted::Feature2DEncrypted(CkksContext* context_in,
                                       int ct_level,
                                       const Duo& skip_in,
                                       const Duo& invalid_fill_in,
                                       PackType packing_type_in)
    : skip(skip_in), invalid_fill(invalid_fill_in), packing_type(packing_type_in) {
    dim = 2;
    context = context_in;
    level = ct_level;
}

vector<CkksPlaintext> Feature2DEncrypted::encode_multiple_channel(const Array<double, 3>& feature_mg, double scale_in) {
    int n_slot = context->get_parameter().get_n() / 2;
    const int N_THREAD = 4;

    auto input_shape = feature_mg.get_shape();
    n_channel = input_shape[0];
    shape = {uint32_t(input_shape[1]), uint32_t(input_shape[2])};
    skip = {1, 1};
    n_channel_per_ct = n_slot / prod(shape);
    uint32_t n_ct = div_ceil(n_channel, n_channel_per_ct);

    vector<CkksPlaintext> pt_vec(n_ct);
    parallel_for(n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        vector<double> image_flat;
        image_flat.reserve(n_channel_per_ct * shape[0] * shape[1]);
        for (int k = 0; k < n_channel_per_ct; k++) {
            if (ct_idx * n_channel_per_ct + k < n_channel) {
                for (int i = 0; i < shape[0]; i++) {
                    for (int j = 0; j < shape[1]; j++) {
                        image_flat.push_back(feature_mg.get(ct_idx * n_channel_per_ct + k, i, j));
                    }
                }
            } else {
                for (int i = 0; i < shape[0]; i++) {
                    for (int j = 0; j < shape[1]; j++) {
                        image_flat.push_back(feature_mg.get((ct_idx * n_channel_per_ct + k) % n_channel, i, j));
                    }
                }
            }
        }
        pt_vec[ct_idx] = ctx_copy.encode(image_flat, level, scale_in);
    });
    return pt_vec;
}

vector<CkksPlaintext> Feature2DEncrypted::encode_multiplexed(const Array<double, 3>& feature_mg, double scale_in) {
    int n_slot = context->get_parameter().get_n() / 2;
    const int N_THREAD = 4;

    auto input_shape = feature_mg.get_shape();
    n_channel = input_shape[0];
    shape = {uint32_t(input_shape[1]), uint32_t(input_shape[2])};

    int n_channel_per_block = prod(skip) / prod(invalid_fill);
    int n_channel_per_block_col = skip[1] / invalid_fill[1];
    n_channel_per_ct = n_slot / prod(shape) / prod(invalid_fill);
    int n_block_per_ct = n_channel_per_ct / n_channel_per_block;

    int n_ct = div_ceil(n_channel, n_channel_per_ct);
    vector<CkksPlaintext> pt_vec(n_ct);

    parallel_for(n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        vector<double> feature_pack(n_slot);
        for (int block_idx = 0; block_idx < n_block_per_ct; block_idx++) {
            for (uint32_t x0 = 0; x0 < shape[0]; x0++) {
                for (uint32_t x1 = 0; x1 < shape[1]; x1++) {
                    Duo x = {x0, x1};
                    for (int channel_idx_in_block = 0; channel_idx_in_block < n_channel_per_block;
                         channel_idx_in_block++) {
                        int channel_idx =
                            ct_idx * n_channel_per_ct + block_idx * n_channel_per_block + channel_idx_in_block;
                        if (channel_idx >= (int)n_channel) {
                            continue;
                        }
                        Duo channel_offset = div_mod(channel_idx_in_block, n_channel_per_block_col);
                        Duo x_in_block = x * skip + channel_offset;
                        int slot =
                            block_idx * prod(shape * skip) + x_in_block[0] * (shape[1] * skip[1]) + x_in_block[1];
                        feature_pack[slot] = feature_mg.get(channel_idx, x0, x1);
                    }
                }
            }
        }
        pt_vec[ct_idx] = ctx_copy.encode(feature_pack, level, scale_in);
    });
    return pt_vec;
}

vector<CkksPlaintext> Feature2DEncrypted::encode_interleaved(const Array<double, 3>& feature_mg,
                                                             const Duo& block_shape,
                                                             const Duo& stride,
                                                             double scale_in) {
    const int N_THREAD = 4;

    auto input_shape = feature_mg.get_shape();
    n_channel = input_shape[0];
    shape = {uint32_t(input_shape[1]), uint32_t(input_shape[2])};

    n_channel_per_ct = 1;
    int f_ct_num = n_channel * prod(stride);
    vector<CkksPlaintext> pt_vec(f_ct_num);

    parallel_for(f_ct_num, N_THREAD, *context, [&](CkksContext& ctx_copy, int i) {
        vector<double> slots(ctx_copy.get_parameter().get_n() / 2);
        int channel_idx = i / prod(stride);
        int grid_idx = i % prod(stride);
        Duo grid_idx_2d = div_mod(grid_idx, stride[1]);
        for (uint32_t x0 = 0; x0 < shape[0]; x0++) {
            int block_row_idx = x0 / stride[0];
            for (uint32_t x1 = 0; x1 < shape[1]; x1++) {
                Duo x = {x0, x1};
                int block_col_idx = x1 / stride[1];
                if (x % stride == grid_idx_2d) {
                    slots[block_row_idx * block_shape[1] + block_col_idx] = feature_mg.get(channel_idx, x0, x1);
                }
            }
        }
        pt_vec[i] = ctx_copy.encode(slots, level, scale_in);
    });
    return pt_vec;
}

void Feature2DEncrypted::pack_multiple_channel(const Array<double, 3>& feature_mg, bool is_symmetric, double scale_in) {
    packing_type = PackType::MultipleChannelPacking;
    vector<CkksPlaintext> pt_vec = encode_multiple_channel(feature_mg, scale_in);
    uint32_t n_ct = pt_vec.size();
    const int N_THREAD = 4;

    data.clear();
    data_compress.clear();
    if (is_symmetric) {
        data_compress.resize(n_ct);
    } else {
        data.resize(n_ct);
    }

    parallel_for(n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        if (is_symmetric) {
            data_compress[ct_idx] = ctx_copy.encrypt_symmetric_compressed(pt_vec[ct_idx]);
        } else {
            data[ct_idx] = ctx_copy.encrypt_symmetric(pt_vec[ct_idx]);
        }
    });
}

void Feature2DEncrypted::pack_interleaved(const Array<double, 3>& feature_mg,
                                          const Duo& block_shape,
                                          const Duo& stride,
                                          bool is_symmetric,
                                          double scale_in) {
    packing_type = PackType::InterleavedPacking;
    vector<CkksPlaintext> pt_vec = encode_interleaved(feature_mg, block_shape, stride, scale_in);

    const int N_THREAD = 4;
    data.clear();
    data_compress.clear();
    if (is_symmetric) {
        data_compress.resize(pt_vec.size());
    } else {
        data.resize(pt_vec.size());
    }
    parallel_for(pt_vec.size(), N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        if (is_symmetric) {
            data_compress[ct_idx] = ctx_copy.encrypt_symmetric_compressed(pt_vec[ct_idx]);
        } else {
            data[ct_idx] = ctx_copy.encrypt_symmetric(pt_vec[ct_idx]);
        }
    });
}

void Feature2DEncrypted::pack_multiplexed(const Array<double, 3>& feature_mg, bool is_symmetric, double scale_in) {
    packing_type = PackType::MultiplexedPacking;
    vector<CkksPlaintext> pt_vec = encode_multiplexed(feature_mg, scale_in);

    const int N_THREAD = 4;
    data.clear();
    data_compress.clear();
    if (is_symmetric) {
        data_compress.resize(pt_vec.size());
    } else {
        data.resize(pt_vec.size());
    }
    parallel_for(pt_vec.size(), N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        if (is_symmetric) {
            data_compress[ct_idx] = ctx_copy.encrypt_symmetric_compressed(pt_vec[ct_idx]);
        } else {
            data[ct_idx] = ctx_copy.encrypt_symmetric(pt_vec[ct_idx]);
        }
    });
}

void Feature2DEncrypted::column_pack(const Array<double, 2>& feature_mg, bool is_symmetric, double scale_in) {
    uint64_t tol_size = feature_mg.get_shape()[0] * feature_mg.get_shape()[1];
    int pack_num = div_ceil(tol_size, (context->get_parameter().get_n() / 2));
    vector<vector<double>> feature_mg_pack(pack_num);
    vector<CkksCiphertext> out_ct;
    int T = 0;
    const int N_THREAD = 4;

    int n_copy = div_ceil((context->get_parameter().get_n() / 2), tol_size);
    for (int k = 0; k < n_copy; k++) {
        for (int i = 0; i < feature_mg.get_shape()[1]; i++) {
            for (int j = 0; j < feature_mg.get_shape()[0]; j++) {
                T = i * feature_mg.get_shape()[0] + j;
                feature_mg_pack[floor(T / (context->get_parameter().get_n() / 2))].push_back(feature_mg.get(j, i));
            }
        }
    }

    for (int i = 0; i < pack_num; i++) {
        auto enc = context->encode(feature_mg_pack[i], level, scale_in);
        data.push_back(context->encrypt_asymmetric(enc));
    }
}

void Feature2DEncrypted::row_pack(const Array<double, 2>& feature_mg, bool is_symmetric, double scale_in) {
    int N = context->get_parameter().get_n();
    uint64_t tol_size = feature_mg.get_shape()[0] * feature_mg.get_shape()[1];
    int pack_num = div_ceil(tol_size, (N / 2));
    vector<vector<double>> feature_mg_pack(pack_num);
    vector<CkksCiphertext> out_ct;
    int T = 0;
    const int N_THREAD = 4;
    int n_copy = div_ceil((context->get_parameter().get_n() / 2), tol_size);
    for (int k = 0; k < n_copy; k++) {
        for (int i = 0; i < feature_mg.get_shape()[0]; i++) {
            for (int j = 0; j < feature_mg.get_shape()[1]; j++) {
                T = i * feature_mg.get_shape()[1] + j;
                feature_mg_pack[floor(T / (context->get_parameter().get_n() / 2))].push_back(feature_mg.get(i, j));
            }
        }
    }
    for (int i = 0; i < pack_num; i++) {
        auto enc = context->encode(feature_mg_pack[i], level, scale_in);
        data.push_back(context->encrypt_asymmetric(enc));
    }
}

Array<double, 3> Feature2DEncrypted::unpack_multiple_channel() const {
    const int N_THREAD = 4;
    int n_ct = data.size();
    Duo pre_skip_shape = shape * skip;

    Array<double, 3> result({n_channel, shape[0], shape[1]});
    parallel_for(n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        CkksPlaintext x_pt = ctx_copy.decrypt(data[ct_idx]);
        Array1D x_mg = ctx_copy.decode(x_pt);
        for (int i = 0; i < n_channel_per_ct; i++) {
            int channel_idx = ct_idx * n_channel_per_ct + i;
            if (channel_idx >= n_channel) {
                continue;
            }
            for (int j = 0; j < shape[0]; j++) {
                for (int k = 0; k < shape[1]; k++) {
                    result.set(channel_idx, j, k,
                               x_mg[i * pre_skip_shape[0] * pre_skip_shape[1] + j * pre_skip_shape[1] * skip[0] +
                                    k * skip[1]]);
                }
            }
        }
    });
    return result;
}

Array<double, 2> Feature2DEncrypted::unpack_row() const {
    const int N_THREAD = 1;
    int n_ct = data.size();
    Duo pre_skip_shape = shape * skip;
    int n_slot = context->get_parameter().get_n() / 2;

    Array<double, 2> result({shape[0], shape[1]});
    parallel_for(n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        CkksPlaintext x_pt = ctx_copy.decrypt(data[ct_idx]);
        Array1D x_mg = ctx_copy.decode(x_pt);
        for (int i = 0; i < x_mg.size(); i++) {
            int idx = ct_idx * n_slot + i;
            int row = idx / pre_skip_shape[1];
            int col = idx % pre_skip_shape[1];
            if (row >= pre_skip_shape[0]) {
                continue;
            }
            result.set(row, col, x_mg[i]);
        }
    });
    return result;
}

Array<double, 3> Feature2DEncrypted::unpack_multiplexed() const {
    const int N_THREAD = 4;
    int n_ct = data.size();
    Array<double, 3> result({n_channel, shape[0], shape[1]});
    int n_channel_per_block = prod(skip) / prod(invalid_fill);
    int n_channel_per_block_col = skip[1] / invalid_fill[1];
    int n_block_per_ct = n_channel_per_ct / n_channel_per_block;

    parallel_for(n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        CkksPlaintext x_pt = ctx_copy.decrypt(data[ct_idx]);
        Array1D x_mg = ctx_copy.decode(x_pt);
        for (int block_idx = 0; block_idx < n_block_per_ct; block_idx++) {
            for (uint32_t x0_in_block = 0; x0_in_block < shape[0] * skip[0]; x0_in_block++) {
                for (uint32_t x1_in_block = 0; x1_in_block < shape[1] * skip[1]; x1_in_block++) {
                    Duo x_in_block = {x0_in_block, x1_in_block};
                    Duo channel_offset = x_in_block % skip;
                    // skip invalid slots: valid offsets are within [0, skip[d]/invalid_fill[d])
                    if (channel_offset[0] >= (int)(skip[0] / invalid_fill[0]) ||
                        channel_offset[1] >= (int)(skip[1] / invalid_fill[1])) {
                        continue;
                    }
                    int channel_idx_in_block = channel_offset[0] * n_channel_per_block_col + channel_offset[1];
                    int channel_idx =
                        ct_idx * n_channel_per_ct + block_idx * n_channel_per_block + channel_idx_in_block;
                    if (channel_idx >= (int)n_channel) {
                        continue;
                    }
                    Duo x = x_in_block / skip;
                    int slot = block_idx * prod(shape * skip) + x0_in_block * (shape[1] * skip[1]) + x1_in_block;
                    result.set(channel_idx, x[0], x[1], x_mg[slot]);
                }
            }
        }
    });
    return result;
}

Array<double, 3> Feature2DEncrypted::unpack_interleaved(const Duo& block_shape, const Duo& stride) const {
    const int N_THREAD = 4;
    int n_ct = data.size();
    Array<double, 3> result({n_channel, shape[0], shape[1]});

    parallel_for(n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        CkksPlaintext x_pt = ctx_copy.decrypt(data[ct_idx]);
        Array1D x_mg = ctx_copy.decode(x_pt);
        int channel_idx = ct_idx / prod(stride);
        int grid_idx = ct_idx % prod(stride);
        Duo grid_idx_2d = div_mod(grid_idx, stride[1]);
        for (int j = 0; j < block_shape[0]; j++) {
            for (int k = 0; k < block_shape[1]; k++) {
                result.set(channel_idx, j * stride[0] + grid_idx_2d[0], k * stride[1] + grid_idx_2d[1],
                           x_mg[j * block_shape[1] + k]);
            }
        }
    });
    return result;
}

Array<double, 2> Feature2DEncrypted::unpack_column() const {
    const int N_THREAD = 1;
    int n_ct = data.size();
    Duo pre_skip_shape = shape * skip;

    Array<double, 2> result({shape[0], shape[1]});
    parallel_for(n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        CkksPlaintext x_pt = ctx_copy.decrypt(data[ct_idx]);
        Array1D x_mg = ctx_copy.decode(x_pt);
        for (int i = 0; i < n_channel_per_ct; i++) {
            int col = ct_idx * n_channel_per_ct + i;
            if (col >= shape[1]) {
                continue;
            }
            for (int j = 0; j < shape[0]; j++) {
                int pos = i * shape[0] + j;
                result.set(j, col, x_mg[pos]);
            }
        }
    });
    return result;
}

Feature2DEncrypted Feature2DEncrypted::refresh_ciphertext() const {
    CkksBtpContext* ctx = dynamic_cast<CkksBtpContext*>(context);
    if (ctx == nullptr) {
        throw std::runtime_error("refresh_ciphertext() requires CkksBtpContext");
    }
    int new_level = 9;
    Feature2DEncrypted result(ctx, new_level);
    result.data.resize(data.size());
    parallel_for(data.size(), th_nums, *ctx, [&](CkksBtpContext& ctx_copy, int ct_idx) {
        result.data[ct_idx] = ctx_copy.bootstrap(data[ct_idx]);
        assert(new_level == result.data[ct_idx].get_level());
    });
    result.skip = skip;
    result.shape = shape;
    result.n_channel = n_channel;
    result.n_channel_per_ct = n_channel_per_ct;
    return result;
}

Feature2DEncrypted Feature2DEncrypted::drop_level(int n_level_to_drop) const {
    int new_level = level - n_level_to_drop;
    Feature2DEncrypted result(context, new_level);
    result.n_channel = n_channel;
    result.n_channel_per_ct = n_channel_per_ct;
    result.shape = shape;
    result.skip = skip;
    result.data.resize(data.size());
    parallel_for(data.size(), th_nums, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        auto ct_tmp = data[ct_idx].copy();
        for (int j = 0; j < n_level_to_drop; j++) {
            ct_tmp = ctx_copy.drop_level(ct_tmp);
        }
        result.data[ct_idx] = move(ct_tmp);
        assert(new_level == result.data[ct_idx].get_level());
    });
    return result;
}

Feature2DEncrypted Feature2DEncrypted::copy() const {
    Feature2DEncrypted result(context, level);
    result.dim = dim;
    result.n_channel = n_channel;
    result.n_channel_per_ct = n_channel_per_ct;
    result.shape = shape;
    result.skip = skip;
    for (int i = 0; i < data.size(); i++) {
        result.data.push_back(data[i].copy());
    }
    return result;
}

Feature2DShare::Feature2DShare(uint64_t q, int s) : FeatureShare{q, s} {}

void Feature2DEncrypted::split_to_shares(Feature2DEncrypted* share0, Feature2DShare* share1) const {
    int n_slot = context->get_parameter().get_n() / 2;
    double share_scale = ENC_TO_SHARE_SCALE;
    int feature_bitlength = ENC_TO_SHARE_SCALE_BIT + 1;
    int sigma = SIGMA;

    Duo pre_skip_shape = shape * skip;
    size_t n_share_feature = n_channel * shape[0] * shape[1];
    size_t n_mask = n_channel * pre_skip_shape[0] * pre_skip_shape[1];

    vector<double> mask_d(n_mask);
    vector<int64_t> r(n_mask);
    for (int i = 0; i < n_mask; i++) {
        r[i] = int64_t(gen_random_uint(feature_bitlength + sigma)) - int64_t(1ull << (feature_bitlength + sigma - 1));
        mask_d[i] = double(r[i]) / share_scale;
    }

    share0->n_channel = n_channel;
    share0->n_channel_per_ct = n_channel_per_ct;
    share0->shape = shape;
    share0->skip = skip;
    share0->level = level;
    share0->data.clear();
    vector<double> mask_d_span(mask_d);
    for (int i = 0; i < data.size(); i++) {
        size_t start = i * n_slot;
        size_t length = i == data.size() - 1 ? (mask_d_span.size() - start) : n_slot;
        std::vector<double> mask_mg_vec(mask_d_span.begin() + start, mask_d_span.begin() + start + length);
        CkksPlaintext mask_pt = context->encode(mask_mg_vec, level, ENC_TO_SHARE_SCALE);
        CkksCiphertext share0_ct = context->add_plain(data[i], mask_pt);
        share0->data.push_back(move(share0_ct));
    }

    share1->shape = shape;
    share1->data.resize({n_share_feature});
    double scale = pow(2, share1->scale_ord);
    for (int i = 0; i < n_channel; i++) {
        for (int j = 0; j < shape[0]; j++) {
            for (int k = 0; k < shape[1]; k++) {
                int skipped_index = i * shape[0] * shape[1] + j * shape[1] + k;
                int pre_skip_index =
                    i * pre_skip_shape[0] * pre_skip_shape[1] + j * pre_skip_shape[1] * skip[0] + k * skip[1];
                share1->data[skipped_index] =
                    (-int64_t(r[pre_skip_index]) % share1->ring_mod + share1->ring_mod) % share1->ring_mod;
            }
        }
    }
}

static vector<CkksPlaintext> multi_pack_to_pt(const Array<double, 3>& feature_mg,
                                              Feature2DEncrypted& f2d,
                                              int n_channel,
                                              Duo shape,
                                              Duo skip,
                                              CkksContext& context,
                                              int level,
                                              double scale_in,
                                              PackType pack_type) {
    if (pack_type == PackType::MultipleChannelPacking) {
        return f2d.encode_multiple_channel(feature_mg, scale_in);
    } else if (pack_type == PackType::MultiplexedPacking) {
        return f2d.encode_multiplexed(feature_mg, scale_in);
    } else {
        Duo block_expansion = {(uint32_t)ceil(shape[0] / (double)BLOCK_SHAPE[0]),
                               (uint32_t)ceil(shape[1] / (double)BLOCK_SHAPE[1])};
        return f2d.encode_interleaved(feature_mg, BLOCK_SHAPE, block_expansion, scale_in);
    }
}

void Feature2DEncrypted::split_to_shares_for_multi_channel_pack(Feature2DEncrypted* share0,
                                                                Feature2DShare* share1) const {
    assert(this->packing_type == PackType::MultipleChannelPacking);
    int n_slot = context->get_parameter().get_n() / 2;
    double share_scale = ENC_TO_SHARE_SCALE;
    int feature_bitlength = ENC_TO_SHARE_SCALE_BIT + 1;
    int sigma = SIGMA;
    Duo pre_skip_shape = shape * skip;
    // cppcheck-suppress duplicateAssignExpression
    size_t n_share_feature = n_channel * shape[0] * shape[1];
    size_t n_mask = n_channel * shape[0] * shape[1];

    vector<double> mask_d(n_mask);
    vector<int64_t> r(n_mask);
    for (int i = 0; i < n_mask; i++) {
        r[i] = int64_t(gen_random_uint(feature_bitlength + sigma)) - int64_t(1ull << (feature_bitlength + sigma - 1));
        mask_d[i] = double(r[i]) / share_scale;
    }
    share0->n_channel = n_channel;
    share0->n_channel_per_ct = n_channel_per_ct;
    share0->shape = shape;
    share0->skip = skip;
    share0->level = level;
    share0->data.clear();
    auto mask_d_array = Array<double, 1>::from_array_1d(mask_d).reshape<3>({n_channel, shape[0], shape[1]});
    auto mask_pt =
        multi_pack_to_pt(mask_d_array, *share0, n_channel, shape, skip, *context, level, DEFAULT_SCALE, packing_type);
    for (int i = 0; i < data.size(); i++) {
        CkksCiphertext share0_ct = context->add_plain(data[i], mask_pt[i]);
        share0->data.push_back(move(share0_ct));
    }

    share1->shape = shape;
    share1->data.resize({n_mask});
    double scale = pow(2, share1->scale_ord);
    for (int i = 0; i < n_mask; i++) {
        share1->data[i] = (-int64_t(r[i]) % share1->ring_mod + share1->ring_mod) % share1->ring_mod;
    }
}

Feature2DEncrypted Feature2DEncrypted::combine_with_share(const Feature2DShare& share) const {
    const int N_THREAD = 4;
    int n_slot = context->get_parameter().get_n() / 2;
    Feature2DEncrypted result(this->context, this->level);
    result.n_channel = this->n_channel;
    result.n_channel_per_ct = this->n_channel_per_ct;
    result.shape = this->shape;
    result.skip = this->skip;
    double scale = pow(2, share.scale_ord);
    int n_ct = this->data.size();

    result.data.clear();
    result.data.resize(n_ct);
    parallel_for(n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, int i) {
        vector<double> mask_d(n_slot);
        for (int j = 0; j < n_slot; j++) {
            uint64_t v;
            if (i * n_slot + j >= share.data.get_size()) {
                v = share.data.get((i * n_slot + j) % share.data.get_size());
            } else {
                v = share.data.get(i * n_slot + j);
            }
            mask_d[j] = uint64_to_double(v, scale, share.ring_mod);
        }
        CkksPlaintext mask_pt = ctx_copy.encode(mask_d, level, ctx_copy.get_parameter().get_default_scale());
        result.data[i] = ctx_copy.add_plain(data[i], mask_pt);
    });
    return result;
}

Feature2DEncrypted Feature2DEncrypted::combine_with_share_new_protocol(const Feature2DShare& share,
                                                                       const Feature2DEncrypted& f2d,
                                                                       const Bytes& b1) const {
    const int N_THREAD = 8;
    int n_slot = context->get_parameter().get_n() / 2;
    Feature2DEncrypted result(this->context, this->level);
    result.n_channel = this->n_channel;
    result.n_channel_per_ct = this->n_channel_per_ct;
    result.shape = this->shape;
    result.skip = this->skip;
    double scale = ENC_TO_SHARE_SCALE;
    double encode_scale = pow(2, DEFAULT_SCALE_BIT);
    int n_ct = this->data.size();

    result.data.clear();
    result.data.resize(n_ct);

    parallel_for_with_extra_level_context(
        n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, CkksContext& extra_level_ctx_copy, int i) {
            vector<double> mask_d(n_slot, 0);
            vector<double> b1_value(n_slot, 0);
            for (int j = 0; j < n_slot; j++) {
                int mg_idx = (i * n_slot + j) % share.data.get_size();
                b1_value[j] = 2 * b1[mg_idx] - 1;
                int64_t mask_value = int64_t(share.data.get(mg_idx)) - int64_t(b1[mg_idx] * share.ring_mod);
                mask_d[j] = double(mask_value) / scale;
            }
            CkksPlaintext mask_pt = ctx_copy.encode(mask_d, level, encode_scale);
            result.data[i] = ctx_copy.add_plain(data[i], mask_pt);

            CkksPlaintext b1_pt =
                extra_level_ctx_copy.encode(b1_value, level + 1, extra_level_ctx_copy.get_parameter().get_q(level + 1));

            auto f2d_mult = extra_level_ctx_copy.mult_plain(f2d.data[i], b1_pt);
            f2d_mult = extra_level_ctx_copy.rescale(f2d_mult, encode_scale);

            result.data[i] = ctx_copy.add(result.data[i], f2d_mult);
        });
    return result;
}

Feature2DEncrypted Feature2DEncrypted::combine_with_share_new_protocol_for_multi_pack(const Feature2DShare& share,
                                                                                      const Feature2DEncrypted& f2d,
                                                                                      const Bytes& b1) const {
    assert(this->packing_type == PackType::MultipleChannelPacking);
    const int N_THREAD = 8;
    int n_slot = context->get_parameter().get_n() / 2;
    Feature2DEncrypted result(this->context, this->level);
    result.n_channel = this->n_channel;
    result.n_channel_per_ct = this->n_channel_per_ct;
    result.shape = this->shape;
    result.skip = this->skip;
    double scale = ENC_TO_SHARE_SCALE;
    double encode_scale = pow(2, DEFAULT_SCALE_BIT);
    int n_ct = this->data.size();

    result.data.clear();
    result.data.resize(n_ct);

    Array<double, 1> mask_d({share.data.get_size()});
    for (int i = 0; i < share.data.get_size(); i++) {
        int64_t mask_value = int64_t(share.data.get(i)) - int64_t(b1[i] * share.ring_mod);
        mask_d.set(i, (mask_value) / scale);
    }
    auto f2d_copy = f2d.copy();
    Array<double, 3> mask_d_3d = mask_d.reshape<3>({this->n_channel, this->shape[0], this->shape[1]});
    auto mask_pt = multi_pack_to_pt(mask_d_3d, f2d_copy, this->n_channel, this->shape, this->skip, *context, level,
                                    DEFAULT_SCALE, PackType::MultipleChannelPacking);
    Array<double, 1> b1_value({b1.size()});
    for (int i = 0; i < b1.size(); i++) {
        b1_value.set(i, 2 * b1[i] - 1);
    }
    Array<double, 3> b1_value_3d = b1_value.reshape<3>({this->n_channel, this->shape[0], this->shape[1]});
    CkksContext& extra_level_context = context->get_extra_level_context();
    auto mask_b1 = multi_pack_to_pt(
        b1_value_3d, f2d_copy, this->n_channel, this->shape, this->skip, extra_level_context, level + 1,
        extra_level_context.get_parameter().get_q(level + 1), PackType::MultipleChannelPacking);
    for (int i = 0; i < data.size(); i++) {
        auto f2d_mult = extra_level_context.mult_plain(f2d.data[i], mask_b1[i]);
        f2d_mult = extra_level_context.rescale(f2d_mult, encode_scale);
        result.data[i] = (*context).add_plain(data[i], mask_pt[i]);
        result.data[i] = (*context).add(result.data[i], f2d_mult);
    }
    return result;
}

void Feature2DEncrypted::decrypt_to_share(Feature2DShare* share, PackType pack_type) const {
    uint64_t ring_mod = RING_MOD;
    int n_slot = context->get_parameter().get_n() / 2;
    share->shape = shape;
    Array<double, 3> x_double_matrix;
    if (pack_type == PackType::MultiplexedPacking) {
        x_double_matrix = this->unpack_multiplexed();
    } else if (pack_type == PackType::MultipleChannelPacking) {
        x_double_matrix = this->unpack_multiple_channel();
    } else if (pack_type == PackType::InterleavedPacking) {
        Duo block_expansion = {(uint32_t)ceil(shape[0] / (double)BLOCK_SHAPE[0]),
                               (uint32_t)ceil(shape[1] / (double)BLOCK_SHAPE[1])};
        x_double_matrix = this->unpack_interleaved(BLOCK_SHAPE, block_expansion);
    }

    share->data = array_double_to_uint64(x_double_matrix, share->scale_ord, share->ring_mod).reshape<1>({0});
}

Array<uint64_t, 1> Feature2DEncrypted::encrypt_from_share(const Feature2DShare& share,
                                                          int n_channel,
                                                          const Duo& input_shape,
                                                          PackType pack_type) {
    int n_slot = context->get_parameter().get_n() / 2;

    this->shape = input_shape;
    Array<double, 1> y0_sub_mod_div_s(share.data.get_shape());
    Array<uint64_t, 1> y0_add_mod(share.data.get_shape());
    double scale = ENC_TO_SHARE_SCALE;
    for (int i = 0; i < share.data.get_size(); i++) {
        uint64_t y0_add_mod_value = (share.data[i] + (share.ring_mod / 2)) % share.ring_mod;
        y0_add_mod.set(i, y0_add_mod_value);
        double y0_sub = double(int64_t(y0_add_mod_value) - int64_t(share.ring_mod / 2)) / scale;
        y0_sub_mod_div_s.set(i, y0_sub);
    }

    Array<double, 3> y3 = y0_sub_mod_div_s.reshape<3>({uint64_t(n_channel), input_shape[0], input_shape[1]});
    if (pack_type == PackType::MultiplexedPacking) {
        this->pack_multiplexed(y3, true, DEFAULT_SCALE);
    } else if (pack_type == PackType::MultipleChannelPacking) {
        this->pack_multiple_channel(y3, true, DEFAULT_SCALE);
    } else if (pack_type == PackType::InterleavedPacking) {
        Duo block_expansion = {(uint32_t)ceil(input_shape[0] / (double)BLOCK_SHAPE[0]),
                               (uint32_t)ceil(input_shape[1] / (double)BLOCK_SHAPE[1])};
        this->pack_interleaved(y3, BLOCK_SHAPE, block_expansion, true);
    }

    return y0_add_mod;
}

void Feature2DEncrypted::decompress() {
    assert(data.size() == 0 && data_compress.size() > 0);
    size_t n_ct = data_compress.size();
    for (int i = 0; i < n_ct; i++) {
        data.push_back(context->compressed_ciphertext_to_ciphertext(data_compress[i]));
    }
    data_compress.clear();
}

Bytes Feature2DEncrypted::serialize() const {
    stringstream ss;
    ss_write(ss, dim);
    ss_write(ss, n_channel);
    ss_write(ss, n_channel_per_ct);
    ss_write(ss, level);
    for (int i = 0; i < 2; i++) {
        ss_write(ss, shape[i]);
    }
    for (int i = 0; i < 2; i++) {
        ss_write(ss, skip[i]);
    }
    uint32_t n_ct = data.size();
    ss_write(ss, n_ct);
    for (const CkksCiphertext& ct : data) {
        Bytes ct_data = ct.serialize(context->get_parameter());
        ss_write_vector(ss, ct_data);
    }
    uint32_t n_cct = data_compress.size();
    ss_write(ss, n_cct);
    for (const CkksCompressedCiphertext& cct : data_compress) {
        Bytes cct_data = cct.serialize(context->get_parameter());
        ss_write_vector(ss, cct_data);
    }

    Bytes bytes = ss_to_bytes(ss);
    return bytes;
}

void Feature2DEncrypted::deserialize(const Bytes& bytes) {
    stringstream ss;
    bytes_to_ss(bytes, ss);
    ss_read(ss, &dim);
    ss_read(ss, &n_channel);
    ss_read(ss, &n_channel_per_ct);
    ss_read(ss, &level);
    for (int i = 0; i < 2; i++) {
        ss_read(ss, &shape[i]);
    }
    for (int i = 0; i < 2; i++) {
        ss_read(ss, &skip[i]);
    }
    uint32_t n_ct;
    ss_read(ss, &n_ct);
    for (int i = 0; i < n_ct; i++) {
        Bytes ct_data;
        ss_read_vector(ss, &ct_data);
        auto y_ct = CkksCiphertext::deserialize(ct_data);
        data.push_back(move(y_ct));
    }
    uint32_t n_cct;
    ss_read(ss, &n_cct);
    for (int i = 0; i < n_cct; i++) {
        Bytes cct_data;
        ss_read_vector(ss, &cct_data);
        auto y_ct = CkksCompressedCiphertext::deserialize(cct_data);
        data_compress.push_back(move(y_ct));
    }
}
