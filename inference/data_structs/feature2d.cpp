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

#include "feature2d.h"
#include "util.h"
#include <sstream>

using namespace std;

Feature2DEncrypted::Feature2DEncrypted(CkksContext* context_in, int ct_level, Duo skip_in, Duo invalid_fill_in)
    : skip(skip_in), invalid_fill(invalid_fill_in) {
    dim = 2;
    context = context_in;
    level = ct_level;
}

vector<vector<double>> Feature2DEncrypted::pack_feature(PackType& packtype,
                                                        const Array<double, 3>& feature_mg,
                                                        const Duo& block_shape = {128, 128},
                                                        const Duo& stride = {1, 1}) {
    vector<vector<double>> feature_tmp_pack;
    int n_slot = context->get_parameter().get_n() / 2;
    const int N_THREAD = 4;

    auto input_shape = feature_mg.get_shape();
    n_channel = input_shape[0];
    shape[0] = input_shape[1];
    shape[1] = input_shape[2];

    if (packtype == PackType::MultChannelPack) {
        skip[0] = 1;
        skip[1] = 1;
        n_channel_per_ct = n_slot / (shape[0] * shape[1]);
        uint32_t n_ct = div_ceil(n_channel, n_channel_per_ct);

        feature_tmp_pack.resize(n_ct);

#pragma omp parallel for num_threads(N_THREAD)
        for (int ct_idx = 0; ct_idx < n_ct; ct_idx++) {
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
            feature_tmp_pack[ct_idx] = image_flat;
        }
    } else if (packtype == PackType::SinglePack) {
        n_channel_per_ct = 1;
        feature_tmp_pack.resize(n_channel);

#pragma omp parallel for num_threads(N_THREAD)
        for (int i = 0; i < n_channel; i++) {
            feature_tmp_pack[i].resize(context->get_parameter().get_n() / 2);
            for (int h = 0; h < shape[0]; h++) {
                for (int k = 0; k < shape[1]; k++) {
                    feature_tmp_pack[i][h * shape[1] * skip[1] * skip[0] + k * skip[1]] = feature_mg.get(i, h, k);
                }
            }
        }
    } else if (packtype == PackType::MultiplexedPack) {
        n_channel_per_ct = n_slot / (shape[0] * shape[1]);

        int f_ct_num = div_ceil(n_channel, skip[0] * skip[1]);
        feature_tmp_pack.resize(f_ct_num);

#pragma omp parallel for num_threads(N_THREAD)
        for (int i = 0; i < f_ct_num; i++) {
            feature_tmp_pack[i].resize(n_slot);
            for (int h = 0; h < shape[0] * skip[0]; h++) {
                for (int k = 0; k < shape[1] * skip[1]; k++) {
                    if ((skip[0] * skip[1] * i + skip[0] * (h % skip[0]) + k % skip[0]) >= n_channel) {
                        continue;
                    }
                    feature_tmp_pack[i][h * shape[0] * skip[0] + k] = feature_mg.get(
                        skip[0] * skip[1] * i + skip[0] * (h % skip[0]) + k % skip[0], h / skip[0], k / skip[1]);
                }
            }
        }
    } else if (packtype == PackType::ParMultiplexedPack) {
        int n_channel_per_block = (skip[0] * skip[1]) / (invalid_fill[0] * invalid_fill[1]);
        int n_channel_per_block_col = skip[1] / invalid_fill[1];
        n_channel_per_ct = n_slot / (shape[0] * shape[1]) / (invalid_fill[0] * invalid_fill[1]);
        int n_block_per_ct = n_channel_per_ct / n_channel_per_block;

        int f_ct_num = div_ceil(n_channel, n_channel_per_ct);
        feature_tmp_pack.resize(f_ct_num);

        for (int i = 0; i < f_ct_num; i++) {
            feature_tmp_pack[i].resize(n_slot);
#pragma omp parallel for num_threads(N_THREAD)
            for (int j = 0; j < n_block_per_ct; j++) {
                for (int x = 0; x < (int)shape[0]; x++) {
                    for (int y = 0; y < (int)shape[1]; y++) {
                        for (int channel_idx_in_block = 0; channel_idx_in_block < n_channel_per_block;
                             channel_idx_in_block++) {
                            int channel_idx = i * n_channel_per_ct + j * n_channel_per_block + channel_idx_in_block;
                            if (channel_idx >= (int)n_channel) {
                                continue;
                            }
                            int channel_x_offset = channel_idx_in_block / n_channel_per_block_col;
                            int channel_y_offset = channel_idx_in_block % n_channel_per_block_col;
                            int x_in_block = x * skip[0] + channel_x_offset;
                            int y_in_block = y * skip[1] + channel_y_offset;
                            int slot = j * (shape[0] * skip[0]) * (shape[1] * skip[1]) +
                                       x_in_block * (shape[1] * skip[1]) + y_in_block;
                            feature_tmp_pack[i][slot] = feature_mg.get(channel_idx, x, y);
                        }
                    }
                }
            }
        }
    } else if (packtype == PackType::InterleavedDecompositionPack) {
        n_segment[0] = stride[0];
        n_segment[1] = stride[1];
        n_channel_per_ct = 1;
        int f_ct_num = n_channel * stride[0] * stride[1];
        feature_tmp_pack.resize(f_ct_num);

#pragma omp parallel for num_threads(N_THREAD)
        for (int i = 0; i < f_ct_num; i++) {
            feature_tmp_pack[i].resize(context->get_parameter().get_n() / 2);
            int channel_idx = i / (stride[0] * stride[1]);
            int seg_idx = i % (stride[0] * stride[1]);
            int row_seg_idx = seg_idx / stride[1];
            int col_seg_idx = seg_idx % stride[1];
            for (int h = 0; h < shape[0]; h++) {
                int block_row_idx = h / stride[0];
                for (int k = 0; k < shape[1]; k++) {
                    int block_col_idx = k / stride[1];
                    if (h % stride[0] == row_seg_idx && k % stride[1] == col_seg_idx) {
                        feature_tmp_pack[i][block_row_idx * block_shape[1] + block_col_idx] =
                            feature_mg.get(channel_idx, h, k);
                    }
                }
            }
        }
    }
    return feature_tmp_pack;
}

void Feature2DEncrypted::pack(const Array<double, 3>& feature_mg, bool is_symmetric, double scale_in) {
    auto pack_type = PackType::MultChannelPack;
    vector<vector<double>> feature_tmp_pack = pack_feature(pack_type, feature_mg);
    uint32_t n_ct = feature_tmp_pack.size();
    const int N_THREAD = 4;

    data.clear();
    data_compress.clear();
    if (is_symmetric) {
        data_compress.resize(n_ct);
    } else {
        data.resize(n_ct);
    }

    parallel_for(n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        auto image_flat_pt = ctx_copy.encode(feature_tmp_pack[ct_idx], level, scale_in);
        if (is_symmetric) {
            auto image_flat_ct = ctx_copy.encrypt_symmetric_compressed(image_flat_pt);
            data_compress[ct_idx] = move(image_flat_ct);
        } else {
            auto image_flat_ct = ctx_copy.encrypt_symmetric(image_flat_pt);
            data[ct_idx] = move(image_flat_ct);
        }
    });
}

void Feature2DEncrypted::single_pack(const Array<double, 3>& feature_mg, bool is_sysmmetric, double scale_in) {
    auto pack_type = PackType::SinglePack;
    vector<vector<double>> feature_tmp_pack = pack_feature(pack_type, feature_mg);

    for (int i = 0; i < n_channel; i++) {
        auto enc = context->encode(feature_tmp_pack[i], level, scale_in);
        data.push_back(context->encrypt_asymmetric(enc));
    }
}

void Feature2DEncrypted::pack_interleaved(const Array<double, 3>& feature_mg,
                                          const Duo& block_shape,
                                          const Duo& stride,
                                          bool is_sysmmetric,
                                          double scale_in) {
    auto pack_type = PackType::InterleavedDecompositionPack;
    vector<vector<double>> feature_tmp_pack = pack_feature(pack_type, feature_mg, block_shape, stride);

    int N_THREAD = 4;
    data.clear();
    data_compress.clear();
    if (is_sysmmetric) {
        data_compress.resize(feature_tmp_pack.size());
    } else {
        data.resize(feature_tmp_pack.size());
    }
    parallel_for(feature_tmp_pack.size(), N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        cxx_sdk_v2::CkksPlaintext enc = ctx_copy.encode(feature_tmp_pack[ct_idx], level, scale_in);
        if (is_sysmmetric) {
            data_compress[ct_idx] = ctx_copy.encrypt_symmetric_compressed(enc);
        } else {
            data[ct_idx] = ctx_copy.encrypt_symmetric(enc);
        }
    });
}

void Feature2DEncrypted::split_with_overlap_pack(const Array<double, 3>& feature_mg,
                                                 const Duo& block_shape,
                                                 const Duo& n_overlap,
                                                 bool is_sysmmetric,
                                                 double scale_in) {
    auto input_shape = feature_mg.get_shape();
    n_channel = input_shape[0];
    shape[0] = input_shape[1];
    shape[1] = input_shape[2];
    n_channel_per_ct = (shape[0] * shape[1] >= context->get_parameter().get_n() / 2) ?
                           1 :
                           context->get_parameter().get_n() / 2 / (shape[0] * shape[1]);

    int row_step = block_shape[0] - n_overlap[0];
    int col_step = block_shape[1] - n_overlap[1];

    int n_row_block =
        (shape[0] <= block_shape[0]) ? 1 : std::ceil((shape[0] - block_shape[0]) / static_cast<float>(row_step)) + 1;
    int n_col_block =
        (shape[1] <= block_shape[1]) ? 1 : std::ceil((shape[1] - block_shape[1]) / static_cast<float>(col_step)) + 1;
    n_segment[0] = n_row_block;
    n_segment[1] = n_col_block;

    segment_valid_range.resize(n_segment[0] * n_segment[1]);
    for (int seg_idx = 0; seg_idx < n_segment[0] * n_segment[1]; seg_idx++) {
        segment_valid_range[seg_idx].resize(4);
    }

    for (int i = 0; i < n_row_block; ++i) {
        int row_start = i * row_step;
        int row_end = std::min(row_start + block_shape[0], shape[0]);
        if (i == n_row_block - 1) {
            row_start = shape[0] - block_shape[0];
            if (row_start < 0)
                row_start = 0;
        }

        for (int j = 0; j < n_col_block; ++j) {
            int col_start = j * col_step;
            int col_end = std::min(col_start + block_shape[1], shape[1]);
            if (j == n_col_block - 1) {
                col_start = shape[1] - block_shape[1];
                if (col_start < 0)
                    col_start = 0;
            }

            int segment_idx = i * n_col_block + j;
            segment_valid_range[segment_idx][0] = row_start;
            segment_valid_range[segment_idx][1] = row_end;
            segment_valid_range[segment_idx][2] = col_start;
            segment_valid_range[segment_idx][3] = col_end;
        }
    }

    int f_ct_num = n_channel * n_segment[0] * n_segment[1];
    vector<vector<double>> feature_tmp_pack(f_ct_num);

    for (int i = 0; i < f_ct_num; i++) {
        int channel_idx = i / (n_segment[0] * n_segment[1]);
        int segment_idx = i % (n_segment[0] * n_segment[1]);
        feature_tmp_pack[i].resize(context->get_parameter().get_n() / 2, 0.0);

        int row_start = segment_valid_range[segment_idx][0];
        int row_end = segment_valid_range[segment_idx][1];
        int col_start = segment_valid_range[segment_idx][2];
        int col_end = segment_valid_range[segment_idx][3];

        int actual_height = row_end - row_start;
        int actual_width = col_end - col_start;

        for (int h = 0; h < actual_height; h++) {
            for (int k = 0; k < actual_width; k++) {
                int pos = h * block_shape[1] + k;
                feature_tmp_pack[i][pos] = feature_mg.get(channel_idx, row_start + h, col_start + k);
            }
        }
    }

    for (int i = 0; i < f_ct_num; i++) {
        cxx_sdk_v2::CkksPlaintext enc = context->encode(feature_tmp_pack[i], level, scale_in);
        data.push_back(context->encrypt_asymmetric(enc));
    }
}

Array<double, 3> Feature2DEncrypted::split_with_overlap_unpack(const Duo& block_shape) const {
    const int N_THREAD = 4;
    int n_ct = data.size();
    Array<double, 3> result({n_channel, shape[0], shape[1]});

    parallel_for(n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        CkksPlaintext x_pt = ctx_copy.decrypt(data[ct_idx]);
        Array1D x_mg = ctx_copy.decode(x_pt);
        int unique_block_idx = ct_idx / (n_segment[0] * n_segment[1]);
        int segment_idx = ct_idx % (n_segment[0] * n_segment[1]);

        int row_start = segment_valid_range[segment_idx][0];
        int row_end = segment_valid_range[segment_idx][1];
        int col_start = segment_valid_range[segment_idx][2];
        int col_end = segment_valid_range[segment_idx][3];

        int actual_height = row_end - row_start;
        int actual_width = col_end - col_start;

        for (int j = 0; j < actual_height; j++) {
            for (int k = 0; k < actual_width; k++) {
                int channel_idx = unique_block_idx * skip[0] * skip[1] + (j % skip[0]) * skip[1] + k % skip[1];
                int row_idx = row_start + j / skip[0];
                int col_idx = col_start + k / skip[1];
                if (channel_idx >= n_channel) {
                    continue;
                }
                result.set(channel_idx, row_idx, col_idx, x_mg[j * block_shape[1] + k]);
            }
        }
    });
    return result;
}

void Feature2DEncrypted::pack_multiplexed(const Array<double, 3>& feature_mg, bool is_sysmmetric, double scale_in) {
    auto pack_type = PackType::ParMultiplexedPack;
    vector<vector<double>> feature_tmp_pack = pack_feature(pack_type, feature_mg);

    for (int i = 0; i < feature_tmp_pack.size(); i++) {
        auto enc = context->encode(feature_tmp_pack[i], level, scale_in);
        if (is_sysmmetric) {
            auto image_flat_ct = context->encrypt_symmetric_compressed(enc);
            data_compress.push_back(move(image_flat_ct));
        } else {
            auto image_flat_ct = context->encrypt_symmetric(enc);
            data.push_back(move(image_flat_ct));
        }
    }
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

Array<double, 3> Feature2DEncrypted::unpack() const {
    const int N_THREAD = 4;
    int n_ct = data.size();
    Duo pre_skip_shape = {shape[0] * skip[0], shape[1] * skip[1]};

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
    Duo pre_skip_shape = {shape[0] * skip[0], shape[1] * skip[1]};
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

Array<double, 3> Feature2DEncrypted::single_unpack() const {
    const int N_THREAD = 4;
    int n_ct = data.size();
    Duo pre_skip_shape = {shape[0] * skip[0], shape[1] * skip[1]};
    Array<double, 3> result({n_channel, shape[0], shape[1]});

    parallel_for(n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        CkksPlaintext x_pt = ctx_copy.decrypt(data[ct_idx]);
        Array1D x_mg = ctx_copy.decode(x_pt);
        int channel_idx = ct_idx;
        for (int j = 0; j < pre_skip_shape[0]; j++) {
            for (int k = 0; k < pre_skip_shape[1]; k++) {
                if (j % skip[0] == 0 && k % skip[1] == 0) {
                    result.set(channel_idx, j / skip[0], k / skip[1], x_mg[j * pre_skip_shape[1] + k]);
                }
            }
        }
    });
    return result;
}

Array<double, 3> Feature2DEncrypted::unpack_multiplexed() const {
    const int N_THREAD = 4;
    int n_ct = data.size();
    Array<double, 3> result({n_channel, shape[0], shape[1]});
    int n_channel_per_block = (skip[0] * skip[1]) / (invalid_fill[0] * invalid_fill[1]);
    int n_channel_per_block_col = skip[1] / invalid_fill[1];
    int n_block_per_ct = n_channel_per_ct / n_channel_per_block;

    parallel_for(n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        CkksPlaintext x_pt = ctx_copy.decrypt(data[ct_idx]);
        Array1D x_mg = ctx_copy.decode(x_pt);
        for (int j = 0; j < n_block_per_ct; j++) {
            for (int x_in_block = 0; x_in_block < (int)(shape[0] * skip[0]); x_in_block++) {
                for (int y_in_block = 0; y_in_block < (int)(shape[1] * skip[1]); y_in_block++) {
                    int channel_x_offset = x_in_block % skip[0];
                    int channel_y_offset = y_in_block % skip[1];
                    // skip invalid slots: valid offsets are within [0, skip[d]/invalid_fill[d])
                    if (channel_x_offset >= (int)(skip[0] / invalid_fill[0]) ||
                        channel_y_offset >= (int)(skip[1] / invalid_fill[1])) {
                        continue;
                    }
                    int channel_idx_in_block = channel_x_offset * n_channel_per_block_col + channel_y_offset;
                    int channel_idx = ct_idx * n_channel_per_ct + j * n_channel_per_block + channel_idx_in_block;
                    if (channel_idx >= (int)n_channel) {
                        continue;
                    }
                    int x = x_in_block / skip[0];
                    int y = y_in_block / skip[1];
                    int slot = j * (shape[0] * skip[0]) * (shape[1] * skip[1]) + x_in_block * (shape[1] * skip[1]) +
                               y_in_block;
                    result.set(channel_idx, x, y, x_mg[slot]);
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
        int channel_idx = ct_idx / (stride[0] * stride[1]);
        int seg_idx = ct_idx % (stride[0] * stride[1]);
        int seg_row_idx = seg_idx / stride[1];
        int seg_col_idx = seg_idx % stride[1];
        for (int j = 0; j < block_shape[0]; j++) {
            for (int k = 0; k < block_shape[1]; k++) {
                result.set(channel_idx, j * stride[0] + seg_row_idx, k * stride[1] + seg_col_idx,
                           x_mg[j * block_shape[1] + k]);
            }
        }
    });
    return result;
}

Array<double, 2> Feature2DEncrypted::unpack_column() const {
    const int N_THREAD = 1;
    int n_ct = data.size();
    Duo pre_skip_shape = {shape[0] * skip[0], shape[1] * skip[1]};

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

void Feature2DEncrypted::block_col_major_pack(const Array<double, 2>& matrix,
                                              uint32_t d,
                                              bool is_symmetric,
                                              double scale_in) {
    uint32_t m = matrix.get_shape()[0];
    uint32_t n_cols = matrix.get_shape()[1];
    uint32_t num_block_rows = div_ceil(m, d);
    uint32_t num_block_cols = div_ceil(n_cols, d);
    int n_slot = context->get_parameter().get_n() / 2;
    uint32_t chunk_size = d * d;
    const int N_THREAD = 4;

    uint32_t total_blocks = num_block_rows * num_block_cols;
    vector<vector<double>> block_vecs(total_blocks);

    // Column-major block order: for bj in [0, num_block_cols), for bi in [0, num_block_rows)
    for (uint32_t bj = 0; bj < num_block_cols; bj++) {
        for (uint32_t bi = 0; bi < num_block_rows; bi++) {
            uint32_t block_idx = bi + num_block_rows * bj;
            vector<double> vec(n_slot, 0.0);
            uint32_t num_chunks = n_slot / chunk_size;
            for (uint32_t c = 0; c < num_chunks; c++) {
                for (uint32_t col = 0; col < d; col++) {
                    for (uint32_t row = 0; row < d; row++) {
                        uint32_t r = bi * d + row;
                        uint32_t c_col = bj * d + col;
                        if (r < m && c_col < n_cols) {
                            vec[c * chunk_size + row + d * col] = matrix.get(r, c_col);
                        }
                    }
                }
            }
            block_vecs[block_idx] = move(vec);
        }
    }

    data.clear();
    data_compress.clear();
    if (is_symmetric) {
        data_compress.resize(total_blocks);
    } else {
        data.resize(total_blocks);
    }

    parallel_for(total_blocks, N_THREAD, *context, [&](CkksContext& ctx_copy, int idx) {
        auto enc = ctx_copy.encode(block_vecs[idx], level, scale_in);
        if (is_symmetric) {
            data_compress[idx] = ctx_copy.encrypt_symmetric_compressed(enc);
        } else {
            data[idx] = ctx_copy.encrypt_symmetric(enc);
        }
    });
}

Array<double, 2> Feature2DEncrypted::block_col_major_unpack(uint32_t m, uint32_t n, uint32_t d) const {
    uint32_t num_block_rows = div_ceil(m, d);
    uint32_t num_block_cols = div_ceil(n, d);
    const int N_THREAD = 4;
    uint32_t total_blocks = num_block_rows * num_block_cols;

    Array<double, 2> result({(uint64_t)m, (uint64_t)n});

    parallel_for(total_blocks, N_THREAD, *context, [&](CkksContext& ctx_copy, int idx) {
        // Recover bi, bj from column-major block index
        uint32_t bi = idx % num_block_rows;
        uint32_t bj = idx / num_block_rows;

        CkksPlaintext x_pt = ctx_copy.decrypt(data[idx]);
        Array1D x_mg = ctx_copy.decode(x_pt);
        // Extract first d*d elements (column-major within block)
        for (uint32_t col = 0; col < d; col++) {
            for (uint32_t row = 0; row < d; row++) {
                uint32_t r = bi * d + row;
                uint32_t c_col = bj * d + col;
                if (r < m && c_col < n) {
                    result.set(r, c_col, x_mg[row + d * col]);
                }
            }
        }
    });
    return result;
}

static uint32_t next_power_of_2(uint32_t x) {
    uint32_t p = 1;
    while (p < x)
        p *= 2;
    return p;
}

void Feature2DEncrypted::par_block_col_major_pack(const Array<double, 2>& matrix,
                                                  uint32_t d,
                                                  uint32_t n_heads,
                                                  bool is_symmetric,
                                                  double scale_in) {
    uint32_t m = matrix.get_shape()[0];
    uint32_t total_cols = matrix.get_shape()[1];
    uint32_t cols_per_head = total_cols / n_heads;
    uint32_t n_h_padded = next_power_of_2(n_heads);
    int n_slot = context->get_parameter().get_n() / 2;
    const int N_THREAD = 4;

    // Determine chunk sizing and n_blocks_per_chunk(S)
    uint32_t S, chunk_size, n_cts_per_block_idx;
    if ((uint32_t)n_slot >= n_h_padded * d * d) {
        S = n_h_padded;
        chunk_size = n_h_padded * d * d;
        n_cts_per_block_idx = 1;
    } else {
        S = n_slot / (d * d);
        chunk_size = n_slot;
        n_cts_per_block_idx = n_h_padded / S;
    }
    uint32_t num_chunks = n_slot / chunk_size;

    uint32_t num_block_rows = div_ceil(m, d);
    uint32_t num_block_cols = div_ceil(cols_per_head, d);
    uint32_t total_vecs = num_block_rows * num_block_cols * n_cts_per_block_idx;

    vector<vector<double>> block_vecs(total_vecs);

    // Column-major block order: for bj, for bi, for g (group number of cts for the same block idx)
    for (uint32_t bj = 0; bj < num_block_cols; bj++) {
        for (uint32_t bi = 0; bi < num_block_rows; bi++) {
            for (uint32_t g = 0; g < n_cts_per_block_idx; g++) {
                uint32_t vec_idx = (bi + num_block_rows * bj) * n_cts_per_block_idx + g;
                vector<double> vec(n_slot, 0.0);

                for (uint32_t h_local = 0; h_local < S; h_local++) {
                    uint32_t h = g * S + h_local;  // global head index
                    for (uint32_t col = 0; col < d; col++) {
                        for (uint32_t row = 0; row < d; row++) {
                            uint32_t r = bi * d + row;
                            uint32_t c = bj * d + col;
                            double val = 0.0;
                            if (h < n_heads && r < m && c < cols_per_head) {
                                val = matrix.get(r, h * cols_per_head + c);
                            }
                            uint32_t base_slot = (row + d * col) * S + h_local;
                            for (uint32_t ci = 0; ci < num_chunks; ci++) {
                                vec[ci * chunk_size + base_slot] = val;
                            }
                        }
                    }
                }
                block_vecs[vec_idx] = move(vec);
            }
        }
    }

    data.clear();
    data_compress.clear();
    if (is_symmetric) {
        data_compress.resize(total_vecs);
    } else {
        data.resize(total_vecs);
    }

    parallel_for(total_vecs, N_THREAD, *context, [&](CkksContext& ctx_copy, int idx) {
        auto enc = ctx_copy.encode(block_vecs[idx], level, scale_in);
        if (is_symmetric) {
            data_compress[idx] = ctx_copy.encrypt_symmetric_compressed(enc);
        } else {
            data[idx] = ctx_copy.encrypt_symmetric(enc);
        }
    });
}

Array<double, 2>
Feature2DEncrypted::par_block_col_major_unpack(uint32_t m, uint32_t n_per_head, uint32_t d, uint32_t n_heads) const {
    uint32_t n_h_padded = next_power_of_2(n_heads);
    int n_slot = context->get_parameter().get_n() / 2;
    const int N_THREAD = 4;

    uint32_t S, chunk_size, n_cts_per_block_idx;
    if ((uint32_t)n_slot >= n_h_padded * d * d) {
        S = n_h_padded;
        chunk_size = n_h_padded * d * d;
        n_cts_per_block_idx = 1;
    } else {
        S = n_slot / (d * d);
        chunk_size = n_slot;
        n_cts_per_block_idx = n_h_padded / S;
    }

    uint32_t num_block_rows = div_ceil(m, d);
    uint32_t num_block_cols = div_ceil(n_per_head, d);
    uint32_t total_vecs = num_block_rows * num_block_cols * n_cts_per_block_idx;
    uint32_t total_cols = n_heads * n_per_head;

    Array<double, 2> result({(uint64_t)m, (uint64_t)total_cols});

    parallel_for(total_vecs, N_THREAD, *context, [&](CkksContext& ctx_copy, int vec_idx) {
        // Recover bi, bj, g from vec_idx
        uint32_t block_idx = vec_idx / n_cts_per_block_idx;
        uint32_t g = vec_idx % n_cts_per_block_idx;
        uint32_t bi = block_idx % num_block_rows;
        uint32_t bj = block_idx / num_block_rows;

        CkksPlaintext x_pt = ctx_copy.decrypt(data[vec_idx]);
        Array1D x_mg = ctx_copy.decode(x_pt);

        for (uint32_t h_local = 0; h_local < S; h_local++) {
            uint32_t h = g * S + h_local;
            if (h >= n_heads)
                continue;
            for (uint32_t col = 0; col < d; col++) {
                for (uint32_t row = 0; row < d; row++) {
                    uint32_t r = bi * d + row;
                    uint32_t c = bj * d + col;
                    if (r < m && c < n_per_head) {
                        uint32_t slot = (row + d * col) * S + h_local;
                        result.set(r, h * n_per_head + c, x_mg[slot]);
                    }
                }
            }
        }
    });
    return result;
}

Feature2DShare::Feature2DShare(uint64_t q, int s) : FeatureShare{q, s} {}

Feature3DShare::Feature3DShare(uint64_t q, int s) : FeatureShare{q, s} {}

void Feature2DEncrypted::split_to_shares(Feature2DEncrypted* share0, Feature2DShare* share1) const {
    int n_slot = context->get_parameter().get_n() / 2;
    double share_scale = ENC_TO_SHARE_SCALE;
    int feature_bitlength = ENC_TO_SHARE_SCALE_BIT + 1;
    int sigma = SIGMA;

    Duo pre_skip_shape = {shape[0] * skip[0], shape[1] * skip[1]};
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
    vector<vector<double>> packed;
    Duo block_expansion = {(uint32_t)ceil(shape[0] / (double)BLOCK_SHAPE[0]),
                           (uint32_t)ceil(shape[1] / (double)BLOCK_SHAPE[1])};
    packed = f2d.pack_feature(pack_type, feature_mg, BLOCK_SHAPE, block_expansion);

    vector<CkksPlaintext> pt_vec;
    for (auto& vec : packed) {
        pt_vec.push_back(context.encode(vec, level, scale_in));
    }
    return pt_vec;
}

void Feature2DEncrypted::split_to_shares_for_multi_channel_pack(Feature2DEncrypted* share0,
                                                                Feature2DShare* share1,
                                                                PackType pack_type_in) const {
    int n_slot = context->get_parameter().get_n() / 2;
    double share_scale = ENC_TO_SHARE_SCALE;
    int feature_bitlength = ENC_TO_SHARE_SCALE_BIT + 1;
    int sigma = SIGMA;
    Duo pre_skip_shape = {shape[0] * skip[0], shape[1] * skip[1]};
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
        multi_pack_to_pt(mask_d_array, *share0, n_channel, shape, skip, *context, level, DEFAULT_SCALE, pack_type_in);
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
                                                                                      const Bytes& b1,
                                                                                      PackType pack_type) const {
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
                                    DEFAULT_SCALE, pack_type);
    Array<double, 1> b1_value({b1.size()});
    for (int i = 0; i < b1.size(); i++) {
        b1_value.set(i, 2 * b1[i] - 1);
    }
    Array<double, 3> b1_value_3d = b1_value.reshape<3>({this->n_channel, this->shape[0], this->shape[1]});
    CkksContext& extra_level_context = context->get_extra_level_context();
    auto mask_b1 =
        multi_pack_to_pt(b1_value_3d, f2d_copy, this->n_channel, this->shape, this->skip, extra_level_context,
                         level + 1, extra_level_context.get_parameter().get_q(level + 1), pack_type);
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
    if (pack_type == PackType::ParMultiplexedPack) {
        x_double_matrix = this->unpack_multiplexed();
    } else if (pack_type == PackType::SinglePack) {
        x_double_matrix = this->unpack();
    } else if (pack_type == PackType::InterleavedDecompositionPack) {
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
    if (pack_type == PackType::SinglePack) {
        this->skip = {1, 1};
    }

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
    if (pack_type == PackType::ParMultiplexedPack) {
        this->pack_multiplexed(y3, true, DEFAULT_SCALE);
    } else if (pack_type == PackType::SinglePack) {
        this->pack(y3, true, DEFAULT_SCALE);
    } else if (pack_type == PackType::InterleavedDecompositionPack) {
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
