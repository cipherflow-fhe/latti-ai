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

using namespace std;

Feature2DEncrypted::Feature2DEncrypted(CkksContext* context_in, int ct_level, Duo skip_in) : skip(skip_in) {
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
        n_channel_per_ct = n_slot / (shape[0] * shape[1]);

        int n_mult_pack_per_ct = n_channel_per_ct / (skip[0] * skip[1]);

        int f_ct_num = div_ceil(n_channel, n_channel_per_ct);
        feature_tmp_pack.resize(f_ct_num);

        for (int i = 0; i < f_ct_num; i++) {
            feature_tmp_pack[i].resize(context->get_parameter().get_n() / 2);
#pragma omp parallel for num_threads(N_THREAD)
            for (int j = 0; j < n_mult_pack_per_ct; j++) {
                for (int h = 0; h < shape[0] * skip[0]; h++) {
                    for (int k = 0; k < shape[1] * skip[1]; k++) {
                        int channel =
                            i * n_channel_per_ct + j * skip[0] * skip[1] + (skip[1] * (h % skip[0]) + k % skip[1]);
                        if (channel >= n_channel) {
                            continue;
                        }
                        feature_tmp_pack[i]
                                        [j * shape[0] * skip[0] * shape[1] * skip[1] + (h * shape[1] * skip[1] + k)] =
                                            feature_mg.get(channel, h / skip[0], k / skip[1]);
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

void Feature2DEncrypted::mult_pack(const Array<double, 3>& feature_mg, bool is_sysmmetric, double scale_in) {
    auto pack_type = PackType::MultiplexedPack;
    vector<vector<double>> feature_tmp_pack = pack_feature(pack_type, feature_mg);

    for (int i = 0; i < feature_tmp_pack.size(); i++) {
        auto enc = context->encode(feature_tmp_pack[i], level, scale_in);
        data.push_back(context->encrypt_asymmetric(enc));
    }
}

void Feature2DEncrypted::split_with_stride_pack(const Array<double, 3>& feature_mg,
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

void Feature2DEncrypted::par_mult_pack(const Array<double, 3>& feature_mg, bool is_sysmmetric, double scale_in) {
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

void Feature2DEncrypted::zero_inserted_mult_pack(const Array<double, 3>& feature_mg,
                                                 const Duo stride,
                                                 bool is_sysmmetric,
                                                 double scale_in) {
    auto input_shape = feature_mg.get_shape();
    n_channel = input_shape[0];
    shape[0] = input_shape[1];
    shape[1] = input_shape[2];
    Duo zero_inserted_shape = {shape[0] * stride[0], shape[1] * stride[1]};
    Duo zero_inserted_skip = {skip[0] / stride[0], skip[1] / stride[1]};
    n_channel_per_ct = context->get_parameter().get_n() / 2 / (zero_inserted_shape[0] * zero_inserted_shape[1]);

    int n_mult_pack_per_ct = n_channel_per_ct / (zero_inserted_skip[0] * zero_inserted_skip[1]);
    int f_ct_num = div_ceil(n_channel, n_channel_per_ct);
    vector<vector<double>> feature_tmp_pack(f_ct_num);

    for (int i = 0; i < f_ct_num; i++) {
        feature_tmp_pack[i].resize(context->get_parameter().get_n() / 2);
        for (int j = 0; j < n_mult_pack_per_ct; j++) {
            for (int h = 0; h < shape[0] * skip[0]; h++) {
                for (int k = 0; k < shape[1] * skip[1]; k++) {
                    int channel = i * n_channel_per_ct + j * zero_inserted_skip[0] * zero_inserted_skip[1] +
                                  (zero_inserted_skip[1] * (h % zero_inserted_skip[0]) + k % zero_inserted_skip[1]);
                    if (channel >= n_channel || (h % skip[0]) >= zero_inserted_skip[0] ||
                        (k % skip[1]) >= zero_inserted_skip[1]) {
                        continue;
                    }
                    feature_tmp_pack[i][j * shape[0] * skip[0] * shape[1] * skip[1] + (h * shape[0] * skip[0] + k)] =
                        feature_mg.get(channel, h / skip[0], k / skip[1]);
                }
            }
        }
    }

    for (int i = 0; i < f_ct_num; i++) {
        auto enc = context->encode(feature_tmp_pack[i], level, scale_in);
        data.push_back(context->encrypt_asymmetric(enc));
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

Array<double, 3> Feature2DEncrypted::mult_unpack() const {
    const int N_THREAD = 4;
    int n_ct = data.size();
    Duo pre_skip_shape = {shape[0] * skip[0], shape[1] * skip[1]};
    Array<double, 3> result({n_channel, shape[0], shape[1]});

    parallel_for(n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        CkksPlaintext x_pt = ctx_copy.decrypt(data[ct_idx]);
        Array1D x_mg = ctx_copy.decode(x_pt);
        for (int j = 0; j < pre_skip_shape[0]; j++) {
            for (int k = 0; k < pre_skip_shape[1]; k++) {
                int channel_idx = ct_idx * n_channel_per_ct + j % skip[0] * skip[1] + k % skip[0];
                if (channel_idx >= n_channel) {
                    continue;
                }
                result.set(channel_idx, j / skip[0], k / skip[0], x_mg[j * pre_skip_shape[0] + k]);
            }
        }
    });
    return result;
}

Array<double, 3> Feature2DEncrypted::par_mult_unpack() const {
    const int N_THREAD = 4;
    int n_ct = data.size();
    Duo pre_skip_shape = {shape[0] * skip[0], shape[1] * skip[1]};
    Array<double, 3> result({n_channel, shape[0], shape[1]});
    int n_mult_pack_per_ct = n_channel_per_ct / (skip[0] * skip[1]);

    parallel_for(n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        CkksPlaintext x_pt = ctx_copy.decrypt(data[ct_idx]);
        Array1D x_mg = ctx_copy.decode(x_pt);
        for (int i = 0; i < n_mult_pack_per_ct; i++) {
            for (int j = 0; j < pre_skip_shape[0]; j++) {
                for (int k = 0; k < pre_skip_shape[1]; k++) {
                    int channel_idx =
                        i * skip[0] * skip[1] + ct_idx * n_channel_per_ct + (j % skip[0]) * skip[1] + k % skip[1];
                    if (channel_idx >= n_channel) {
                        continue;
                    }
                    result.set(channel_idx, j / skip[0], k / skip[1],
                               x_mg[i * pre_skip_shape[0] * pre_skip_shape[1] + j * pre_skip_shape[1] + k]);
                }
            }
        }
    });
    return result;
}

Array<double, 3> Feature2DEncrypted::split_with_stride_unpack(const Duo& block_shape, const Duo& stride) const {
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

Array<double, 3> Feature2DEncrypted::zero_inserted_mult_unpack(const Duo stride_next) const {
    const int N_THREAD = 4;
    int n_ct = data.size();
    Duo pre_skip_shape = {shape[0] * skip[0], shape[1] * skip[1]};
    Array<double, 3> result({n_channel, shape[0], shape[1]});
    Duo next_skip = {skip[0] / stride_next[0], skip[1] / stride_next[1]};
    int n_mult_pack_per_ct = n_channel_per_ct / (next_skip[0] * next_skip[1]);

    parallel_for(n_ct, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        CkksPlaintext x_pt = ctx_copy.decrypt(data[ct_idx]);
        Array1D x_mg = ctx_copy.decode(x_pt);
        for (int i = 0; i < n_mult_pack_per_ct; i++) {
            for (int j = 0; j < pre_skip_shape[0]; j++) {
                for (int k = 0; k < pre_skip_shape[1]; k++) {
                    int channel_idx = i * next_skip[0] * next_skip[1] + ct_idx * n_channel_per_ct +
                                      (j % next_skip[0]) * next_skip[1] + k % next_skip[1];
                    if (channel_idx >= n_channel || (j % skip[0]) >= next_skip[0] || (k % skip[1]) >= next_skip[1]) {
                        continue;
                    }
                    result.set(channel_idx, j / skip[0], k / skip[1],
                               x_mg[i * pre_skip_shape[0] * pre_skip_shape[1] + j * pre_skip_shape[1] + k]);
                }
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
    uint32_t num_block_rows = m / d;
    uint32_t num_block_cols = n_cols / d;
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
                        vec[c * chunk_size + row + d * col] = matrix.get(bi * d + row, bj * d + col);
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
    uint32_t num_block_rows = m / d;
    uint32_t num_block_cols = n / d;
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
                result.set(bi * d + row, bj * d + col, x_mg[row + d * col]);
            }
        }
    });
    return result;
}
