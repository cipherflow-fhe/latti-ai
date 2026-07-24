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

#include "mpc_wrapper/enc_share_conversion.h"

#include <cmath>
#include <iostream>
#include <random>

#include "mpc/fhe_mpc.h"
#include "mpc_array_bridge.h"
#include "mpc/mpc_session.h"
#include "util.h"

using namespace std;
using namespace lattisense;

namespace {

vector<CkksPlaintext> multi_pack_to_pt(const Array<double, 3>& feature_mg,
                                       Feature2DEncrypted& f2d,
                                       int n_channel,
                                       Duo shape,
                                       Duo skip,
                                       CkksContext& context,
                                       int level,
                                       double scale_in,
                                       PackType pack_type) {
    CkksContext* old_context = f2d.context;
    int old_level = f2d.level;
    f2d.context = &context;
    f2d.level = level;

    vector<CkksPlaintext> pt_vec;
    if (pack_type == PackType::MultipleChannelPacking) {
        pt_vec = f2d.encode_multiple_channel(feature_mg, scale_in);
    } else if (pack_type == PackType::MultiplexedPacking) {
        pt_vec = f2d.encode_multiplexed(feature_mg, scale_in);
    } else {
        Duo block_expansion = {(uint32_t)ceil(shape[0] / (double)BLOCK_SHAPE[0]),
                               (uint32_t)ceil(shape[1] / (double)BLOCK_SHAPE[1])};
        pt_vec = f2d.encode_interleaved(feature_mg, BLOCK_SHAPE, block_expansion, scale_in);
    }

    f2d.context = old_context;
    f2d.level = old_level;
    return pt_vec;
}

}  // namespace

EncToShareClient::EncToShareClient(map<string, unique_ptr<CkksContext>>& ckks_contexts,
                                   CkksContext*& context_in,
                                   CkksContext*& context_out,
                                   int scale_ord,
                                   uint64_t ring_mod)
    : ckks_contexts_(ckks_contexts),
      context_in_(context_in),
      context_out_(context_out),
      scale_ord_(scale_ord),
      ring_mod_(ring_mod) {}

ShareToEncClient::ShareToEncClient(CkksContext& context_out, int scale_ord, uint64_t ring_mod, double pt_range)
    : context_out_(context_out), scale_ord_(scale_ord), ring_mod_(ring_mod), pt_range_(pt_range) {}

EncToShareServer::EncToShareServer(CkksContext& context, int scale_ord, uint64_t ring_mod)
    : context_(context), scale_ord_(scale_ord), ring_mod_(ring_mod) {}

ShareToEncServer::ShareToEncServer(CkksContext& context, int scale_ord, uint64_t ring_mod, double pt_range)
    : context_(context), scale_ord_(scale_ord), ring_mod_(ring_mod), pt_range_(pt_range) {}

Array<uint64_t, 1> ShareToEncClient::encrypt_from_share(Feature2DEncrypted& x_enc,
                                                        const Feature2DShare& share,
                                                        int n_channel,
                                                        const Duo& input_shape,
                                                        PackType pack_type) const {
    x_enc.shape = input_shape;
    Array<double, 1> y0_sub_mod_div_s(share.data.get_shape());
    Array<uint64_t, 1> y0_add_mod(share.data.get_shape());
    double scale = mpc::DEFAULT_SCALE;
    for (int i = 0; i < share.data.get_size(); i++) {
        uint64_t y0_add_mod_value = (share.data[i] + (share.ring_mod / 2)) % share.ring_mod;
        y0_add_mod.set(i, y0_add_mod_value);
        double y0_sub = double(int64_t(y0_add_mod_value) - int64_t(share.ring_mod / 2)) / scale;
        y0_sub_mod_div_s.set(i, y0_sub);
    }

    Array<double, 3> y3 = y0_sub_mod_div_s.reshape<3>({uint64_t(n_channel), input_shape[0], input_shape[1]});
    if (pack_type == PackType::MultiplexedPacking) {
        x_enc.pack_multiplexed(y3, true, mpc::DEFAULT_SCALE);
    } else if (pack_type == PackType::MultipleChannelPacking) {
        x_enc.pack_multiple_channel(y3, true, mpc::DEFAULT_SCALE);
    } else if (pack_type == PackType::InterleavedPacking) {
        Duo block_expansion = {(uint32_t)ceil(input_shape[0] / (double)BLOCK_SHAPE[0]),
                               (uint32_t)ceil(input_shape[1] / (double)BLOCK_SHAPE[1])};
        x_enc.pack_interleaved(y3, BLOCK_SHAPE, block_expansion, true);
    }

    return y0_add_mod;
}

void ShareToEncClient::encrypt_from_share_simple(Feature2DEncrypted& x_enc,
                                                 const Feature2DShare& share,
                                                 int n_channel,
                                                 const Duo& input_shape,
                                                 PackType pack_type,
                                                 bool use_recode) const {
    x_enc.shape = input_shape;
    Array<double, 1> share_mg = share.data_double.copy();

    Array<double, 3> y3 = share_mg.reshape<3>({uint64_t(n_channel), input_shape[0], input_shape[1]});
    if (pack_type == PackType::MultipleChannelPacking) {
        x_enc.pack_multiple_channel(y3, true, mpc::DEFAULT_SCALE, use_recode);
    } else if (pack_type == PackType::MultiplexedPacking) {
        x_enc.pack_multiplexed(y3, true, mpc::DEFAULT_SCALE, use_recode);
    } else if (pack_type == PackType::InterleavedPacking) {
        Duo block_expansion = {(uint32_t)ceil(input_shape[0] / (double)BLOCK_SHAPE[0]),
                               (uint32_t)ceil(input_shape[1] / (double)BLOCK_SHAPE[1])};
        x_enc.pack_interleaved(y3, BLOCK_SHAPE, block_expansion, true, mpc::DEFAULT_SCALE, use_recode);
    }
}

Array<uint64_t, 1> ShareToEncClient::encrypt_from_share(Feature0DEncrypted& x_enc,
                                                        const Feature0DShare& share,
                                                        int n_channel) const {
    int n_slot = x_enc.context->get_parameter().get_n() / 2;
    x_enc.skip = 1;

    Array<double, 1> out_data_mg(share.data.get_shape());
    Array<uint64_t, 1> data_add(share.data.get_shape());
    double scale = mpc::DEFAULT_SCALE;
    for (int i = 0; i < share.data.get_size(); i++) {
        uint64_t data_add_value = (share.data[i] + (share.ring_mod / 2)) % share.ring_mod;
        data_add.set(i, data_add_value);
        double out_data_value = double(int64_t(data_add_value) - int64_t(share.ring_mod / 2)) / scale;
        out_data_mg.set(i, out_data_value);
    }

    double encode_scale = pow(2, mpc::DEFAULT_SCALE_BIT);
    x_enc.pack_cyclic(out_data_mg.to_array_1d(), true, encode_scale);
    x_enc.n_channel = n_channel;
    x_enc.n_channel_per_ct = n_slot;
    return data_add;
}

Feature2DEncrypted ShareToEncServer::combine_with_share(const Feature2DEncrypted& y_share1_enc,
                                                        const Feature2DShare& share) const {
    const int N_THREAD = 4;
    int n_slot = y_share1_enc.context->get_parameter().get_n() / 2;
    Feature2DEncrypted result(y_share1_enc.context, y_share1_enc.level);
    result.n_channel = y_share1_enc.n_channel;
    result.n_channel_per_ct = y_share1_enc.n_channel_per_ct;
    result.shape = y_share1_enc.shape;
    result.skip = y_share1_enc.skip;
    double scale = pow(2, share.scale_ord);
    int n_ct = y_share1_enc.data.size();

    result.data.clear();
    result.data.resize(n_ct);
    parallel_for(n_ct, N_THREAD, *y_share1_enc.context, [&](CkksContext& ctx_copy, int i) {
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
        CkksPlaintext mask_pt =
            ctx_copy.encode(mask_d, y_share1_enc.level, ctx_copy.get_parameter().get_default_scale());
        result.data[i] = ctx_copy.add_plain(y_share1_enc.data[i], mask_pt);
    });
    return result;
}

Feature2DEncrypted ShareToEncServer::combine_with_share_simple(const Feature2DEncrypted& y_share1_enc,
                                                               const Feature2DShare& share) const {
    const int N_THREAD = 4;
    int n_slot = y_share1_enc.context->get_parameter().get_n() / 2;
    Feature2DEncrypted result(y_share1_enc.context, y_share1_enc.level);
    result.n_channel = y_share1_enc.n_channel;
    result.n_channel_per_ct = y_share1_enc.n_channel_per_ct;
    result.shape = y_share1_enc.shape;
    result.skip = y_share1_enc.skip;
    int n_ct = y_share1_enc.data.size();

    result.data.clear();
    result.data.resize(n_ct);
    parallel_for(n_ct, N_THREAD, *y_share1_enc.context, [&](CkksContext& ctx_copy, int i) {
        vector<double> mask_d(n_slot);
        for (int j = 0; j < n_slot; j++) {
            int share_idx = (i * n_slot + j) % share.data_double.get_size();
            mask_d[j] = share.data_double.get(share_idx);
        }
        CkksPlaintext mask_pt =
            ctx_copy.encode(mask_d, y_share1_enc.level, ctx_copy.get_parameter().get_default_scale());
        result.data[i] = ctx_copy.add_plain(y_share1_enc.data[i], mask_pt);
    });
    return result;
}

Feature2DEncrypted ShareToEncServer::combine_with_share_simple_for_multi_pack(
    const Feature2DEncrypted& y_share1_enc,
    const Feature2DShare& share,
    PackType pack_type) const {
    Feature2DEncrypted result(y_share1_enc.context, y_share1_enc.level, y_share1_enc.skip, y_share1_enc.invalid_fill,
                              pack_type);
    result.n_channel = y_share1_enc.n_channel;
    result.n_channel_per_ct = y_share1_enc.n_channel_per_ct;
    result.shape = y_share1_enc.shape;
    result.skip = y_share1_enc.skip;
    result.packing_type = pack_type;

    Array<double, 1> share_mg = share.data_double.copy();
    Array<double, 3> share_mg_3d =
        share_mg.reshape<3>({y_share1_enc.n_channel, y_share1_enc.shape[0], y_share1_enc.shape[1]});

    int n_ct = y_share1_enc.data.size();

    auto result_copy = result.copy();
    auto mask_pt = multi_pack_to_pt(share_mg_3d, result_copy, y_share1_enc.n_channel, y_share1_enc.shape,
                                    y_share1_enc.skip, *y_share1_enc.context, y_share1_enc.level,
                                    y_share1_enc.context->get_parameter().get_default_scale(), pack_type);

    result.data.clear();
    result.data.resize(n_ct);
    if ((int)mask_pt.size() != n_ct) {
        throw runtime_error("combine_with_share_simple_for_multi_pack mask/plaintext count mismatch: mask_pt=" +
                            to_string(mask_pt.size()) + ", ct=" + to_string(n_ct));
    }
    for (int i = 0; i < n_ct; i++) {
        result.data[i] = y_share1_enc.context->add_plain(y_share1_enc.data[i], mask_pt[i]);
    }
    return result;
}

Feature2DEncrypted ShareToEncServer::combine_with_share_new_protocol(const Feature2DEncrypted& y_share1_enc,
                                                                     const Feature2DShare& share,
                                                                     const Feature2DEncrypted& y_share2_enc,
                                                                     const Bytes& b1) const {
    const int N_THREAD = 8;
    int n_slot = y_share1_enc.context->get_parameter().get_n() / 2;
    Feature2DEncrypted result(y_share1_enc.context, y_share1_enc.level);
    result.n_channel = y_share1_enc.n_channel;
    result.n_channel_per_ct = y_share1_enc.n_channel_per_ct;
    result.shape = y_share1_enc.shape;
    result.skip = y_share1_enc.skip;
    double scale = mpc::DEFAULT_SCALE;
    double encode_scale = pow(2, mpc::DEFAULT_SCALE_BIT);
    int n_ct = y_share1_enc.data.size();

    result.data.clear();
    result.data.resize(n_ct);

    parallel_for_with_extra_level_context(
        n_ct, N_THREAD, *y_share1_enc.context, [&](CkksContext& ctx_copy, CkksContext& extra_level_ctx_copy, int i) {
            vector<double> mask_d(n_slot, 0);
            vector<double> b1_value(n_slot, 0);
            for (int j = 0; j < n_slot; j++) {
                int mg_idx = (i * n_slot + j) % share.data.get_size();
                b1_value[j] = 2 * b1[mg_idx] - 1;
                int64_t mask_value = int64_t(share.data.get(mg_idx)) - int64_t(b1[mg_idx] * share.ring_mod);
                mask_d[j] = double(mask_value) / scale;
            }
            CkksPlaintext mask_pt = ctx_copy.encode(mask_d, y_share1_enc.level, encode_scale);
            result.data[i] = ctx_copy.add_plain(y_share1_enc.data[i], mask_pt);

            CkksPlaintext b1_pt = extra_level_ctx_copy.encode(
                b1_value, y_share1_enc.level + 1, extra_level_ctx_copy.get_parameter().get_q(y_share1_enc.level + 1));

            auto f2d_mult = extra_level_ctx_copy.mult_plain(y_share2_enc.data[i], b1_pt);
            f2d_mult = extra_level_ctx_copy.rescale(f2d_mult, encode_scale);

            result.data[i] = ctx_copy.add(result.data[i], f2d_mult);
        });
    return result;
}

Feature2DEncrypted ShareToEncServer::combine_with_share_new_protocol_for_multi_pack(
    const Feature2DEncrypted& y_share1_enc,
    const Feature2DShare& share,
    const Feature2DEncrypted& y_share2_enc,
    const Bytes& b1,
    PackType pack_type) const {
    double scale = mpc::DEFAULT_SCALE;
    double encode_scale = pow(2, mpc::DEFAULT_SCALE_BIT);
    Feature2DEncrypted result(y_share1_enc.context, y_share1_enc.level);
    result.n_channel = y_share1_enc.n_channel;
    result.n_channel_per_ct = y_share1_enc.n_channel_per_ct;
    result.shape = y_share1_enc.shape;
    result.skip = y_share1_enc.skip;

    result.data.clear();
    result.data.resize(y_share1_enc.data.size());

    Array<double, 1> mask_d({share.data.get_size()});
    for (int i = 0; i < share.data.get_size(); i++) {
        int64_t mask_value = int64_t(share.data.get(i)) - int64_t(b1[i] * share.ring_mod);
        mask_d.set(i, mask_value / scale);
    }
    auto y_share2_copy = y_share2_enc.copy();
    Array<double, 3> mask_d_3d =
        mask_d.reshape<3>({y_share1_enc.n_channel, y_share1_enc.shape[0], y_share1_enc.shape[1]});
    auto mask_pt = multi_pack_to_pt(mask_d_3d, y_share2_copy, y_share1_enc.n_channel, y_share1_enc.shape,
                                    y_share1_enc.skip, *y_share1_enc.context, y_share1_enc.level, mpc::DEFAULT_SCALE,
                                    pack_type);
    Array<double, 1> b1_value({b1.size()});
    for (int i = 0; i < b1.size(); i++) {
        b1_value.set(i, 2 * b1[i] - 1);
    }
    Array<double, 3> b1_value_3d =
        b1_value.reshape<3>({y_share1_enc.n_channel, y_share1_enc.shape[0], y_share1_enc.shape[1]});
    CkksContext& extra_level_context = y_share1_enc.context->get_extra_level_context();
    auto mask_b1 = multi_pack_to_pt(b1_value_3d, y_share2_copy, y_share1_enc.n_channel, y_share1_enc.shape,
                                    y_share1_enc.skip, extra_level_context, y_share1_enc.level + 1,
                                    extra_level_context.get_parameter().get_q(y_share1_enc.level + 1), pack_type);
    for (int i = 0; i < y_share1_enc.data.size(); i++) {
        auto f2d_mult = extra_level_context.mult_plain(y_share2_enc.data[i], mask_b1[i]);
        f2d_mult = extra_level_context.rescale(f2d_mult, encode_scale);
        result.data[i] = y_share1_enc.context->add_plain(y_share1_enc.data[i], mask_pt[i]);
        result.data[i] = y_share1_enc.context->add(result.data[i], f2d_mult);
    }
    return result;
}

Feature0DEncrypted ShareToEncServer::combine_with_share(const Feature0DEncrypted& y_share1_enc,
                                                        const Feature0DShare& share) const {
    int n_slot = y_share1_enc.context->get_parameter().get_n() / 2;
    Feature0DEncrypted result(y_share1_enc.context, y_share1_enc.level);
    result.n_channel = y_share1_enc.n_channel;
    result.n_channel_per_ct = y_share1_enc.n_channel_per_ct;
    result.skip = y_share1_enc.skip;
    double scale = pow(2, share.scale_ord);

    for (int i = 0; i < y_share1_enc.data.size(); i++) {
        vector<double> mask_d(n_slot, 0.0);
        for (int j = 0; j < n_slot; j++) {
            if (i * n_slot + j >= share.data.get_size()) {
                mask_d[j] =
                    uint64_to_double(share.data.get((i * n_slot + j) % share.data.get_size()), scale, share.ring_mod);
            } else {
                mask_d[j] = uint64_to_double(share.data.get(i * n_slot + j), scale, share.ring_mod);
            }
        }
        CkksPlaintext mask_pt =
            y_share1_enc.context->encode(mask_d, y_share1_enc.level,
                                         y_share1_enc.context->get_parameter().get_default_scale());
        result.data.push_back(y_share1_enc.context->add_plain(y_share1_enc.data[i], mask_pt));
    }
    return result;
}

Feature0DEncrypted ShareToEncServer::combine_with_share_new_protocol(const Feature0DEncrypted& y_share1_enc,
                                                                     const Feature0DShare& share,
                                                                     const Feature0DEncrypted& y_share2_enc,
                                                                     const Bytes& b1) const {
    int n_slot = y_share1_enc.context->get_parameter().get_n() / 2;
    Feature0DEncrypted result(y_share1_enc.context, y_share1_enc.level);
    result.n_channel = y_share1_enc.n_channel;
    result.n_channel_per_ct = y_share1_enc.n_channel_per_ct;
    result.skip = y_share1_enc.skip;
    double scale = mpc::DEFAULT_SCALE;
    double encode_scale = pow(2, mpc::DEFAULT_SCALE_BIT);

    for (int i = 0; i < y_share1_enc.data.size(); i++) {
        vector<double> b1_value(n_slot, 0);
        vector<double> mask_d(n_slot, 0.0);
        for (int j = 0; j < n_slot; j++) {
            int64_t mask_value;
            if (i * n_slot + j >= share.data.get_size()) {
                b1_value[j] = b1[(i * n_slot + j) % share.data.get_size()];
                mask_value = int64_t(share.data.get((i * n_slot + j) % share.data.get_size())) -
                             int64_t(b1_value[j] * share.ring_mod);
            } else {
                b1_value[j] = b1[i * n_slot + j];
                mask_value = int64_t(share.data.get(i * n_slot + j)) - int64_t(b1[i * n_slot + j] * share.ring_mod);
            }
            b1_value[j] = 2 * b1_value[j] - 1;
            mask_d[j] = double(mask_value) / scale;
        }
        CkksPlaintext mask_pt = y_share1_enc.context->encode(mask_d, y_share1_enc.level, encode_scale);
        result.data.push_back(y_share1_enc.context->add_plain(y_share1_enc.data[i], mask_pt));

        CkksContext& ctx_extra = y_share1_enc.context->get_extra_level_context();
        CkksPlaintext b1_pt =
            ctx_extra.encode(b1_value, y_share1_enc.level + 1, ctx_extra.get_parameter().get_q(y_share1_enc.level + 1));
        auto f2d_mult = ctx_extra.mult_plain(y_share2_enc.data[i], b1_pt);
        f2d_mult = ctx_extra.rescale(f2d_mult, encode_scale);

        result.data[i] = y_share1_enc.context->add(result.data[i], f2d_mult);
    }
    return result;
}

Feature2DShare EncToShareClient::decrypt_to_share(const Feature2DEncrypted& x_enc, PackType pack_type) const {
    Feature2DShare share(ring_mod_, scale_ord_);
    share.shape = x_enc.shape;

    Array<double, 3> x_double_matrix;
    if (pack_type == PackType::MultiplexedPacking) {
        x_double_matrix = x_enc.unpack_multiplexed();
    } else if (pack_type == PackType::MultipleChannelPacking) {
        x_double_matrix = x_enc.unpack_multiple_channel();
    } else if (pack_type == PackType::InterleavedPacking) {
        Duo block_expansion = {(uint32_t)ceil(x_enc.shape[0] / (double)BLOCK_SHAPE[0]),
                               (uint32_t)ceil(x_enc.shape[1] / (double)BLOCK_SHAPE[1])};
        x_double_matrix = x_enc.unpack_interleaved(BLOCK_SHAPE, block_expansion);
    }

    share.data = array_double_to_uint64(x_double_matrix, share.scale_ord, share.ring_mod).reshape<1>({0});
    return share;
}

Feature2DShare EncToShareClient::decrypt_to_share_simple(const Feature2DEncrypted& x_enc, PackType pack_type) const {
    Feature2DShare share(ring_mod_, scale_ord_);
    share.shape = x_enc.shape;

    Array<double, 3> x_double_matrix;
    if (pack_type == PackType::MultiplexedPacking) {
        x_double_matrix = x_enc.unpack_multiplexed();
    } else if (pack_type == PackType::MultipleChannelPacking) {
        x_double_matrix = x_enc.unpack_multiple_channel();
    } else if (pack_type == PackType::InterleavedPacking) {
        Duo block_expansion = {(uint32_t)ceil(x_enc.shape[0] / (double)BLOCK_SHAPE[0]),
                               (uint32_t)ceil(x_enc.shape[1] / (double)BLOCK_SHAPE[1])};
        x_double_matrix = x_enc.unpack_interleaved(BLOCK_SHAPE, block_expansion);
    }

    share.data_double.resize({x_double_matrix.get_size()});
    for (int i = 0; i < x_double_matrix.get_size(); i++) {
        share.data_double.set(i, x_double_matrix.get_data()[i]);
    }
    return share;
}

Feature0DShare EncToShareClient::decrypt_to_share(const Feature0DEncrypted& x_enc, int n_channel) const {
    (void)n_channel;
    Feature0DShare share(ring_mod_, scale_ord_);
    Array<double, 1> x_double_vec = x_enc.unpack();
    share.data = array_double_to_uint64(x_double_vec, share.scale_ord, share.ring_mod);
    return share;
}

void EncToShareServer::split_to_shares(const Feature2DEncrypted& x_enc,
                                       Feature2DEncrypted* share0,
                                       Feature2DShare* share1) const {
    int n_slot = x_enc.context->get_parameter().get_n() / 2;
    double share_scale = mpc::DEFAULT_SCALE;
    int feature_bitlength = mpc::DEFAULT_SCALE_BIT + 1;
    int sigma = mpc::SIGMA;

    Duo pre_skip_shape = x_enc.shape * x_enc.skip;
    size_t n_share_feature = x_enc.n_channel * x_enc.shape[0] * x_enc.shape[1];
    size_t n_mask = x_enc.n_channel * pre_skip_shape[0] * pre_skip_shape[1];

    vector<double> mask_d(n_mask);
    vector<int64_t> r(n_mask);
    for (int i = 0; i < n_mask; i++) {
        r[i] = int64_t(gen_random_uint(feature_bitlength + sigma)) - int64_t(1ull << (feature_bitlength + sigma - 1));
        mask_d[i] = double(r[i]) / share_scale;
    }

    share0->n_channel = x_enc.n_channel;
    share0->n_channel_per_ct = x_enc.n_channel_per_ct;
    share0->shape = x_enc.shape;
    share0->skip = x_enc.skip;
    share0->level = x_enc.level;
    share0->data.clear();
    vector<double> mask_d_span(mask_d);
    for (int i = 0; i < x_enc.data.size(); i++) {
        size_t start = i * n_slot;
        size_t length = i == x_enc.data.size() - 1 ? (mask_d_span.size() - start) : n_slot;
        vector<double> mask_mg_vec(mask_d_span.begin() + start, mask_d_span.begin() + start + length);
        CkksPlaintext mask_pt = x_enc.context->encode(mask_mg_vec, x_enc.level, mpc::DEFAULT_SCALE);
        CkksCiphertext share0_ct = x_enc.context->add_plain(x_enc.data[i], mask_pt);
        share0->data.push_back(move(share0_ct));
    }

    share1->shape = x_enc.shape;
    share1->data.resize({n_share_feature});
    for (int i = 0; i < x_enc.n_channel; i++) {
        for (int j = 0; j < x_enc.shape[0]; j++) {
            for (int k = 0; k < x_enc.shape[1]; k++) {
                int skipped_index = i * x_enc.shape[0] * x_enc.shape[1] + j * x_enc.shape[1] + k;
                int pre_skip_index = i * pre_skip_shape[0] * pre_skip_shape[1] +
                                     j * pre_skip_shape[1] * x_enc.skip[0] + k * x_enc.skip[1];
                share1->data[skipped_index] =
                    (-int64_t(r[pre_skip_index]) % share1->ring_mod + share1->ring_mod) % share1->ring_mod;
            }
        }
    }
}

void EncToShareServer::split_to_shares_simple(const Feature2DEncrypted& x_enc,
                                              Feature2DEncrypted* share0,
                                              Feature2DShare* share1) const {
    int n_slot = x_enc.context->get_parameter().get_n() / 2;
    int sigma = mpc::SIGMA;

    Duo pre_skip_shape = x_enc.shape * x_enc.skip;
    size_t n_share_feature = x_enc.n_channel * x_enc.shape[0] * x_enc.shape[1];
    size_t n_mask = x_enc.n_channel * pre_skip_shape[0] * pre_skip_shape[1];

    vector<double> mask_d(n_mask);
    vector<double> r(n_mask);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(-pow(2, mpc::DATA_BIT + sigma), pow(2, mpc::DATA_BIT + sigma));
    for (int i = 0; i < n_mask; i++) {
        r[i] = dis(gen);
        mask_d[i] = -r[i];
    }

    share0->n_channel = x_enc.n_channel;
    share0->n_channel_per_ct = x_enc.n_channel_per_ct;
    share0->shape = x_enc.shape;
    share0->skip = x_enc.skip;
    share0->level = x_enc.level;
    share0->data.clear();
    vector<double> mask_d_span(mask_d);
    for (int i = 0; i < x_enc.data.size(); i++) {
        size_t start = i * n_slot;
        size_t length = i == x_enc.data.size() - 1 ? (mask_d_span.size() - start) : n_slot;
        vector<double> mask_mg_vec(mask_d_span.begin() + start, mask_d_span.begin() + start + length);
        CkksPlaintext mask_pt = x_enc.context->encode(mask_mg_vec, x_enc.level, mpc::DEFAULT_SCALE);
        CkksCiphertext share0_ct = x_enc.context->add_plain(x_enc.data[i], mask_pt);
        share0->data.push_back(move(share0_ct));
    }

    share1->shape = x_enc.shape;
    share1->data_double.resize({n_share_feature});
    for (int i = 0; i < x_enc.n_channel; i++) {
        for (int j = 0; j < x_enc.shape[0]; j++) {
            for (int k = 0; k < x_enc.shape[1]; k++) {
                int skipped_index = i * x_enc.shape[0] * x_enc.shape[1] + j * x_enc.shape[1] + k;
                int pre_skip_index = i * pre_skip_shape[0] * pre_skip_shape[1] +
                                     j * pre_skip_shape[1] * x_enc.skip[0] + k * x_enc.skip[1];
                share1->data_double[skipped_index] = r[pre_skip_index];
            }
        }
    }
}

void EncToShareServer::split_to_shares_for_multi_channel_pack(const Feature2DEncrypted& x_enc,
                                                              Feature2DEncrypted* share0,
                                                              Feature2DShare* share1,
                                                              PackType pack_type) const {
    double share_scale = mpc::DEFAULT_SCALE;
    int feature_bitlength = mpc::DEFAULT_SCALE_BIT + 1;
    int sigma = mpc::SIGMA;
    size_t n_mask = x_enc.n_channel * x_enc.shape[0] * x_enc.shape[1];

    vector<double> mask_d(n_mask);
    vector<int64_t> r(n_mask);
    for (int i = 0; i < n_mask; i++) {
        r[i] = int64_t(gen_random_uint(feature_bitlength + sigma)) - int64_t(1ull << (feature_bitlength + sigma - 1));
        mask_d[i] = double(r[i]) / share_scale;
    }
    share0->n_channel = x_enc.n_channel;
    share0->n_channel_per_ct = x_enc.n_channel_per_ct;
    share0->shape = x_enc.shape;
    share0->skip = x_enc.skip;
    share0->level = x_enc.level;
    share0->data.clear();
    auto mask_d_array = Array<double, 1>::from_array_1d(mask_d).reshape<3>(
        {x_enc.n_channel, x_enc.shape[0], x_enc.shape[1]});
    auto mask_pt = multi_pack_to_pt(mask_d_array, *share0, x_enc.n_channel, x_enc.shape, x_enc.skip, *x_enc.context,
                                    x_enc.level, mpc::DEFAULT_SCALE, pack_type);
    if (mask_pt.size() != x_enc.data.size()) {
        throw runtime_error("split_to_shares_for_multi_channel_pack mask/plaintext count mismatch: mask_pt=" +
                            to_string(mask_pt.size()) + ", ct=" + to_string(x_enc.data.size()));
    }
    for (int i = 0; i < x_enc.data.size(); i++) {
        CkksCiphertext share0_ct = x_enc.context->add_plain(x_enc.data[i], mask_pt[i]);
        share0->data.push_back(move(share0_ct));
    }

    share1->shape = x_enc.shape;
    share1->data.resize({n_mask});
    for (int i = 0; i < n_mask; i++) {
        share1->data[i] = (-int64_t(r[i]) % share1->ring_mod + share1->ring_mod) % share1->ring_mod;
    }
}

void EncToShareServer::split_to_shares_for_multi_channel_pack_simple(const Feature2DEncrypted& x_enc,
                                                                     Feature2DEncrypted* share0,
                                                                     Feature2DShare* share1,
                                                                     PackType pack_type) const {
    int sigma = mpc::SIGMA;
    size_t n_share_feature = x_enc.n_channel * x_enc.shape[0] * x_enc.shape[1];

    vector<double> mask_d(n_share_feature);
    vector<double> r(n_share_feature);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(-pow(2, mpc::DATA_BIT + sigma), pow(2, mpc::DATA_BIT + sigma));
    for (int i = 0; i < n_share_feature; i++) {
        r[i] = dis(gen);
        mask_d[i] = -r[i];
    }

    share0->n_channel = x_enc.n_channel;
    share0->n_channel_per_ct = x_enc.n_channel_per_ct;
    share0->shape = x_enc.shape;
    share0->skip = x_enc.skip;
    share0->level = x_enc.level;
    share0->data.clear();
    share0->packing_type = pack_type;

    auto mask_d_array = Array<double, 1>::from_array_1d(mask_d).reshape<3>(
        {x_enc.n_channel, x_enc.shape[0], x_enc.shape[1]});
    auto mask_pt = multi_pack_to_pt(mask_d_array, *share0, x_enc.n_channel, x_enc.shape, x_enc.skip, *x_enc.context,
                                    x_enc.level, mpc::DEFAULT_SCALE, pack_type);
    if (mask_pt.size() != x_enc.data.size()) {
        throw runtime_error("split_to_shares_for_multi_channel_pack_simple mask/plaintext count mismatch: mask_pt=" +
                            to_string(mask_pt.size()) + ", ct=" + to_string(x_enc.data.size()));
    }
    for (int i = 0; i < x_enc.data.size(); i++) {
        CkksCiphertext share0_ct = x_enc.context->add_plain(x_enc.data[i], mask_pt[i]);
        share0->data.push_back(move(share0_ct));
    }

    share1->shape = x_enc.shape;
    share1->data_double.resize({n_share_feature});
    for (int i = 0; i < n_share_feature; i++) {
        share1->data_double[i] = r[i];
    }
}

void EncToShareServer::split_to_shares(const Feature0DEncrypted& x_enc,
                                       Feature0DEncrypted* share0,
                                       Feature0DShare* share1) const {
    int n_slot = x_enc.context->get_parameter().get_n() / 2;
    double share_scale = pow(2, share1->scale_ord);
    int feature_bitlength = mpc::DEFAULT_SCALE_BIT + 1;
    int sigma = mpc::SIGMA;
    share0->n_channel = x_enc.n_channel;
    share0->n_channel_per_ct = x_enc.n_channel_per_ct;
    share0->skip = x_enc.skip;
    share0->level = x_enc.level;
    share0->data.clear();
    vector<vector<double>> mask_d_mat;
    vector<vector<int64_t>> r_mat;
    for (int i = 0; i < x_enc.data.size(); i++) {
        vector<double> mask_d(n_slot);
        vector<int64_t> r(n_slot);
        for (int j = 0; j < n_slot; j++) {
            r[j] =
                int64_t(gen_random_uint(feature_bitlength + sigma)) - int64_t(1ull << (feature_bitlength + sigma - 1));
            mask_d[j] = double(r[j]) / share_scale;
        }
        mask_d_mat.push_back(mask_d);
        r_mat.push_back(r);
        CkksPlaintext mask_pt = x_enc.context->encode(mask_d, x_enc.level, mpc::DEFAULT_SCALE);
        CkksCiphertext share0_ct = x_enc.context->add_plain(x_enc.data[i], mask_pt);

        share0->data.push_back(move(share0_ct));
    }

    share1->data.resize({x_enc.n_channel});
    int T = 0;
    for (int i = 0; i < mask_d_mat.size(); i++) {
        for (int j = 0; j < x_enc.n_channel_per_ct; j++) {
            if (T >= x_enc.n_channel) {
                break;
            }
            uint64_t neg_r = (-r_mat[i][j * x_enc.skip] % share1->ring_mod + share1->ring_mod) % share1->ring_mod;
            share1->data.set(T, neg_r);
            T = T + 1;
        }
    }
}

void EncToShareServer::split_to_shares_reshape(const Feature0DEncrypted& x_enc,
                                               Feature0DEncrypted* share0,
                                               Feature0DShare* share1) const {
    int n_slot = x_enc.context->get_parameter().get_n() / 2;
    double share_scale = pow(2, share1->scale_ord);
    int feature_bitlength = mpc::DEFAULT_SCALE_BIT + 1;
    int sigma = mpc::SIGMA;
    share0->n_channel = x_enc.n_channel;
    share0->n_channel_per_ct = x_enc.n_channel_per_ct;
    share0->skip = x_enc.skip;
    share0->level = x_enc.level;
    share0->data.clear();
    vector<vector<double>> mask_d_mat;
    vector<vector<int64_t>> r_mat;
    for (int i = 0; i < x_enc.data.size(); i++) {
        vector<double> mask_d(n_slot);
        vector<int64_t> r(n_slot);
        for (int j = 0; j < n_slot; j++) {
            r[j] =
                int64_t(gen_random_uint(feature_bitlength + sigma)) - int64_t(1ull << (feature_bitlength + sigma - 1));
            mask_d[j] = double(r[j]) / share_scale;
        }
        r_mat.push_back(r);
        mask_d_mat.push_back(mask_d);
        CkksPlaintext mask_pt = x_enc.context->encode(mask_d, x_enc.level, mpc::DEFAULT_SCALE);
        CkksCiphertext share0_ct = x_enc.context->add_plain(x_enc.data[i], mask_pt);
        share0->data.push_back(move(share0_ct));
    }

    share1->data.resize({x_enc.n_channel});
    int T = 0;

    for (int i = 0; i < mask_d_mat.size(); i++) {
        for (int j = 0; j < div_ceil(x_enc.n_channel, x_enc.data.size()); j++) {
            if (T >= x_enc.n_channel) {
                break;
            }
            uint64_t neg_r = (-r_mat[i][j * x_enc.skip] % share1->ring_mod + share1->ring_mod) % share1->ring_mod;
            share1->data.set(T, neg_r);
            T += 1;
        }
    }
}

Feature2DShare EncToShareClient::client_enc_to_share(const Bytes& meta_data_bytes) {
    DataTransmission data_trans = ::mpc::data_transmission();

    uint8_t level;
    uint8_t param_id_in;
    uint8_t param_id_out;
    bytes_to_va(meta_data_bytes, {"u8", "u8", "u8"}, &level, &param_id_in, &param_id_out);

    string param_in = param_to_string(param_id_in);
    string param_out = param_to_string(param_id_out);
    context_in_ = ckks_contexts_[param_in].get();
    context_out_ = ckks_contexts_[param_out].get();

    Bytes x_e_bytes = data_trans.receive_bytes();
    Feature2DEncrypted x_e(context_in_, level);
    x_e.deserialize(x_e_bytes);
    return decrypt_to_share(x_e, PackType::MultipleChannelPacking);
}

Feature2DShare EncToShareClient::client_enc_to_share_simple(const Bytes& meta_data_bytes) {
    DataTransmission data_trans = ::mpc::data_transmission();

    uint8_t level;
    uint8_t param_id_in;
    uint8_t param_id_out;
    bytes_to_va(meta_data_bytes, {"u8", "u8", "u8"}, &level, &param_id_in, &param_id_out);

    string param_in = param_to_string(param_id_in);
    string param_out = param_to_string(param_id_out);
    context_in_ = ckks_contexts_[param_in].get();
    context_out_ = ckks_contexts_[param_out].get();

    Bytes x_e_bytes = data_trans.receive_bytes();
    Feature2DEncrypted x_e(context_in_, level);
    x_e.deserialize(x_e_bytes);
    return decrypt_to_share_simple(x_e, PackType::MultipleChannelPacking);
}

Feature2DShare EncToShareClient::client_enc_to_share_for_multi_channel_pack(const Bytes& meta_data_bytes) {
    DataTransmission data_trans = ::mpc::data_transmission();
    uint8_t level;
    uint8_t param_id_in;
    uint8_t param_id_out;
    uint8_t temp_int = 0;
    bytes_to_va(meta_data_bytes, {"u8", "u8", "u8", "u8"}, &level, &param_id_in, &param_id_out, &temp_int);
    PackType pack_type = (PackType)temp_int;
    cout << "[mpc_refresh][client] enc_to_share pack_type=" << static_cast<int>(pack_type)
         << ", level=" << static_cast<int>(level) << endl;

    string param_in = param_to_string(param_id_in);
    string param_out = param_to_string(param_id_out);
    context_in_ = ckks_contexts_[param_in].get();
    context_out_ = ckks_contexts_[param_out].get();

    Bytes x_e_bytes = data_trans.receive_bytes();
    Feature2DEncrypted x_e(context_in_, level);
    x_e.deserialize(x_e_bytes);
    return decrypt_to_share(x_e, pack_type);
}

Feature2DShare EncToShareClient::client_enc_to_share_for_multi_channel_pack_simple(const Bytes& meta_data_bytes) {
    DataTransmission data_trans = ::mpc::data_transmission();
    uint8_t level;
    uint8_t param_id_in;
    uint8_t param_id_out;
    uint8_t temp_int = 0;
    bytes_to_va(meta_data_bytes, {"u8", "u8", "u8", "u8"}, &level, &param_id_in, &param_id_out, &temp_int);
    PackType pack_type = (PackType)temp_int;

    string param_in = param_to_string(param_id_in);
    string param_out = param_to_string(param_id_out);
    context_in_ = ckks_contexts_[param_in].get();
    context_out_ = ckks_contexts_[param_out].get();

    Bytes x_e_bytes = data_trans.receive_bytes();
    Feature2DEncrypted x_e(context_in_, level);
    x_e.deserialize(x_e_bytes);
    return decrypt_to_share_simple(x_e, pack_type);
}

Feature0DShare EncToShareClient::client_enc_to_share_0d(const Bytes& meta_data_bytes) {
    DataTransmission data_trans = ::mpc::data_transmission();
    uint8_t level;
    uint32_t n_channal;
    uint8_t param_in_id;
    uint8_t param_out_id;

    bytes_to_va(meta_data_bytes, {"u8", "u32", "u8", "u8"}, &level, &n_channal, &param_in_id, &param_out_id);
    string param_in = param_to_string(param_in_id);
    string param_out = param_to_string(param_out_id);
    context_in_ = ckks_contexts_.at(param_in).get();
    context_out_ = ckks_contexts_.at(param_out).get();

    Bytes x_e_bytes = data_trans.receive_bytes();
    if (context_in_ == nullptr) {
        cout << "wrong ptr" << endl;
    }
    Feature0DEncrypted x_e(context_in_, level);
    x_e.deserialize(x_e_bytes);
    return decrypt_to_share(x_e, n_channal);
}

void ShareToEncClient::client_share_to_enc(Feature2DShare& share, const Bytes& meta_data_bytes) {
    DataTransmission data_trans = ::mpc::data_transmission();
    uint8_t level;
    uint32_t n_channel;
    bytes_to_va(meta_data_bytes, {"u8", "u32"}, &level, &n_channel);

    Feature2DEncrypted x_e(&context_out_, level, {1, 1}, {1, 1}, PackType::MultipleChannelPacking);
    for (int i = 0; i < share.data.get_size(); i++) {
        share.data.set(i, share.data.get(i) * mpc::T_SCALE % ring_mod_);
    }
    auto data_process = encrypt_from_share(x_e, share, n_channel, share.shape, PackType::MultipleChannelPacking);

    MPC mpc_protocol(scale_ord_, ring_mod_, pt_range_);
    auto b0 = mpc_protocol.wrap_protocol(data_process.to_array_1d(), ::mpc::current_party());

    Array<double, 1> b0_mult_mod_div_s_mg(share.data.get_shape());
    double scale = mpc::DEFAULT_SCALE;
    for (int i = 0; i < b0.size(); i++) {
        double temp_res = double(b0[i] * ring_mod_) / scale;
        b0_mult_mod_div_s_mg.set(i, temp_res);
    }
    CkksContext& ctx_extra = context_out_.get_extra_level_context();
    Feature2DEncrypted send_ct(&ctx_extra, level + 1, {1, 1}, {1, 1}, PackType::MultipleChannelPacking);
    send_ct.pack_multiple_channel(
        b0_mult_mod_div_s_mg.reshape<3>({(uint64_t)n_channel, (uint64_t)share.shape[0], (uint64_t)share.shape[1]}),
        false, mpc::DEFAULT_SCALE);
    data_trans.send_bytes(x_e.serialize());
    data_trans.send_bytes(send_ct.serialize());
}

void ShareToEncClient::client_share_to_enc_simple(Feature2DShare& share, const Bytes& meta_data_bytes) {
    DataTransmission data_trans = ::mpc::data_transmission();
    uint8_t level;
    uint32_t n_channel;
    bytes_to_va(meta_data_bytes, {"u8", "u32"}, &level, &n_channel);

    Feature2DEncrypted x_e(&context_out_, level, {1, 1}, {1, 1}, PackType::MultipleChannelPacking);
    encrypt_from_share_simple(x_e, share, n_channel, share.shape, PackType::MultipleChannelPacking,
                              mpc::MPC_REFRESH_USE_RECODE);

    data_trans.send_bytes(x_e.serialize());
}

void ShareToEncClient::client_share_to_enc_for_multi_channel_pack(Feature2DShare& share,
                                                                  const Bytes& meta_data_bytes) {
    DataTransmission data_trans = ::mpc::data_transmission();
    uint8_t level;
    uint32_t n_channel;
    Duo skip;

    uint8_t temp_int = 0;
    bytes_to_va(meta_data_bytes, {"u8", "u32", "duo", "u8"}, &level, &n_channel, &skip, &temp_int);
    PackType pack_type = (PackType)temp_int;
    Feature2DEncrypted x_e(&context_out_, level, skip);
    for (int i = 0; i < share.data.get_size(); i++) {
        share.data.set(i, share.data.get(i) * mpc::T_SCALE % ring_mod_);
    }
    auto data_process = encrypt_from_share(x_e, share, n_channel, share.shape, pack_type);

    MPC mpc_protocol(scale_ord_, ring_mod_, pt_range_);
    auto b0 = mpc_protocol.wrap_protocol(data_process.to_array_1d(), ::mpc::current_party());

    Array<double, 1> b0_mult_mod_div_s_mg(share.data.get_shape());
    double scale = mpc::DEFAULT_SCALE;
    for (int i = 0; i < b0.size(); i++) {
        double temp_res = double(b0[i] * ring_mod_) / scale;
        b0_mult_mod_div_s_mg.set(i, temp_res);
    }
    CkksContext& ctx_extra = context_out_.get_extra_level_context();
    Feature2DEncrypted send_ct(&ctx_extra, level + 1, skip, {1, 1}, pack_type);
    auto send_mg =
        b0_mult_mod_div_s_mg.reshape<3>({(uint64_t)n_channel, (uint64_t)share.shape[0], (uint64_t)share.shape[1]});
    if (pack_type == PackType::MultipleChannelPacking) {
        send_ct.pack_multiple_channel(send_mg, false, mpc::DEFAULT_SCALE);
    } else if (pack_type == PackType::MultiplexedPacking) {
        send_ct.pack_multiplexed(send_mg, false, mpc::DEFAULT_SCALE);
    } else if (pack_type == PackType::InterleavedPacking) {
        Duo block_expansion = {(uint32_t)ceil(share.shape[0] / (double)BLOCK_SHAPE[0]),
                               (uint32_t)ceil(share.shape[1] / (double)BLOCK_SHAPE[1])};
        send_ct.pack_interleaved(send_mg, BLOCK_SHAPE, block_expansion, false, mpc::DEFAULT_SCALE);
    }
    data_trans.send_bytes(x_e.serialize());
    data_trans.send_bytes(send_ct.serialize());
}

void ShareToEncClient::client_share_to_enc_for_multi_channel_pack_simple(Feature2DShare& share,
                                                                         const Bytes& meta_data_bytes) {
    DataTransmission data_trans = ::mpc::data_transmission();
    uint8_t level;
    uint32_t n_channel;
    Duo skip;

    uint8_t temp_int = 0;
    bytes_to_va(meta_data_bytes, {"u8", "u32", "duo", "u8"}, &level, &n_channel, &skip, &temp_int);
    PackType pack_type = (PackType)temp_int;
    cout << "[mpc_refresh][client] share_to_enc pack_type=" << static_cast<int>(pack_type)
         << ", level=" << static_cast<int>(level) << ", n_channel=" << n_channel
         << ", skip=(" << skip[0] << "," << skip[1] << ")" << endl;
    Feature2DEncrypted x_e(&context_out_, level, skip, {1, 1}, pack_type);
    encrypt_from_share_simple(x_e, share, n_channel, share.shape, pack_type, mpc::MPC_REFRESH_USE_RECODE);

    data_trans.send_bytes(x_e.serialize());
}

void ShareToEncClient::client_share_to_enc_0d(Feature0DShare& share, const Bytes& meta_data_bytes) {
    DataTransmission data_trans = ::mpc::data_transmission();
    uint8_t level = 0;
    uint32_t n_channel = 0;
    bytes_to_va(meta_data_bytes, {"u8", "u32"}, &level, &n_channel);
    Feature0DEncrypted x_e(&context_out_, level);
    x_e.skip = 1;
    for (int i = 0; i < share.data.get_size(); i++) {
        share.data.set(i, share.data.get(i) * mpc::T_SCALE % ring_mod_);
    }
    auto data_process = encrypt_from_share(x_e, share, n_channel);
    MPC mpc_protocol(scale_ord_, ring_mod_, pt_range_);
    data_trans.flush();
    auto b0 = mpc_protocol.wrap_protocol(data_process.to_array_1d(), ::mpc::current_party());
    data_trans.flush();

    Array<double, 1> send_mg(share.data.get_shape());
    double scale = mpc::DEFAULT_SCALE;
    for (int i = 0; i < b0.size(); i++) {
        double temp_res = double(b0[i] * ring_mod_) / scale;
        send_mg.set(i, temp_res);
    }
    CkksContext& ctx_extra = context_out_.get_extra_level_context();
    Feature0DEncrypted send_ct(&ctx_extra, level + 1);
    send_ct.skip = 1;
    double encode_scale = pow(2, mpc::DEFAULT_SCALE_BIT);
    send_ct.pack_cyclic(send_mg.to_array_1d(), false, encode_scale);
    data_trans.send_bytes(x_e.serialize());
    data_trans.send_bytes(send_ct.serialize());
}

Feature2DShare EncToShareServer::server_enc_to_share_multi_pack(const Feature2DEncrypted& x_enc,
                                                                 PackType pack_type) {
    DataTransmission data_trans = ::mpc::data_transmission();

    Feature2DEncrypted x_share1_enc(&context_, x_enc.level, x_enc.skip, x_enc.invalid_fill, pack_type);
    Feature2DShare x_share0(ring_mod_, scale_ord_);

    split_to_shares_for_multi_channel_pack(x_enc, &x_share1_enc, &x_share0, pack_type);
    data_trans.send_bytes(x_share1_enc.serialize());
    data_trans.flush();

    return x_share0;
}

Feature2DShare EncToShareServer::server_enc_to_share_multi_pack_simple(const Feature2DEncrypted& x_enc,
                                                                       PackType pack_type) {
    DataTransmission data_trans = ::mpc::data_transmission();

    Feature2DEncrypted x_share1_enc(&context_, x_enc.level, x_enc.skip, x_enc.invalid_fill, pack_type);
    Feature2DShare x_share0(ring_mod_, scale_ord_);

    split_to_shares_for_multi_channel_pack_simple(x_enc, &x_share1_enc, &x_share0, pack_type);
    data_trans.send_bytes(x_share1_enc.serialize());
    data_trans.flush();

    return x_share0;
}

Feature2DEncrypted ShareToEncServer::server_share_to_enc_multi_pack(Feature2DShare& y_share0,
                                                                    int level,
                                                                    PackType pack_type) {
    DataTransmission data_trans = ::mpc::data_transmission();
    for (int i = 0; i < y_share0.data.get_size(); i++) {
        y_share0.data.set(i, (y_share0.data.get(i) * mpc::T_SCALE) % ring_mod_);
    }

    MPC mpc_protocol(scale_ord_, ring_mod_, pt_range_);
    auto b1 = mpc_protocol.wrap_protocol(y_share0.data.to_array_1d(), SERVER);

    Feature2DEncrypted y_share1_enc(&context_, level, {1, 1}, {1, 1}, pack_type);
    y_share1_enc.deserialize(data_trans.receive_bytes());
    y_share1_enc.packing_type = pack_type;
    y_share1_enc.decompress();

    CkksContext& extra_context = context_.get_extra_level_context();
    Feature2DEncrypted y_share2_enc(&extra_context, level + 1, y_share1_enc.skip, {1, 1}, pack_type);
    y_share2_enc.deserialize(data_trans.receive_bytes());
    y_share2_enc.packing_type = pack_type;

    Feature2DEncrypted y_ct = combine_with_share_new_protocol_for_multi_pack(y_share1_enc, y_share0, y_share2_enc, b1,
                                                                             pack_type);
    y_ct.packing_type = pack_type;
    return y_ct;
}

Feature2DEncrypted ShareToEncServer::server_share_to_enc_multi_pack_simple(Feature2DShare& y_share0,
                                                                           int level,
                                                                           PackType pack_type) {
    DataTransmission data_trans = ::mpc::data_transmission();
    Feature2DEncrypted y_share1_enc(&context_, level, {1, 1}, {1, 1}, pack_type);
    y_share1_enc.deserialize(data_trans.receive_bytes());
    y_share1_enc.packing_type = pack_type;
    y_share1_enc.decompress();

    Feature2DEncrypted y_ct = combine_with_share_simple_for_multi_pack(y_share1_enc, y_share0, pack_type);
    y_ct.packing_type = pack_type;
    return y_ct;
}

Feature2DEncrypted ShareToEncServer::server_share_to_enc_simple(Feature2DShare& y_share0, int level) {
    DataTransmission data_trans = ::mpc::data_transmission();
    Feature2DEncrypted y_share1_enc(&context_, level, {1, 1}, {1, 1}, PackType::MultipleChannelPacking);
    y_share1_enc.deserialize(data_trans.receive_bytes());
    y_share1_enc.packing_type = PackType::MultipleChannelPacking;
    y_share1_enc.decompress();

    Feature2DEncrypted y_ct = combine_with_share_simple(y_share1_enc, y_share0);
    y_ct.packing_type = PackType::MultipleChannelPacking;
    return y_ct;
}

Feature2DShare EncToShareServer::server_enc_to_share(const Feature2DEncrypted& x_enc) {
    DataTransmission data_trans = ::mpc::data_transmission();

    Feature2DEncrypted x_share1_enc(&context_, x_enc.level, x_enc.skip, x_enc.invalid_fill,
                                    PackType::MultipleChannelPacking);
    Feature2DShare x_share0(ring_mod_, scale_ord_);

    split_to_shares(x_enc, &x_share1_enc, &x_share0);
    data_trans.send_bytes(x_share1_enc.serialize());

    return x_share0;
}

Feature2DShare EncToShareServer::server_enc_to_share_simple(const Feature2DEncrypted& x_enc) {
    DataTransmission data_trans = ::mpc::data_transmission();

    Feature2DEncrypted x_share1_enc(&context_, x_enc.level, x_enc.skip, x_enc.invalid_fill,
                                    PackType::MultipleChannelPacking);
    Feature2DShare x_share0(ring_mod_, scale_ord_);

    split_to_shares_simple(x_enc, &x_share1_enc, &x_share0);
    data_trans.send_bytes(x_share1_enc.serialize());

    return x_share0;
}

Feature0DShare EncToShareServer::server_enc_to_share(const Feature0DEncrypted& x_enc) {
    DataTransmission data_trans = ::mpc::data_transmission();

    Feature0DShare x_share0(ring_mod_, scale_ord_);
    Feature0DEncrypted x_share1_enc(&context_, x_enc.level);

    split_to_shares(x_enc, &x_share1_enc, &x_share0);
    data_trans.send_bytes(x_share1_enc.serialize());

    return x_share0;
}

Feature0DEncrypted ShareToEncServer::share_to_enc(Feature0DShare& y_share0, int level) {
    DataTransmission data_trans = ::mpc::data_transmission();
    for (int i = 0; i < y_share0.data.get_size(); i++) {
        y_share0.data.set(i, (y_share0.data.get(i) * mpc::T_SCALE) % ring_mod_);
    }

    MPC mpc_protocol(scale_ord_, ring_mod_, pt_range_);
    auto b1 = mpc_protocol.wrap_protocol(y_share0.data.to_array_1d(), SERVER);

    Feature0DEncrypted y_share1_enc(&context_, level);
    y_share1_enc.deserialize(data_trans.receive_bytes());
    y_share1_enc.decompress();

    CkksContext& extra_context = context_.get_extra_level_context();
    Feature0DEncrypted y_share2_enc(&extra_context, level + 1);
    y_share2_enc.deserialize(data_trans.receive_bytes());

    return combine_with_share_new_protocol(y_share1_enc, y_share0, y_share2_enc, b1);
}
